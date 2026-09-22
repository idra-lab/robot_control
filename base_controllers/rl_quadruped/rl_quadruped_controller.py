# -*- coding: utf-8 -*-
"""State-machine controller running the Isaac Lab velocity policy on the Aliengo.

The controller is deliberately thin.  The policy trained in
``safe_rl/tasks/manager_based/velocity`` is a single fully proprioceptive network - it regresses the
base linear velocity internally - so this controller needs no state estimator, no leg odometry, no
inverse kinematics and no whole-body controller.  What it does need is a reliable sequence for
getting the robot from lying on the floor to walking and back, and a way out if something goes wrong.

State machine::

    INIT ──▶ CALIBRATE ──▶ FOLD ──▶ STAND_UP ──▶ STANDING
                              ▲                          │
                              │                          ▼
                              └── STAND_DOWN ◀───── RL ⇄ SAFE_STOP

    any state ──[emergency]──▶ DAMPING ──▶ DONE

* **INIT** pushes the joint PD gains (no integral term anywhere) and latches the *measured* posture,
  so nothing moves when the controller takes over.
* **CALIBRATE** estimates the accelerometer bias, and it runs before any commanded motion.  The
  bias is only observable while the base is not accelerating, so the procedure assumes the robot
  has been placed flat and left alone, and commands nothing but a hold of the posture it was placed
  in.  The policy consumes raw specific force, so this bias goes straight into the observation.
  ``--skip-calibration`` leaves it at zero.
* **FOLD** eases into the resting posture once the bias is known, on the soft gains, because the
  posture the robot was placed in is generally not one it can hold indefinitely.  It then *stays*
  there waiting for a command, and it is where a stand-down returns to - so it is the robot's
  resting state, not a step on the way to one.
* **STAND_UP** / **STAND_DOWN** walk a list of joint waypoints with quintic interpolation (zero
  velocity and acceleration at every waypoint).
* **RL** runs the walking policy at 50 Hz on top of the faster control loop and follows the
  operator's velocity command.
* **SAFE_STOP** runs a second network trained in the same environment with the command range pinned
  to zero and much harsher resets and pushes: a standstill that actively rejects disturbances.  It
  has its own operator command and is *not* the emergency path - the robot stays standing.  Because
  the two variants share the observation contract, the hand-over is a pointer move on one
  continuous history and can happen mid-stride, in either direction.
* **DAMPING** is the emergency path: gains drop to pure joint damping so the robot sinks under its
  own weight instead of dropping, and any feed-forward torque is ramped out.  It is reachable from
  every state, including the safe stop, so the softer option never costs the operator the harder one.

Run it::

    python3 rl_quadruped_controller.py                 # gazebo, keyboard
    python3 rl_quadruped_controller.py --input joy     # gazebo, Xbox pad
    python3 rl_quadruped_controller.py --real --input joy

See ``--help`` for the rest.  The control loop is always 500 Hz and the policy always 50 Hz
(decimation 10, checked rather than assumed).  On the real robot pass ``--real``: the controller
then drops every computation the policy does not need, clips commands to the joint limits and asks
for a real-time scheduling class.
"""

import os

# Must precede numpy: the policy is a small MLP, and letting the BLAS/OMP pools spin up threads only
# adds jitter to a hard-real-time loop.  onnxruntime is pinned to one thread separately.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import gc
import queue
import sys
import threading
import time as wall_time
from enum import Enum

import numpy as np
import pinocchio as pin
import rospkg
import rospy as ros
from gazebo_msgs.srv import ApplyBodyWrench, ApplyBodyWrenchRequest, GetLinkProperties
from geometry_msgs.msg import TwistStamped, Vector3, Vector3Stamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64, Float64MultiArray, MultiArrayDimension, String
from termcolor import colored
from tf.transformations import euler_from_quaternion

import base_controllers.rl_quadruped.rl_controller_config as conf
from base_controllers.base_controller import BaseController
from base_controllers.rl_quadruped.rl_controller_config import get_config
from base_controllers.utils.common_functions import getRobotModelFloating
from base_controllers.components.imu_utils import IMU_utils
from base_controllers.rl_quadruped.velocity_policy import VelocityPolicy
from base_controllers.rl_quadruped import training_model
from base_controllers.rl_quadruped.commandInterface import Event, create_command_interface
from base_controllers.utils.pidManager import PidManager
from ros_impedance_controller.msg import EffortPid


CONTROL_RATE_HZ = 500.0
"""Rate the control loop runs at.  Fixed: it has to be an integer multiple of the policy rate."""

POLICY_RATE_HZ = 50.0
"""Rate the policy was trained at (Isaac: decimation 4 on a 5 ms physics step = 20 ms).  Everything
the network sees is tied to this period, so it is not a free parameter."""


class State(Enum):
    INIT = "init"
    CALIBRATE = "calibrate"
    FOLD = "fold"
    STAND_UP = "stand_up"
    STANDING = "standing"
    RL = "rl"
    SAFE_STOP = "safe_stop"
    STAND_DOWN = "stand_down"
    DAMPING = "damping"
    DONE = "done"


POLICY_STATES = (State.RL, State.SAFE_STOP)
"""States in which a network is driving the joints.  A transition *between* them is a hand-over,
not a start: the observation history has to survive it."""


_ZERO_COMMAND = np.zeros(3)
"""The command the safe stop feeds its policy.  Module-level so the hot path allocates nothing."""


def quintic(alpha):
    """Quintic scaling and its time-normalised derivative for ``alpha`` in [0, 1].

    ``s(0) = 0``, ``s(1) = 1`` with zero first and second derivative at both ends, so a waypoint is
    entered and left at rest - what you want when the feet are loaded.
    """
    alpha = min(max(alpha, 0.0), 1.0)
    a2 = alpha * alpha
    s = a2 * alpha * (10.0 - 15.0 * alpha + 6.0 * a2)
    sd = 30.0 * a2 * (1.0 - alpha) * (1.0 - alpha)
    return s, sd


class RlQuadrupedController(BaseController):
    """Minimal quadruped controller whose walking gait comes from an ONNX policy."""

    def __init__(self, robot_name="aliengo", input_device="keyboard", policy_name=None,
                 real_robot=False, realtime=None, telemetry=None, pin_cpu=None,
                 skip_calibration=None, use_rviz=None, estimate_velocity=None,
                 inference_timing=None):
        # Resolve the configuration *before* the base class runs: rl_controller_config is handed in
        # as external_conf, which makes BaseController read this module's robot_params instead of
        # params.py.  From here on params.py plays no part in this controller.
        self.cfg = get_config(robot_name, real_robot=real_robot)
        super(RlQuadrupedController, self).__init__(robot_name, external_conf=conf,
                                                   broadcast_world=False)
        self.input_device = input_device

        if policy_name is not None:
            self.policy_name = policy_name
        elif self.cfg["policy_name"] is not None:
            self.policy_name = self.cfg["policy_name"]
        else:
            self.policy_name = robot_name

        # The control rate is fixed at 500 Hz and the policy at 50 Hz, so the decimation is 10.
        # Both are checked rather than assumed: a control period that is not an exact multiple of
        # the policy period puts the joint velocities, the IMU history and the action history on a
        # different time base than training, which the policy has no way to signal.
        self.dt = self.cfg["control_dt"]
        expected_dt = 1.0 / CONTROL_RATE_HZ
        if abs(self.dt - expected_dt) > 1e-12:
            raise ValueError(
                f"control_dt is {self.dt * 1e3:.4f} ms but this controller runs at "
                f"{CONTROL_RATE_HZ:.0f} Hz ({expected_dt * 1e3:.4f} ms). Change CONTROL_RATE_HZ in "
                f"{__file__} if you really mean to run at another rate, and keep it an integer "
                f"multiple of {POLICY_RATE_HZ:.0f} Hz.")
        self.expected_decimation = int(round(CONTROL_RATE_HZ / POLICY_RATE_HZ))
        print(colored(f"Control loop at {1.0 / self.dt:.0f} Hz, policy at "
                      f"{POLICY_RATE_HZ:.0f} Hz, decimation {self.expected_decimation} "
                      f"({'real robot' if self.real_robot else 'simulation'})", "yellow"))

        def resolve(name, override):
            value = self.cfg[name] if override is None else override
            return self.real_robot if value is None else value

        self.realtime = resolve("realtime", realtime)
        self.pin_cpu = self.cfg["pin_cpu"] if pin_cpu is None else pin_cpu
        self.clip_to_joint_limits = resolve("clip_to_joint_limits", None)
        self.enable_telemetry = (self.cfg["publish_telemetry"] if telemetry is None
                                 else telemetry)
        # Two independent diagnostics, both off the control path and both switchable: the actor's
        # state-estimation head, and the per-tick inference timing.  Resolved here rather than read
        # from self.cfg at each use so there is one answer for the whole run.
        self.estimate_velocity = (self.cfg["estimate_base_velocity"] if estimate_velocity is None
                                  else bool(estimate_velocity))
        self.measure_inference_timing = (self.cfg["measure_inference_timing"]
                                         if inference_timing is None else bool(inference_timing))

        self._page_size = os.sysconf("SC_PAGE_SIZE")
        self.state = State.INIT
        self.prev_state = None
        self.state_start_time = 0.0
        self.state_first_tick = True
        self._quit_requested = False
        self._fold_moving = False
        self._last_push = np.zeros(3)
        self._push_stamp = -1e9
        self.push_enabled = False
        # Whether the robot is already in the fold posture.  Tracked explicitly rather than
        # inferred from a joint distance: the fold is held at kp_fold against gravity and the
        # contacts, so the measured posture sits about 0.17 rad from the target even when the
        # target *is* q_fold, while the start-up posture is 0.89 rad away - and picking a threshold
        # between two numbers that close is guessing.  What matters is not where the joints are but
        # whether the fold has been commanded and not left since.
        self._folded = False
        self._calibrated = False
        self.skip_calibration = (self.cfg["skip_calibration"] if skip_calibration is None
                                 else bool(skip_calibration))
        self.use_rviz = self.cfg["use_rviz"] if use_rviz is None else bool(use_rviz)
        self.publish_tf = self.cfg["publish_tf"]
        self._zero_cmd_since = None

    # ---------------------------------------------------------------------------------------------
    # setup
    # ---------------------------------------------------------------------------------------------
    def check_faulty_ping(self, ip):
        """Return a non-zero code when the robot does not answer, mirroring QuadrupedController."""
        response = os.system("ping -c 1 -W 1 " + ip)
        if response == 0:
            print(colored(f"Robot network active: ping {ip} ok", "green"))
        else:
            print(colored(f"Cannot ping {ip}: bring up the local network with that gateway", "red"))
        return response

    def startController(self, world_name=None, additional_args=None):
        if self.real_robot:
            self.use_ground_truth_contacts = False
            if self.check_faulty_ping(conf.robot_params[self.robot_name]['ip']):
                sys.exit("Cannot reach the robot")
        else:
            self.use_ground_truth_contacts = False

        if world_name is None:
            world_name = self.cfg["world_name"]
        if not self.real_robot and self.cfg["fix_spawn_height"]:
            self._fixSpawnHeight()

        args = ["rviz:=" + ("true" if self.use_rviz else "false")]
        if self.use_rviz and self.cfg["rviz_conf"] is not None:
            args.append("rviz_conf:=" + str(self.cfg["rviz_conf"]))
        if self.real_robot:
            args.append("task_period:=0.002")
        if additional_args:
            args.extend(additional_args)

        self.startSimulator(world_name=None if self.real_robot else world_name,
                            additional_args=args)
        self.loadModelAndPublishers()
        # The policy is loaded before initVars: it carries the training default posture, which is
        # both the stand-up target and the offset the actions are applied to.
        self.policy = VelocityPolicy(self.policy_name, dt=self.dt,
                                     enable_estimator=self.estimate_velocity,
                                     measure_timing=self.measure_inference_timing)
        self.initVars()
        self.initSubscribers()
        self._initCommandMessage()
        self.rate = ros.Rate(1.0 / self.dt)

        if abs(self.policy.policy_rate - POLICY_RATE_HZ) > 1e-9:
            raise ValueError(
                f"{self.policy_name} was exported for {self.policy.policy_rate:.1f} Hz but this "
                f"controller assumes {POLICY_RATE_HZ:.1f} Hz; fix 'policy_rate' in the policy json "
                f"or POLICY_RATE_HZ in {__file__}.")
        if self.policy.decimation != self.expected_decimation:
            raise ValueError(
                f"Policy decimation is {self.policy.decimation}, expected "
                f"{self.expected_decimation} for {CONTROL_RATE_HZ:.0f} Hz control and "
                f"{POLICY_RATE_HZ:.0f} Hz policy")

        # Fail here rather than at the moment the operator asks for a safe stop: that request is
        # made when something is already going wrong, and it is not a moment to discover that the
        # variant was never exported.
        self.safe_variant = self.cfg["safe_policy_variant"]
        if self.safe_variant not in self.policy.networks:
            raise ValueError(
                f"No policy variant '{self.safe_variant}' in {self.policy_name}: the contract "
                f"offers {self.policy.variants}. Add it to 'onnx_variants' in the policy json, or "
                f"set 'safe_policy_variant' in rl_controller_config to one of those.")
        print(colored(f"Safe stop will run the '{self.safe_variant}' variant "
                      f"({os.path.basename(self.policy.networks[self.safe_variant].model_path)}) "
                      f"with the command held at zero", "yellow"))
        self._warnIfSafeVariantIsACopy()

        if self.cfg["kp_rl"] is None:
            self.cfg["kp_rl"] = np.full(self.robot.na, self.policy.kp)
        if self.cfg["kd_rl"] is None:
            self.cfg["kd_rl"] = np.full(self.robot.na, self.policy.kd)

        self._checkImpedanceController()
        self._checkTrainingModel()
        # Benchmark before the loop starts, while nothing is competing: this separates what the
        # policy costs from what the machine does to it.  Part of the timing instrumentation, so it
        # goes with it.
        if self.measure_inference_timing:
            self.policy.benchmark()
            print(colored(f"Policy inference benchmark: "
                          f"{self.policy.benchmark_ms * 1e3:.0f} us of the "
                          f"{self.dt * 1e6:.0f} us control tick "
                          f"({100.0 * self.policy.benchmark_ms / (self.dt * 1e3):.1f} %), fastest "
                          f"of 400 back-to-back runs; median "
                          f"{self.policy.benchmark_median_ms * 1e3:.0f} us", "cyan"))
        else:
            print(colored("Inference timing is off: no benchmark, no policy/infer_* topics and "
                          "no timing summary at shutdown", "yellow"))
        if not self.estimate_velocity:
            print(colored("The state-estimation head is off: the policy still uses its internal "
                          "estimate, but it is not evaluated separately or published", "yellow"))
        self.command = create_command_interface(self.input_device, cfg=self.cfg)
        self._initTelemetry()
        # After the publishers: the push worker publishes the disturbance it applies.
        self._initPushes()
        self._waitForSensors()
        self.pid = PidManager(self.joint_names)
        if self.realtime:
            if not self.real_robot:
                print(colored("Real-time scheduling is on in simulation. Gazebo is essential work "
                              "here, so putting this process above it makes timing worse, not "
                              "better - it starves the simulator that feeds us. Prefer the default "
                              "(real-time on only with --real).", "yellow"))
            self._applyRealtimeTuning()
        # Baseline after everything is allocated, so growth measured against it is real growth.
        self._rss_baseline_mb = self._rssMb()
        print(colored(f"RlQuadrupedController ready (resident memory "
                      f"{self._rss_baseline_mb:.1f} MB)", "green"))

    def _warnIfSafeVariantIsACopy(self):
        """Say so when the safe stop is the walking policy under another filename.

        Two different files is the whole premise of the safe stop: the operator reaches for it when
        something is already going wrong, and what makes it worth reaching for is that it was
        trained on zero command with far harsher resets and pushes, so it covers failures the
        walking policy shares rather than repeating them.  Copying one file over the other is a
        perfectly reasonable placeholder while the recovery task is retrained - it still holds
        station - but it is invisible from the outside, and 'the fallback is the thing it was
        supposed to fall back from' is not something to discover mid-run.
        """
        import hashlib

        def digest(name):
            with open(self.policy.networks[name].model_path, "rb") as handle:
                return hashlib.md5(handle.read()).hexdigest()

        default = self.policy.default_variant
        if self.safe_variant == default:
            return
        try:
            if digest(self.safe_variant) != digest(default):
                return
        except OSError:
            return
        print(colored(
            f"WARNING: the '{self.safe_variant}' network is byte-for-byte identical to "
            f"'{default}'. The safe stop will hold station, but it is the walking policy at zero "
            f"command, not a recovery policy - it shares the walking policy's failure modes "
            f"instead of covering them. Export the recovery task over "
            f"{os.path.basename(self.policy.networks[self.safe_variant].model_path)} to restore "
            f"it.", "red", attrs=["bold"]))

    def _initCommandMessage(self):
        """Preallocate the /command message and the clipping bounds.

        At 250 Hz a fresh JointState per tick is pure allocation churn, and the base-class clipping
        path recomputes the soft limits from the model on every call.  Both are hoisted out here.
        """
        self._cmd_msg = JointState()
        self._cmd_msg.name = self.joint_names
        soft = 0.9
        na = self.robot.na
        lower = self.robot.model.lowerPositionLimit[-na:] * soft
        upper = self.robot.model.upperPositionLimit[-na:] * soft
        # A soft factor on a signed bound must always shrink the interval, so order the pair.
        self._q_min = np.minimum(lower, upper)
        self._q_max = np.maximum(lower, upper)
        self._qd_max = self.robot.model.velocityLimit[-na:] * soft
        self._tau_max = self.robot.model.effortLimit[-na:] * soft

    def send_des_jstate(self, q_des, qd_des, tau_ffwd, soft_limits=0.9, clip_commands=False):
        """Publish the joint command, reusing one message and precomputed limits."""
        msg = self._cmd_msg
        if clip_commands:
            msg.position = np.clip(q_des, self._q_min, self._q_max)
            msg.velocity = np.clip(qd_des, -self._qd_max, self._qd_max)
            msg.effort = np.clip(tau_ffwd, -self._tau_max, self._tau_max)
        else:
            msg.position = q_des
            msg.velocity = qd_des
            msg.effort = tau_ffwd
        self.pub_des_jstate.publish(msg)

    def _checkImpedanceController(self):
        """Confirm the low-level joint controller implements the law the policy was trained with.

        ``ros_impedance_controller`` computes, in its default branch::

            tau = kp (q_des - q) + kd (qd_des - qd) + ki integral(q_des - q) + tau_ffwd

        which with ``ki = 0`` is exactly the PD actuator Isaac used in training, so the policy's
        gains carry over directly.  Its other branch, selected by ``/pid_discrete_implementation``,
        instead takes a filtered derivative of the *position error* - a different controller, which
        the policy has never seen.  This never fails loudly on its own, so it is checked here.
        """
        discrete = ros.get_param("/pid_discrete_implementation", False)
        if discrete:
            print(colored("WARNING: /pid_discrete_implementation is true, so the joint controller "
                          "differentiates the position error instead of using (qd_des - qd). The "
                          "policy was trained against tau = kp(q_des - q) + kd(qd_des - qd); "
                          "relaunch with pid_discrete_implementation:=false.", "red",
                          attrs=["bold"]))
        else:
            print(colored("Joint controller: tau = kp(q_des - q) + kd(qd_des - qd) + tau_ffwd, "
                          "ki = 0 - matches the training actuator", "green"))

    def initSubscribers(self):
        self.sub_jstate = ros.Subscriber("/" + self.robot_name + "/joint_states", JointState,
                                         callback=self._receive_jstate, queue_size=1,
                                         tcp_nodelay=True)
        self.sub_pid_effort = ros.Subscriber("/" + self.robot_name + "/effort_pid", EffortPid,
                                             callback=self._receive_pid_effort, queue_size=1,
                                             tcp_nodelay=True)
        if self.real_robot:
            # The hardware interface splits the IMU: orientation and rate on /imu, the raw
            # accelerometer on /trunk_imu as a bare Vector3.
            self.sub_imu = ros.Subscriber("/" + self.robot_name + "/imu", Imu,
                                          callback=self._receive_imu, queue_size=1,
                                          tcp_nodelay=True)
            self.sub_imu_acc = ros.Subscriber("/" + self.robot_name + "/trunk_imu", Vector3,
                                              callback=self._receive_imu_acc_real, queue_size=1,
                                              tcp_nodelay=True)
        else:
            # Gazebo publishes one complete Imu message on /trunk_imu.
            self.sub_imu = ros.Subscriber("/" + self.robot_name + "/trunk_imu", Imu,
                                          callback=self._receive_imu_sim, queue_size=1,
                                          tcp_nodelay=True)
            # Ground truth is logged for reference only; nothing in the loop reads it.
            self.sub_pose = ros.Subscriber("/" + self.robot_name + "/ground_truth", Odometry,
                                           callback=self._receive_ground_truth, queue_size=1,
                                           tcp_nodelay=True)


    def initVars(self):
        """Allocate the controller's state.

        Deliberately does **not** call ``BaseController.initVars``: that sizes about twenty logging
        arrays from ``buffer_size`` (some 70 MB for the stock 50001-sample buffer) plus a pile of
        whole-body-control state this controller never touches.  Telemetry goes out over ROS topics
        instead, so nothing here grows with runtime and the loop has a fixed memory footprint.
        """
        na = self.robot.na
        self.time = 0.0

        # measured state
        self.q = np.zeros(na)
        self.qd = np.zeros(na)
        self.tau = np.zeros(na)
        self.tau_fb = np.zeros(na)
        self.quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        self.euler = np.zeros(3)
        self.angVelB = np.zeros(3)
        self.baseLinAccB = np.zeros(3)
        self.projected_gravity = np.array([0.0, 0.0, -1.0])
        self.b_R_w = np.eye(3)
        self.basePoseW = np.zeros(6)
        self.baseTwistW = np.zeros(6)
        self.gt_baseTwistW = np.zeros(6)

        # commands
        self.q_des = np.zeros(na)
        self.qd_des = np.zeros(na)
        self.tau_ffwd = np.zeros(na)

        self.imu_utils = IMU_utils(timeout=int(round(self.cfg["calibration_duration"] / self.dt)),
                                   dt=self.dt)

        self.velocity_cmd = np.zeros(3)
        self.rl_action = np.zeros(na)
        self.q_policy = np.zeros(na)

        self.q_fold = np.asarray(self.cfg["q_fold"], dtype=np.float64)
        q_stand = self.cfg["q_stand"]
        self.q_stand = np.asarray(self.policy.q_default if q_stand is None else q_stand,
                                  dtype=np.float64)
        self.stand_up_waypoints = ([self.q_fold, self.q_stand] if self.cfg["stand_up_via_fold"]
                                   else [self.q_stand])
        self.stand_down_waypoints = [self.q_fold]

        self._traj_from = np.zeros(na)
        self._traj_to = np.zeros(na)
        self._traj_index = 0
        self._sensors_ready = False
        self._got_jstate = False
        self._got_imu = False
        self._got_imu_acc = False
        self._last_inference_count = 0
        self._active_gains = None

        # loop health, tracked as running aggregates rather than a history
        self.loop_dt = self.dt
        self._last_loop_stamp = None
        self.loop_overruns = 0
        self.worst_loop_dt = 0.0
        self.tick_count = 0
        self._stats_dt_sum = 0.0
        self._stats_dt_max = 0.0
        self._stats_ticks = 0
        self._stats_overruns = 0
        self._stats_compute_ticks = 0
        # Computation time per tick, i.e. everything except rate.sleep().  Unlike the loop period
        # this means the same thing in simulation and on hardware: in Gazebo the period is set by
        # the simulator's real-time factor, because rate.sleep() sleeps on sim time.
        self.compute_dt = 0.0
        self._stats_compute_sum = 0.0
        self._stats_compute_max = 0.0
        # Steady-state worst case, excluding the first tick of each state.  A state entry pays for a
        # /set_pids service round-trip and a full garbage collection, which is the whole reason those
        # are done at a transition; lumping them in would hide the number that actually matters.
        self.worst_compute_dt = 0.0
        self.worst_entry_compute_dt = 0.0
        self._rss_baseline_mb = None
        self._memory_warned = False

        # calibration accumulators
        self._calib_sum = np.zeros(3)
        self._calib_sq_sum = np.zeros(3)
        self._calib_acc_sum = np.zeros(3)
        self._calib_n = 0
        self._calib_stable_since = None
        self._calib_warned = -1e9

    def _standHeight(self, q, robot=None):
        """Distance from the feet to the base frame for a joint configuration, in metres."""
        robot = self.robot if robot is None else robot
        fb = np.hstack([pin.neutral(robot.model)[0:7], np.asarray(q, dtype=np.float64)])
        pin.forwardKinematics(robot.model, robot.data, fb)
        pin.updateFramePlacements(robot.model, robot.data)
        feet_z = [robot.data.oMf[robot.model.getFrameId(frame)].translation[2]
                  for frame in conf.robot_params[self.robot_name]['ee_frames']]
        return -float(np.mean(feet_z))

    def _srdfHomePosture(self):
        """The joint posture the launch actually starts the robot in, or None if unreadable.

        ``spawn_model`` places the base at ``spawn_z`` while the description's ``go0`` script
        independently applies the SRDF ``home`` group state - and ignores the ``go0_conf`` argument
        the launch hands it.  So ``home`` is what the robot is really in at t=0, whatever the
        controller asked for, and it is the posture the spawn height has to match.
        """
        import xml.etree.ElementTree as ElementTree
        try:
            package = rospkg.RosPack().get_path(f"{self.robot_name}_description")
            path = os.path.join(package, "robots", f"{self.robot_name}.srdf.xacro")
            root = ElementTree.parse(path).getroot()
            values = {}
            for state in root.iter("group_state"):
                if state.get("name") != "home":
                    continue
                for joint in state.iter("joint"):
                    values[joint.get("name")] = float(joint.get("value"))
            if not values:
                return None
            # Reorder into the locosim joint order this controller works in.
            return np.array([values[name] for name in self.joint_names], dtype=np.float64)
        except (KeyError, ValueError, TypeError, OSError,
                ElementTree.ParseError, rospkg.ResourceNotFound) as exc:
            print(colored(f"Could not read the SRDF home posture ({exc})", "yellow"))
            return None

    def _fixSpawnHeight(self):
        """Correct ``spawn_z`` before the launch runs, so the robot does not drop on start-up.

        ``params.py`` carries a ``spawn_z`` sized for the ``q_0`` standing posture (0.37 m for the
        Aliengo), but the robot is actually spawned in the sprawled SRDF ``home`` posture whose feet
        are only 0.084 m below the base.  The robot therefore free-falls a quarter of a metre and
        lands hard, which is what makes it ring for the first few seconds - and what made the
        accelerometer bias unobservable during calibration.

        Setting the height to match the posture removes the fall entirely.  This has to happen
        before :meth:`startSimulator`, which is why the kinematic model is loaded standalone here
        rather than reused from :meth:`loadModelAndPublishers`.
        """
        posture = self._srdfHomePosture()
        if posture is None:
            print(colored("Leaving spawn_z alone; the robot may drop on start-up", "yellow"))
            return
        robot = getRobotModelFloating(self.robot_name)
        height = (self._standHeight(posture, robot=robot) + self.cfg["foot_radius"]
                  + self.cfg["spawn_clearance"])
        previous = conf.robot_params[self.robot_name]['spawn_z']
        conf.robot_params[self.robot_name]['spawn_z'] = height
        self.base_offset[2] = height
        print(colored(f"Spawn height corrected: {previous:.3f} -> {height:.4f} m, removing a "
                      f"{previous - height:.3f} m drop onto the ground "
                      f"(spawn posture haa/hfe/kfe = {np.round(posture[:3], 2)})", "cyan"))

    def _waitForSensors(self, timeout=40.0):
        """Block until joint states and both IMU signals are flowing.

        Timed on the wall clock, deliberately: ``/use_sim_time`` is on and a fast world advances sim
        time several times faster than real time, so a sim-time budget here would expire long before
        the controller spawner has actually come up.
        """
        print("Waiting for joint states and IMU...")
        deadline = wall_time.monotonic() + timeout
        announced = 0.0
        while not ros.is_shutdown() and wall_time.monotonic() < deadline:
            if self._got_jstate:
                # Claim the joint target the moment a measurement exists, still at the launch's own
                # soft gains.  Until something commands it, the low-level controller drives towards
                # the home posture in its yaml - which for this robot has the hips 0.8 rad away from
                # where the model was spawned, so it spends the whole start-up dragging the splayed
                # legs inward under the robot's weight.  Asking it to hold what it already measures
                # costs nothing and keeps the robot still until the state machine takes over.
                self.q_des[:] = self.q
                self.qd_des[:] = 0.0
                self.tau_ffwd[:] = 0.0
                self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
            if self._got_jstate and self._got_imu and self._got_imu_acc:
                self._sensors_ready = True
                print(colored("Sensors alive", "green"))
                return
            wall_time.sleep(0.05)
            waited = timeout - (deadline - wall_time.monotonic())
            if waited - announced > 5.0:
                announced = waited
                pending = [name for name, got in (("joint_states", self._got_jstate),
                                                  ("imu orientation", self._got_imu),
                                                  ("imu acceleration", self._got_imu_acc))
                           if not got]
                print(f"  still waiting after {waited:.0f} s for: {', '.join(pending)}")
        missing = []
        if not self._got_jstate:
            missing.append("joint_states")
        if not self._got_imu:
            missing.append("imu orientation")
        if not self._got_imu_acc:
            missing.append("imu acceleration")
        raise RuntimeError(f"No data on: {', '.join(missing)}")

    def _applyRealtimeTuning(self):
        """Best-effort real-time hygiene.  Every step is optional and failure is not fatal."""
        if self.cfg["gc_managed"]:
            gc.collect()
            if hasattr(gc, "freeze"):
                # Move the startup heap - imports, the ONNX session, the ROS machinery - into the
                # permanent generation so no later collection ever walks it again.
                gc.freeze()
            # Keep generations 0 and 1 collecting: both are bounded by their thresholds and cost
            # microseconds, and generation 1 is what stops objects that survive a gen0 pass from
            # piling up for the whole of a long walk.  Only the generation-2 full-heap scan is
            # pushed out of the loop; it is taken by hand at resting states instead, see
            # _collectAtSafePoint.  Disabling the collector outright, or deferring generation 1 too,
            # leaves memory growing for as long as the robot keeps walking.
            gc.set_threshold(self.cfg["gc_gen0_threshold"], self.cfg["gc_gen1_threshold"],
                             1 << 30)
            print(colored(f"GC: startup heap frozen, gen0/gen1 thresholds "
                          f"{self.cfg['gc_gen0_threshold']}/{self.cfg['gc_gen1_threshold']}, "
                          f"full collection deferred to resting states", "yellow"))
        self._applyRealtimePriority()
        if self.pin_cpu:
            try:
                cpus = sorted(os.sched_getaffinity(0))
                index = self.cfg["pin_cpu_index"]
                core = cpus[-1] if index is None else index
                if core not in cpus:
                    raise OSError(f"core {core} is not in the available set {cpus}")
                # Affinity with pid 0 applies to this thread only, so the rospy callback threads
                # keep the rest of the machine.
                os.sched_setaffinity(0, {core})
                print(colored(f"Pinned the control thread to CPU {core}", "yellow"))
            except (OSError, AttributeError, IndexError) as exc:
                print(colored(f"Could not set CPU affinity ({exc})", "yellow"))

    def _checkTrainingModel(self):
        """Compare the simulated robot against the model the policy was trained on.

        The comparison is between *physics bodies*, not URDF links: both engines merge fixed-joint
        groups before integrating, so the things that move are 'calf + foot' and 'trunk + imu' on
        this side and 'thigh + rotor' and friends on the training side.  Comparing raw links would
        compare quantities neither simulator uses.

        This only ever runs in simulation, and it only ever reports.  On hardware the robot *is*
        the reference; and in simulation a mismatch means the robot description is wrong, which is
        where it should be fixed - a controller that quietly rewrote the physics engine's numbers at
        start-up would leave every other tool on the robot looking at a different machine.
        """
        if self.cfg["training_model_check"] == "off" or self.real_robot:
            return
        model_path = training_model.default_model_path(self.policy.policy_dir, self.policy_name)
        if model_path is None:
            print(colored(f"No training model vendored for '{self.policy_name}' "
                          f"({self.policy_name}_velocity_model.urdf); skipping the model check",
                          "yellow"))
            return

        name_map = self.cfg["training_body_map"]
        if not name_map:
            print(colored(f"No training_body_map for '{self.robot_name}'; skipping the model "
                          f"check (it maps Gazebo body names onto the training URDF's links)",
                          "yellow"))
            return
        try:
            training = training_model.load_bodies(model_path)
            ros.wait_for_service("/gazebo/get_link_properties", timeout=10.0)
            get_link = ros.ServiceProxy("/gazebo/get_link_properties", GetLinkProperties)
            deployed = {}
            for body_name in name_map:
                response = get_link(link_name=f"{self.robot_name}::{body_name}")
                if not response.success:
                    continue
                values = np.array([response.ixx, response.ixy, response.ixz,
                                   response.iyy, response.iyz, response.izz])
                com = np.array([response.com.position.x, response.com.position.y,
                                response.com.position.z])
                deployed[body_name] = training_model.Body(
                    body_name, response.mass, com,
                    np.array([[values[0], values[1], values[2]],
                              [values[1], values[3], values[4]],
                              [values[2], values[4], values[5]]]))
        except (ros.ROSException, ros.ServiceException, OSError, ValueError, KeyError) as exc:
            print(colored(f"Could not compare against the training model ({exc}); continuing",
                          "yellow"))
            return

        tol = self.cfg["training_model_tol"]
        report = training_model.compare(training, deployed, name_map,
                                        mass_tol=1e-3, inertia_tol=tol)
        differing = [row for row in report if row["differs"]]
        if not differing:
            print(colored(f"Deployed model matches the training model on all "
                          f"{len(report)} bodies (within {100 * tol:.0f} %)", "green"))
            return

        # One line per distinct discrepancy: the four legs are always identical, so listing all
        # twelve leg bodies would bury the one number that matters.
        seen = set()
        print(colored(f"Deployed model differs from the training model "
                      f"({os.path.basename(model_path)}) on {len(differing)} of {len(report)} "
                      f"bodies:", "yellow"))
        print(colored(f"  {'body':24s} {'mass sim/train':>20s} {'ixx':>7s} {'iyy':>7s} "
                      f"{'izz':>7s}   (ratio sim/train)", "yellow"))
        for row in differing:
            key = (round(row["mass"][0], 6), tuple(np.round(row["ratio"], 4)))
            if key in seen:
                continue
            seen.add(key)
            print(colored(f"  {row['deployed'] + ' / ' + row['training']:24s} "
                          f"{row['mass'][0]:8.4f}/{row['mass'][1]:<11.4f} "
                          f"{row['ratio'][0]:7.3f} {row['ratio'][3]:7.3f} {row['ratio'][5]:7.3f}",
                          "yellow"))
        sim_mass = sum(b.mass for b in training_model.unique_bodies(deployed))
        train_mass = sum(b.mass for b in training_model.unique_bodies(training))
        print(colored(f"  total mass {sim_mass:.3f} kg simulated against {train_mass:.3f} kg "
                      f"trained ({sim_mass - train_mass:+.3f} kg)", "yellow"))

        print(colored("  The policy is a feedback law tuned to the trained inertias, so this "
                      "makes a Gazebo run test a different robot. Fix the description rather "
                      "than the run: the values belong in the robot's own xacro.", "yellow"))

    # ---------------------------------------------------------------------------------------------
    # disturbances (simulation only)
    # ---------------------------------------------------------------------------------------------
    def _initPushes(self):
        """Start the worker that turns push requests into Gazebo wrenches.

        The service call goes on its own thread for the obvious reason: ``/gazebo/apply_body_wrench``
        is a round-trip to another process, and the control loop has 2 ms.  A push is asynchronous
        by nature anyway - the request says "shove the robot", and when the shove lands is the
        simulator's business - so the loop posts to a queue and never waits.
        """
        self.push_enabled = not self.real_robot
        self._push_queue = queue.Queue()
        self._push_count = 0
        if not self.push_enabled:
            return
        body = self.cfg["push_body"] or f"{self.robot_name}::base_link"
        self._push_body = body
        self._push_rng = np.random.default_rng()
        self._push_thread = threading.Thread(target=self._pushWorker, name="push", daemon=True)
        self._push_thread.start()
        print(colored(f"Disturbances armed on {body}: 'p' one random push, 'P' a burst of "
                      f"{self.cfg['push_burst_count']}, -/+ to change the "
                      f"{self.cfg['push_force']:.0f} N magnitude", "cyan"))

    def _pushWorker(self):
        """Apply queued pushes.  Runs off the control thread; see :meth:`_initPushes`."""
        try:
            ros.wait_for_service("/gazebo/apply_body_wrench", timeout=20.0)
            apply_wrench = ros.ServiceProxy("/gazebo/apply_body_wrench", ApplyBodyWrench)
        except ros.ROSException as exc:
            print(colored(f"No /gazebo/apply_body_wrench ({exc}); pushes are unavailable", "red"))
            self.push_enabled = False
            return

        while not ros.is_shutdown():
            try:
                count, force = self._push_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            for index in range(count):
                if ros.is_shutdown():
                    return
                if index:
                    low, high = self.cfg["push_burst_interval"]
                    wall_time.sleep(float(self._push_rng.uniform(low, high)))
                self._applyPush(apply_wrench, force)

    def _applyPush(self, apply_wrench, force):
        """One shove of ``force`` newtons in a random direction, held for ``push_duration``."""
        # Uniform in heading rather than a uniform vector in the plane: the point is to probe every
        # direction equally, and a uniformly sampled xy vector clusters towards the diagonals.
        heading = float(self._push_rng.uniform(0.0, 2.0 * np.pi))
        vector = np.array([np.cos(heading), np.sin(heading), 0.0])
        if self.cfg["push_vertical"]:
            vector[2] = float(self._push_rng.uniform(-1.0, 1.0))
            vector /= np.linalg.norm(vector)
        vector *= force

        request = ApplyBodyWrenchRequest()
        request.body_name = self._push_body
        # The world frame is the only one this service handles correctly - it silently ignores any
        # other, which is why the direction is sampled in world coordinates here.
        request.reference_frame = "world"
        request.wrench.force.x, request.wrench.force.y, request.wrench.force.z = vector
        request.start_time = ros.Time(0)          # now
        request.duration = ros.Duration(self.cfg["push_duration"])
        try:
            response = apply_wrench(request)
        except (ros.ServiceException, TypeError) as exc:
            print(colored(f"Push failed: {exc}", "red"))
            return
        if not response.success:
            print(colored(f"Push refused: {response.status_message}", "red"))
            return

        self._push_count += 1
        self._last_push[:] = vector
        self._push_stamp = self.time
        if self.enable_telemetry:
            # Published from this thread rather than handed to the control loop: rospy publishers
            # are thread-safe, and the stamp is worth more taken when the push actually landed.
            self._msg_push.header.stamp = ros.Time.now()
            (self._msg_push.vector.x, self._msg_push.vector.y,
             self._msg_push.vector.z) = vector
            self.pub_push.publish(self._msg_push)
        print(colored(f"PUSH #{self._push_count}: {force:.0f} N at {np.degrees(heading):5.0f} deg "
                      f"for {self.cfg['push_duration']:.2f} s "
                      f"[{np.round(vector, 1)} N, state '{self.state.value}']", "magenta"))

    def requestPush(self, count=1):
        """Queue ``count`` random pushes.  Safe to call from the control loop: it never blocks."""
        if not self.push_enabled:
            print(colored("Pushes are simulation only - there is no wrench service on the robot",
                          "yellow"))
            return
        self._push_queue.put((int(count), float(self.command.push_force)))

    def _applyRealtimePriority(self):
        """Put the whole process on SCHED_FIFO - every thread, or none of them.

        Raising only the control thread is actively harmful here, and subtly so.  rospy delivers
        joint states and the IMU on background threads; a real-time main thread holding the GIL
        starves them, so the loop keeps running at 500 Hz on an observation that has stopped
        updating.  In simulation that reliably threw the robot over about a second into walking,
        while reporting 200 ms "computation" times that were really the loop waiting on threads it
        had itself starved.

        So the control thread is raised to ``sched_fifo_priority`` and every other thread of the
        process to ten below it: still above everything else on the machine, but never able to
        preempt the control loop.  If any thread cannot be raised, the ones already raised are put
        back - a partially real-time process is the dangerous configuration, not a safer one.

        Launching through ``run_rl_controller.sh`` (``chrt -f 80 python3 ...``) avoids this entirely,
        because threads inherit the policy from the process at creation.
        """
        priority = self.cfg["sched_fifo_priority"]
        helper_priority = max(1, priority - 10)
        main_tid = os.getpid()
        try:
            tids = [int(entry) for entry in os.listdir("/proc/self/task")]
        except OSError as exc:
            print(colored(f"Cannot enumerate this process's threads ({exc}); leaving scheduling "
                          f"alone", "yellow"))
            return

        raised = []
        try:
            for tid in tids:
                target = priority if tid == main_tid else helper_priority
                os.sched_setscheduler(tid, os.SCHED_FIFO, os.sched_param(target))
                raised.append(tid)
        except (OSError, AttributeError, PermissionError) as exc:
            for tid in raised:
                try:
                    os.sched_setscheduler(tid, os.SCHED_OTHER, os.sched_param(0))
                except OSError:
                    pass
            print(colored(f"Could not put every thread on SCHED_FIFO ({exc}); reverted to normal "
                          f"scheduling for all of them, because raising only the control thread "
                          f"starves the ROS threads that feed it. Launch through "
                          f"./run_rl_controller.sh for real-time priority.", "yellow"))
            return

        print(colored(f"SCHED_FIFO: control thread at {priority}, {len(tids) - 1} ROS thread(s) at "
                      f"{helper_priority}", "yellow"))

    def _collectAtSafePoint(self):
        """Take a full garbage collection where a pause cannot hurt.

        The older generations are deferred out of the control loop, so they are collected here
        instead: on entry to a state where the robot is resting or merely holding a posture and a
        millisecond of jitter costs nothing.
        """
        if not (self.realtime and self.cfg["gc_managed"]):
            return
        if self.state in (State.INIT, State.CALIBRATE, State.FOLD, State.STANDING,
                          State.DONE):
            gc.collect()

    # ---------------------------------------------------------------------------------------------
    # callbacks
    # ---------------------------------------------------------------------------------------------
    def _receive_jstate(self, msg):
        for msg_idx in range(len(msg.name)):
            for joint_idx in range(len(self.joint_names)):
                if self.joint_names[joint_idx] == msg.name[msg_idx]:
                    self.q[joint_idx] = msg.position[msg_idx]
                    self.qd[joint_idx] = msg.velocity[msg_idx]
                    self.tau[joint_idx] = msg.effort[msg_idx]
        self._got_jstate = True

    def _receive_pid_effort(self, msg):
        for msg_idx in range(len(msg.name)):
            for joint_idx in range(len(self.joint_names)):
                if self.joint_names[joint_idx] == msg.name[msg_idx]:
                    self.tau_fb[joint_idx] = msg.effort_pid[msg_idx]

    def _set_orientation(self, quat):
        self.quaternion[0] = quat.x
        self.quaternion[1] = quat.y
        self.quaternion[2] = quat.z
        self.quaternion[3] = quat.w
        self.euler[:] = euler_from_quaternion(self.quaternion)
        self.basePoseW[3:6] = self.euler
        # b_R_w maps world into base (math_tools.rpyToRot returns the world-to-base rotation).
        self.b_R_w = self.math_utils.rpyToRot(self.euler)
        # Unit gravity direction in the base frame: Isaac's GRAVITY_VEC_W is normalised (0,0,-1).
        self.projected_gravity[:] = -self.b_R_w[:, 2]

    def _receive_imu(self, msg):
        """Real robot: orientation and body angular rate."""
        self._set_orientation(msg.orientation)
        self.angVelB[0] = msg.angular_velocity.x
        self.angVelB[1] = msg.angular_velocity.y
        self.angVelB[2] = msg.angular_velocity.z
        self.baseTwistW[3:6] = self.b_R_w.T @ self.angVelB
        self._got_imu = True

    def _receive_imu_acc_real(self, msg):
        self.baseLinAccB[0] = msg.x
        self.baseLinAccB[1] = msg.y
        self.baseLinAccB[2] = msg.z
        self._got_imu_acc = True

    def _receive_imu_sim(self, msg):
        """Gazebo: one Imu message carries orientation, rate and acceleration."""
        self._receive_imu(msg)
        self.baseLinAccB[0] = msg.linear_acceleration.x
        self.baseLinAccB[1] = msg.linear_acceleration.y
        self.baseLinAccB[2] = msg.linear_acceleration.z
        self._got_imu_acc = True

    def _receive_ground_truth(self, msg):
        self.basePoseW[0] = msg.pose.pose.position.x
        self.basePoseW[1] = msg.pose.pose.position.y
        self.basePoseW[2] = msg.pose.pose.position.z
        self.gt_baseTwistW[0] = msg.twist.twist.linear.x
        self.gt_baseTwistW[1] = msg.twist.twist.linear.y
        self.gt_baseTwistW[2] = msg.twist.twist.linear.z

    # ---------------------------------------------------------------------------------------------
    # observations
    # ---------------------------------------------------------------------------------------------
    def imu_lin_acc(self):
        """Specific force in the base frame, bias removed - the accelerometer, as it reads.

        Gravity is *not* subtracted: a real accelerometer measures specific force, so a level robot
        reads about ``(0, 0, +9.81)``.  The only correction here is the sensor bias measured in
        :meth:`_calibrate`.

        This is the honest measurement and not, by itself, the policy's ``imu_lin_acc`` observation
        term: the training sensor reported something else (see ``imu_lin_acc_model`` in the policy
        contract and :meth:`VelocityPolicy._imuLinAcc`).  Converting one into the other is
        :class:`VelocityPolicy`'s business, which is why this is handed to it raw, on every control
        tick - the conversion needs the samples between inferences, not only the ones on them.
        """
        return self.baseLinAccB - self.imu_utils.IMU_accelerometer_bias

    # ---------------------------------------------------------------------------------------------
    # state machine
    # ---------------------------------------------------------------------------------------------
    def transition(self, new_state):
        if new_state is self.state:
            return
        print(colored(f"[t={self.time:7.2f}] {self.state.value} -> {new_state.value}", "blue"))
        self.prev_state = self.state
        self.state = new_state
        self.state_start_time = self.time
        self.state_first_tick = True

    @property
    def state_elapsed(self):
        return self.time - self.state_start_time

    def _setGains(self, kp, kd):
        """Push joint gains.  ki is always zero: this controller never uses an integral term.

        ``/set_pids`` is a service round-trip, so gains that have not actually changed are not
        re-sent: consecutive states often want the same stiffness.
        """
        kp = np.asarray(kp, dtype=np.float64)
        kd = np.asarray(kd, dtype=np.float64)
        if self._active_gains is not None:
            if np.array_equal(self._active_gains[0], kp) and \
                    np.array_equal(self._active_gains[1], kd):
                return
        self.pid.setPDjoints(kp, kd, np.zeros(self.robot.na))
        self._active_gains = (kp.copy(), kd.copy())

    def handleEvents(self):
        for event in self.command.poll():
            if event is Event.STOP:
                print(colored("EMERGENCY: collapsing on joint damping", "red", attrs=["bold"]))
                self.transition(State.DAMPING)
                return
            if event is Event.PUSH:
                self.requestPush(1)
                continue
            if event is Event.PUSH_BURST:
                self.requestPush(self.cfg["push_burst_count"])
                continue
            if event is Event.QUIT:
                print(colored("Quit requested: collapsing on joint damping first", "yellow"))
                self._quit_requested = True
                self.transition(State.DAMPING)
                return
            if event is Event.CALIBRATE:
                # Not while the fold is still moving: the procedure needs a still robot, so it
                # would only burn the whole timeout and report failure.
                if self.state is State.INIT or (self.state is State.FOLD
                                                and not self._fold_moving):
                    self.transition(State.CALIBRATE)
                elif self.state is State.FOLD:
                    print(colored("Wait for the fold to finish, then calibrate", "yellow"))
                else:
                    print(colored("Calibration only from the folded pose or at start-up",
                                  "yellow"))
            elif event is Event.STAND_UP:
                if self.state is State.FOLD:
                    self.transition(State.STAND_UP)
                else:
                    print(colored(f"Cannot stand up from '{self.state.value}'", "yellow"))
            elif event is Event.START_RL:
                # From the safe stop this is a resume: the walking policy takes the motion back
                # over from the standstill policy without either of them being reset.
                if self.state in (State.STANDING, State.SAFE_STOP):
                    self.transition(State.RL)
                else:
                    print(colored(f"Cannot start the policy from '{self.state.value}'", "yellow"))
            elif event is Event.SAFE_STOP:
                if self.state is State.SAFE_STOP:
                    print(colored("Already in the safe stop", "yellow"))
                elif self.state in (State.RL, State.STANDING):
                    self.transition(State.SAFE_STOP)
                else:
                    print(colored(f"The safe stop needs the robot standing, not "
                                  f"'{self.state.value}'. It keeps the robot up on the "
                                  f"zero-command policy; for a collapse use the emergency "
                                  f"damping instead.", "yellow"))
            elif event is Event.STAND_DOWN:
                if self.state in (State.RL, State.SAFE_STOP, State.STANDING):
                    self.transition(State.STAND_DOWN)
                else:
                    print(colored(f"Cannot stand down from '{self.state.value}'", "yellow"))

    # -- individual states ------------------------------------------------------------------------
    def _init(self):
        """Freeze the posture the robot was placed in, let it settle, then calibrate.

        Calibration comes before any commanded motion, because the accelerometer bias is only
        observable while the base is not accelerating, and the quietest the robot will ever be is
        the moment before the controller first asks it to move.  So the procedure takes the
        operator at their word - the robot has been set down flat and left alone - and commands
        nothing but "hold exactly where you are".  That hold is not a no-op: the low-level
        controller has been running since the launch with its own default target, so the robot is
        usually still drifting when we take over, and latching the measured position is what stops
        it.  Folding into the resting posture happens afterwards, in :meth:`_fold`, once the bias
        is known.
        """
        if self.state_first_tick:
            # The soft, well-damped fold gains: holding a posture the robot was placed in means
            # the legs are already against their contacts, and stiff position control there is
            # what makes the base ring.
            self._setGains(self.cfg["kp_fold"], self.cfg["kd_fold"])
            self.q_des[:] = self.q
            self.qd_des[:] = 0.0
            self.tau_ffwd[:] = 0.0
            print(colored("Gains set (ki = 0). Holding the measured posture while it settles.",
                          "cyan"))

        if self.state_elapsed < self.cfg["settle_duration"]:
            return

        if self.skip_calibration:
            print(colored("Skipping the accelerometer calibration on request: the bias stays "
                          "zero, so the policy sees the accelerometer exactly as it reads.",
                          "yellow"))
            self.transition(State.FOLD)
        else:
            self.transition(State.CALIBRATE)

    def _isQuiescent(self):
        """True when the robot is still enough for the accelerometer bias to be observable.

        A bias estimate is only meaningful while the base is not accelerating: any real motion in the
        window is indistinguishable from sensor bias and would be baked straight into the observation
        the policy sees.
        """
        return (np.linalg.norm(self.angVelB) < self.cfg["calib_gyro_threshold"] and
                np.max(np.abs(self.qd)) < self.cfg["calib_joint_vel_threshold"])

    def _calibrate(self):
        """Estimate the accelerometer bias over a continuously still window.

        The window has to be *continuous*.  An earlier version accumulated every sample that happened
        to look quiet and skipped the rest, which finishes faster but averages across a robot that is
        still settling - and a bias that absorbed part of a transient is worse than no bias at all,
        because everything downstream then trusts it.  So: wait for ``calib_stable_window`` seconds
        of uninterrupted stillness, then integrate for ``calibration_duration`` more, and start over
        from scratch if stillness is lost at any point.

        Two independent checks decide whether the result is usable: the spread of the samples
        (a still robot has almost none) and the magnitude residual ``|acc - bias| - |g|``, which must
        come out near zero because gravity is the only thing a static accelerometer measures.
        """
        if self.state_first_tick:
            print(colored(f"Calibrating the accelerometer bias - the robot must lie still for "
                          f"{self.cfg['calib_stable_window']:.1f} s + "
                          f"{self.cfg['calibration_duration']:.1f} s", "cyan"))
            # At start-up, hold the posture the robot was placed in.  Re-triggered from the
            # fold, keep holding q_fold: re-latching the measurement would put a step back into
            # the reference on the way out, since the fold is held with a standing tracking error.
            if not self._folded:
                self.q_des[:] = self.q
            self.qd_des[:] = 0.0
            self.tau_ffwd[:] = 0.0
            self.imu_utils.IMU_accelerometer_bias[:] = 0.0
            self._calibReset(announce=False)


        if not self._isQuiescent():
            if self._calib_n > 0 or self._calib_stable_since is not None:
                self._calibReset(announce=True)
            if self.state_elapsed - self._calib_warned > 3.0:
                self._calib_warned = self.state_elapsed
                print(colored(f"Waiting for the robot to be still: "
                              f"|gyro|={np.linalg.norm(self.angVelB):.3f} rad/s (limit "
                              f"{self.cfg['calib_gyro_threshold']}), "
                              f"max|qd|={np.max(np.abs(self.qd)):.3f} rad/s (limit "
                              f"{self.cfg['calib_joint_vel_threshold']})", "yellow"))
            # One attempt.  The procedure assumes the robot has been laid flat and left alone,
            # so if it never goes still that assumption is wrong and the honest thing is to say so
            # and carry on with a zero bias, not to go looking for a posture that does settle.
            timeout = self.cfg["calib_wait_timeout"]
            if self.state_elapsed > timeout:
                print(colored(f"The robot never became still within {timeout:.0f} s: skipping "
                              f"calibration, the accelerometer bias stays zero and the policy "
                              f"will see the accelerometer as it comes. Lay the robot flat on the "
                              f"ground and re-trigger the calibration with 'c' (Y on the pad) "
                              f"from the folded hold - it can be repeated as often as you like.",
                              "red"))
                self.transition(State.FOLD)
            return

        # Stillness has held since _calib_stable_since; wait out the settling window before
        # believing any of it.
        if self._calib_stable_since is None:
            self._calib_stable_since = self.state_elapsed
        if self.state_elapsed - self._calib_stable_since < self.cfg["calib_stable_window"]:
            return

        residual_vector = self.baseLinAccB - self.b_R_w @ self.imu_utils.g0
        self._calib_sum += residual_vector
        self._calib_sq_sum += residual_vector * residual_vector
        self._calib_acc_sum += self.baseLinAccB
        self._calib_n += 1

        if self._calib_n < self.imu_utils.timeout:
            return

        count = float(self._calib_n)
        bias = self._calib_sum / count
        spread = np.sqrt(np.maximum(self._calib_sq_sum / count - bias * bias, 0.0))
        mean_acc = self._calib_acc_sum / count
        residual = np.linalg.norm(mean_acc - bias) - np.linalg.norm(self.imu_utils.g0)

        if np.linalg.norm(bias) > self.cfg["calib_max_bias"]:
            print(colored(f"Rejecting an implausible accelerometer bias {np.round(bias, 3)} m/s^2 "
                          f"(|bias| > {self.cfg['calib_max_bias']} m/s^2). Check that the IMU topic "
                          f"reports specific force, i.e. about +9.81 m/s^2 on z when level.", "red"))
            self.imu_utils.IMU_accelerometer_bias[:] = 0.0
        elif np.max(spread) > self.cfg["calib_max_spread"]:
            print(colored(f"Rejecting the calibration: sample spread {np.round(spread, 3)} m/s^2 "
                          f"exceeds {self.cfg['calib_max_spread']} m/s^2, so the robot was not "
                          f"actually still. Retrying.", "yellow"))
            self._calibReset(announce=False)
            return
        elif abs(residual) > self.cfg["calib_max_residual"]:
            print(colored(f"Calibration residual {residual:+.3f} m/s^2 exceeds "
                          f"{self.cfg['calib_max_residual']} m/s^2 (mean reading "
                          f"{np.round(mean_acc, 3)}). Keeping the bias, but treat the policy input "
                          f"as suspect.", "yellow"))
            self.imu_utils.IMU_accelerometer_bias[:] = bias
            self._calibrated = True
        else:
            self.imu_utils.IMU_accelerometer_bias[:] = bias
            self._calibrated = True
            print(colored(f"Accelerometer bias = {np.round(bias, 4)} m/s^2 over "
                          f"{self._calib_n} still samples (mean reading {np.round(mean_acc, 3)}, "
                          f"spread {np.round(spread, 4)}, residual {residual:+.4f} m/s^2)", "green"))
        # Either the first calibration, which the fold follows, or one the operator
        # re-triggered from the fold, which returns there.  Both go to the same place.
        self.transition(State.FOLD)

    def _calibReset(self, announce):
        """Discard the current window and wait for stillness again."""
        if announce and self._calib_n > 0:
            print(colored(f"Motion during calibration after {self._calib_n} samples: "
                          f"discarding the window and waiting for the robot to be still again",
                          "yellow"))
        self._calib_sum = np.zeros(3)
        self._calib_sq_sum = np.zeros(3)
        self._calib_acc_sum = np.zeros(3)
        self._calib_n = 0
        self._calib_stable_since = None
        self._calib_warned = -1e9

    def _fold(self):
        """Ease into the resting fold posture, then hold it and wait for a command.

        This is both the move and the resting state, because they are the same thing: a separate
        "holding the fold" state held exactly the posture this one arrives at, so it was two names
        for one behaviour and one more transition to read in a log.

        Whatever posture the robot was placed in is generally not one it can hold indefinitely - a
        splayed leg fights its contact at full stiffness - so once the bias is known the robot is
        brought to a configuration it can sit in.  The move drags loaded feet across the ground,
        which is why it runs on the fold gains rather than the stand gains.

        Arriving already folded - which is how a stand-down ends, since its last waypoint *is*
        q_fold - skips the move rather than replaying a three-second interpolation to a posture the
        robot is already holding.
        """
        if self.state_first_tick:
            self._setGains(self.cfg["kp_fold"], self.cfg["kd_fold"])
            self._fold_moving = not self._folded
            if self._fold_moving:
                self._startTrajectory([self.q_fold], self.cfg["init_fold_duration"])
                print(colored(f"Easing into the fold posture {np.round(self.q_fold, 3)}", "cyan"))
            else:
                print(colored("Already folded. Command a stand up when the robot is clear.",
                              "cyan"))

        if self._fold_moving:
            if self._stepTrajectory():
                self._fold_moving = False
                self._folded = True
                print(colored("Folded. Command a stand up when the robot is clear.", "cyan"))
            return

        self.q_des[:] = self.q_fold
        self.qd_des[:] = 0.0
        self.tau_ffwd[:] = 0.0

    def _startTrajectory(self, waypoints, duration):
        self._traj_waypoints = waypoints
        self._traj_index = 0
        self._traj_duration = duration / max(len(waypoints), 1)
        self._traj_from[:] = self.q
        self._traj_to[:] = waypoints[0]
        self._traj_leg_start = self.time

    def _stepTrajectory(self):
        """Advance the quintic waypoint interpolation; returns True when the last one is reached."""
        alpha = (self.time - self._traj_leg_start) / self._traj_duration
        s, sd = quintic(alpha)
        delta = self._traj_to - self._traj_from
        self.q_des[:] = self._traj_from + s * delta
        self.qd_des[:] = delta * sd / self._traj_duration
        self.tau_ffwd[:] = 0.0
        if alpha >= 1.0:
            self._traj_index += 1
            if self._traj_index >= len(self._traj_waypoints):
                self.qd_des[:] = 0.0
                return True
            self._traj_from[:] = self._traj_to
            self._traj_to[:] = self._traj_waypoints[self._traj_index]
            self._traj_leg_start = self.time
        return False

    def _stand_up(self):
        if self.state_first_tick:
            self._folded = False
            if not self._calibrated:
                print(colored("Standing up without a calibrated IMU: the policy will see a "
                              "biased acceleration", "yellow"))
            self._setGains(self.cfg["kp_stand"], self.cfg["kd_stand"])
            self._startTrajectory(self.stand_up_waypoints, self.cfg["stand_up_duration"])
            print(colored(f"Standing up to {np.round(self.q_stand, 3)}", "cyan"))
        if self._stepTrajectory():
            self.transition(State.STANDING)

    def _standing(self):
        if self.state_first_tick:
            self.q_des[:] = self.q_stand
            self.qd_des[:] = 0.0
            self.tau_ffwd[:] = 0.0
            print(colored("Standing. Command the RL policy to start walking.", "cyan"))
        if self._checkSafety():
            self.transition(State.DAMPING)

    def _enterPolicy(self, variant, announce):
        """Shared entry for the two policy-driven states.

        The one decision that matters here is whether to reset.  Arriving from a standstill, the
        history must be seeded the way an Isaac Lab episode reset does it, or the first inference
        runs on an empty buffer.  Arriving from the *other* policy it must not be: the two
        variants share the observation contract, so the incoming network should pick the motion up
        as it actually is.  Telling a recovery policy that the robot has been standing still is
        exactly wrong at the moment you reach for it.

        Nothing is eased: the command is live from the first tick and the policy's own joint target
        is applied as it comes.  A policy that has been asked for is a policy that is driving.

        The stored command is zeroed on *every* entry, and that is not the same thing as easing.
        Whatever the operator had set is discarded, so a policy never inherits a velocity - not from
        an earlier walk, and in particular not across a hand-back from the safe stop, which is
        precisely the moment a stale command must not launch the robot.  Whatever they ask for after
        that arrives on the next tick with nothing in between.  On a pad the stick is re-read every
        message, so a stick genuinely being held takes effect immediately anyway.

        The observation history is a separate question and is *not* reset on a hand-over between the
        two policies: they share the observation contract, so the incoming network should pick the
        motion up as it actually is.
        """
        self._setGains(self.cfg["kp_rl"], self.cfg["kd_rl"])
        self.command.zero()
        self.velocity_cmd[:] = 0.0
        if self.prev_state not in POLICY_STATES:
            self.policy.reset(self.q, self.qd, self.imu_lin_acc(), self.angVelB,
                              self.projected_gravity, velocity_cmd=self.velocity_cmd)
        self.policy.select(variant)
        self.publishPolicyVariant()
        print(colored(announce, "cyan"))

    def _stepPolicy(self, command):
        """Advance the active policy one control tick on ``command``.

        The joint target the network returns is applied unmodified.  There is no blend into it and
        no ramp on the command: the operator asked for this policy, so this policy has the robot.
        Any smoothing the *command* wants belongs in the input device, where the operator can see
        it, not in the state machine where it silently delays what they asked for.
        """
        self.velocity_cmd[:] = command

        self.q_policy[:] = self.policy.step(self.q, self.qd, self.imu_lin_acc(), self.angVelB,
                                            self.projected_gravity, self.velocity_cmd)
        self.rl_action[:] = self.policy.action
        self.q_des[:] = self.q_policy
        self.qd_des[:] = 0.0
        self.tau_ffwd[:] = 0.0

        if self._checkSafety():
            self.transition(State.DAMPING)

    def _commandIsZero(self):
        """True while the *operator* command is inside the dead band.

        The operator command, not ``velocity_cmd``: the latter is what the policy is fed, and the
        safe stop holds it at zero by construction, so testing it there would be circular.
        """
        return bool(np.max(np.abs(self.command.get_velocity_cmd()))
                    < self.cfg["auto_safe_stop_deadband"])

    def _rl(self):
        if self.state_first_tick:
            self._zero_cmd_since = None
            self._enterPolicy(
                self.policy.default_variant,
                f"Policy '{self.policy.default_variant}' running: "
                f"kp={self.cfg['kp_rl'][0]:.1f} kd={self.cfg['kd_rl'][0]:.2f}, "
                f"{self.policy.policy_rate:.0f} Hz, command reset to zero and then followed "
                f"immediately")

        self._stepPolicy(self.command.get_velocity_cmd())

        # Optional hand-over to the standstill policy once the command has been zero for a while.
        # The dwell time matters: switching on the first zero tick would fire between two pushes of
        # the stick, and the walking policy stops perfectly well most of the time.
        dwell = self.cfg["auto_safe_stop_after"]
        if dwell > 0.0 and self.state is State.RL:
            if self._commandIsZero():
                if self._zero_cmd_since is None:
                    self._zero_cmd_since = self.state_elapsed
                elif self.state_elapsed - self._zero_cmd_since >= dwell:
                    print(colored(f"Command has been zero for {dwell:.1f} s: handing over to the "
                                  f"standstill policy", "cyan"))
                    self.transition(State.SAFE_STOP)
            else:
                self._zero_cmd_since = None

    def _safe_stop(self):
        """Hold station on the zero-command policy.

        This is deliberately *not* the emergency path, and it is reached by its own command.
        Damping gives up on the robot and lets it sink; this keeps it standing and actively
        rejects what is happening to it, which is the right answer to a command that has gone
        wrong, an unexpected push, or an operator who just wants the machine to stop where it is.
        The variant comes from the same training environment as the walking policy with the
        command range pinned to zero and much harsher resets and pushes, so it consumes an
        identical observation - which is what makes the mid-stride hand-over possible at all.

        The emergency damping is still one command away from here, and it still wins from any
        state, so choosing the safe stop never costs the operator the harder option.

        Leaving is always deliberate: resume walking, stand down, or collapse.  There is no
        automatic hand-back on the operator's command, because every entry into a policy zeroes
        that command - so a hand-back driven by it would immediately discard the very command that
        triggered it and drop straight back here on the next dwell.
        """
        if self.state_first_tick:
            self._enterPolicy(
                self.cfg["safe_policy_variant"],
                f"SAFE STOP: policy '{self.cfg['safe_policy_variant']}' holding station with the "
                f"command forced to zero. Resume walking, stand down, or trigger the emergency "
                f"damping - all still available.")

        # Zero is not "ignore the operator": it is held at zero because that is the only command
        # this variant ever saw in training, so anything else is off-distribution.  This is the one
        # command the state machine overrides, and it overrides it outright rather than fading it.
        self._stepPolicy(_ZERO_COMMAND)


    def _checkSafety(self):
        """Bail out to damping if the robot is clearly losing it.

        The attitude envelope is wider during the safe stop.  That policy is trained to recover
        from tilts and pushes that the walking policy would never see, so applying the walking
        limit there would trip damping on exactly the excursions the safe stop exists to catch -
        the operator would ask for a recovery and get a collapse.
        """
        limit = (self.cfg["safe_max_roll_pitch"] if self.state is State.SAFE_STOP
                 else self.cfg["max_roll_pitch"])
        if abs(self.euler[0]) > limit or abs(self.euler[1]) > limit:
            print(colored(f"Safety: roll/pitch {np.round(self.euler[:2], 2)} rad exceeds "
                          f"{limit} rad", "red"))
            return True
        if np.max(np.abs(self.qd)) > self.cfg["max_joint_vel"]:
            print(colored(f"Safety: joint velocity {np.max(np.abs(self.qd)):.1f} rad/s exceeds "
                          f"{self.cfg['max_joint_vel']} rad/s", "red"))
            return True
        return False

    def _stand_down(self):
        if self.state_first_tick:
            self._setGains(self.cfg["kp_stand"], self.cfg["kd_stand"])
            self._startTrajectory(self.stand_down_waypoints, self.cfg["stand_down_duration"])
            print(colored("Standing down", "cyan"))
        if self._stepTrajectory():
            # The last stand-down waypoint is q_fold, so the fold is reached by arriving here.
            self._folded = True
            self.transition(State.FOLD)

    def _damping(self):
        """Emergency: pure joint damping, so the robot sinks under gravity instead of dropping.

        With kp = 0 the joint torque is ``-kd * qd``: gravity pulls the legs in and the damping
        bleeds the energy off, which is the standard soft-collapse for these machines.  The gains are
        pushed once (``/set_pids`` is a service call and has no business in a 250 Hz loop) and any
        feed-forward torque still in flight is ramped out inside the loop.
        """
        if self.state_first_tick:
            self._damping_tau0 = self.tau_ffwd.copy()
            self._setGains(np.zeros(self.robot.na), self.cfg["kd_damping"])
            self.q_des[:] = self.q
            self.qd_des[:] = 0.0
            print(colored(f"Damping mode: kp=0, kd={self.cfg['kd_damping'][0]:.1f}", "red"))

        ramp = self.cfg["damping_ramp_duration"]
        scale = max(0.0, 1.0 - self.state_elapsed / ramp) if ramp > 0.0 else 0.0
        self.tau_ffwd[:] = scale * self._damping_tau0
        self.qd_des[:] = 0.0
        # q_des is irrelevant with kp = 0, but tracking the measurement keeps the log honest.
        self.q_des[:] = self.q

        if self.state_elapsed >= ramp + self.cfg["damping_settle_duration"]:
            print(colored("Robot settled", "green"))
            self.transition(State.DONE)

    def _done(self):
        self._setGains(np.zeros(self.robot.na), np.zeros(self.robot.na))
        self.q_des[:] = self.q
        self.qd_des[:] = 0.0
        self.tau_ffwd[:] = 0.0

    # ---------------------------------------------------------------------------------------------
    # main loop
    # ---------------------------------------------------------------------------------------------
    HANDLERS = {
        State.INIT: _init,
        State.CALIBRATE: _calibrate,
        State.FOLD: _fold,
        State.STAND_UP: _stand_up,
        State.STANDING: _standing,
        State.RL: _rl,
        State.SAFE_STOP: _safe_stop,
        State.STAND_DOWN: _stand_down,
        State.DAMPING: _damping,
        State.DONE: _done,
    }

    def mainLoop(self):
        print(colored("Starting the control loop", "green"))
        self.transition(State.INIT)
        while not ros.is_shutdown():
            tick_start = wall_time.monotonic()
            self.command.update()
            self.handleEvents()

            state_before = self.state
            entry_tick = self.state_first_tick
            if entry_tick:
                self._collectAtSafePoint()
                self.publishState()
            self.HANDLERS[state_before](self)
            # A handler that transitioned has already armed the first-tick flag for the state it
            # moved to, so clearing it is only correct when we stayed put.
            if self.state is state_before:
                self.state_first_tick = False

            self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd,
                                 clip_commands=self.clip_to_joint_limits)
            self.publishTelemetry()

            self.compute_dt = wall_time.monotonic() - tick_start
            if entry_tick:
                if self.compute_dt > self.worst_entry_compute_dt:
                    self.worst_entry_compute_dt = self.compute_dt
            else:
                self._stats_compute_sum += self.compute_dt
                self._stats_compute_ticks += 1
                if self.compute_dt > self._stats_compute_max:
                    self._stats_compute_max = self.compute_dt
                if self.compute_dt > self.worst_compute_dt:
                    self.worst_compute_dt = self.compute_dt

            # DONE is checked on the tick *after* it was entered, so its handler runs once and
            # releases the joints before the loop ends.
            if state_before is State.DONE:
                break

            self.rate.sleep()
            self._trackLoopTime()
            self.time += self.dt

        self.shutdown()

    def _trackLoopTime(self):
        self.tick_count += 1
        stamp = wall_time.monotonic()
        if self._last_loop_stamp is not None:
            self.loop_dt = stamp - self._last_loop_stamp
            self._stats_dt_sum += self.loop_dt
            self._stats_ticks += 1
            if self.loop_dt > self._stats_dt_max:
                self._stats_dt_max = self.loop_dt
            if self.loop_dt > 1.5 * self.dt:
                self.loop_overruns += 1
                self._stats_overruns += 1
            if self.loop_dt > self.worst_loop_dt:
                self.worst_loop_dt = self.loop_dt
        self._last_loop_stamp = stamp

    def shutdown(self):
        print(colored("\nShutting down", "yellow"))
        self.command.shutdown()
        if self.tick_count:
            budget = self.dt * 1e3
            headroom = 100.0 * self.worst_compute_dt * 1e3 / budget
            print(colored(f"Compute per tick: worst {self.worst_compute_dt * 1e3:.3f} ms of the "
                          f"{budget:.2f} ms budget ({headroom:.1f} %)",
                          "green" if headroom < 50.0 else "yellow"))
            print(colored(f"  state-entry ticks peaked at "
                          f"{self.worst_entry_compute_dt * 1e3:.3f} ms (a /set_pids round-trip "
                          f"plus a full collection, by design at a transition)", "cyan"))
            if (getattr(self, "policy", None) is not None and self.policy.inference_count
                    and self.measure_inference_timing):
                summary = self.policy.timingSummary(simulated=not self.real_robot)
                bench = self.policy.benchmark_ms
                fits = bench is not None and bench < 0.5 * self.dt * 1e3
                print(colored("  " + summary, "green" if fits else "yellow"))
            if self.loop_overruns:
                note = ("" if self.real_robot else
                        " - in simulation the period follows Gazebo's real-time factor, so judge "
                        "timing by the compute figure above")
                print(colored(f"Loop period over 1.5x nominal on {self.loop_overruns} of "
                              f"{self.tick_count} ticks "
                              f"({100.0 * self.loop_overruns / self.tick_count:.2f} %), worst "
                              f"{self.worst_loop_dt * 1e3:.2f} ms{note}", "yellow"))
            else:
                print(colored(f"Loop period held in {self.tick_count} ticks "
                              f"(nominal {self.dt * 1e3:.2f} ms)", "green"))
        rss = self._rssMb()
        if self._rss_baseline_mb is not None and rss == rss:
            print(colored(f"Resident memory {rss:.1f} MB, "
                          f"{rss - self._rss_baseline_mb:+.1f} MB against the baseline",
                          "green" if rss - self._rss_baseline_mb < 16.0 else "yellow"))
        if self.realtime and self.cfg["gc_managed"]:
            gc.set_threshold(700, 10, 10)
            if hasattr(gc, "unfreeze"):
                gc.unfreeze()
        ros.signal_shutdown("controller finished")

    def _initTelemetry(self):
        """Create the telemetry publishers and preallocate every message.

        Telemetry is published, not accumulated: that is what makes it recordable with ``rosbag``
        and readable live in PlotJuggler, and it is also why the controller's memory footprint does
        not depend on how long it runs.  Joint data is already on ``/command`` and
        ``/<robot>/joint_states``, so it is not duplicated here.

        Every message object is allocated once and refilled in place, so a publish costs a
        serialisation and no allocation.
        """
        ns = f"/{self.robot_name}/{self.cfg['topic_namespace']}"
        self.telemetry_topics = []

        def publisher(name, msg_type, latch=False):
            topic = f"{ns}/{name}"
            self.telemetry_topics.append(topic)
            return ros.Publisher(topic, msg_type, queue_size=1, latch=latch, tcp_nodelay=True)

        # state machine: published on entry and as a slow heartbeat, latched so a late subscriber
        # (or PlotJuggler starting mid-run) still sees where the controller is
        self.pub_state = publisher("state", String, latch=True)
        self._msg_state = String()
        # Which network is driving, latched for the same reason: a bag or a plot that does not say
        # whether the walking or the safe-stop policy produced an action is hard to read back.
        self.pub_variant = publisher("policy_variant", String, latch=True)
        self._msg_variant = String()

        # operator command and policy output, at the policy rate
        self.pub_cmd = publisher("velocity_command", TwistStamped)
        self._msg_cmd = TwistStamped()
        self.pub_action = publisher("action", Float64MultiArray)
        # The base linear velocity the actor regresses internally, in the base frame.  This is the
        # single most useful signal for judging whether the policy is reading the robot correctly:
        # if it disagrees with reality, the policy is acting on a wrong belief about its own motion.
        self.publish_estimate = (self.cfg["publish_estimated_velocity"]
                                 and self.policy.has_estimator)
        if self.publish_estimate:
            self.pub_estimate = publisher("estimated_base_lin_vel", Vector3Stamped)
            self._msg_estimate = Vector3Stamped()
        self._msg_action = Float64MultiArray()
        dim = MultiArrayDimension()
        dim.label = "|".join(self.joint_names)
        dim.size = self.robot.na
        dim.stride = self.robot.na
        self._msg_action.layout.dim.append(dim)
        self._msg_action.data = [0.0] * self.robot.na

        # the three measured observation terms the policy actually consumes
        self.pub_imu_acc = publisher("imu_lin_acc", Vector3Stamped)
        self.pub_proj_grav = publisher("projected_gravity", Vector3Stamped)
        self.pub_ang_vel = publisher("ang_vel_b", Vector3Stamped)
        self.pub_rpy = publisher("base_rpy", Vector3Stamped)
        self._msg_imu_acc = Vector3Stamped()
        self._msg_proj_grav = Vector3Stamped()
        self._msg_ang_vel = Vector3Stamped()
        self._msg_rpy = Vector3Stamped()
        # The accelerometer as the *network* sees it, which for these policies is not the
        # accelerometer as the robot reads it - see imu_lin_acc_model in the policy contract.  Two
        # topics rather than one, because the interesting failure is exactly the two disagreeing in
        # the wrong way, and that is unreadable if only one of them is recorded.  Published at the
        # policy rate, since that is when it changes.
        self.pub_imu_acc_obs = publisher("policy/imu_lin_acc", Vector3Stamped)
        self._msg_imu_acc_obs = Vector3Stamped()

        # loop health and memory, as separate scalars so each is one named series in PlotJuggler
        self.pub_loop_mean = publisher("loop/dt_mean_ms", Float64)
        self.pub_loop_max = publisher("loop/dt_max_ms", Float64)
        self.pub_loop_overruns = publisher("loop/overruns", Float64)
        self.pub_compute_mean = publisher("loop/compute_mean_ms", Float64)
        self.pub_compute_max = publisher("loop/compute_max_ms", Float64)
        self.pub_rss = publisher("memory/rss_mb", Float64)
        self.pub_policy_rate = publisher("policy/inferences", Float64)
        # Inference cost on the tick it runs, against the control period.  Published so the
        # question "does the policy fit in a tick" is answerable from a bag instead of a benchmark.
        # Not created when the timing is off: the counters behind them would every one read zero,
        # and a topic publishing a constant zero is worse than an absent one.
        if self.measure_inference_timing:
            self.pub_infer_ms = publisher("policy/infer_ms", Float64)
            self.pub_infer_late = publisher("policy/infer_over_budget", Float64)
        # The disturbance, so a bag shows the push next to the response to it.  Latched: a push is
        # an event, and a plot of the recovery is unreadable without knowing when it was hit.
        self.pub_push = publisher("push", Vector3Stamped, latch=True)
        self._msg_push = Vector3Stamped()
        self._msg_scalar = Float64()

        self._telemetry_decim = max(1, int(round(1.0 / (self.cfg["telemetry_rate"] * self.dt))))
        self._stats_decim = max(1, int(round(1.0 / (self.cfg["stats_rate"] * self.dt))))
        self._memory_decim = max(1, int(round(1.0 / (self.cfg["memory_check_rate"] * self.dt))))
        print(colored(f"Telemetry on {ns}/* at {1.0 / (self._telemetry_decim * self.dt):.0f} Hz "
                      f"(stats {1.0 / (self._stats_decim * self.dt):.0f} Hz)", "green"))
        print(colored("  rosbag record " + " ".join(
            [f"/{self.robot_name}/joint_states", "/command"] + self.telemetry_topics), "cyan"))

    def _rssMb(self):
        """Resident set size in MiB, read straight from procfs to avoid pulling in psutil."""
        try:
            with open("/proc/self/statm", "r") as handle:
                pages = int(handle.readline().split()[1])
            return pages * self._page_size / (1024.0 * 1024.0)
        except (OSError, IndexError, ValueError):
            return float("nan")

    def publishState(self):
        if not self.enable_telemetry:
            return
        self._msg_state.data = self.state.value
        self.pub_state.publish(self._msg_state)

    def publishPolicyVariant(self):
        """Announce the active network.  Only changes at a state entry, so that is where it goes."""
        if not self.enable_telemetry:
            return
        self._msg_variant.data = self.policy.active_variant
        self.pub_variant.publish(self._msg_variant)

    def publishTelemetry(self):
        """Push one round of telemetry.  Called every tick; decimates internally."""
        if not self.enable_telemetry:
            return

        # Policy-rate signals, emitted exactly when a new action exists.
        if self.policy.inference_count != self._last_inference_count:
            self._last_inference_count = self.policy.inference_count
            stamp = ros.Time.now()
            self._msg_cmd.header.stamp = stamp
            self._msg_cmd.twist.linear.x = self.velocity_cmd[0]
            self._msg_cmd.twist.linear.y = self.velocity_cmd[1]
            self._msg_cmd.twist.angular.z = self.velocity_cmd[2]
            self.pub_cmd.publish(self._msg_cmd)
            # A plain list assignment here would allocate; the slice refills the existing one.
            self._msg_action.data[:] = self.rl_action
            self.pub_action.publish(self._msg_action)
            observed = self.policy.imu_lin_acc_obs
            self._msg_imu_acc_obs.header.stamp = stamp
            (self._msg_imu_acc_obs.vector.x, self._msg_imu_acc_obs.vector.y,
             self._msg_imu_acc_obs.vector.z) = observed
            self.pub_imu_acc_obs.publish(self._msg_imu_acc_obs)
            if self.publish_estimate:
                estimate = self.policy.estimated_base_lin_vel
                self._msg_estimate.header.stamp = stamp
                self._msg_estimate.vector.x = estimate[0]
                self._msg_estimate.vector.y = estimate[1]
                self._msg_estimate.vector.z = estimate[2]
                self.pub_estimate.publish(self._msg_estimate)

        if self.publish_tf and self.tick_count % self._telemetry_decim == 0:
            self._publishWorldTransform()

        if self.tick_count % self._telemetry_decim == 0:
            stamp = ros.Time.now()
            acc = self.imu_lin_acc()
            for msg, vec in ((self._msg_imu_acc, acc),
                             (self._msg_proj_grav, self.projected_gravity),
                             (self._msg_ang_vel, self.angVelB),
                             (self._msg_rpy, self.euler)):
                msg.header.stamp = stamp
                msg.vector.x, msg.vector.y, msg.vector.z = vec[0], vec[1], vec[2]
            self.pub_imu_acc.publish(self._msg_imu_acc)
            self.pub_proj_grav.publish(self._msg_proj_grav)
            self.pub_ang_vel.publish(self._msg_ang_vel)
            self.pub_rpy.publish(self._msg_rpy)

        if self.tick_count % self._stats_decim == 0 and self._stats_ticks > 0:
            self._publishScalar(self.pub_loop_mean,
                                1e3 * self._stats_dt_sum / self._stats_ticks)
            self._publishScalar(self.pub_loop_max, 1e3 * self._stats_dt_max)
            self._publishScalar(self.pub_loop_overruns, float(self._stats_overruns))
            if self._stats_compute_ticks:
                self._publishScalar(self.pub_compute_mean,
                                    1e3 * self._stats_compute_sum / self._stats_compute_ticks)
                self._publishScalar(self.pub_compute_max, 1e3 * self._stats_compute_max)
            self._publishScalar(self.pub_policy_rate, float(self.policy.inference_count))
            if self.measure_inference_timing:
                self._publishScalar(self.pub_infer_ms, self.policy.last_infer_ms)
                self._publishScalar(self.pub_infer_late, float(self.policy.infer_over_budget))
            self._stats_dt_sum = 0.0
            self._stats_dt_max = 0.0
            self._stats_compute_sum = 0.0
            self._stats_compute_max = 0.0
            self._stats_compute_ticks = 0
            self._stats_ticks = 0
            self._stats_overruns = 0

        if self.tick_count % self._memory_decim == 0:
            self._checkMemory()

    def _publishWorldTransform(self):
        """Broadcast world -> base_link, so rviz has a path from its fixed frame to the robot.

        ``robot_state_publisher`` turns the joint states into the transforms *below* base_link; the
        one thing it cannot know is where the base is.  In simulation that comes from the
        ground-truth odometry.  On the robot there is nothing that measures absolute position - the
        policy does not need it and this controller deliberately runs no odometry - so the
        translation stays at the origin and only the measured attitude is broadcast.  That is the
        honest thing to draw: the robot rendered in place, leaning the way it actually leans.
        """
        self.broadcaster.sendTransform(self.basePoseW[:3], self.quaternion,
                                       ros.Time.now(), '/base_link', '/world')

    def _publishScalar(self, publisher, value):
        self._msg_scalar.data = value
        publisher.publish(self._msg_scalar)

    def _checkMemory(self):
        """Publish resident memory and complain once if it has grown.

        Steady state allocates nothing, so growth past the threshold means something is leaking -
        which on a robot has to be visible rather than discovered when the machine starts swapping.
        """
        rss = self._rssMb()
        self._publishScalar(self.pub_rss, rss)
        if self._rss_baseline_mb is None or rss != rss:
            return
        growth = rss - self._rss_baseline_mb
        if growth > self.cfg["memory_growth_warn_mb"] and not self._memory_warned:
            self._memory_warned = True
            print(colored(f"Resident memory has grown {growth:.1f} MB above the "
                          f"{self._rss_baseline_mb:.1f} MB baseline; something is leaking",
                          "red"))

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="aliengo",
                        help="robot name; needs an entry in ROBOTS of rl_controller_config.py")
    parser.add_argument("--policy", default=None,
                        help="policy basename under rl_quadruped/policies "
                             "(default: the robot name)")
    parser.add_argument("--input", default="keyboard", choices=["keyboard", "joy", "none"],
                        help="operator input device")
    parser.add_argument("--real", action="store_true", help="run on the real robot")
    parser.add_argument("--skip-calibration", dest="skip_calibration", action="store_true",
                        default=None,
                        help="do not measure the accelerometer bias; it stays zero and therefore "
                             "shows up in the observation the policy consumes. Bring-up only")
    parser.add_argument("--world", default=None,
                        help="gazebo world (simulation only); default from the config")
    parser.add_argument("--gui", action="store_true", help="show the gazebo gui")
    parser.add_argument("--rviz", dest="use_rviz", action="store_true", default=None,
                        help="start rviz with the simulator (the default; see the config key "
                             "use_rviz)")
    parser.add_argument("--no-rviz", dest="use_rviz", action="store_false",
                        help="do not start rviz - one less process competing with the control "
                             "loop, for timing work or a loaded real-robot control PC")
    parser.add_argument("--no-telemetry", action="store_true",
                        help="do not publish the telemetry topics")
    parser.add_argument("--no-estimator", dest="estimate_velocity", action="store_false",
                        default=None,
                        help="do not evaluate the actor's state-estimation head: the estimated "
                             "base linear velocity is neither computed nor published. The policy "
                             "is unaffected - it uses the estimate inside its own graph either way")
    parser.add_argument("--no-inference-timing", dest="inference_timing", action="store_false",
                        default=None,
                        help="do not time the inference ticks: drops the start-up benchmark, the "
                             "policy/infer_* topics and the shutdown timing summary")
    parser.add_argument("--bag", nargs="?", const="", default=None, metavar="PREFIX",
                        help="record a rosbag of the telemetry and joint topics; optional filename "
                             "prefix")
    parser.add_argument("--realtime", dest="realtime", action="store_true", default=None,
                        help="force the real-time tuning on (default: on with --real)")
    parser.add_argument("--no-realtime", dest="realtime", action="store_false",
                        help="skip the real-time scheduling and GC tuning")
    parser.add_argument("--pin-cpu", dest="pin_cpu", action="store_true", default=None,
                        help="pin the control thread to one core (best with isolcpus; only helps "
                             "once SCHED_FIFO is granted)")
    return parser.parse_args(argv)


def start_rosbag(robot_name, topics, prefix=""):
    """Record the controller's topics with ``rosbag record``, returning the process."""
    import subprocess
    name = prefix if prefix else f"rl_{robot_name}"
    command = ["rosbag", "record", "-O", name] + topics
    print(colored("Recording: " + " ".join(command), "cyan"))
    return subprocess.Popen(command)


if __name__ == "__main__":
    args = parse_args()

    p = RlQuadrupedController(args.robot,
                              input_device=args.input,
                              policy_name=args.policy,
                              real_robot=args.real,
                              realtime=args.realtime,
                              telemetry=not args.no_telemetry,
                              pin_cpu=args.pin_cpu,
                              skip_calibration=args.skip_calibration,
                              use_rviz=args.use_rviz,
                              estimate_velocity=args.estimate_velocity,
                              inference_timing=args.inference_timing)
    recorder = None
    try:
        p.startController(world_name=args.world,
                          additional_args=["gui:=" + str(args.gui).lower()])
        if args.bag is not None:
            recorder = start_rosbag(args.robot,
                                    [f"/{args.robot}/joint_states", "/command"]
                                    + p.telemetry_topics,
                                    prefix=args.bag)
        p.mainLoop()
    except (ros.ROSInterruptException, ros.service.ServiceException, KeyboardInterrupt):
        print(colored("\nInterrupted", "yellow"))
        try:
            p.command.shutdown()
        except Exception:
            pass
        ros.signal_shutdown("killed")
    finally:
        if recorder is not None:
            recorder.send_signal(2)   # SIGINT, so rosbag closes the file cleanly
            recorder.wait(timeout=10)
