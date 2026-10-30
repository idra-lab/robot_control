# -*- coding: utf-8 -*-
"""State-machine controller running the Isaac Lab velocity policy on a locosim quadruped.

The policy (``policies/<robot>_velocity.json``) is a single proprioceptive network that estimates
the base velocity internally, so the controller needs no state estimator.  It brings the robot
from the floor to walking and back::

    INIT ──▶ CALIBRATE ──▶ FOLD ──▶ STAND_UP ──▶ STANDING
                              ▲                          │
                              │                          ▼
                              └── STAND_DOWN ◀───── RL ⇄ SAFE_STOP

    any state ──[emergency]──▶ DAMPING ──▶ DONE

* INIT holds the measured posture, then CALIBRATE estimates the accelerometer bias while the robot
  lies still on its belly, legs folded.
* FOLD lifts the body into ``q_fold``, a crouch, and waits there.  From here on haa keeps the feet
  at the lateral position they lay with (the stance), so they are not dragged sideways.
* STAND_UP / STAND_DOWN rise straight to the policy's default posture and back, with quintic timing.
  From the fold on, a feed-forward torque carries the robot's weight, so moderate gains reach the
  default posture instead of sagging below it.
* RL runs the walking policy at 50 Hz on the 500 Hz loop; SAFE_STOP runs the ``safe`` variant
  (trained at zero command) on the same observation history, so the hand-over works mid-stride.
* DAMPING (kp = 0) lets the robot sink under its own weight, from any state.
* With --safety-filter (off by default), RL hands over to SAFE_STOP by itself when the value
  function (components/safe_rl) says the walking policy is about to fall, and back to RL, at zero
  command, once it has recovered.  A safe stop the operator asked for stays until the operator ends
  it.

    python3 rl_quadruped_controller.py --robot go2                 # gazebo, keyboard
    python3 rl_quadruped_controller.py --robot go2 --safety-filter # gazebo, with the safety filter
    ./run_rl_controller.sh --robot go2 --real --input joy          # real robot
    ./run_rl_controller.sh --robot go2 --real --input joy --safety-filter
"""

import os

# Before numpy: one thread for the numeric libraries, the policy is a small MLP.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import gc
import queue
import subprocess
import sys
import threading
import time as wall_time
import xml.etree.ElementTree as ElementTree
from enum import Enum

import numpy as np
import pinocchio as pin
import rospkg
import rospy as ros
import yaml
from gazebo_msgs.srv import ApplyBodyWrench, ApplyBodyWrenchRequest
from geometry_msgs.msg import TwistStamped, Vector3, Vector3Stamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64, Float64MultiArray, MultiArrayDimension, String
from termcolor import colored
from tf.transformations import euler_from_quaternion

import base_controllers.rl_quadruped.rl_controller_config as conf
from base_controllers.base_controller import BaseController
from base_controllers.components.imu_utils import IMU_utils
from base_controllers.rl_quadruped.commandInterface import Event, create_command_interface
from base_controllers.rl_quadruped.rl_controller_config import get_config, known_robots
from base_controllers.rl_quadruped.velocity_policy import VelocityPolicy
from base_controllers.utils.common_functions import getRobotModelFloating
from base_controllers.utils.pidManager import PidManager
from ros_impedance_controller.msg import EffortPid


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


POLICY_STATES = (State.RL, State.SAFE_STOP)   # switching between these keeps the history
SAFE_VARIANT = "safe"
ZERO_COMMAND = np.zeros(3)

TELEMETRY_RATE = 100.0          # Hz, IMU terms and attitude
STATS_RATE = 10.0               # Hz, loop timing
SCHED_FIFO_PRIORITY = 80
GC_THRESHOLDS = (2000, 10)      # gen0, gen1; the full collection runs only at resting states

# safety filter (--safety-filter, components/safe_rl/README.md): V at FILTER_RATE from
# FILTER_START_DELAY s after the walking policy starts, as it was tested (the value function never
# saw the gait start)
FILTER_RATE = 100.0             # Hz
FILTER_START_DELAY = 1.0        # s

# _checkActuation
ACTUATION_TORQUE_MIN = 2.0      # Nm commanded before judging
ACTUATION_MEASURED_MIN = 0.5    # Nm measured below which the motors produce nothing
ACTUATION_SPEED_MIN = 0.05      # rad/s below which a joint is still
ACTUATION_SETTLE = 1.0          # s the command must have been loaded


def quintic(alpha):
    """Quintic scaling s(alpha) and ds/dalpha, zero velocity and acceleration at both ends."""
    alpha = min(max(alpha, 0.0), 1.0)
    a2 = alpha * alpha
    s = a2 * alpha * (10.0 - 15.0 * alpha + 6.0 * a2)
    sd = 30.0 * a2 * (1.0 - alpha) * (1.0 - alpha)
    return s, sd


def lowest_point(shape, placement):
    """Height of the lowest point of a collision primitive (hpp-fcl) at ``placement``; meshes
    count by their origin."""
    kind, rotation, centre = type(shape).__name__, placement.rotation, placement.translation
    if kind == "Sphere":
        return centre[2] - shape.radius
    if kind == "Box":
        return centre[2] - np.abs(rotation[2]) @ shape.halfSide
    if kind in ("Cylinder", "Capsule"):
        tilt = abs(rotation[2, 2])
        rim = shape.radius if kind == "Capsule" else shape.radius * np.sqrt(max(0.0, 1.0 - tilt**2))
        return centre[2] - shape.halfLength * tilt - rim
    return centre[2]


class RlQuadrupedController(BaseController):

    def __init__(self, robot_name="aliengo", input_device="keyboard", real_robot=False,
                 realtime=None, telemetry=None, skip_calibration=None, use_rviz=None,
                 network_interface=None, safety_filter=False, filter_threshold=None,
                 filter_model=None):
        self.cfg = get_config(robot_name, real_robot=real_robot)
        super(RlQuadrupedController, self).__init__(robot_name, external_conf=conf,
                                                   broadcast_world=False)
        self.input_device = input_device
        self.dt = self.cfg["dt"]

        def pick(override, key):
            return self.cfg[key] if override is None else override

        self.realtime = pick(realtime, "realtime")
        if self.realtime is None:
            self.realtime = self.real_robot
        self.enable_telemetry = pick(telemetry, "publish_telemetry")
        self.skip_calibration = pick(skip_calibration, "skip_calibration")
        self.use_rviz = pick(use_rviz, "use_rviz")
        self.network_interface = pick(network_interface, "network_interface")
        # the safety filter's ValueFunctionManager, None when off
        self.use_safety_filter = safety_filter
        self.filter_threshold = filter_threshold
        self.filter_model = filter_model
        self.safety_filter = None

        self.state = State.INIT
        self.prev_state = None
        self.state_start_time = 0.0
        self.state_first_tick = True
        self._quit_requested = False
        self._fold_moving = False
        # Tracked rather than measured: the fold is held with a sag of ~0.17 rad against gravity.
        self._folded = False
        self._calibrated = False
        self.push_enabled = False
        self._actuation_reported = False
        self._actuation_loaded_since = None

    # ---------------------------------------------------------------------------------------------
    # setup
    # ---------------------------------------------------------------------------------------------
    def check_faulty_ping(self, ip):
        response = os.system("ping -c 1 -W 1 " + ip)
        if response == 0:
            print(colored(f"Robot network active: ping {ip} ok", "green"))
        else:
            print(colored(f"Cannot ping {ip}: bring up the local network with that gateway", "red"))
        return response

    def _checkNetworkInterface(self):
        """A driver bound to a missing NIC waits for lowstate forever, so check it exists."""
        name = self.network_interface
        if not name or os.path.isdir(os.path.join("/sys/class/net", name)):
            return
        present = sorted(entry for entry in os.listdir("/sys/class/net")
                         if not entry.startswith(("veth", "br-", "docker", "lo")))
        sys.exit(f"Network interface '{name}' does not exist here (have {present}). Set "
                 f"'network_interface' in rl_controller_config.py or pass --network-interface.")

    def startController(self, world_name=None, additional_args=None):
        if self.real_robot:
            self._checkNetworkInterface()
            if self.check_faulty_ping(self.cfg["ip"]):
                sys.exit("Cannot reach the robot")
        else:
            self._fixSpawnHeight()

        args = ["rviz:=" + ("true" if self.use_rviz else "false")]
        if self.real_robot:
            args.append("task_period:=" + str(self.dt))
            if self.network_interface:
                args.append("network_interface:=" + self.network_interface)
        else:
            args.append("go0_conf:=" + self.cfg["spawn_posture_name"])
        args.extend(additional_args or [])
        self.startSimulator(world_name=None if self.real_robot else
                            (world_name or self.cfg["world_name"]), additional_args=args)
        self.loadModelAndPublishers()

        self.policy = VelocityPolicy(self.robot_name, dt=self.dt)
        if SAFE_VARIANT not in self.policy.networks:
            raise ValueError(f"{self.robot_name}_velocity.json has no '{SAFE_VARIANT}' variant "
                             f"(have {self.policy.variants})")
        self.kp_rl = np.full(self.robot.na, self.policy.kp)
        self.kd_rl = np.full(self.robot.na, self.policy.kd)
        if self.use_safety_filter:
            self._initSafetyFilter()

        self.initVars()
        self.initSubscribers()
        self._initCommandMessage()
        self.rate = ros.Rate(1.0 / self.dt)
        self.command = create_command_interface(self.input_device, self.cfg)
        self._initTelemetry()
        self._initPushes()
        self._waitForSensors()
        self.pid = PidManager(self.joint_names)
        if ros.get_param("/pid_discrete_implementation", False):
            print(colored("WARNING: /pid_discrete_implementation is true, so the joint PD is not "
                          "the one the policy was trained with. Relaunch with "
                          "pid_discrete_implementation:=false.", "red", attrs=["bold"]))
        if self.realtime:
            if not self.real_robot:
                print(colored("Real-time scheduling in simulation starves Gazebo; it is meant for "
                              "--real", "yellow"))
            self._applyRealtimeTuning()
        print(colored("RlQuadrupedController ready", "green"))

    def _initCommandMessage(self):
        """One reused /command message and the clipping bounds (90 % of the URDF limits)."""
        self._cmd_msg = JointState()
        self._cmd_msg.name = self.joint_names
        na = self.robot.na
        lower = self.robot.model.lowerPositionLimit[-na:] * 0.9
        upper = self.robot.model.upperPositionLimit[-na:] * 0.9
        self._q_min = np.minimum(lower, upper)
        self._q_max = np.maximum(lower, upper)
        self._qd_max = self.robot.model.velocityLimit[-na:] * 0.9
        self._tau_max = self.robot.model.effortLimit[-na:] * 0.9

    def send_des_jstate(self, q_des, qd_des, tau_ffwd):
        """Publish the joint command; on the real robot it is clipped to the joint limits."""
        msg = self._cmd_msg
        if self.real_robot:
            msg.position = np.clip(q_des, self._q_min, self._q_max)
            msg.velocity = np.clip(qd_des, -self._qd_max, self._qd_max)
            msg.effort = np.clip(tau_ffwd, -self._tau_max, self._tau_max)
        else:
            msg.position = q_des
            msg.velocity = qd_des
            msg.effort = tau_ffwd
        self.pub_des_jstate.publish(msg)

    def initSubscribers(self):
        ns = "/" + self.robot_name
        self.sub_jstate = ros.Subscriber(ns + "/joint_states", JointState, self._receive_jstate,
                                         queue_size=1, tcp_nodelay=True)
        self.sub_pid_effort = ros.Subscriber(ns + "/effort_pid", EffortPid,
                                             self._receive_pid_effort, queue_size=1,
                                             tcp_nodelay=True)
        if self.real_robot:
            # orientation and rate on /imu, the accelerometer on /trunk_imu
            self.sub_imu = ros.Subscriber(ns + "/imu", Imu, self._receive_imu, queue_size=1,
                                          tcp_nodelay=True)
            self.sub_imu_acc = ros.Subscriber(ns + "/trunk_imu", Vector3,
                                              self._receive_imu_acc_real, queue_size=1,
                                              tcp_nodelay=True)
        else:
            self.sub_imu = ros.Subscriber(ns + "/trunk_imu", Imu, self._receive_imu_sim,
                                          queue_size=1, tcp_nodelay=True)
            self.sub_pose = ros.Subscriber(ns + "/ground_truth", Odometry,
                                           self._receive_ground_truth, queue_size=1,
                                           tcp_nodelay=True)

    def initVars(self):
        """Replaces BaseController.initVars, which allocates large log buffers."""
        na = self.robot.na
        self.time = 0.0

        self.q = np.zeros(na)
        self.qd = np.zeros(na)
        self.tau = np.zeros(na)
        self.tau_fb = np.zeros(na)
        self.quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        self.euler = np.zeros(3)
        self.angVelB = np.zeros(3)
        self.baseLinAccB = np.zeros(3)
        self._acc_sum = np.zeros(3)
        self._acc_count = 0
        self._acc_lock = threading.Lock()
        self.projected_gravity = np.array([0.0, 0.0, -1.0])
        self.b_R_w = np.eye(3)
        self.basePoseW = np.zeros(6)
        self.gt_baseTwistW = np.zeros(6)

        self.q_des = np.zeros(na)
        self.qd_des = np.zeros(na)
        self.tau_ffwd = np.zeros(na)
        self.velocity_cmd = np.zeros(3)
        self.rl_action = np.zeros(na)
        self.filter_value = np.nan
        self._filter_walk = True
        self._filter_backup = False      # SAFE_STOP entered by the safety filter, not the operator
        self._filter_from = 0.0

        self.imu_utils = IMU_utils(timeout=int(round(self.cfg["calibration_duration"] / self.dt)),
                                   dt=self.dt)

        # The fold and stand postures take haa from the stance (_keepStance) once it is measured.
        self._haa = [i for i, name in enumerate(self.joint_names) if name.endswith("haa_joint")]
        self._haa_joints = [self.robot.model.getJointId(self.joint_names[i]) for i in self._haa]
        self._feet = [self.robot.model.getFrameId(frame) for frame in self.cfg["ee_frames"]]
        self.stance_y = None
        self.stance_haa = None
        self.q_fold = np.asarray(self.cfg["q_fold"], dtype=np.float64)
        self.q_stand = self.policy.q_default.copy()
        self.tau_fold = self._standingTorques(self.q_fold)
        self.tau_stand = self._standingTorques(self.q_stand)
        self._traj_from = np.zeros(na)
        self._traj_to = np.zeros(na)
        self._traj_load = (0.0, 0.0)

        self._got_jstate = False
        self._got_imu = False
        self._got_imu_acc = False
        self._last_inference_count = 0
        self._active_gains = None

        # loop health
        self.tick_count = 0
        self._last_loop_stamp = None
        self.loop_overruns = 0
        self.worst_loop_dt = 0.0
        self.worst_compute_dt = 0.0          # excluding state-entry ticks
        self.worst_entry_compute_dt = 0.0
        self._stats_dt_sum = 0.0
        self._stats_dt_max = 0.0
        self._stats_ticks = 0
        self._stats_overruns = 0
        self._stats_compute_sum = 0.0
        self._stats_compute_max = 0.0
        self._stats_compute_ticks = 0

        self._calibReset(announce=False)

    # ---------------------------------------------------------------------------------------------
    # simulation start-up
    # ---------------------------------------------------------------------------------------------
    def _startupPosture(self):
        """The posture go0 applies at spawn, in joint order."""
        package = rospkg.RosPack().get_path(f"{self.robot_name}_description")
        name = self.cfg["spawn_posture_name"]
        if self.cfg["spawn_posture_source"] == "srdf":
            path = os.path.join(package, "robots", f"{self.robot_name}.srdf.xacro")
            values = {joint.get("name"): joint.get("value")
                      for state in ElementTree.parse(path).getroot().iter("group_state")
                      if state.get("name") == name for joint in state.iter("joint")}
        else:
            with open(os.path.join(package, "config", "ros_impedance_controller.yaml")) as f:
                values = yaml.safe_load(f)["ros_impedance_controller"][name]
        return np.array([float(values[joint]) for joint in self.joint_names])

    def _ensureGeneratedUrdf(self):
        """getRobotModelFloating falls back to robot_urdf/generated_urdf/<robot>.urdf before the
        launch has loaded /robot_description; generate it if missing."""
        urdf = os.path.join(os.environ["LOCOSIM_DIR"], "robot_urdf", "generated_urdf",
                            f"{self.robot_name}.urdf")
        if os.path.isfile(urdf):
            return True
        try:
            xacro = os.path.join(rospkg.RosPack().get_path(f"{self.robot_name}_description"),
                                 "robots", f"{self.robot_name}.urdf.xacro")
            os.makedirs(os.path.dirname(urdf), exist_ok=True)
            subprocess.check_output(["rosrun", "xacro", "xacro", xacro, "-o", urdf],
                                    stderr=subprocess.STDOUT)
        except (rospkg.ResourceNotFound, subprocess.CalledProcessError, OSError) as exc:
            print(colored(f"Could not generate {urdf} ({exc})", "yellow"))
            return False
        return True

    def _fixSpawnHeight(self):
        """Set spawn_z so the robot starts on the ground in the start-up posture, on whatever
        touches it first (the belly when lying): a drop onto the floor makes the robot ring and
        spoils the IMU calibration."""
        try:
            posture = self._startupPosture()
        except (OSError, KeyError, ValueError, TypeError, ElementTree.ParseError, yaml.YAMLError,
                rospkg.ResourceNotFound) as exc:
            print(colored(f"Cannot read the start-up posture ({exc}); keeping spawn_z", "yellow"))
            return
        if not self._ensureGeneratedUrdf():
            return
        robot = getRobotModelFloating(self.robot_name)
        pin.framesForwardKinematics(robot.model, robot.data,
                                    np.hstack([pin.neutral(robot.model)[:7], posture]))
        if robot.collision_model.ngeoms:
            pin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model,
                                         robot.collision_data)
            bottom = min(lowest_point(obj.geometry, placement) for obj, placement in
                         zip(robot.collision_model.geometryObjects, robot.collision_data.oMg))
        else:
            bottom = min(robot.data.oMf[robot.model.getFrameId(frame)].translation[2]
                         for frame in self.cfg["ee_frames"]) - self.cfg["foot_radius"]
        height = -float(bottom) + self.cfg["spawn_clearance"]
        conf.robot_params[self.robot_name]["spawn_z"] = height
        self.base_offset[2] = height
        print(colored(f"Spawn height {height:.3f} m for the {self.cfg['spawn_posture_source']} "
                      f"'{self.cfg['spawn_posture_name']}' posture", "cyan"))

    def _waitForSensors(self, timeout=40.0):
        """Block until joint states and both IMU signals arrive (wall clock: sim time can run
        faster than the spawner comes up)."""
        print("Waiting for joint states and IMU...")
        deadline = wall_time.monotonic() + timeout
        while not ros.is_shutdown() and wall_time.monotonic() < deadline:
            if self._got_jstate:
                # Hold the measured posture; otherwise the low-level controller drives to the
                # yaml home posture while we start up.
                self.q_des[:] = self.q
                self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
                if self._got_imu and self._got_imu_acc:
                    print(colored("Sensors alive", "green"))
                    return
            wall_time.sleep(0.05)
        missing = [name for name, got in (("joint_states", self._got_jstate),
                                          ("imu orientation", self._got_imu),
                                          ("imu acceleration", self._got_imu_acc)) if not got]
        raise RuntimeError(f"No data on: {', '.join(missing)}")

    # ---------------------------------------------------------------------------------------------
    # real-time tuning
    # ---------------------------------------------------------------------------------------------
    def _applyRealtimeTuning(self):
        """Freeze the start-up heap out of the GC, defer the full collection to resting states and
        put the process on SCHED_FIFO."""
        gc.collect()
        gc.freeze()
        gc.set_threshold(*GC_THRESHOLDS, 1 << 30)
        self._applyRealtimePriority()

    def _applyRealtimePriority(self):
        """SCHED_FIFO for every thread or none: raising only the control thread starves the rospy
        threads that deliver the sensors.  The control thread gets SCHED_FIFO_PRIORITY, the others
        ten less."""
        helper_priority = SCHED_FIFO_PRIORITY - 10
        main_tid = os.getpid()
        tids = [int(entry) for entry in os.listdir("/proc/self/task")]
        raised = []
        try:
            for tid in tids:
                priority = SCHED_FIFO_PRIORITY if tid == main_tid else helper_priority
                os.sched_setscheduler(tid, os.SCHED_FIFO, os.sched_param(priority))
                raised.append(tid)
        except OSError as exc:
            for tid in raised:
                try:
                    os.sched_setscheduler(tid, os.SCHED_OTHER, os.sched_param(0))
                except OSError:
                    pass
            print(colored(f"Could not set SCHED_FIFO ({exc}); normal scheduling. Launch through "
                          f"./run_rl_controller.sh for real-time priority.", "yellow"))
            return
        print(colored(f"SCHED_FIFO: control thread at {SCHED_FIFO_PRIORITY}, {len(tids) - 1} "
                      f"other thread(s) at {helper_priority}", "yellow"))

    def _collectAtSafePoint(self):
        if self.realtime and self.state in (State.INIT, State.CALIBRATE, State.FOLD,
                                            State.STANDING, State.DONE):
            gc.collect()

    # ---------------------------------------------------------------------------------------------
    # pushes (simulation only)
    # ---------------------------------------------------------------------------------------------
    def _initPushes(self):
        """Pushes are Gazebo wrenches applied from a worker thread, so the loop never waits on the
        service."""
        self.push_enabled = not self.real_robot
        self._push_queue = queue.Queue()
        self._push_count = 0
        if not self.push_enabled:
            return
        self._push_rng = np.random.default_rng()
        threading.Thread(target=self._pushWorker, name="push", daemon=True).start()

    def _pushWorker(self):
        try:
            ros.wait_for_service("/gazebo/apply_body_wrench", timeout=20.0)
        except ros.ROSException as exc:
            print(colored(f"No /gazebo/apply_body_wrench ({exc}); pushes are unavailable", "red"))
            self.push_enabled = False
            return
        apply_wrench = ros.ServiceProxy("/gazebo/apply_body_wrench", ApplyBodyWrench)
        while not ros.is_shutdown():
            try:
                count, force = self._push_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            for index in range(count):
                if ros.is_shutdown():
                    return
                if index:
                    wall_time.sleep(float(self._push_rng.uniform(*self.cfg["push_burst_interval"])))
                self._applyPush(apply_wrench, force)

    def _applyPush(self, apply_wrench, force):
        """One horizontal shove in a uniformly random heading, held for push_duration."""
        heading = float(self._push_rng.uniform(0.0, 2.0 * np.pi))
        vector = force * np.array([np.cos(heading), np.sin(heading), 0.0])
        request = ApplyBodyWrenchRequest()
        request.body_name = f"{self.robot_name}::base_link"
        request.reference_frame = "world"      # the only frame the service honours
        request.wrench.force.x, request.wrench.force.y, request.wrench.force.z = vector
        request.start_time = ros.Time(0)
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
        if self.enable_telemetry:
            self._msg_push.header.stamp = ros.Time.now()
            self._msg_push.vector.x, self._msg_push.vector.y, self._msg_push.vector.z = vector
            self.pub_push.publish(self._msg_push)
        print(colored(f"PUSH #{self._push_count}: {force:.0f} N at {np.degrees(heading):5.0f} deg "
                      f"for {self.cfg['push_duration']:.2f} s (state '{self.state.value}')",
                      "magenta"))

    def requestPush(self, count=1):
        if not self.push_enabled:
            print(colored("Pushes are simulation only", "yellow"))
            return
        self._push_queue.put((int(count), float(self.command.push_force)))

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

    def _receive_imu(self, msg):
        quat = msg.orientation
        self.quaternion[:] = quat.x, quat.y, quat.z, quat.w
        self.euler[:] = euler_from_quaternion(self.quaternion)
        self.basePoseW[3:6] = self.euler
        self.b_R_w = self.math_utils.rpyToRot(self.euler)       # world -> base
        self.projected_gravity[:] = -self.b_R_w[:, 2]
        self.angVelB[:] = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        self._got_imu = True

    def _receive_imu_acc_real(self, msg):
        self._accumulateImuAcc(msg.x, msg.y, msg.z)

    def _receive_imu_sim(self, msg):
        self._receive_imu(msg)
        acc = msg.linear_acceleration
        self._accumulateImuAcc(acc.x, acc.y, acc.z)

    def _accumulateImuAcc(self, x, y, z):
        with self._acc_lock:
            self._acc_sum += (x, y, z)
            self._acc_count += 1
        self._got_imu_acc = True

    def _latchImuAcc(self):
        """This tick's accelerometer reading is the mean of every sample since the last tick (Gazebo
        publishes at 1 kHz); a tick without samples keeps the previous one."""
        with self._acc_lock:
            if self._acc_count:
                np.divide(self._acc_sum, self._acc_count, out=self.baseLinAccB)
                self._acc_sum[:] = 0.0
                self._acc_count = 0

    def _receive_ground_truth(self, msg):
        position, linear = msg.pose.pose.position, msg.twist.twist.linear
        self.basePoseW[:3] = position.x, position.y, position.z
        self.gt_baseTwistW[:3] = linear.x, linear.y, linear.z

    def imu_lin_acc(self):
        """Specific force in the base frame, bias removed (about (0, 0, 9.81) when level)."""
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
        """Push the joint gains (ki = 0), skipping the /set_pids call if they did not change.
        The gains go through a service and the target through a topic, so the joint PD can pair
        the new gains with the old target for a tick (e.g. stiff gains on the policy's offset
        target).  An error-free target that feeds forward the torque the joints get now bridges
        the switch whichever arrives first."""
        kp = np.asarray(kp, dtype=np.float64)
        kd = np.asarray(kd, dtype=np.float64)
        if self._active_gains is not None and np.array_equal(self._active_gains[0], kp) \
                and np.array_equal(self._active_gains[1], kd):
            return
        self.send_des_jstate(self.q, np.zeros(self.robot.na), self.tau_fb + self.tau_ffwd)
        self.pid.setPDjoints(kp, kd, np.zeros(self.robot.na))
        self._active_gains = (kp.copy(), kd.copy())

    def _hold(self, q, tau_ffwd=0.0):
        self.q_des[:] = q
        self.qd_des[:] = 0.0
        self.tau_ffwd[:] = tau_ffwd

    def _standingTorques(self, q):
        """Joint torques that carry the robot's weight on its four feet in posture ``q``, base
        level: static balance g(q) = S^T tau + J^T f, with the minimum-norm foot forces f."""
        model, data = self.robot.model, self.robot.data
        q_full = np.hstack([pin.neutral(model)[:7], q])
        pin.computeJointJacobians(model, data, q_full)
        pin.updateFramePlacements(model, data)
        gravity = pin.computeGeneralizedGravity(model, data, q_full)
        jac = np.vstack([pin.getFrameJacobian(model, data, model.getFrameId(frame),
                                              pin.LOCAL_WORLD_ALIGNED)[:3]
                         for frame in self.cfg["ee_frames"]])
        forces = np.linalg.lstsq(jac[:, :6].T, gravity[:6], rcond=None)[0]
        return gravity[6:] - jac[:, 6:].T @ forces

    def _measureStance(self):
        """Record the feet's lateral position in the base frame and fit the fold and stand
        postures to it, so no later move drags the feet sideways against friction."""
        model, data = self.robot.model, self.robot.data
        pin.framesForwardKinematics(model, data, np.hstack([pin.neutral(model)[:7], self.q]))
        self.stance_y = np.array([data.oMf[foot].translation[1] for foot in self._feet])
        self.stance_haa = self.q[self._haa].copy()
        self.q_fold = self._keepStance(np.asarray(self.cfg["q_fold"], dtype=np.float64))
        self.q_stand = self._keepStance(self.policy.q_default)
        self.tau_fold = self._standingTorques(self.q_fold)
        self.tau_stand = self._standingTorques(self.q_stand)

    def _keepStance(self, q):
        """``q`` with each haa set so its foot keeps the stance.  With haa at zero the foot is at
        p in the haa frame (axis x); haa = theta puts it at p_y cos(theta) - p_z sin(theta), the
        solution nearer the given haa.  Tilting the leg rolls the foot sphere, which moves its
        centre by -radius * (theta - stance_haa) without slipping: a few Newton steps add that."""
        model, data, radius = self.robot.model, self.robot.data, self.cfg["foot_radius"]
        q = np.array(q, dtype=np.float64)
        flat = q.copy()
        flat[self._haa] = 0.0
        pin.framesForwardKinematics(model, data, np.hstack([pin.neutral(model)[:7], flat]))
        for leg, (i, joint, foot) in enumerate(zip(self._haa, self._haa_joints, self._feet)):
            hip = data.oMi[joint]
            p = hip.actInv(data.oMf[foot].translation)
            reach, phase = np.hypot(p[1], p[2]), np.arctan2(p[2], p[1])
            lateral = (self.stance_y[leg] - hip.translation[1]) / reach
            spread = np.arccos(np.clip(lateral, -1.0, 1.0))
            haa = min((spread - phase, -spread - phase), key=lambda theta: abs(theta - q[i]))
            for _ in range(3):
                error = (reach * np.cos(haa + phase) + hip.translation[1] - self.stance_y[leg]
                         + radius * (haa - self.stance_haa[leg]))
                haa -= error / (radius - reach * np.sin(haa + phase))
            q[i] = haa
        return q

    def handleEvents(self):
        for event in self.command.poll():
            if event is Event.STOP:
                print(colored("EMERGENCY: collapsing on joint damping", "red", attrs=["bold"]))
                self.transition(State.DAMPING)
                return
            if event is Event.QUIT:
                print(colored("Quit requested: collapsing on joint damping first", "yellow"))
                self._quit_requested = True
                self.transition(State.DAMPING)
                return
            if event is Event.PUSH:
                self.requestPush(1)
            elif event is Event.PUSH_BURST:
                self.requestPush(self.cfg["push_burst_count"])
            elif event is Event.CALIBRATE:
                if self.state is State.INIT or (self.state is State.FOLD and not self._fold_moving):
                    self.transition(State.CALIBRATE)
                else:
                    print(colored("Calibrate only at start-up or once folded", "yellow"))
            elif event is Event.STAND_UP:
                if self.state is State.FOLD:
                    self.transition(State.STAND_UP)
                else:
                    print(colored(f"Cannot stand up from '{self.state.value}'", "yellow"))
            elif event is Event.START_RL:
                if self.state in (State.STANDING, State.SAFE_STOP):
                    self._filter_backup = False     # the operator resumes: the filter starts over
                    self.transition(State.RL)
                else:
                    print(colored(f"Cannot start the policy from '{self.state.value}'", "yellow"))
            elif event is Event.SAFE_STOP:
                if self.state in (State.RL, State.STANDING, State.SAFE_STOP):
                    self._filter_backup = False     # the operator's now: the filter will not end it
                    self.transition(State.SAFE_STOP)
                else:
                    print(colored(f"No safe stop from '{self.state.value}': it needs the robot "
                                  f"standing", "yellow"))
            elif event is Event.STAND_DOWN:
                if self.state in (State.RL, State.SAFE_STOP, State.STANDING):
                    self.transition(State.STAND_DOWN)
                else:
                    print(colored(f"Cannot stand down from '{self.state.value}'", "yellow"))

    def _init(self):
        """Hold the posture the robot was placed in while it settles, then calibrate."""
        if self.state_first_tick:
            self._setGains(self.cfg["kp_fold"], self.cfg["kd_fold"])
            self._hold(self.q)
            print(colored("Holding the measured posture while it settles", "cyan"))
        if self.state_elapsed < self.cfg["settle_duration"]:
            return
        if self.skip_calibration:
            print(colored("Skipping the accelerometer calibration: the bias stays zero", "yellow"))
            self.transition(State.FOLD)
        else:
            self.transition(State.CALIBRATE)

    def _isQuiescent(self):
        return (np.linalg.norm(self.angVelB) < self.cfg["calib_gyro_threshold"] and
                np.max(np.abs(self.qd)) < self.cfg["calib_joint_vel_threshold"])

    def _calibrate(self):
        """Accelerometer bias over a continuously still window: wait calib_stable_window s of
        stillness, then average calibration_duration s; any motion starts over.  The result is
        checked on its spread and on the residual |acc - bias| - |g|."""
        if self.state_first_tick:
            print(colored(f"Calibrating the accelerometer bias: keep the robot still for "
                          f"{self.cfg['calib_stable_window']:.1f} + "
                          f"{self.cfg['calibration_duration']:.1f} s", "cyan"))
            # Re-triggered from the fold, keep holding it.
            if self._folded:
                self._hold(self.q_fold, self.tau_fold)
            else:
                self._hold(self.q)
            self.imu_utils.IMU_accelerometer_bias[:] = 0.0
            self._calibReset(announce=False)

        if not self._isQuiescent():
            if self._calib_n > 0 or self._calib_stable_since is not None:
                self._calibReset(announce=True)
            if self.state_elapsed - self._calib_warned > 3.0:
                self._calib_warned = self.state_elapsed
                print(colored(f"Waiting for the robot to be still: |gyro| "
                              f"{np.linalg.norm(self.angVelB):.3f} rad/s, max|qd| "
                              f"{np.max(np.abs(self.qd)):.3f} rad/s", "yellow"))
            if self.state_elapsed > self.cfg["calib_wait_timeout"]:
                print(colored("The robot never became still: no calibration, the bias stays zero. "
                              "Re-trigger it from the fold with 'c' / Y.", "red"))
                self.transition(State.FOLD)
            return

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

        bias = self._calib_sum / self._calib_n
        spread = np.sqrt(np.maximum(self._calib_sq_sum / self._calib_n - bias * bias, 0.0))
        mean_acc = self._calib_acc_sum / self._calib_n
        residual = np.linalg.norm(mean_acc - bias) - np.linalg.norm(self.imu_utils.g0)

        if np.linalg.norm(bias) > self.cfg["calib_max_bias"]:
            print(colored(f"Rejecting an implausible bias {np.round(bias, 3)} m/s^2: check that the "
                          f"IMU reports specific force (+9.81 on z when level)", "red"))
            self.imu_utils.IMU_accelerometer_bias[:] = 0.0
        elif np.max(spread) > self.cfg["calib_max_spread"]:
            if self.state_elapsed < self.cfg["calib_wait_timeout"]:
                print(colored(f"Sample spread {np.round(spread, 3)} m/s^2 is too large: the robot "
                              f"was not still. Retrying.", "yellow"))
                self._calibReset(announce=False)
                return
            print(colored(f"Sample spread {np.round(spread, 3)} m/s^2 still too large after "
                          f"{self.cfg['calib_wait_timeout']:.0f} s: no calibration, the bias stays "
                          f"zero. Re-trigger it from the fold with 'c' / Y.", "red"))
        else:
            self.imu_utils.IMU_accelerometer_bias[:] = bias
            self._calibrated = True
            if abs(residual) > self.cfg["calib_max_residual"]:
                print(colored(f"Calibration residual {residual:+.3f} m/s^2 is large (mean reading "
                              f"{np.round(mean_acc, 3)}): keeping the bias, but check the IMU",
                              "yellow"))
            else:
                print(colored(f"Accelerometer bias {np.round(bias, 4)} m/s^2 ({self._calib_n} "
                              f"samples, spread {np.round(spread, 4)}, residual "
                              f"{residual:+.4f})", "green"))
        self.transition(State.FOLD)

    def _calibReset(self, announce):
        if announce and self._calib_n > 0:
            print(colored(f"Motion after {self._calib_n} calibration samples: starting over",
                          "yellow"))
        self._calib_sum = np.zeros(3)
        self._calib_sq_sum = np.zeros(3)
        self._calib_acc_sum = np.zeros(3)
        self._calib_n = 0
        self._calib_stable_since = None
        self._calib_warned = -1e9

    def _fold(self):
        """From lying on the belly, lift the body into q_fold on the fold gains while the weight
        feed-forward ramps in, the feet keeping the stance they lay with; then hold it."""
        if self.state_first_tick:
            self._setGains(self.cfg["kp_fold"], self.cfg["kd_fold"])
            self._fold_moving = not self._folded
            if self._fold_moving:
                self._measureStance()
                self._startTrajectory(self.q_fold, self.cfg["init_fold_duration"], load=(0.0, 1.0))
                print(colored(f"Folding to {np.round(self.q_fold, 3)}", "cyan"))
            else:
                print(colored("Folded. Command a stand up when the robot is clear.", "cyan"))
        if self._fold_moving:
            if self._stepTrajectory():
                self._fold_moving = False
                self._folded = True
                print(colored("Folded. Command a stand up when the robot is clear.", "cyan"))
            return
        self._hold(self.q_fold, self.tau_fold)

    def _startTrajectory(self, target, duration, load):
        """Quintic move from the measured posture to ``target``; ``load`` (start, end) scales the
        weight feed-forward along it."""
        self._traj_from[:] = self.q
        self._traj_to[:] = target
        self._traj_duration = duration
        self._traj_load = load
        self._traj_start = self.time

    def _stepTrajectory(self):
        """Advance the move, haa following the stance; True once the target is reached."""
        alpha = (self.time - self._traj_start) / self._traj_duration
        s, sd = quintic(alpha)
        delta = self._traj_to - self._traj_from
        q_des = self._traj_from + s * delta
        qd_des = np.zeros_like(delta) if alpha >= 1.0 else delta * sd / self._traj_duration
        if self.stance_y is not None:
            ahead = self._keepStance(q_des + qd_des * self.dt)
            q_des = self._keepStance(q_des)
            qd_des[self._haa] = (ahead - q_des)[self._haa] / self.dt
        self.q_des[:] = q_des
        self.qd_des[:] = qd_des
        load = self._traj_load[0] + s * (self._traj_load[1] - self._traj_load[0])
        self.tau_ffwd[:] = load * self._standingTorques(self.q_des) if load > 0.0 else 0.0
        return alpha >= 1.0

    def _stand_up(self):
        if self.state_first_tick:
            self._folded = False
            if not self._calibrated:
                print(colored("Standing up without a calibrated accelerometer", "yellow"))
            self._setGains(self.cfg["kp_stand"], self.cfg["kd_stand"])
            self._startTrajectory(self.q_stand, self.cfg["stand_up_duration"], load=(1.0, 1.0))
            print(colored(f"Standing up to {np.round(self.q_stand, 3)}", "cyan"))
        if self._stepTrajectory():
            self.transition(State.STANDING)

    def _standing(self):
        if self.state_first_tick:
            self._hold(self.q_stand, self.tau_stand)
            print(colored("Standing. Command the RL policy to start walking.", "cyan"))
        if self._checkSafety():
            self.transition(State.DAMPING)

    def _enterPolicy(self, variant, announce):
        """Start a policy state.  The history is reset only when coming from a non-policy state:
        a hand-over between the two variants keeps it.  The stored command is always zeroed."""
        self._setGains(self.kp_rl, self.kd_rl)
        self.command.zero()
        self.velocity_cmd[:] = 0.0
        if self.prev_state not in POLICY_STATES:
            self.policy.reset(self.q, self.qd, self.imu_lin_acc(), self.angVelB,
                              self.projected_gravity, velocity_cmd=self.velocity_cmd)
        self.policy.select(variant)
        self.publishPolicyVariant()
        print(colored(announce, "cyan"))

    def _stepPolicy(self, command):
        self.velocity_cmd[:] = command
        self.q_des[:] = self.policy.step(self.q, self.qd, self.imu_lin_acc(), self.angVelB,
                                         self.projected_gravity, self.velocity_cmd)
        self.rl_action[:] = self.policy.action
        self.qd_des[:] = 0.0
        self.tau_ffwd[:] = 0.0
        if self._checkSafety():
            self.transition(State.DAMPING)

    def _rl(self):
        if self.state_first_tick:
            back_from_filter = self._filter_backup and self.prev_state is State.SAFE_STOP
            self._enterPolicy(self.policy.default_variant,
                              f"Policy '{self.policy.default_variant}' running: kp={self.policy.kp:.1f} "
                              f"kd={self.policy.kd:.2f}, {self.policy.policy_rate:.0f} Hz"
                              + (", the safety filter ended its safe stop" if back_from_filter else ""))
            self._filter_backup = False
            if not back_from_filter:
                self._resetSafetyFilter()
        self._stepPolicy(self.command.get_velocity_cmd())
        if self.state is State.RL and not self._filterSaysWalk():
            self._filter_backup = True
            self.transition(State.SAFE_STOP)

    def _safe_stop(self):
        """Hold station on the zero-command policy.  Leaving is an operator command, or, when the
        safety filter entered it, the filter's once V has recovered."""
        if self.state_first_tick:
            if self.prev_state is not State.RL:
                self._filter_backup = False
            if self._filter_backup:
                announce = (f"SAFE STOP by the safety filter (V {self.filter_value:.2f} below "
                            f"{self.safety_filter.threshold}): policy '{SAFE_VARIANT}' at zero command, "
                            f"back to walking once V has recovered.")
            else:
                announce = (f"SAFE STOP: policy '{SAFE_VARIANT}' at zero command. Resume, stand down or "
                            f"emergency damping.")
            self._enterPolicy(SAFE_VARIANT, announce)
        self._stepPolicy(ZERO_COMMAND)
        if self._filter_backup and self.state is State.SAFE_STOP and self._filterSaysWalk():
            self.transition(State.RL)

    # ---------------------------------------------------------------------------------------------
    # safety filter
    # ---------------------------------------------------------------------------------------------
    def _initSafetyFilter(self):
        """Load the value function (ONNX, onnxruntime on one thread like the policy) and run it
        once, so the first evaluation in RL does not stall the loop."""
        from base_controllers.components.safe_rl.value_function.value_function_manager import (
            ValueFunctionManager)
        self.safety_filter = ValueFunctionManager(robot_name=self.robot_name, model_path=self.filter_model,
                                                  threshold=self.filter_threshold)
        self._filter_decim = max(1, int(round(1.0 / (FILTER_RATE * self.dt))))
        self.safety_filter.value(np.array([0.0, 0.0, -1.0]), np.zeros(3), np.asarray(self.policy.q_default),
                                 np.zeros(self.robot.na))
        print(colored(f"Safety filter on: V below {self.safety_filter.threshold} switches to the "
                      f"'{SAFE_VARIANT}' policy", "cyan"))

    def _resetSafetyFilter(self):
        """Walking again, from the operator: the filter starts over, FILTER_START_DELAY s from now."""
        if self.safety_filter is not None:
            self.safety_filter.reset()
            self._filter_walk = True
            self._filter_from = self.time + FILTER_START_DELAY

    def _filterSaysWalk(self):
        """The safety filter's verdict: True to walk, False for the safe policy (always True when
        the filter is off).  V is evaluated at FILTER_RATE; the manager switches after one value
        below the threshold, and back after 2 s above it."""
        if self.safety_filter is None or self.time < self._filter_from:
            return True
        if self.tick_count % self._filter_decim == 0:
            self._filter_walk, self.filter_value = self.safety_filter.computeValueFnc(
                self.projected_gravity, self.angVelB, self.q, self.qd)
            if self.enable_telemetry:
                self._publishScalar(self.pub_filter_value, self.filter_value)
                self._publishScalar(self.pub_filter_backup, float(not self._filter_walk))
        return self._filter_walk

    def _checkSafety(self):
        limit = self.cfg["max_roll_pitch"]
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
            self._measureStance()
            self._startTrajectory(self.q_fold, self.cfg["stand_down_duration"], load=(1.0, 1.0))
            print(colored("Standing down", "cyan"))
        if self._stepTrajectory():
            self._folded = True
            self.transition(State.FOLD)

    def _damping(self):
        """kp = 0, so the torque is -kd qd and the robot sinks; feed-forward is ramped out."""
        if self.state_first_tick:
            self._damping_tau0 = self.tau_ffwd.copy()
            self._setGains(np.zeros(self.robot.na), self.cfg["kd_damping"])
            print(colored(f"Damping mode: kp=0, kd={self.cfg['kd_damping'][0]:.1f}", "red"))
        ramp = self.cfg["damping_ramp_duration"]
        scale = max(0.0, 1.0 - self.state_elapsed / ramp) if ramp > 0.0 else 0.0
        self.tau_ffwd[:] = scale * self._damping_tau0
        self.qd_des[:] = 0.0
        self.q_des[:] = self.q
        if self.state_elapsed >= ramp + self.cfg["damping_settle_duration"]:
            print(colored("Robot settled", "green"))
            self.transition(State.DONE)

    def _done(self):
        self._setGains(np.zeros(self.robot.na), np.zeros(self.robot.na))
        self._hold(self.q)

    def _checkActuation(self):
        """Real robot, once: compare the commanded feedback torque with the motors' tau_est."""
        if self._actuation_reported or not self.real_robot:
            return
        commanded = float(np.max(np.abs(self.tau_fb)))
        if commanded < ACTUATION_TORQUE_MIN:
            self._actuation_loaded_since = None
            return
        if self._actuation_loaded_since is None:
            self._actuation_loaded_since = self.time
        if self.time - self._actuation_loaded_since < ACTUATION_SETTLE:
            return
        self._actuation_reported = True
        measured = float(np.max(np.abs(self.tau)))
        speed = float(np.max(np.abs(self.qd)))
        print(colored(f"Actuation check: commanded |tau| up to {commanded:.2f} Nm, motors report "
                      f"{measured:.2f} Nm, fastest joint {speed:.3f} rad/s", "cyan"))
        if float(np.max(np.abs(self.q))) < 1e-6:
            print(colored("  -> all joint positions are zero: no lowstate from the robot",
                          "red", attrs=["bold"]))
        elif measured < ACTUATION_MEASURED_MIN:
            print(colored("  -> the motors produce no torque: the robot is not executing the "
                          "commands", "red", attrs=["bold"]))
        elif speed < ACTUATION_SPEED_MIN:
            print(colored("  -> torque but no motion: the gains are too low for the joint friction",
                          "yellow", attrs=["bold"]))
        else:
            print(colored("  -> actuation is working", "green"))

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
            self._latchImuAcc()
            self.command.update()
            self.handleEvents()

            state_before = self.state
            entry_tick = self.state_first_tick
            if entry_tick:
                self._collectAtSafePoint()
                self.publishState()
            self.HANDLERS[state_before](self)
            # a handler that transitioned has armed the flag for the next state
            if self.state is state_before:
                self.state_first_tick = False

            self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
            self._checkActuation()
            self.publishTelemetry()

            compute_dt = wall_time.monotonic() - tick_start
            if entry_tick:
                self.worst_entry_compute_dt = max(self.worst_entry_compute_dt, compute_dt)
            else:
                self._stats_compute_sum += compute_dt
                self._stats_compute_ticks += 1
                self._stats_compute_max = max(self._stats_compute_max, compute_dt)
                self.worst_compute_dt = max(self.worst_compute_dt, compute_dt)

            # DONE runs once, releasing the joints, before the loop ends
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
            loop_dt = stamp - self._last_loop_stamp
            self._stats_dt_sum += loop_dt
            self._stats_ticks += 1
            self._stats_dt_max = max(self._stats_dt_max, loop_dt)
            self.worst_loop_dt = max(self.worst_loop_dt, loop_dt)
            if loop_dt > 1.5 * self.dt:
                self.loop_overruns += 1
                self._stats_overruns += 1
        self._last_loop_stamp = stamp

    def shutdown(self):
        print(colored("\nShutting down", "yellow"))
        self.command.shutdown()
        if self.tick_count:
            budget = self.dt * 1e3
            print(colored(f"Compute per tick: worst {self.worst_compute_dt * 1e3:.3f} ms of "
                          f"{budget:.2f} ms ({self.worst_entry_compute_dt * 1e3:.3f} ms on "
                          f"state-entry ticks)", "green"))
            note = "" if self.real_robot else " (in simulation the period follows Gazebo)"
            print(colored(f"Loop period over 1.5x nominal on {self.loop_overruns} of "
                          f"{self.tick_count} ticks, worst {self.worst_loop_dt * 1e3:.2f} ms{note}",
                          "yellow" if self.loop_overruns else "green"))
        if self.realtime:
            gc.set_threshold(700, 10, 10)
            gc.unfreeze()
        ros.signal_shutdown("controller finished")

    # ---------------------------------------------------------------------------------------------
    # telemetry, under /<robot>/rl/
    # ---------------------------------------------------------------------------------------------
    def _initTelemetry(self):
        """Create the publishers with preallocated messages."""
        ns = f"/{self.robot_name}/rl"
        self.telemetry_topics = []

        def publisher(name, msg_type, latch=False):
            self.telemetry_topics.append(f"{ns}/{name}")
            return ros.Publisher(f"{ns}/{name}", msg_type, queue_size=1, latch=latch,
                                 tcp_nodelay=True)

        self.pub_state = publisher("state", String, latch=True)
        self._msg_state = String()
        self.pub_variant = publisher("policy_variant", String, latch=True)
        self._msg_variant = String()

        # at the policy rate
        self.pub_cmd = publisher("velocity_command", TwistStamped)
        self._msg_cmd = TwistStamped()
        self.pub_action = publisher("action", Float64MultiArray)
        self._msg_action = Float64MultiArray()
        dim = MultiArrayDimension(label="|".join(self.joint_names), size=self.robot.na,
                                  stride=self.robot.na)
        self._msg_action.layout.dim.append(dim)
        self._msg_action.data = [0.0] * self.robot.na
        self.pub_imu_acc_obs = publisher("policy/imu_lin_acc", Vector3Stamped)  # what the net sees
        self._msg_imu_acc_obs = Vector3Stamped()
        if self.policy.has_estimator:
            self.pub_estimate = publisher("estimated_base_lin_vel", Vector3Stamped)
            self._msg_estimate = Vector3Stamped()

        # at TELEMETRY_RATE
        self.pub_imu_acc = publisher("imu_lin_acc", Vector3Stamped)
        self.pub_proj_grav = publisher("projected_gravity", Vector3Stamped)
        self.pub_ang_vel = publisher("ang_vel_b", Vector3Stamped)
        self.pub_rpy = publisher("base_rpy", Vector3Stamped)
        self._msg_imu_acc = Vector3Stamped()
        self._msg_proj_grav = Vector3Stamped()
        self._msg_ang_vel = Vector3Stamped()
        self._msg_rpy = Vector3Stamped()

        # at STATS_RATE
        self.pub_loop_mean = publisher("loop/dt_mean_ms", Float64)
        self.pub_loop_max = publisher("loop/dt_max_ms", Float64)
        self.pub_loop_overruns = publisher("loop/overruns", Float64)
        self.pub_compute_mean = publisher("loop/compute_mean_ms", Float64)
        self.pub_compute_max = publisher("loop/compute_max_ms", Float64)
        self.pub_inferences = publisher("policy/inferences", Float64)
        self._msg_scalar = Float64()

        if self.safety_filter is not None:
            # at FILTER_RATE: V and whether the filter wants the safe policy (1)
            self.pub_filter_value = publisher("safety_filter/value", Float64)
            self.pub_filter_backup = publisher("safety_filter/backup", Float64)

        self.pub_push = publisher("push", Vector3Stamped, latch=True)
        self._msg_push = Vector3Stamped()

        self._telemetry_decim = max(1, int(round(1.0 / (TELEMETRY_RATE * self.dt))))
        self._stats_decim = max(1, int(round(1.0 / (STATS_RATE * self.dt))))

    def publishState(self):
        if self.enable_telemetry:
            self._msg_state.data = self.state.value
            self.pub_state.publish(self._msg_state)

    def publishPolicyVariant(self):
        if self.enable_telemetry:
            self._msg_variant.data = self.policy.active_variant
            self.pub_variant.publish(self._msg_variant)

    def _publishVector(self, publisher, msg, vector, stamp):
        msg.header.stamp = stamp
        msg.vector.x, msg.vector.y, msg.vector.z = vector[0], vector[1], vector[2]
        publisher.publish(msg)

    def _publishScalar(self, publisher, value):
        self._msg_scalar.data = value
        publisher.publish(self._msg_scalar)

    def publishTelemetry(self):
        if not self.enable_telemetry:
            return

        if self.policy.inference_count != self._last_inference_count:
            self._last_inference_count = self.policy.inference_count
            stamp = ros.Time.now()
            self._msg_cmd.header.stamp = stamp
            self._msg_cmd.twist.linear.x = self.velocity_cmd[0]
            self._msg_cmd.twist.linear.y = self.velocity_cmd[1]
            self._msg_cmd.twist.angular.z = self.velocity_cmd[2]
            self.pub_cmd.publish(self._msg_cmd)
            self._msg_action.data[:] = self.rl_action
            self.pub_action.publish(self._msg_action)
            self._publishVector(self.pub_imu_acc_obs, self._msg_imu_acc_obs,
                                self.policy.imu_lin_acc_obs, stamp)
            if self.policy.has_estimator:
                self._publishVector(self.pub_estimate, self._msg_estimate,
                                    self.policy.estimated_base_lin_vel, stamp)

        if self.tick_count % self._telemetry_decim == 0:
            stamp = ros.Time.now()
            self._publishVector(self.pub_imu_acc, self._msg_imu_acc, self.imu_lin_acc(), stamp)
            self._publishVector(self.pub_proj_grav, self._msg_proj_grav, self.projected_gravity,
                                stamp)
            self._publishVector(self.pub_ang_vel, self._msg_ang_vel, self.angVelB, stamp)
            self._publishVector(self.pub_rpy, self._msg_rpy, self.euler, stamp)
            # world -> base_link for rviz: ground-truth position in simulation, origin on the robot
            self.broadcaster.sendTransform(self.basePoseW[:3], self.quaternion, stamp,
                                           '/base_link', '/world')

        if self.tick_count % self._stats_decim == 0 and self._stats_ticks > 0:
            self._publishScalar(self.pub_loop_mean, 1e3 * self._stats_dt_sum / self._stats_ticks)
            self._publishScalar(self.pub_loop_max, 1e3 * self._stats_dt_max)
            self._publishScalar(self.pub_loop_overruns, float(self._stats_overruns))
            if self._stats_compute_ticks:
                self._publishScalar(self.pub_compute_mean,
                                    1e3 * self._stats_compute_sum / self._stats_compute_ticks)
                self._publishScalar(self.pub_compute_max, 1e3 * self._stats_compute_max)
            self._publishScalar(self.pub_inferences, float(self.policy.inference_count))
            self._stats_dt_sum = self._stats_dt_max = 0.0
            self._stats_compute_sum = self._stats_compute_max = 0.0
            self._stats_ticks = self._stats_overruns = self._stats_compute_ticks = 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="aliengo", choices=known_robots())
    parser.add_argument("--input", default="keyboard", choices=["keyboard", "joy", "none"],
                        help="operator input device")
    parser.add_argument("--real", action="store_true", help="run on the real robot")
    parser.add_argument("--network-interface", dest="network_interface", default=None,
                        metavar="NAME", help="NIC the robot is wired to (Go2); default from the "
                                             "config")
    parser.add_argument("--skip-calibration", dest="skip_calibration", action="store_true",
                        default=None, help="leave the accelerometer bias at zero")
    parser.add_argument("--world", default=None, help="gazebo world; default from the config")
    parser.add_argument("--gui", action="store_true", help="show the gazebo gui")
    parser.add_argument("--no-rviz", dest="use_rviz", action="store_false", default=None,
                        help="do not start rviz")
    parser.add_argument("--no-telemetry", dest="telemetry", action="store_false", default=None,
                        help="do not publish the telemetry topics")
    parser.add_argument("--bag", nargs="?", const="", default=None, metavar="PREFIX",
                        help="record the telemetry and joint topics with rosbag")
    parser.add_argument("--realtime", dest="realtime", action="store_true", default=None,
                        help="force the real-time tuning on (default: only with --real)")
    parser.add_argument("--no-realtime", dest="realtime", action="store_false",
                        help="skip the real-time tuning")
    parser.add_argument("--safety-filter", dest="safety_filter", action="store_true",
                        help="switch to the safe policy when the value function says the walking "
                             "policy is about to fall (components/safe_rl); off by default")
    parser.add_argument("--filter-threshold", dest="filter_threshold", type=float, default=None,
                        help="the safety filter's threshold on V; default per robot")
    parser.add_argument("--filter-model", dest="filter_model", default=None, metavar="ONNX",
                        help="the value function; default "
                             "components/safe_rl/value_function/models/VF_push_end_<robot>.onnx")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    p = RlQuadrupedController(args.robot, input_device=args.input, real_robot=args.real,
                              realtime=args.realtime, telemetry=args.telemetry,
                              skip_calibration=args.skip_calibration, use_rviz=args.use_rviz,
                              network_interface=args.network_interface,
                              safety_filter=args.safety_filter, filter_threshold=args.filter_threshold,
                              filter_model=args.filter_model)
    recorder = None
    try:
        p.startController(world_name=args.world,
                          additional_args=["gui:=" + str(args.gui).lower()])
        if args.bag is not None:
            topics = [f"/{args.robot}/joint_states", "/command"] + p.telemetry_topics
            command = ["rosbag", "record", "-O", args.bag or f"rl_{args.robot}"] + topics
            print(colored("Recording: " + " ".join(command), "cyan"))
            recorder = subprocess.Popen(command)
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
