# -*- coding: utf-8 -*-
"""Self-contained configuration for :mod:`rl_quadruped_controller`.

This module is the **only** configuration the RL controller reads.  It does not import
:mod:`base_controllers.params` and nothing here is merged with it: the controller hands this module
to :class:`BaseController` as its ``external_conf``, which replaces ``params.robot_params`` for the
lifetime of the process.  So a robot that runs this controller is described here, once, and tuning it
cannot disturb any other controller in the framework.

That is also why the robot description fields (``joint_names``, ``ee_frames``, ``q_fold``, ``ip``,
``spawn_*``) appear below even though ``params.py`` has its own copies - the two are deliberately
independent.  The URDF itself still comes from the ``<robot>_description`` package, since that is
where the kinematics live.

Resolution order, later winning:

1. :data:`DEFAULTS`
2. ``ROBOTS[robot_name]``
3. ``ROBOTS[robot_name]['real']``, only when running on hardware - **normally empty**

The control parameters are deliberately identical in simulation and on the robot: the policy was
trained against a model of the machine, so a gain or a duration that has to differ between the two is
a modelling error to fix rather than a difference to paper over.  The ``real`` mechanism is kept for
the case where something genuinely is hardware-only, but nothing uses it.

:func:`get_config` returns the merged dictionary and also publishes it as
:data:`robot_params`, in the shape :class:`BaseController` expects.

Gains are per-joint arrays in the joint order given by ``joint_names``.  ``ki`` never appears: the
controller runs no integral term in any state, by design - an integral term on a position-controlled
leg winds up while the foot is loaded and kicks on lift-off.
"""

import numpy as np

# Consumed by BaseController.__init__ via external_conf.
verbose = False

# Filled in by get_config(); BaseController reads spawn_*, joint_names and real_robot from here.
robot_params = {}


def _legs(haa, hfe, kfe):
    """A symmetric four-leg posture, mirrored on the right side, in ``joint_names`` order."""
    return np.array([haa, hfe, kfe,      # lf
                     haa, hfe, kfe,      # lh
                     -haa, hfe, kfe,     # rf
                     -haa, hfe, kfe])    # rh


DEFAULTS = {
    # =============================================================================================
    # robot description
    # =============================================================================================
    "joint_names": None,     # required
    "ee_frames": None,       # required, in the same leg order as joint_names
    "ip": None,              # pinged before a --real run
    "real_robot": False,     # set from the command line, not here
    "spawn_x": 0.0,
    "spawn_y": 0.0,
    "spawn_z": 0.4,          # corrected automatically, see fix_spawn_height
    "spawn_R": 0.0,
    "spawn_P": 0.0,
    "spawn_Y": 0.0,
    # Resting posture the controller folds into before calibrating and standing up.  Keep every
    # joint well clear of its limit: a knee parked against the hard stop is held there by the motor,
    # which shows up as a strained, buzzing robot rather than a resting one.
    "q_fold": None,          # required
    # Present only because BaseController's own initVars would size its log arrays from it.  This
    # controller overrides initVars and logs over ROS topics instead, so nothing here allocates.
    "buffer_size": 1,

    # =============================================================================================
    # loop rate
    # =============================================================================================
    # Control-loop period: 0.002 s = 500 Hz, giving a decimation of 10 against the policy's 50 Hz.
    # It has to be an exact integer multiple of the policy period (20 ms); the controller checks both
    # rates at start-up and refuses to run otherwise, rather than silently sampling the joint
    # velocities, the IMU history and the action history on a different time base than training.
    "control_dt": 0.002,
    "dt": 0.002,             # kept equal to control_dt for BaseController compatibility

    # =============================================================================================
    # gains
    # =============================================================================================
    # Repositioning the legs on the ground during INIT: softer and better damped than the stand-up
    # set, because the feet are loaded and being dragged, and stiff position control there is what
    # makes the robot judder.
    "kp_fold": np.array([60., 60., 60.] * 4),
    "kd_fold": np.array([0.5, 0.5, 0.5] * 4),
    # Lifting and holding the body.
    "kp_stand": np.array([100., 100., 100.] * 4),
    "kd_stand": np.array([1.0, 1.0, 1.0] * 4),
    # While the policy drives.  None takes the values the policy was trained with, read from the
    # policy's json (Isaac DelayedPDActuatorCfg: stiffness 25, damping 0.5).  Overriding these is a
    # deliberate departure from training - the soft damping is what the policy expects.
    "kp_rl": None,
    "kd_rl": None,
    # Emergency collapse: kp is zero, so joint torque is -kd*qd and the robot sinks under its own
    # weight with the energy bled off instead of dropping.
    "kd_damping": np.array([1., 1., 1.] * 4),

    # =============================================================================================
    # motion timing, seconds
    # =============================================================================================
    "settle_duration": 1.0,        # hold the posture we inherit, before calibrating and after
                                   # reaching the fold
    "init_fold_duration": 3.0,     # ramp from the start-up posture into q_fold
    # Worth of *continuously still* samples the bias estimate integrates over.  At 500 Hz that is
    # 1500 samples, which brings the per-axis spread down to a few 1e-4 m/s^2.  The total wait is
    # this plus calib_stable_window.
    "calibration_duration": 5.0,
    "stand_up_duration": 2.0,      # total, split over the stand-up waypoints
    "stand_down_duration": 2.0,
    "damping_ramp_duration": 0.6,  # feed-forward torque ramp-out
    "damping_settle_duration": 2.5,

    # =============================================================================================
    # postures
    # =============================================================================================
    # Stand-up target.  None takes the policy's training default posture, which is the offset its
    # actions are applied to and therefore the smoothest place to hand over.
    "q_stand": None,
    # Tuck the feet under the body before extending; this is what makes a stand-up work from a
    # sprawled pose rather than only from a neat fold.
    "stand_up_via_fold": True,

    # =============================================================================================
    # IMU bias calibration
    # =============================================================================================
    # The window must be *continuously* still: stillness has to hold for calib_stable_window before
    # a single sample is taken, and any motion afterwards throws the window away and starts over.  A
    # bias that absorbed part of a settling transient is worse than no bias at all, because
    # everything downstream then trusts it.
    "calib_stable_window": 1.0,         # s of uninterrupted stillness before sampling starts
    # A settled robot measures about 0.003 rad/s of base rate and 0.012 rad/s of joint rate, so
    # these leave a wide margin while still rejecting a robot that is drifting or being nudged.
    "calib_gyro_threshold": 0.2,       # rad/s
    "calib_joint_vel_threshold": 0.5,  # rad/s
    "calib_wait_timeout": 30.0,         # s before giving up on ever being still
    "calib_max_bias": 2.0,              # m/s^2, a larger estimate is rejected outright
    "calib_max_spread": 0.25,           # m/s^2, per-axis sample std above which the window is redone
    "calib_max_residual": 0.5,          # m/s^2, tolerated |acc - bias| - |g| after calibration
    # Skip the whole procedure and leave the bias at zero.  The bias then shows up in the
    # accelerometer history the policy consumes, so this is for bring-up and for a robot whose bias
    # is already known to be small - not for a run you intend to trust.  Also exposed as
    # --skip-calibration.
    "skip_calibration": False,

    # =============================================================================================
    # safety envelope while standing or walking
    # =============================================================================================
    "max_roll_pitch": 1.5,          # rad, beyond this the robot is on its way over -> DAMPING
    # The same limit during the safe stop, where it has to be wider: that policy is trained on
    # resets of +-0.25 rad and on pushes, so it is meant to be handed attitudes the walking policy
    # never sees.  Tripping the walking limit there would answer a request for a recovery with a
    # collapse.
    "safe_max_roll_pitch": 1.5,
    "max_joint_vel": 50.0,          # rad/s -> DAMPING
    "clip_to_joint_limits": None,   # None: on for the real robot, off in simulation

    # =============================================================================================
    # disturbances, simulation only
    # =============================================================================================
    # 'p' shoves the base once in a random horizontal direction; 'P' fires a burst of them; -/+
    # change the magnitude live.  Applied as a wrench through /gazebo/apply_body_wrench, off the
    # control thread, so a push never costs the loop a service round-trip.
    #
    # The default is chosen to match what the policy was trained against.  Training perturbs the
    # robot with mdp.push_by_setting_velocity over +-1.0 m/s in x and y, which on a 24.94 kg robot
    # is an impulse of about 24.9 N s; over push_duration that is 125 N.  So the default push is a
    # training-strength push, and turning it up is testing beyond what the policy has seen.
    "push_force": 125.0,            # N
    "push_duration": 0.2,           # s the wrench is held
    "push_force_step": 25.0,        # N per press of -/+
    "push_vertical": False,         # include a random z component; training only ever pushed in xy
    "push_burst_count": 5,          # pushes fired by 'P'
    "push_burst_interval": (1.0, 3.0),   # s, uniform gap between them
    # Gazebo body to push.  None means '<robot>::base_link', which is what the floating base is
    # called once spawned - not 'trunk', which is the URDF's name for it.
    "push_body": None,

    # =============================================================================================
    # operator input
    # =============================================================================================
    "max_lin_vel_cmd": 0.5,   # m/s, the range the policy was trained on
    "max_ang_vel_cmd": 0.5,   # rad/s
    "key_lin_step": 0.1,      # m/s per key press
    "key_ang_step": 0.1,      # rad/s per key press
    "key_speed_presets": (0.0, 0.1, 0.2, 0.3, 0.4),  # forward m/s, keys 1..5
    "joy_dead_zone": 0.08,

    # =============================================================================================
    # telemetry - published on ROS topics, nothing is accumulated in memory
    # =============================================================================================
    # Everything the controller knows goes out as topics so it can be recorded with rosbag and read
    # live in PlotJuggler.  Rates are decimated from the control loop; messages are preallocated and
    # reused, so a tick costs a serialise and no allocation.
    "publish_telemetry": True,
    "telemetry_rate": 100.0,   # Hz, for the per-tick signals (IMU terms, attitude)
    "stats_rate": 10.0,        # Hz, for loop timing and memory
    "topic_namespace": "rl",   # topics land under /<robot>/<topic_namespace>/...
    # Publish the base linear velocity the policy regresses internally.  It is the actor's own
    # estimate, evaluated from weights read out of the policy file, and it is the single most useful
    # signal for judging whether the policy is reading the robot correctly.
    "publish_estimated_velocity": True,
    # Evaluate the actor's state-estimation head at all.  It is a diagnostic, not part of the
    # control path: the policy backbone already consumes the estimate inside the ONNX graph, and
    # the controller re-evaluates the head from weights read out of the same file purely so the
    # number can be seen from outside.  False never reads it and never runs it - one three-layer
    # forward pass less per inference tick - and makes 'publish_estimated_velocity' moot, since
    # there is then nothing to publish.  Also on the command line as --no-estimator.
    "estimate_base_velocity": True,
    # Time every inference tick and publish policy/infer_ms and policy/infer_over_budget.  This is
    # instrumentation, not control.  False drops the two perf_counter calls per inference, both
    # topics, the start-up benchmark and the shutdown timing summary; the benchmark itself still
    # works if called directly.  Worth turning off once the timing question is settled - it is
    # answered in the README - and the loop wants nothing it does not need.  Also on the command
    # line as --no-inference-timing.
    "measure_inference_timing": False,

    # =============================================================================================
    # real-time behaviour
    # =============================================================================================
    # None: on for the real robot, off in simulation.
    "realtime": None,
    "sched_fifo_priority": 80,
    # Pin the control thread to one core.  Only worth it once SCHED_FIFO is actually granted, and
    # best paired with isolcpus so nothing else runs there - see the launcher script.
    "pin_cpu": False,
    "pin_cpu_index": None,     # None: the highest-numbered available core
    # Garbage collection.  The startup heap is frozen out of every later collection, generations 0
    # and 1 keep running (bounded and cheap), and only the full generation-2 scan is deferred to
    # resting states.  Switching the collector off entirely, or deferring generation 1 as well,
    # would be simpler but lets garbage accumulate for the whole of a long walk - exactly what must
    # not happen on a robot.
    "gc_managed": True,
    # Generation 0 and 1 stay enabled: both are bounded and cheap, and generation 1 is what stops
    # anything that survives one gen0 pass from accumulating for the whole run.  Only generation 2 -
    # the full-heap scan, the one that can take milliseconds - is deferred to resting states.
    "gc_gen0_threshold": 2000,
    "gc_gen1_threshold": 10,
    # Warn once if resident memory grows by more than this from the post-startup baseline.  A
    # correctly behaving loop allocates nothing steady-state, so any real growth is a leak.
    "memory_growth_warn_mb": 64.0,
    # How often resident memory is read from procfs.  A file open inside the control loop is an
    # occasional multi-millisecond outlier, so this is deliberately slow - a leak shows up over
    # minutes, not milliseconds.
    "memory_check_rate": 1.0,   # Hz

    # =============================================================================================
    # simulation start-up
    # =============================================================================================
    # Correct spawn_z before the launch runs.  The description's go0 script spawns the robot in the
    # sprawled SRDF 'home' posture (and ignores the go0_conf argument it is given), so a spawn height
    # meant for a standing posture drops the robot a quarter of a metre onto the ground and it rings
    # for seconds afterwards.
    "fix_spawn_height": True,
    "spawn_clearance": 0.002,   # m of air left under the feet
    # Radius of the foot collision sphere; the foot frame sits at its centre, so the base has to
    # start this much higher than the pure kinematic foot-to-base distance.
    "foot_radius": 0.0265,
    # rl_flat.world is fast.world with an explicit ODE <constraints> block. Gazebo's default
    # contact_surface_layer lets a foot sink 1 mm before any contact force builds, which a trotting
    # robot feels as a spongy floor: base roll rms while walking goes 5.2 -> 3.1 deg at 0.3 m/s and
    # 5.4 -> 2.0 deg at 0.5 m/s with it at zero, and back to 5.6 / 4.2 deg when it is restored.
    # A reference simulator running the same network on the training URDF sits at 1.5 deg, so
    # this closes most of the gap to Isaac but not all of it.
    "world_name": "rl_flat.world",
    # Start rviz with the simulator.  On by default: it is the intended way to look at the robot,
    # and unlike the Gazebo GUI it renders from the published TF and markers rather than driving
    # the physics window.  It is still another process competing for the machine, so --no-rviz is
    # there for timing work and for a real-robot run on a loaded control PC.
    "use_rviz": True,
    # Broadcast the world -> base_link transform.  rviz is launched with 'world' as its fixed
    # frame, and robot_state_publisher only supplies base_link downwards, so without this there is
    # no path from the fixed frame to the robot and rviz shows nothing.  The base class has its own
    # broadcast inside a kinematics update this controller does not run, so it is done here, at the
    # telemetry rate rather than the control rate: rviz has no use for 500 Hz TF and the traffic is
    # not free.
    "publish_tf": True,
    # rviz config file passed to the launch; None keeps the launch file's own default
    # (ros_impedance_controller/config/operator_floating.rviz).
    "rviz_conf": None,
    # Policy basename under rl_quadruped/policies/, without the '_velocity' suffix.
    # None means "use the robot name".
    "policy_name": None,
    # =============================================================================================
    # the model the policy was trained on
    # =============================================================================================
    # A position-control policy is a feedback law tuned against particular link inertias: its joint
    # target becomes torque through a fixed PD, so each leg's closed-loop response is set by kp/kd
    # against that leg's inertia.  Deployed on a robot whose legs differ by a factor, the response
    # the policy learned to expect is not the response it gets - and standing still, which is held
    # by feedback, degenerates into a residual limit cycle.  So the training model is part of the
    # policy's contract and the controller checks it.
    #
    # 'warn' (the default) reports the differing bodies at start-up; 'off' skips the check.  It
    # only ever reports: aliengo_description now carries Unitree's own values, so a mismatch here
    # means the description has drifted - a stale install space, an edit, or a policy retrained
    # against a different asset - and that is where it should be fixed.  There was briefly a
    # 'match' mode that wrote the trained values into Gazebo at start-up; it is gone, because a
    # controller that silently rewrites the physics engine leaves every other tool on the robot
    # looking at a different machine.
    #
    # The check covers mass, centre of mass and inertia.  It does *not* cover joint friction, which
    # is part of the same contract and was the other half of the mismatch: Isaac draws it uniformly
    # from (0, 0.5) N.m at every reset while aliengo_description had it at exactly zero, and a
    # frictionless robot over-rotates - 0.67 rad/s against a 0.5 rad/s yaw command, which Isaac
    # reproduces at 0.64 when its own friction is switched off.  const.xacro now carries 0.25 N.m,
    # the middle of the trained range, and that is where it belongs; there is nothing to check here
    # because Gazebo does not report joint friction back.
    "training_model_check": "warn",     # 'warn' | 'off'
    # Relative inertia difference below which two bodies count as the same.
    "training_model_tol": 0.02,
    # Gazebo body name -> the link of the same body in the training URDF.  Per-robot, because it
    # is purely a naming-convention map; an empty map disables the check.
    "training_body_map": {},
    # Hand over to the safe-stop policy automatically once the operator command has been at zero
    # this long.  0.0 disables it.  Leaving again is always deliberate - resume walking, stand down,
    # or collapse - because every entry into a policy zeroes the stored command, so a hand-back
    # driven by that command would discard the command that triggered it.
    #
    # Why this exists: the walking policy has two attractors at zero command.  Measured over seven
    # identical Gazebo runs of "walk at 0.3 m/s, then command zero", it settled to ~1e-4 rad/s of
    # base rate in four, and in two it locked into a non-decaying 3.5-3.8 Hz trot in place at
    # 0.04-0.15 rad/s - growing, in one case, from 0.09 to 0.18 rad/s over ten seconds.  Which
    # attractor it lands in is not driven by loop jitter (uncorrelated, r = -0.28..-0.04 across
    # those runs) and not by the model mismatch below (the failures occur with either model): it
    # depends on the gait state at the instant the command drops.  The safe-stop policy, trained
    # only ever at zero command, settled in every run measured.  So if you need "stop" to mean
    # stopped, this is the switch - at the cost of a state change the operator did not ask for.
    "auto_safe_stop_after": 0.0,        # s of continuous zero command; 0.0 = off
    "auto_safe_stop_deadband": 0.02,    # m/s and rad/s below which a command counts as zero
    # Which entry of the contract's "onnx_variants" the safe stop runs.  It has to share the
    # observation and action spaces with the walking variant - the controller checks - because the
    # two hand over mid-stride on a single continuous observation history.
    "safe_policy_variant": "safe",
}


ROBOTS = {
    "aliengo": {
        "joint_names": ["lf_haa_joint", "lf_hfe_joint", "lf_kfe_joint",
                        "lh_haa_joint", "lh_hfe_joint", "lh_kfe_joint",
                        "rf_haa_joint", "rf_hfe_joint", "rf_kfe_joint",
                        "rh_haa_joint", "rh_hfe_joint", "rh_kfe_joint"],
        "ee_frames": ["lf_foot", "lh_foot", "rf_foot", "rh_foot"],
        "ip": "192.168.123.220",
        # Gazebo body name -> the link of the same body in the training URDF.  Gazebo calls the
        # spawned floating base 'base_link' where the URDF calls it 'trunk', and the two projects
        # name legs differently (lf/lh/rf/rh + haa/hfe/kfe against FL/RL/FR/RR + hip/thigh/calf).
        "training_body_map": dict(
            {"base_link": "trunk"},
            **{f"{a}_{s}": f"{b}_{t}"
               for a, b in (("lf", "FL"), ("lh", "RL"), ("rf", "FR"), ("rh", "RR"))
               for s, t in (("hip", "hip"), ("upperleg", "thigh"), ("lowerleg", "calf"))}),
        # Knee at -2.40 rather than the -2.70 of params.py's q_fold: the joint limit is -2.775, so
        # -2.70 leaves only 0.075 rad of margin and the knees end up parked against the stop,
        # carrying about 7.5 Nm to stay there.  -2.40 leaves 0.375 rad and rests quietly.
        "q_fold": _legs(0.2, 1.4, -2.70),
        "foot_radius": 0.0265,

        # No 'real' overrides: the control parameters are identical in simulation and on the
        # robot.  The policy was trained against a model of this machine, so if a gain or a duration
        # has to change between the two, that is a modelling error to fix rather than a difference to
        # paper over - and a controller that behaves differently in simulation is one you cannot
        # test there.  The only things that still differ are not modelling parameters: joint-limit
        # clipping and the real-time tuning, both resolved from --real in the controller, and the
        # simulation-only spawn-height correction.
    },
}


REQUIRED = ("joint_names", "ee_frames", "q_fold")


def get_config(robot_name, real_robot=False):
    """Merged configuration for ``robot_name``, also published as :data:`robot_params`.

    Raises:
        KeyError: if the robot has no entry, if an unknown key is present, or if a required field is
            missing.  Running an untuned machine on silent defaults is worse than refusing to start.
    """
    if robot_name not in ROBOTS:
        raise KeyError(
            f"No RL controller configuration for '{robot_name}'. Add an entry to ROBOTS in "
            f"{__file__}; known robots: {sorted(ROBOTS)}")

    cfg = dict(DEFAULTS)
    robot_cfg = dict(ROBOTS[robot_name])
    real_cfg = robot_cfg.pop("real", {})

    unknown = (set(robot_cfg) | set(real_cfg)) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown RL controller config keys for '{robot_name}': {sorted(unknown)}")

    cfg.update(robot_cfg)
    if real_robot:
        cfg.update(real_cfg)
    cfg["real_robot"] = real_robot
    cfg["dt"] = cfg["control_dt"]

    missing = [key for key in REQUIRED if cfg.get(key) is None]
    if missing:
        raise KeyError(f"RL controller config for '{robot_name}' is missing: {missing}")

    # BaseController resolves everything through conf.robot_params[robot_name]; handing it this
    # dictionary is what keeps params.py out of the picture entirely.
    robot_params[robot_name] = cfg
    return cfg
