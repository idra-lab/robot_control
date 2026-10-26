# -*- coding: utf-8 -*-
"""Configuration of :mod:`rl_quadruped_controller`, the only one it reads.

The controller hands this module to :class:`BaseController` as ``external_conf``, so
``params.py`` plays no part.  :func:`get_config` merges :data:`DEFAULTS` with ``ROBOTS[robot]`` and
publishes the result as :data:`robot_params`.  Gains are per joint, in ``joint_names`` order.
"""

import numpy as np

verbose = False      # read by BaseController
robot_params = {}    # filled by get_config()

_LEGS = ("lf", "lh", "rf", "rh")


def _legs(haa, hfe, kfe):
    """A posture mirrored on the right side, in joint_names order."""
    return np.array([haa, hfe, kfe, haa, hfe, kfe, -haa, hfe, kfe, -haa, hfe, kfe])


def _gains(haa, hfe, kfe):
    """The same gain triple on every leg."""
    return np.array([haa, hfe, kfe] * 4)


def _joint_names():
    return [f"{leg}_{joint}_joint" for leg in _LEGS for joint in ("haa", "hfe", "kfe")]


def _ee_frames():
    return [f"{leg}_foot" for leg in _LEGS]


DEFAULTS = {
    # --- robot -----------------------------------------------------------------------------------
    "joint_names": None,          # required, in the policy's joint order
    "ee_frames": None,            # required, same leg order
    "ip": None,                   # pinged before a --real run
    "network_interface": None,    # NIC the hardware interface binds to (Go2), None = driver default
    "body_mass": None,            # required, kg; sizes the default push
    "q_fold": None,               # required, resting posture, keep the knees off the hard stop
    "real_robot": False,          # set by get_config()
    "dt": 0.002,                  # control period, must divide the policy period (20 ms)

    # --- simulation start-up -------------------------------------------------------------------
    "world_name": "rl_flat.world",
    "use_rviz": True,
    "spawn_x": 0.0,
    "spawn_y": 0.0,
    "spawn_z": 0.4,               # recomputed from the start-up posture before the launch
    "spawn_R": 0.0,
    "spawn_P": 0.0,
    "spawn_Y": 0.0,
    # Where the description's go0 takes the start-up posture from: 'srdf' (the 'home' group state)
    # or 'yaml' (<spawn_posture_name> in config/ros_impedance_controller.yaml).
    "spawn_posture_source": None,  # required
    "spawn_posture_name": "home",
    "spawn_clearance": 0.002,     # m of air under the feet at spawn
    "foot_radius": 0.0265,        # m, the foot frame is at the centre of the collision sphere

    # --- gains (per robot) ------------------------------------------------------------------------
    # The policy runs on the kp/kd in its json.
    "kp_fold": None,
    "kd_fold": None,
    "kp_stand": None,
    "kd_stand": None,
    "kd_damping": None,           # emergency collapse, kp = 0

    # --- timing, s -------------------------------------------------------------------------------
    "settle_duration": 1.0,       # hold the inherited posture before calibrating
    "init_fold_duration": 3.0,
    "stand_up_duration": 2.0,     # split over the two stand-up waypoints (fold, stand)
    "stand_down_duration": 2.0,
    "damping_ramp_duration": 0.6,  # feed-forward ramp-out
    "damping_settle_duration": 2.5,

    # --- accelerometer bias calibration ----------------------------------------------------------
    # Wait for calib_stable_window s of continuous stillness, then average calibration_duration s.
    # Any motion restarts the window.
    "skip_calibration": False,
    "calibration_duration": 5.0,
    "calib_stable_window": 1.0,
    "calib_gyro_threshold": 0.2,       # rad/s
    "calib_joint_vel_threshold": 0.5,  # rad/s
    "calib_wait_timeout": 30.0,        # s, then go on with a zero bias
    "calib_max_bias": 2.0,             # m/s^2, larger is rejected
    "calib_max_spread": 0.25,          # m/s^2, per-axis std above which the window is redone
    "calib_max_residual": 0.5,         # m/s^2, tolerated |acc - bias| - |g|

    # --- safety, -> DAMPING ----------------------------------------------------------------------
    "max_roll_pitch": None,       # rad
    "max_joint_vel": None,        # rad/s

    # --- pushes (simulation) ---------------------------------------------------------------------
    # None: body_mass * 1 m/s / push_duration, the strongest push used in training.
    "push_force": None,           # N
    "push_force_step": None,      # N per -/+ press, None = push_force / 5
    "push_duration": 0.2,         # s
    "push_burst_count": 5,
    "push_burst_interval": (1.0, 3.0),  # s, uniform

    # --- operator input --------------------------------------------------------------------------
    "max_lin_vel_cmd": 0.5,       # m/s, the training command range
    "max_ang_vel_cmd": 0.5,       # rad/s
    "key_lin_step": 0.1,
    "key_ang_step": 0.1,
    "key_speed_presets": (0.0, 0.1, 0.2, 0.3, 0.4),  # forward m/s on keys 1..5
    "joy_dead_zone": 0.08,

    # --- runtime ---------------------------------------------------------------------------------
    "publish_telemetry": True,
    "realtime": None,             # SCHED_FIFO and GC tuning, None = only with --real
}

ROBOTS = {
    "aliengo": {
        "joint_names": _joint_names(),
        "ee_frames": _ee_frames(),
        "ip": "192.168.123.220",
        "body_mass": 24.937,
        "foot_radius": 0.0265,
        "spawn_posture_source": "srdf",
        "spawn_posture_name": "home",
        # knee limit -2.775
        "q_fold": _legs(0.2, 1.4, -2.70),
        "kp_fold": _gains(20., 20., 20.),
        "kd_fold": _gains(0.5, 0.5, 0.5),
        "kp_stand": _gains(60., 60., 60.),
        "kd_stand": _gains(1.0, 1.0, 1.0),
        "kd_damping": _gains(1.0, 1.0, 1.0),
        "max_roll_pitch": 1.5,
        "max_joint_vel": 50.0,    # ~3x the fastest joint while walking
    },
    "go2": {
        "joint_names": _joint_names(),
        "ee_frames": _ee_frames(),
        "ip": "192.168.123.161",
        "network_interface": "enp11s0",
        "body_mass": 16.088,
        "foot_radius": 0.022,
        "spawn_posture_source": "yaml",
        "spawn_posture_name": "standDown",
        # knee limit -2.7227
        "q_fold": _legs(0.2, 1.4, -2.50),
        "kp_fold": _gains(20., 20., 20.),
        "kd_fold": _gains(0.5, 0.5, 0.5),
        "kp_stand": _gains(50., 50., 50.),
        "kd_stand": _gains(0.5, 0.5, 0.5),
        "kd_damping": _gains(1.0, 1.0, 1.0),
        "max_roll_pitch": 1.5,
        "max_joint_vel": 60.0,    # ~3x the fastest joint while walking
    },
}

REQUIRED = ("joint_names", "ee_frames", "q_fold", "body_mass", "spawn_posture_source",
            "kp_fold", "kd_fold", "kp_stand", "kd_stand", "kd_damping",
            "max_roll_pitch", "max_joint_vel")


def known_robots():
    return sorted(ROBOTS)


def get_config(robot_name, real_robot=False):
    """Merged configuration for ``robot_name``, also stored in :data:`robot_params`."""
    if robot_name not in ROBOTS:
        raise KeyError(f"No RL controller configuration for '{robot_name}'; known robots: "
                       f"{known_robots()}")
    unknown = set(ROBOTS[robot_name]) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown config keys for '{robot_name}': {sorted(unknown)}")
    cfg = dict(DEFAULTS, **ROBOTS[robot_name])
    missing = [key for key in REQUIRED if cfg[key] is None]
    if missing:
        raise KeyError(f"Config for '{robot_name}' is missing: {missing}")

    cfg["real_robot"] = real_robot
    if cfg["push_force"] is None:
        cfg["push_force"] = cfg["body_mass"] * 1.0 / cfg["push_duration"]
    if cfg["push_force_step"] is None:
        cfg["push_force_step"] = cfg["push_force"] / 5.0
    robot_params[robot_name] = cfg
    return cfg
