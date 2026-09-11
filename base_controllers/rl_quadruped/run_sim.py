"""Run the velocity policy through a scripted command sequence in Gazebo and report the gait.

    python3 run_sim.py                      # the default sequence, prints a summary
    python3 run_sim.py --record gz.npz      # and save it for gait_report.py
    python3 run_sim.py --rviz --gui

This drives the real controller - its state machine, ros_impedance_controller and Gazebo - through
the scripted command interface, so what it measures is what the robot does, not what a reduced
model of it does.  The value over driving it by hand is repeatability: the same commands for the
same durations every time, so two runs differing only in a parameter can be compared.

Recordings are read by :mod:`gait_report`, which prints the same sections as safe_rl's
``scripts/rsl_rl/eval_gait.py`` so an Isaac rollout and a Gazebo run can be diffed directly.
"""

import argparse
import os
import sys
import time as wall
import threading

import numpy as np

# (name, velocity command, seconds, policy variant)
PHASES = [("settle",  (0.0, 0.0, 0.0), 2.0, "normal"),
          ("fwd0.3",  (0.3, 0.0, 0.0), 8.0, "normal"),
          ("yaw0.5",  (0.0, 0.0, 0.5), 8.0, "normal"),
          ("lat0.3",  (0.0, 0.3, 0.0), 8.0, "normal"),
          ("fwd0.5",  (0.5, 0.0, 0.0), 8.0, "normal"),
          ("safe",    (0.0, 0.0, 0.0), 8.0, "safe")]
"""The last phase hands over to the standstill variant, which is the only way to see what the
backup policy does: it is trained on a zero command and is what the controller reaches for when
something has gone wrong, so its station-keeping is worth measuring rather than assuming."""

RECORD_KEYS = ("sim", "state", "phase", "pose", "twist", "angvel", "q", "qd", "qdes", "cmd",
               "tau", "est", "action", "grf", "foot_z")
"""``grf`` is the normal force on each foot and ``foot_z`` the sole height, both in leg order
FL, RL, FR, RR - the order the policy's joints are in.  They are what turns a recording into the
gait report that :mod:`gait_report` prints, and therefore what makes a deployment run comparable
with safe_rl's own ``scripts/rsl_rl/eval_gait.py`` output from Isaac."""


# =================================================================================================
# metrics
# =================================================================================================
def summarise(log, title):
    """Per-phase gait metrics."""
    pose, twist, angvel = log["pose"], log["twist"], log["angvel"]
    phase, command, tau = log["phase"], log["cmd"], log["tau"]
    yaw = pose[:, 5]
    cos, sin = np.cos(yaw), np.sin(yaw)
    vx = cos * twist[:, 0] + sin * twist[:, 1]
    vy = -sin * twist[:, 0] + cos * twist[:, 1]
    print(f"\n===== {title}")
    print(f" {'phase':8s} {'command':>16s} | {'vx':>7s} {'vy':>7s} {'wz':>7s} | {'height':>13s} | "
          f"{'roll':>5s} {'pitch':>5s} | {'yaw drift':>9s} | {'|tau|':>12s}")
    for name in dict.fromkeys(phase):
        if not name:
            continue
        mask = phase == name
        index = np.where(mask)[0]
        if len(index) < 20:
            continue
        mask[index[:len(index) // 4]] = False           # drop the transient into the phase
        print(f" {name:8s} {np.array2string(command[mask][0], precision=2):>16s} | "
              f"{vx[mask].mean():+7.3f} {vy[mask].mean():+7.3f} {angvel[mask, 2].mean():+7.3f} | "
              f"{pose[mask, 2].mean():.3f}+-{pose[mask, 2].std():.3f} | "
              f"{np.degrees(pose[mask, 3]).std():5.1f} {np.degrees(pose[mask, 4]).std():5.1f} | "
              f"{np.degrees(yaw[mask][-1] - yaw[mask][0]):+9.1f} | "
              f"{np.abs(tau[mask]).mean():5.2f}/{np.abs(tau[mask]).max():6.2f}")
    print(f" worst over the run: |roll| {np.degrees(np.abs(pose[:, 3])).max():.1f} deg, "
          f"|pitch| {np.degrees(np.abs(pose[:, 4])).max():.1f} deg, "
          f"lowest base {pose[:, 2].min():.3f} m")


# =================================================================================================
# the run
# =================================================================================================
def run_gazebo(args):
    import rospy
    from base_controllers.rl_quadruped.rl_quadruped_controller import RlQuadrupedController, State
    from base_controllers.rl_quadruped.commandInterface import Event

    controller = RlQuadrupedController(args.robot, input_device="none", skip_calibration=True,
                                       use_rviz=args.rviz, realtime=False, inference_timing=False,
                                       estimate_velocity=True)
    controller.startController(additional_args=["gui:=" + ("true" if args.gui else "false")])

    log = {k: [] for k in RECORD_KEYS}
    phase = [""]

    # Foot contact comes from the bumper sensors in the aliengo description.  The controller does
    # not subscribe to them (nothing in its loop uses contact), so the harness does.
    from gazebo_msgs.msg import ContactsState
    grf = np.zeros(4)
    legs = ["lf", "lh", "rf", "rh"]        # locosim order = FL, RL, FR, RR

    def bumper(index):
        def callback(message):
            total = 0.0
            for state in message.states:
                for wrench in state.wrenches:
                    total += abs(wrench.force.z)
            grf[index] = total
        return callback
    for index, leg in enumerate(legs):
        rospy.Subscriber(f"/{args.robot}/{leg}_foot_bumper", ContactsState, bumper(index),
                         queue_size=1)

    original_send = controller.send_des_jstate

    def send(*a, **k):
        if phase[0]:
            log["sim"].append(rospy.Time.now().to_sec())
            log["state"].append(controller.state.value)
            log["phase"].append(phase[0])
            log["pose"].append(controller.basePoseW.copy())
            log["twist"].append(np.concatenate([controller.gt_baseTwistW[:3], controller.angVelB]))
            log["angvel"].append(controller.angVelB.copy())
            log["q"].append(controller.q.copy())
            log["qd"].append(controller.qd.copy())
            log["qdes"].append(controller.q_des.copy())
            log["cmd"].append(controller.velocity_cmd.copy())
            log["tau"].append(controller.tau.copy())
            log["est"].append(controller.policy.estimated_base_lin_vel.copy())
            log["action"].append(controller.rl_action.copy())
            log["grf"].append(grf.copy())
            log["foot_z"].append(np.full(4, np.nan))   # no foot frames published by Gazebo
        return original_send(*a, **k)
    controller.send_des_jstate = send

    def wait_for(states, timeout=90.0):
        end = wall.monotonic() + timeout
        while wall.monotonic() < end:
            if controller.state in states:
                return True
            wall.sleep(0.2)
        return False

    def driver():
        if not wait_for((State.FOLD,)):
            print("gazebo: never reached FOLD"); return
        while controller._fold_moving:
            wall.sleep(0.2)
        wall.sleep(1.0)
        controller.command.post(Event.STAND_UP)
        if not wait_for((State.STANDING,)):
            print("gazebo: never reached STANDING"); return
        wall.sleep(1.5)
        controller.command.post(Event.START_RL)
        if not wait_for((State.RL,)):
            print("gazebo: never reached RL"); return
        wall.sleep(2.0)
        for name, velocity, seconds, variant in PHASES:
            if controller.state not in (State.RL, State.SAFE_STOP):
                print(f"gazebo: left the policy states before '{name}' "
                      f"(now '{controller.state.value}')")
                break
            if variant == "safe" and controller.state is not State.SAFE_STOP:
                controller.command.post(Event.SAFE_STOP)
                wait_for((State.SAFE_STOP,), timeout=5.0)
            print(f"gazebo: phase {name} {velocity} on '{variant}'")
            phase[0] = name
            controller.command.velocity_cmd[:] = velocity
            wall.sleep(seconds)
        phase[0] = ""
        controller.command.post(Event.QUIT)

    threading.Thread(target=driver, daemon=True).start()
    try:
        controller.mainLoop()
    except Exception as exc:                            # a fall ends the loop; keep what we logged
        print("gazebo: mainLoop raised", type(exc).__name__, exc)
    return {k: np.array(v) for k, v in log.items()}



# =================================================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="aliengo")
    parser.add_argument("--rviz", action="store_true", help="start rviz with the simulator")
    parser.add_argument("--gui", action="store_true", help="show the gazebo gui")
    parser.add_argument("--record", default=None, metavar="FILE.npz", help="save the run")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    log = run_gazebo(args)
    if not len(log.get("sim", [])):
        print("nothing recorded")
        return 1
    summarise(log, f"gazebo - {args.robot}")
    if args.record:
        np.savez(args.record, **log)
        print(f" saved {len(log['sim'])} ticks to {args.record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
