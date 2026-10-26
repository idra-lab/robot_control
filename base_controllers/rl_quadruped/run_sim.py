"""Run the controller in Gazebo through a fixed command sequence and print the gait per phase.

    python3 run_sim.py --robot go2                    # prints a summary
    python3 run_sim.py --robot go2 --record gz.npz    # and saves the run for gait_report.py
"""

import argparse
import sys
import time as wall
import threading

import numpy as np

from base_controllers.rl_quadruped.rl_controller_config import known_robots

# (name, velocity command, seconds, policy variant)
PHASES = [("settle",  (0.0, 0.0, 0.0), 2.0, "normal"),
          ("fwd0.3",  (0.3, 0.0, 0.0), 8.0, "normal"),
          ("yaw0.5",  (0.0, 0.0, 0.5), 8.0, "normal"),
          ("lat0.3",  (0.0, 0.3, 0.0), 8.0, "normal"),
          ("fwd0.5",  (0.5, 0.0, 0.0), 8.0, "normal"),
          ("safe",    (0.0, 0.0, 0.0), 8.0, "safe")]

RECORD_KEYS = ("sim", "state", "phase", "pose", "twist", "angvel", "q", "qd", "qdes", "cmd",
               "tau", "est", "action", "grf", "foot_z", "acc", "imu_obs")
"""Per control tick.  ``grf`` (foot normal force) and ``foot_z`` are in leg order FL, RL, FR, RR;
``acc`` is the accelerometer reading handed to the policy, ``imu_obs`` its last imu_lin_acc term."""


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
                                       use_rviz=args.rviz, realtime=False)
    controller.startController(additional_args=["gui:=" + ("true" if args.gui else "false")])

    log = {k: [] for k in RECORD_KEYS}
    phase = [""]

    # foot contact from the bumper sensors of the description
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
            log["acc"].append(controller.imu_lin_acc().copy())
            log["imu_obs"].append(controller.policy.imu_lin_acc_obs.copy())
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
    parser.add_argument("--robot", default="aliengo", choices=known_robots())
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
        np.savez(args.record, robot=np.array(args.robot), **log)   # gait_report reads the robot
        print(f" saved {len(log['sim'])} ticks to {args.record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
