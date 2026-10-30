# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:52:08 2018

@author: mfocchi

Gazebo test of the safety filter, to watch.  The walking policy walks at a command and the robot is
pushed every push_period s, each kick of a growing list in turn to the left, right, front and back
(the push of the dataset and of rl_quadruped/push_sweep.py).  The value function is evaluated at
100 Hz (ValueFunctionManager); below its threshold the safe policy takes over at zero command, until
V has been back above it for 2 s.  A robot that falls is reset and the run goes on.

rviz shows V over the robot, green on the walking policy and red on the safe one, and each push as
a blue arrow.  /safety_filter/value, /safety_filter/threshold and /safety_filter/backup carry them
for rqt_plot (--plot starts it).  components/safe_rl/README.md has the whole pipeline.

    python3 saferl_controller.py [--robot aliengo|go2] [--threshold 0.5] [--kicks 1,1.5,2,2.5,3] [--cmd 0.3,0,0] [--no-filter] [--speed 1] [--plot] [--gui]
"""

from __future__ import print_function
import argparse
import subprocess
import rospy as ros
from base_controllers.utils.math_tools import *
np.set_printoptions(threshold=np.inf, precision=5, linewidth=1000, suppress=True)
from base_controllers.saferl_controller_dataset import SafeRLControllerDataset
from base_controllers.components.rl_velocity_controller.rl_controller import RlVelocityController
from base_controllers.components.safe_rl.value_function import value_function_manager as vfm
from base_controllers.utils.pidManager import PidManager
from std_msgs.msg import Float32
from termcolor import colored
import numpy as np

# the push directions, in turn, in the base frame
DIRECTIONS = (("left", np.pi / 2), ("right", -np.pi / 2), ("forward", 0.0), ("backward", np.pi))


class SafeRLController(SafeRLControllerDataset):
    """The dataset's robot (reset, push, simulation speed), run with the safety filter."""

    def __init__(self, robot_name="myrobot"):
        super().__init__(robot_name=robot_name)

    def startFilterTopics(self, threshold):
        self.threshold = threshold
        self.pub_value = ros.Publisher("/safety_filter/value", Float32, queue_size=1, tcp_nodelay=True)
        self.pub_threshold = ros.Publisher("/safety_filter/threshold", Float32, queue_size=1, tcp_nodelay=True)
        self.pub_backup = ros.Publisher("/safety_filter/backup", Float32, queue_size=1, tcp_nodelay=True)

    def publishFilter(self, V, backup):
        self.pub_value.publish(Float32(V))
        self.pub_threshold.publish(Float32(self.threshold))
        self.pub_backup.publish(Float32(float(backup)))

    def showFilter(self, V, backup, push_arrow=None):
        """V over the robot, and the push if any (start, vector) in rviz.  The same markers every
        tick, updated in place: deleting and re-adding them every tick makes them flicker."""
        base = self.basePoseW[:3]
        v = float(np.clip(V, 0.0, 1.0))
        self.ros_pub.add_marker(base + np.array([0., 0., 0.3]), radius=0.1, color=np.array([1.0 - v, v, 0.0]), alpha=0.9)
        text = f"V {v:.2f}  " + ("SAFE POLICY" if backup else "walking")
        self.ros_pub.add_text(base + np.array([0., 0., 0.45]), text,
                              color=np.array([1.0, 0.25, 0.2]) if backup else np.array([0.3, 1.0, 0.3]))
        if push_arrow is None:   # the arrow stays, invisible
            self.ros_pub.add_arrow(base, np.array([0., 0., 0.01]), "blue", alpha=0.0)
        else:
            self.ros_pub.add_arrow(push_arrow[0], push_arrow[1], "blue", scale=2.0)
        self.ros_pub.publishVisual(delete_markers=False)


def inclination(quaternion):
    return np.degrees(2 * np.arcsin(np.sqrt(quaternion[0] ** 2 + quaternion[1] ** 2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gazebo test of the safety filter")
    parser.add_argument("--robot", default="aliengo", choices=sorted(vfm.DEFAULT_THRESHOLD))
    parser.add_argument("--model", default=None, help="value function, ONNX (default models/VF_push_end_<robot>.onnx)")
    parser.add_argument("--threshold", type=float, default=None, help="default per robot (value_function_manager.py)")
    parser.add_argument("--no-filter", action="store_true", help="show V but never switch to the safe policy")
    parser.add_argument("--kicks", default="1,1.5,2,2.5,3", help="m/s, each pushed in the 4 directions in turn")
    parser.add_argument("--cmd", default="0.3,0,0", help="walking command vx,vy,wz (training range +-0.5)")
    parser.add_argument("--push-period", type=float, default=4.0, help="s between pushes")
    parser.add_argument("--speed", type=float, default=1.0, help="x real time")
    parser.add_argument("--plot", action="store_true", help="start rqt_plot on the filter topics")
    parser.add_argument("--gui", action="store_true", help="Gazebo GUI")
    parser.add_argument("--no-rviz", action="store_true")
    args = parser.parse_args()
    kicks = [float(k) for k in args.kicks.split(",")]
    pushes = [(kick, name, angle) for kick in kicks for name, angle in DIRECTIONS]
    cmd = np.array([float(c) for c in args.cmd.split(",")])
    first_push = 2.0  # s after a reset: the gait has settled
    push_duration = 0.2

    p = SafeRLController(args.robot)
    p.state_estimation = 'ground_truth'
    rl_controller = RlVelocityController(p.robot_name, p.dt, use_nn_se=True, debug=False, policy="velocity")
    vf = vfm.ValueFunctionManager(robot_name=args.robot, model_path=args.model, threshold=args.threshold)
    vf_every = int(round(0.01 / p.dt))  # 100 Hz
    # the rviz markers at 30 Hz: rviz draws no faster, and every message it has to process slows it
    vis_every = int(round(1.0 / (30 * p.dt)))
    log = []  # per push: kick, direction, lowest V, switch time after the push, fell

    try:
        # as the dataset: rl_flat.world, no ground truth contacts (they would force slow.world)
        p.startController(world_name='rl_flat.world', use_ground_truth_contacts=False,
                          additional_args=['gui:=' + str(args.gui), 'rviz:=' + str(not args.no_rviz).lower()])
        p.pid = PidManager(p.joint_names)
        # real time: the loop takes about 0.6 ms of the 2 ms control period (the overruns at the end
        # tell if it kept up here)
        p.setSpeed(args.speed)
        p.startFilterTopics(vf.threshold)
        if args.plot:
            subprocess.Popen(["rqt_plot", "/safety_filter/value/data", "/safety_filter/threshold/data",
                              "/safety_filter/backup/data"])
        q_reset = np.array(rl_controller.q_def, dtype=np.float64)
        base_pose = np.array([0, 0, p.standingHeight(q_reset) + 0.03, 0., 0., 0.])

        def reset():
            """Standing still in the policy's default posture, the walking policy on."""
            p.resetRobot(basePoseDes=base_pose, q_reset=q_reset)
            p.initVars()
            p.dm.wait_for_reset(base_pose, q_reset)
            p.pid.setPDjoints(rl_controller.kp, rl_controller.kd, np.zeros(12))
            rl_controller.reset()
            vf.reset()

        reset()
        k, step, next_push, push_end, arrow = 0, 0, first_push, -1.0, None
        nominal, V = True, 1.0
        ticks, overruns, last_tick = 0, 0, None
        while not ros.is_shutdown() and (k < len(pushes) or float(np.ravel(p.time)[0]) < next_push):
            p.updateKinematics()
            t = float(np.ravel(p.time)[0])
            lin_acc_b = p.baseLinAccB
            lin_vel_b = p.b_R_w.dot(p.baseTwistW[:3])
            ang_vel_b = p.b_R_w.dot(p.baseTwistW[3:6])
            proj_gravity = p.b_R_w.dot(np.array([0, 0, -1]))

            # the safety filter, from 1 s after a reset
            if step % vf_every == 0 and t >= 1.0:
                nominal, V = vf.computeValueFnc(proj_gravity, ang_vel_b, p.q, p.qd)
                nominal = nominal or args.no_filter
                p.publishFilter(V, not nominal)
                if log and not log[-1]["fell"]:
                    log[-1]["v_min"] = min(log[-1]["v_min"], V)
                    if not nominal and np.isnan(log[-1]["switch"]):
                        log[-1]["switch"] = t - log[-1]["t"]

            # the push
            if t >= next_push and k < len(pushes):
                kick, name, angle = pushes[k]
                yaw = p.basePoseW[5]
                direction = np.array([np.cos(yaw + angle), np.sin(yaw + angle), 0.])
                p.push(p.robot.robotMass * kick / push_duration * direction[:2], push_duration)
                print(colored(f"push {k + 1}/{len(pushes)}: {kick} m/s, {name}", "blue"))
                log.append({"kick": kick, "dir": name, "t": t, "v_min": V, "switch": np.nan, "fell": False})
                arrow = (p.basePoseW[:3] - 0.6 * direction, 0.45 * direction)
                push_end, next_push, k = t + 0.5, t + args.push_period, k + 1

            # the walking policy, or the safe one at zero command
            if nominal:
                rl_controller.velocity_cmd = cmd
                p.q_des = rl_controller.action(lin_acc_b, lin_vel_b, ang_vel_b, proj_gravity, p.q, p.qd,
                                               policy_type="default")
            else:
                rl_controller.velocity_cmd = np.zeros(3)
                p.q_des = rl_controller.action(lin_acc_b, lin_vel_b, ang_vel_b, proj_gravity, p.q, p.qd,
                                               policy_type="safe")
            p.send_des_jstate(p.q_des, np.zeros(12), np.zeros(12))
            if step % vis_every == 0:
                p.showFilter(V, not nominal, arrow if t < push_end else None)
            p.rate.sleep()
            # a control tick the loop missed: the simulation went on more than 1.5 periods
            tick = ros.get_time()
            ticks += 1
            overruns += last_tick is not None and tick - last_tick > 1.5 * p.dt
            last_tick = tick
            p.time = np.round(p.time + p.dt, 4)
            step += 1

            if p.dm.check_fallen(p.basePoseW[2], inclination(p.quaternion)):
                print(colored("fallen: reset", "red"))
                if log:
                    log[-1]["fell"] = True
                reset()
                step, next_push, push_end, nominal, V = 0, first_push, -1.0, True, 1.0
                last_tick = None

        print("\npush             lowest V   safe policy after   outcome")
        for e in log:
            switch = "-" if np.isnan(e["switch"]) else f"{e['switch']:.2f} s"
            print(f"{e['kick']:.1f} m/s {e['dir']:9s} {e['v_min']:7.2f}   {switch:>17s}   "
                  f"{'fell' if e['fell'] else 'recovered'}")
        print(f"fell {sum(e['fell'] for e in log)} of {len(log)} pushes, threshold {vf.threshold}"
              f"{', filter off' if args.no_filter else ''}")
        print(colored(f"control ticks missed: {overruns} of {ticks} at {args.speed}x real time"
                      + (" (lower --speed)" if overruns > 0.01 * ticks else ""),
                      "yellow" if overruns > 0.01 * ticks else "green"))

    except (ros.ROSInterruptException, ros.service.ServiceException):
        ros.signal_shutdown("killed")
        p.deregister_node()
