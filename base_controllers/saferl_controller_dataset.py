# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:52:08 2018

@author: mfocchi

Push episodes in Gazebo for the safety filter's value function (components/safe_rl/README.md).
Each episode resets the robot standing in the policy's default posture, walks it at a command drawn
as in training, pushes it once and labels how it ends (DatasetManager, dataset_manager.py).

    python3 saferl_controller_dataset.py --robot go2 --episodes 70 --seed 501 --out <folder>     # dataset
    python3 saferl_controller_dataset.py --robot go2 ... --from-push-start      # test data, walking state included
    python3 saferl_controller_dataset.py --robot go2 ... --fall-check           # run to the ground: fall_rule.py
    python3 saferl_controller_dataset.py --robot go2 ... --filter               # closed loop with the safety filter

The same --seed gives the same commands and pushes, so a run with --filter compares episode by
episode with one without (compare_filter.py).  It runs in real time (--speed); several batches run
in parallel with components/safe_rl/run_batches.sh, the whole pipeline with run_pipeline.sh.
"""

from __future__ import print_function
import argparse
import rospy as ros
from base_controllers.utils.math_tools import *
np.set_printoptions(threshold=np.inf, precision=5, linewidth=1000, suppress=True)
from base_controllers.quadruped_controller import QuadrupedController
from base_controllers.components.rl_velocity_controller.rl_controller import RlVelocityController
from base_controllers.components.safe_rl.dataset_manager import DatasetManager, ROBOT_SETTINGS
from base_controllers.utils.pidManager import PidManager
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelStateRequest, SetPhysicsPropertiesRequest
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Point, Wrench
import numpy as np
import pinocchio as pin
import time
robotName = "aliengo"  # needs to inherit BaseController

class SafeRLControllerDataset(QuadrupedController):

    def __init__(self, robot_name="myrobot"):
        super().__init__(robot_name=robot_name)
        print("Initialized   controller---------------------------------------------------------------")
        self.dm = DatasetManager(quadruped = self)
        self.get_state = ros.ServiceProxy('/gazebo/get_model_state', GetModelState)

    def resetRobot(self, basePoseDes=np.array([0, 0, 0.3, 0., 0., 0.]), q_reset=None):
        """Bring the robot back to q_reset (default q_0) at basePoseDes, at rest, and leave it on
        the stand-alone joint gains (kp_j, kd_j).  The joints are driven there by the PD with the
        robot lifted clear of the ground and gravity off, never teleported: gazebo_ros_control
        unwraps each joint reading by the shortest angle, so a joint teleported by about pi or more
        - a hip left at the +-pi end of its range by a fall, the Aliengo hfe range being [-pi, pi]
        - reads 2 pi off for the rest of the run and the PD then drives it into its stop."""
        q_reset = self.qj_0 if q_reset is None else q_reset
        # drop the policy's last target, far off after a fall, before stiffening the gains
        self.send_des_jstate(self.q, np.zeros(12), np.zeros(12))
        self.pid.setPDjoints(self.kp_j, self.kd_j, self.ki_j)
        self.freezeBase(True, basePoseW=np.array([0., 0., 1.0, 0., 0., 0.]), baseTwistW=np.zeros(6))
        q_start = self.q.copy()
        steps = int(0.6 / self.dt)
        for i in range(1, steps + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * i / steps)
            self.send_des_jstate(q_start + (q_reset - q_start) * s, np.zeros(12), np.zeros(12))
            self.rate.sleep()
        for _ in range(int(0.5 / self.dt)):
            self.send_des_jstate(q_reset, np.zeros(12), np.zeros(12))
            if np.max(np.abs(self.q - q_reset)) < 0.05 and np.max(np.abs(self.qd)) < 0.5:
                break
            self.rate.sleep()
        # this sets the position of the base and turns gravity back on, at rest: without a twist
        # freezeBase keeps the measured one
        self.freezeBase(False, basePoseW=basePoseDes, baseTwistW=np.zeros(6))

    def standingHeight(self, q_joints):
        """Height of the base frame over the foot centres with the joints at q_joints."""
        q = np.hstack([pin.neutral(self.robot.model)[0:7], self.u.mapToRos(q_joints)])
        self.robot.forwardKinematics(q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        return -np.mean([self.robot.data.oMf[i].translation[2] for i in self.robot.getEndEffectorsFrameId])

    def push(self, force_xy, duration):
        """A horizontal force on the base, in the world frame, for duration s from now: the push of
        the push benchmark (rl_quadruped/push_sweep.py).  A push sets no velocity, so the joint
        velocities and the base rotation are left as they are; setting the model twist would set
        them on every link."""
        wrench = Wrench()
        wrench.force.x, wrench.force.y = force_xy
        response = self.apply_body_wrench(body_name=self.robot_name + "::base_link", reference_frame="world",
                                          reference_point=Point(), wrench=wrench, start_time=ros.Time(0),
                                          duration=ros.Duration(duration))
        if not response.success:
            raise RuntimeError(f"push refused: {response.status_message}")

    def setSpeed(self, speed):
        """Run the simulation at speed x real time (1: real time), every other physics setting left
        as the world file has it.  setSimSpeed would also put the 1 mm contact surface layer back
        that rl_flat.world removes."""
        physics = self.get_physics_client()
        req = SetPhysicsPropertiesRequest()
        req.time_step = physics.time_step
        req.max_update_rate = speed / physics.time_step
        req.gravity = physics.gravity
        req.ode_config = physics.ode_config
        self.set_physics_client(req)

    def setBaseTwist(self, baseTwist=np.zeros(6)):
        # Gazebo sets this twist on every link of the model, so the joint velocities go to zero:
        # not a push, use push()
        current = self.get_state(self.robot_name, "")
        req = SetModelStateRequest()
        model_state = ModelState()
        # keep the pose unchanged
        model_state.pose = current.pose
        model_state.model_name = self.robot_name
        # apply your twist only
        model_state.twist.linear.x = baseTwist[0]
        model_state.twist.linear.y = baseTwist[1]
        model_state.twist.linear.z = baseTwist[2]
        model_state.twist.angular.x = baseTwist[3]
        model_state.twist.angular.y = baseTwist[4]
        model_state.twist.angular.z = baseTwist[5]
        req.model_state = model_state
        self.reset_world(req)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Push episodes for the safety value function (components/safe_rl/README.md)")
    parser.add_argument("--robot", default="aliengo", choices=sorted(ROBOT_SETTINGS))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None, help="default: the clock; the same seed gives the same commands and pushes")
    parser.add_argument("--out", default=None, help="folder for observations_dataset_abs_test.npy and its _episodes.csv "
                        "(default components/safe_rl/value_function/observation_datasets/<robot>)")
    parser.add_argument("--noise", type=float, default=10.0, help="torque noise std from the push on, Nm per joint and tick")
    parser.add_argument("--filter", nargs="?", const="default", default=None, metavar="MODEL",
                        help="run the safety filter (default model: models/VF_push_end_<robot>.onnx)")
    parser.add_argument("--threshold", type=float, default=None, help="safety filter threshold (default per robot)")
    parser.add_argument("--from-push-start", action="store_true",
                        help="record from the push start, not its end: test data with the walking state before the push")
    parser.add_argument("--fall-check", action="store_true",
                        help="run every episode to the ground (inclination 85 deg or base 0.1 m) or to 3 s after the "
                             "push, capture off; fall_rule.py checks the fall rule and the capture hold on it")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="x real time; an episode whose loop misses control ticks is discarded")
    parser.add_argument("--no-rviz", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Gazebo GUI")
    args = parser.parse_args()

    p = SafeRLControllerDataset(args.robot)
    world_name = 'rl_flat.world'
    p.state_estimation = 'ground_truth'  # 'odometry','imu', 'pronto', 'ground_truth' (only sim)
    rl_controller = RlVelocityController(p.robot_name, p.dt, use_nn_se=True,debug=False, policy="velocity")
    p.dm.record_from_push_start = args.from_push_start
    if args.fall_check:
        p.dm.INCLINATION_THRESHOLD, p.dm.FALL_HEIGHT_THRESHOLD, p.dm.capture = 85.0, 0.1, False
    if args.filter is not None:
        from base_controllers.components.safe_rl.value_function import value_function_manager as vfm
        p.dm.value_function = vfm.ValueFunctionManager(
            robot_name=args.robot, model_path=None if args.filter == "default" else args.filter, threshold=args.threshold)

    try:
        # rl_flat.world, as the RL controller (rl_quadruped) runs the policy: no 1 mm contact
        # surface layer, which the policy never saw.  Ground truth contacts would force slow.world,
        # and the dataset does not use contacts.
        p.startController(world_name=world_name,
                          use_ground_truth_contacts=False,
                          additional_args=['gui:=' + str(args.gui), 'rviz:=' + str(not args.no_rviz).lower()])

        p.pid = PidManager(p.joint_names)
        # RlVelocityController sets no gains itself: the dataset manager puts the policy's kp, kd
        # (its json) on after every reset
        p.pid.setPDjoints(rl_controller.kp, rl_controller.kd, np.full(12, 0))
        # real time by default: the loop takes about 0.6 ms of the 2 ms control period, also with
        # 8 simulations in parallel; episodes where it missed ticks are discarded
        p.setSpeed(args.speed)
        # the episode (command range, push, horizon, backup policy) is set by DatasetManager's
        # attributes, e.g. p.dm.kick_range = (0.0, 3.0)
        out = args.out or f"components/safe_rl/value_function/observation_datasets/{args.robot}"
        p.dm.run_batch_simulations(rl_controller, n_episodes=args.episodes, save_path=out, noise_std=args.noise,
                                   seed=int(time.time()) if args.seed is None else args.seed)

    except (ros.ROSInterruptException, ros.service.ServiceException):
        ros.signal_shutdown("killed")
        p.deregister_node()
