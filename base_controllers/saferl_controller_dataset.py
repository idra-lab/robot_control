# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:52:08 2018

@author: mfocchi
"""

from __future__ import print_function
import rospy as ros
from base_controllers.utils.math_tools import *
np.set_printoptions(threshold=np.inf, precision=5, linewidth=1000, suppress=True)
from base_controllers.quadruped_controller import QuadrupedController
from base_controllers.components.rl_velocity_controller.rl_controller import RlVelocityController
from base_controllers.components.safe_rl.dataset_manager import DatasetManager
from base_controllers.utils.pidManager import PidManager
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelStateRequest
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
    p = SafeRLControllerDataset('aliengo')
    world_name = 'fast.world'
    use_gui=False
    p.state_estimation = 'ground_truth'  # 'odometry','imu', 'pronto', 'ground_truth' (only sim)
    rl_controller = RlVelocityController(p.robot_name, p.dt, use_nn_se=True,debug=False, policy="velocity")

    try:
        p.startController(world_name=world_name,
                          use_ground_truth_contacts=True,
                          additional_args=['gui:=' + str(use_gui),'rviz:=true'])

        p.pid = PidManager(p.joint_names)
        p.pid.setPDjoints(rl_controller.kp, rl_controller.kd, np.full(12, 0))
        p.setSimSpeed(dt_sim=0.001, max_update_rate=200)
        p.dm.run_batch_simulations(rl_controller, n_episodes=100, save_path="components/safe_rl/value_function/observation_datasets", noise_std=10.0, seed = int(time.time()))

    except (ros.ROSInterruptException, ros.service.ServiceException):
        ros.signal_shutdown("killed")
        p.deregister_node()

    # if conf.plotting:
    #     plotJoint('position', time_log=p.time_log, q_log=p.q_log, q_des_log=p.q_des_log, sharex=True, sharey=False,
    #               start=0, end=-1)
    #     plotFrame('position', time_log=p.time_log, des_Pose_log=p.basePoseW_des_log, Pose_log=p.basePoseW_log,
    #               title='Base', frame='W', sharex=True, sharey=False, start=0, end=-1)
    #     plotFrame('velocity', time_log=p.time_log, des_Twist_log=p.baseTwistW_des_log, Twist_log=p.baseTwistW_log,
    #               title='Base', frame='W', sharex=True, sharey=False, start=0, end=-1)
