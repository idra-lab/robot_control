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
from gazebo_ros import gazebo_interface
from base_controllers.utils.pidManager import PidManager
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelStateRequest
from gazebo_msgs.msg import ModelState
import numpy as np
import time
robotName = "aliengo"  # needs to inherit BaseController

class SafeRLControllerDataset(QuadrupedController):

    def __init__(self, robot_name="myrobot"):
        super().__init__(robot_name=robot_name)
        print("Initialized   controller---------------------------------------------------------------")
        self.dm = DatasetManager(quadruped = self)
        self.get_state = ros.ServiceProxy('/gazebo/get_model_state', GetModelState)

    def resetRobot(self, basePoseDes=np.array([0, 0, 0.3, 0., 0., 0.])):
        # this sets the position of the joints
        gazebo_interface.set_model_configuration_client(self.robot_name, '', self.joint_names, self.qj_0, '/gazebo')
        self.send_des_jstate(self.q_des, np.zeros(12), np.zeros(12))
        #print('self.q_des',self.q_des)
        # this sets the position of the base without removing gravity
        self.freezeBase(False,  basePoseW=basePoseDes)

    def setBaseTwist(self, baseTwist=np.zeros(6)):
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
