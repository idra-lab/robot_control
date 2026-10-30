"""V of the safety filter's value function on a recorded Aliengo run: the joint states
(/aliengo/joint_states) and the Pronto odometry (/state_estimator_pronto/odom) of a playing bag go
through ValueFunctionManager (models/VF_push_end_aliengo.onnx, components/safe_rl), which prints
when V is below its threshold.

    rosbag play <bag> &
    python3 vf_rosbag.py

A run of rl_quadruped/rl_quadruped_controller.py with --safety-filter --bag records V itself, as
/<robot>/rl/safety_filter/value.
"""
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from base_controllers.saferl_controller import SafeRLController
import rospy as ros
from base_controllers.components.safe_rl.value_function.value_function_manager import ValueFunctionManager
import numpy as np
from tf.transformations import euler_from_quaternion
from base_controllers.utils.utils import Utils
from base_controllers.utils.math_tools import *
import base_controllers.params as conf
from base_controllers.utils.common_functions import *

class PubSub():
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.quaternion = np.zeros(4)
        self.basePoseW = np.zeros(6)
        self.euler = np.zeros(3)
        self.baseTwistW = np.zeros(6)
        self.b_R_w = np.eye(3)
        self.u = Utils()
        self.math_utils = Math()
        self.joint_names = conf.robot_params[self.robot_name]['joint_names']
        self.robot = getRobotModelFloating(self.robot_name)
        self.q = np.zeros(self.robot.na)
        self.qd = np.zeros(self.robot.na)
        self.tau = np.zeros(self.robot.na)

    def _receive_pose(self, msg):
        self.quaternion[0] = msg.pose.pose.orientation.x
        self.quaternion[1] = msg.pose.pose.orientation.y
        self.quaternion[2] = msg.pose.pose.orientation.z
        self.quaternion[3] = msg.pose.pose.orientation.w

        self.basePoseW[self.u.sp_crd["LX"]] = msg.pose.pose.position.x
        self.basePoseW[self.u.sp_crd["LY"]] = msg.pose.pose.position.y
        self.basePoseW[self.u.sp_crd["LZ"]] = msg.pose.pose.position.z

        self.euler = np.array(euler_from_quaternion(self.quaternion))

        self.basePoseW[self.u.sp_crd["AX"]] = self.euler[0]
        self.basePoseW[self.u.sp_crd["AY"]] = self.euler[1]
        self.basePoseW[self.u.sp_crd["AZ"]] = self.euler[2]

        self.baseTwistW[self.u.sp_crd["LX"]] = msg.twist.twist.linear.x
        self.baseTwistW[self.u.sp_crd["LY"]] = msg.twist.twist.linear.y
        self.baseTwistW[self.u.sp_crd["LZ"]] = msg.twist.twist.linear.z
        self.baseTwistW[self.u.sp_crd["AX"]] = msg.twist.twist.angular.x
        self.baseTwistW[self.u.sp_crd["AY"]] = msg.twist.twist.angular.y
        self.baseTwistW[self.u.sp_crd["AZ"]] = msg.twist.twist.angular.z

        # compute orientation matrix
        self.b_R_w = self.math_utils.rpyToRot(self.euler)

    def _receive_jstate(self, msg):
        for msg_idx in range(len(msg.name)):
            for joint_idx in range(len(self.joint_names)):
                if self.joint_names[joint_idx] == msg.name[msg_idx]:
                    self.q[joint_idx] = msg.position[msg_idx]
                    self.qd[joint_idx] = msg.velocity[msg_idx]
                    self.tau[joint_idx] = msg.effort[msg_idx]

    def init_subscribers(self):
        self.sub_jstate = ros.Subscriber("/" + self.robot_name + "/joint_states", JointState, callback=self._receive_jstate,queue_size=1, tcp_nodelay=True)
        self.sub_pose = ros.Subscriber("/state_estimator_pronto/odom", Odometry, callback=self._receive_pose, queue_size=1,tcp_nodelay=True)

#
ros.init_node('communicate_aliengo')
pubSub = PubSub('aliengo')
pubSub.init_subscribers()

vf = ValueFunctionManager()
ang_vel_b_prev = np.zeros(3)
proj_gravity_b_prev = np.zeros(3)
q_prev = np.zeros(12)
qd_prev = np.zeros(12)
while not ros.is_shutdown():
    #print('Init')
    #print(pubSub.b_R_w)
    #print(pubSub.baseTwistW[3:6])
    #print(pubSub.q, pubSub.qd)
    ang_vel_b = pubSub.b_R_w.dot(pubSub.baseTwistW[3:6])
    proj_gravity_b = pubSub.b_R_w.dot(np.array([0, 0, -1]))
    q = pubSub.q
    qd = pubSub.qd
    '''print('ang_vel_b',ang_vel_b_prev,ang_vel_b)
    print('proj_gravity_b',proj_gravity_b_prev,proj_gravity_b)
    print('q', q_prev, q)
    print('qd', qd_prev, qd)'''
    #if np.any(ang_vel_b != ang_vel_b_prev) or np.any(proj_gravity_b != proj_gravity_b_prev) or np.any(q != q_prev) or np.any(qd != qd_prev):
    isrec, V_safe = vf.computeValueFnc(body_ang_vel=ang_vel_b, proj_gravity=proj_gravity_b, joint_pos=pubSub.q, joint_vel=pubSub.qd, vf_additional_term = 0.0)
    vf.VF = True
    vf.count = 0
    vf.count_back = 0
        #print(V_safe)
        #ang_vel_b_prev = ang_vel_b
        #proj_gravity_b_prev = proj_gravity_b
        #q_prev = q
        #qd_prev = qd