import pinocchio as pin
from base_controllers.utils.pidManager import PidManager
from base_controllers.base_controller import BaseController
from base_controllers.utils.math_tools import *
from base_controllers.utils.joyManager import JoyManager
from base_controllers.components.whole_body_controller import WholeBodyController
from base_controllers.utils.common_functions import *
from base_controllers.components.inverse_kinematics.inv_kinematics_quadruped import InverseKinematics as AnalyticInverseKinematics
from base_controllers.components.inverse_kinematics.inv_kinematics_pinocchio import robotKinematics as PinocchioInverseKinematics
from base_controllers.components.leg_odometry.leg_odometry import LegOdometry
from base_controllers.components.rl_velocity_controller.rl_controller import RlVelocityController
from base_controllers.components.rl_velocity_controller.LocomotionPolicyWrapper import LocomotionPolicyWrapper
from termcolor import colored
import base_controllers.params as conf
from scipy.io import savemat
#gazebo messages
from gazebo_ros import gazebo_interface
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3, Twist
from sensor_msgs.msg import Imu
from ros_impedance_controller.msg import EffortPid
from base_controllers.components.imu_utils import IMU_utils
from base_controllers.components.quadruped_tasks import QuadrupedTasks
from base_controllers.components.state_machine import StateMachine
from base_controllers.utils.rosbag_recorder import RosbagControlledRecorder


class QuadrupedController(BaseController):
    def __init__(self, robot_name="hyq", launch_file=None):
        super(QuadrupedController, self).__init__(robot_name, launch_file)
        self.qj_0 = conf.robot_params[self.robot_name]['q_0']


        self.ee_frames = conf.robot_params[self.robot_name]['ee_frames']
        self.leg_names = [foot[:2] for foot in self.ee_frames]


        if not self.real_robot:
            self.gravity_comp_duration = 0.5 #1.5
            self.standup_period = 1. #3
            self.dt = conf.robot_params[self.robot_name]['dt']
        else:
            self.gravity_comp_duration = 1.5
            self.standup_period = 3.
            print(colored("overriding default dt: quadruped controller for real robot runs at 250Hz!","red"))
            self.dt = 0.004


        # pid = os.getpid()
        # print(colored("USER IS ROOT: USING RENICE FOR THREAD PRIORITY", "red"))
        # # NOTE: use chrt -r 99 command for rt
        # # need to launch docker as root
        # os.system(f'sudo renice -n -21 -p {str(pid)}')
        # # need to launch docker as root
        # os.system(f'sudo echo -20 > /proc/{str(pid)}/autogroup')


    #####################
    # OVERRIDEN METHODS #
    #####################
    # initVars
    # logData
    # startupProcedure

    def initSubscribers(self):
        self.sub_jstate = ros.Subscriber("/" + self.robot_name + "/joint_states", JointState, callback=self._receive_jstate, queue_size=1, tcp_nodelay=True)
        self.sub_pid_effort = ros.Subscriber("/" + self.robot_name + "/effort_pid", EffortPid, callback=self._receive_pid_effort, queue_size=1, tcp_nodelay=True)

        if self.real_robot:
            self.sub_imu_lin_acc = ros.Subscriber("/" + self.robot_name + "/trunk_imu", Vector3,  callback=self._receive_imu_acc_real, queue_size=1, tcp_nodelay=True)
            if self.state_estimation == 'mocap':  # use pronto for state estimation
                from geometry_msgs.msg import PoseStamped
                launchFileNode("mocap_qualisys", "qualisys.launch")
                self.sub_pose = ros.Subscriber("/qualisys/robot/pose", PoseStamped, callback=self._receive_mocap, queue_size=1, tcp_nodelay=True)
                self.pub_mocap_twist = ros.Publisher("/qualisys/robot/twist", Twist, queue_size=1, tcp_nodelay=True)

            elif self.state_estimation == 'pronto':#use pronto for state estimation
                #start stateest node
                # self.u.putIntoGlobalParamServer("use_sim_time", str("false"))
                # load_rosparams_from_package('pronto_aliengo', 'config/aliengo_state_estimator.yaml', target_namespace='/')
                # startNode(package="pronto_aliengo", executable="pronto_aliengo_node", args='', name="pronto_aliengo")
                # startNode(package="pronto_aliengo", executable="aliengo_joint_swapper", args='',name="aliengo_joint_swapper")
                from pronto_msgs.msg import QuadrupedStance
                self.pronto_contacts_sub = ros.Subscriber("/state_estimator_pronto/stance", QuadrupedStance, callback=self._receive_pronto_contacts, queue_size=1, tcp_nodelay=True)
                #we start the pronto node only after startup!
                self.pronto_config = "aliengo_state_estimator.yaml"
                self.sub_pose = ros.Subscriber("/state_estimator_pronto/odom", Odometry,  callback=self._receive_pose, queue_size=1, tcp_nodelay=True)
            elif self.state_estimation=='odometry':#use odometry
                self.sub_imu_euler = ros.Subscriber("/" + self.robot_name + "/euler_imu", Vector3,  callback=self._receive_euler, queue_size=1, tcp_nodelay=True)
                self.sub_pose = ros.Subscriber("/" + self.robot_name + "/ground_truth", Odometry,   callback=self._receive_pose_real, queue_size=1, tcp_nodelay=True)
            elif self.state_estimation=='imu':#use angular velocity and quaternion coming from IMU (robot_name/imu topic) and write BasePose BaseTwist variables
                self.sub_imu = ros.Subscriber("/" + self.robot_name + "/imu", Imu,  callback=self._receive_imu, queue_size=1, tcp_nodelay=True)
            elif self.state_estimation=='ground_truth':
                print(f"state_estimation ground truth  not possible on real robot!")
            else:
                print(f"state_estimation type not known {self.state_estimation}")
        else:#simulation
            self.sub_imu_lin_acc = ros.Subscriber("/" + self.robot_name + "/trunk_imu", Imu,  callback=self._receive_imu_acc, queue_size=1, tcp_nodelay=True)

            if self.state_estimation == 'pronto':  # use pronto for state estimation
                #  stateest node requires publication of msg type sensor:IMU in topic aliengo/imu we remap
                startNode(package="topic_tools", executable="relay", args="/" + self.robot_name + "/trunk_imu" + "  " + "/" + self.robot_name + "/imu", name="trunk_imu_to_imu")
                self.pronto_config = "aliengo_state_estimator_sim.yaml"
                from pronto_msgs.msg import QuadrupedStance, QuadrupedForceTorqueSensors
                self.pronto_contacts_sub = ros.Subscriber("/state_estimator_pronto/stance", QuadrupedStance, callback=self._receive_pronto_contacts, queue_size=1, tcp_nodelay=True)
                self.sub_pose = ros.Subscriber("/state_estimator_pronto/odom", Odometry, callback=self._receive_pose, queue_size=1, tcp_nodelay=True)
                #these on real robot are published by hw interface
                self.pub_feet_forces = ros.Publisher("/" + self.robot_name + "/feet_forces", QuadrupedForceTorqueSensors, queue_size=1, tcp_nodelay=True)

            elif self.state_estimation=='ground_truth':
                self.sub_pose = ros.Subscriber("/" + self.robot_name + "/ground_truth", Odometry,  callback=self._receive_pose,  queue_size=1, tcp_nodelay=True)
            elif self.state_estimation=='odometry':
                self.sub_pose = ros.Subscriber("/" + self.robot_name + "/ground_truth", Odometry, callback=self._receive_pose_real, queue_size=1, tcp_nodelay=True)
            elif self.state_estimation=='imu':
                self.sub_imu = ros.Subscriber("/" + self.robot_name + "/trunk_imu", Imu,  callback=self._receive_imu, queue_size=1, tcp_nodelay=True)
            else:
                print(f"state_estimation type not known {self.state_estimation}")

            if self.use_ground_truth_contacts:
                self.sub_contact_lf = ros.Subscriber("/" + self.robot_name + "/lf_foot_bumper", ContactsState,
                                                     callback=self._receive_contact_lf, queue_size=1, buff_size=2 ** 24,
                                                     tcp_nodelay=True)
                self.sub_contact_rf = ros.Subscriber("/" + self.robot_name + "/rf_foot_bumper", ContactsState,
                                                     callback=self._receive_contact_rf, queue_size=1, buff_size=2 ** 24,
                                                     tcp_nodelay=True)
                self.sub_contact_lh = ros.Subscriber("/" + self.robot_name + "/lh_foot_bumper", ContactsState,
                                                     callback=self._receive_contact_lh, queue_size=1, buff_size=2 ** 24,
                                                     tcp_nodelay=True)
                self.sub_contact_rh = ros.Subscriber("/" + self.robot_name + "/rh_foot_bumper", ContactsState,
                                                     callback=self._receive_contact_rh, queue_size=1, buff_size=2 ** 24,
                                                     tcp_nodelay=True)
        # if self.real_robot:
        #     self.sub_contact_force_z = ros.Subscriber("/" + self.robot_name + "/contact_force_z", Float64MultiArray,
        #                             callback=self._receive_contact_force_real, queue_size=1, tcp_nodelay=True)

    def _receive_imu_acc_real(self, msg):
        self.baseLinAccB[0] = msg.x
        self.baseLinAccB[1] = msg.y
        self.baseLinAccB[2] = msg.z
        # baseLinAccW is without gravity
        self.baseLinAccW = self.b_R_w.T @ (self.baseLinAccB - self.imu_utils.IMU_accelerometer_bias) - self.imu_utils.g0


    def _receive_imu_acc(self, msg):
        self.baseLinAccB[0] = msg.linear_acceleration.x
        self.baseLinAccB[1] = msg.linear_acceleration.y
        self.baseLinAccB[2] = msg.linear_acceleration.z
        # baseLinAccW is without gravity
        self.baseLinAccW = self.b_R_w.T @ (self.baseLinAccB - self.imu_utils.IMU_accelerometer_bias) - self.imu_utils.g0

    def _receive_euler(self, msg):
        self.euler[0] = msg.x
        self.euler[1] = msg.y
        self.euler[2] = msg.z

    def _receive_pronto_contacts(self, msg):
        self.pronto_contacts[0] = msg.lf
        self.pronto_contacts[1] = msg.rf
        self.pronto_contacts[2] = msg.lh
        self.pronto_contacts[3] = msg.rh

    def _receive_mocap(self, msg):
        self.quaternion[0] = msg.pose.orientation.x
        self.quaternion[1] = msg.pose.orientation.y
        self.quaternion[2] = msg.pose.orientation.z
        self.quaternion[3] = msg.pose.orientation.w
        self.basePoseW[self.u.sp_crd["LX"]] = msg.pose.position.x
        self.basePoseW[self.u.sp_crd["LY"]] = msg.pose.position.y
        self.basePoseW[self.u.sp_crd["LZ"]] = msg.pose.position.z
        self.euler = np.array(euler_from_quaternion(self.quaternion))
        self.basePoseW[self.u.sp_crd["AX"]] = self.euler[0]
        self.basePoseW[self.u.sp_crd["AY"]] = self.euler[1]
        self.basePoseW[self.u.sp_crd["AZ"]] = self.euler[2]
        # filter
        self.basePoseW_f = self.beta * self.basePoseW + (1. - self.beta) * self.basePoseW_f

        Jomega = self.math_utils.Tomega(self.u.angPart(self.basePoseW))
        vel = self.u.linPart(self.basePoseW_f-self.basePoseW_f_old) / self.dt
        omega = Jomega @ self.u.angPart(self.basePoseW_f-self.basePoseW_f_old) / self.dt
        self.baseTwistW[:3] = vel
        self.baseTwistW[3:]  = omega
        self.basePoseW_f_old = self.basePoseW_f.copy()
        msg=Twist()
        msg.linear.x = self.baseTwistW[0]
        msg.linear.y = self.baseTwistW[1]
        msg.linear.z = self.baseTwistW[2]
        msg.angular.x = self.baseTwistW[3]
        msg.angular.y = self.baseTwistW[4]
        msg.angular.z = self.baseTwistW[5]
        self.pub_mocap_twist.publish(msg)
        # compute orientation matrix
        self.b_R_w = self.math_utils.rpyToRot(self.euler)

    # def _receive_contact_force_real(self, msg):
    #     #53.0, 82.0, 87.0, 78.0
    #     self.contact_state[0] = msg.data[0] > 73
    #     self.contact_state[1] = msg.data[1] > 102
    #     self.contact_state[2] = msg.data[2] > 107
    #     self.contact_state[3] = msg.data[3] > 98
    #     #print(self.contact_state)

    def _receive_imu(self, msg):
        self.quaternion[0] = msg.orientation.x
        self.quaternion[1] = msg.orientation.y
        self.quaternion[2] = msg.orientation.z
        self.quaternion[3] = msg.orientation.w

        self.euler = np.array(euler_from_quaternion(self.quaternion))
        self.basePoseW[self.u.sp_crd["AX"]] = self.euler[0]
        self.basePoseW[self.u.sp_crd["AY"]] = self.euler[1]
        self.basePoseW[self.u.sp_crd["AZ"]] = self.euler[2]
        # compute orientation matrix
        self.b_R_w = self.math_utils.rpyToRot(self.euler)
        self.angVelB[0] = msg.angular_velocity.x
        self.angVelB[1] = msg.angular_velocity.y
        self.angVelB[2] = msg.angular_velocity.z
        self.baseTwistW[3:] = self.b_R_w.T.dot(self.angVelB)

    def _receive_pose_real(self, msg):
        self.quaternion[0] = msg.pose.pose.orientation.x
        self.quaternion[1] = msg.pose.pose.orientation.y
        self.quaternion[2] = msg.pose.pose.orientation.z
        self.quaternion[3] = msg.pose.pose.orientation.w

        self.basePoseW[self.u.sp_crd["LX"]] = self.basePoseW_legOdom[0]
        self.basePoseW[self.u.sp_crd["LY"]] = self.basePoseW_legOdom[1]
        self.basePoseW[self.u.sp_crd["LZ"]] = self.basePoseW_legOdom[2]

        self.basePoseW[self.u.sp_crd["AX"]] = self.euler[0]
        self.basePoseW[self.u.sp_crd["AY"]] = self.euler[1]
        self.basePoseW[self.u.sp_crd["AZ"]] = self.euler[2]

        if False:#any(self.contact_state):
            self.baseTwistW[self.u.sp_crd["LX"]] = self.baseTwistW_legOdom[0]
            self.baseTwistW[self.u.sp_crd["LY"]] = self.baseTwistW_legOdom[1]
            self.baseTwistW[self.u.sp_crd["LZ"]] = self.baseTwistW_legOdom[2]
        else:
            self.baseTwistW[self.u.sp_crd["LX"]] = self.imu_utils.baseLinTwistImuW[0]
            self.baseTwistW[self.u.sp_crd["LY"]] = self.imu_utils.baseLinTwistImuW[1]
            self.baseTwistW[self.u.sp_crd["LZ"]] = self.imu_utils.baseLinTwistImuW[2]

        self.baseTwistW[self.u.sp_crd["AX"]] = msg.twist.twist.angular.x
        self.baseTwistW[self.u.sp_crd["AY"]] = msg.twist.twist.angular.y
        self.baseTwistW[self.u.sp_crd["AZ"]] = msg.twist.twist.angular.z

        # compute orientation matrix
        self.b_R_w = self.math_utils.rpyToRot(self.euler)

    def initVars(self):
        super().initVars()
        self.q_des = np.zeros_like(self.q)

        # mocap filter coeff
        ta = 0.15
        self.beta = self.dt / (self.dt + ta)
        self.basePoseW_f_old = np.zeros(6)
        self.basePoseW_f = np.zeros(6)

        self.imu_utils = IMU_utils(dt=conf.robot_params[self.robot_name]['dt'])
        #pinocchio based
        self.ikin = PinocchioInverseKinematics(self.robot, conf.robot_params[self.robot_name]['ee_frames'])
        self.IK = AnalyticInverseKinematics(self.robot)
        self.leg_odom = LegOdometry(self.robot, self.real_robot)
        self.legConfig = {}
        if 'solo' in self.robot_name or  self.robot_name == 'hyq' or self.robot_name == 'anymal_d':  # either solo or solo_fw
            self.legConfig['lf'] = ['HipDown', 'KneeInward']
            self.legConfig['lh'] = ['HipDown', 'KneeInward']
            self.legConfig['rf'] = ['HipDown', 'KneeInward']
            self.legConfig['rh'] = ['HipDown', 'KneeInward']

        elif self.robot_name == 'aliengo' or self.robot_name == 'go1' or self.robot_name == 'go2':
            self.legConfig['lf'] = ['HipDown', 'KneeInward']
            self.legConfig['lh'] = ['HipDown', 'KneeOutward']
            self.legConfig['rf'] = ['HipDown', 'KneeInward']
            self.legConfig['rh'] = ['HipDown', 'KneeOutward']

        else:
            assert False, 'leg configuration is not defined for ' + self.robot_name

        self.euler = np.zeros(3)
        # some extra variables

        self.tau_fb = np.zeros(self.robot.na)
        self.tau_ffwd = np.zeros(self.robot.na)
        self.tau_des = np.zeros(self.robot.na)

        self.basePoseW_des = np.zeros(6) * np.nan
        self.baseTwistW_des = np.zeros(6) * np.nan


        self.comPoseW_des = np.zeros(6) * np.nan
        self.comTwistW_des = np.zeros(6) * np.nan

        self.comPosB = np.zeros(3) * np.nan
        self.comVelB = np.zeros(3) * np.nan

        self.basePoseW_legOdom = np.zeros(3) #* np.nan
        self.baseTwistW_legOdom = np.zeros(3) #* np.nan
        ###W
        self.g_mag = np.linalg.norm(self.robot.model.gravity.vector)

        self.grForcesW_des = np.empty(3 * self.robot.nee) * np.nan
        self.grForcesW_wbc = np.empty(3 * self.robot.nee) * np.nan
        self.grForcesB = np.empty(3 * self.robot.nee) * np.nan #not used
        self.grForcesB_ffwd = np.empty(3 * self.robot.nee) * np.nan

        # load gains
        if self.real_robot:
            real_str = '_real'
        else:
            real_str = ''

        # stand alone joint pid
        self.kp_j = conf.robot_params[self.robot_name].get('kp'+real_str, np.zeros(self.robot.na))
        self.kd_j = conf.robot_params[self.robot_name].get('kd'+real_str, np.zeros(self.robot.na))
        self.ki_j = conf.robot_params[self.robot_name].get('ki'+real_str, np.zeros(self.robot.na))

        ###W
        # virtual impedance wrench control
        self.kp_lin = np.diag(conf.robot_params[self.robot_name].get('kp_lin'+real_str, np.zeros(3)))
        self.kd_lin = np.diag(conf.robot_params[self.robot_name].get('kd_lin'+real_str, np.zeros(3)))

        self.kp_ang = np.diag(conf.robot_params[self.robot_name].get('kp_ang'+real_str, np.zeros(3)))
        self.kd_ang = np.diag(conf.robot_params[self.robot_name].get('kd_ang'+real_str, np.zeros(3)))

        # updated in WBC
        self.kp_linW = np.zeros_like(self.kp_lin)
        self.kd_linW = np.zeros_like(self.kd_lin)

        self.kp_angW = np.zeros_like(self.kp_ang)
        self.kd_angW = np.zeros_like(self.kd_ang)

        # joint pid with wbc
        self.kp_wbc_j = conf.robot_params[self.robot_name].get('kp_wbc'+real_str, np.zeros(self.robot.na))
        self.kd_wbc_j = conf.robot_params[self.robot_name].get('kd_wbc'+real_str, np.zeros(self.robot.na))
        self.ki_wbc_j = conf.robot_params[self.robot_name].get('ki_wbc'+real_str, np.zeros(self.robot.na))


        self.wrench_fbW  = np.zeros(6)
        self.wrench_ffW  = np.zeros(6)
        self.wrench_gW   = np.zeros(6)
        self.wrench_gW[self.u.sp_crd["LZ"]] = self.robot.robotMass * self.g_mag
        self.wrench_desW = np.zeros(6)

        self.wrench_fbW_log = np.full( (6, conf.robot_params[self.robot_name]['buffer_size'] ), np.nan)
        self.wrench_ffW_log = np.full( (6, conf.robot_params[self.robot_name]['buffer_size'] ), np.nan)
        self.wrench_gW_log = np.full( (6, conf.robot_params[self.robot_name]['buffer_size'] ), np.nan)
        self.wrench_desW_log = np.full( (6, conf.robot_params[self.robot_name]['buffer_size'] ), np.nan)

        self.NEMatrix = np.zeros([6, 3*self.robot.nee]) # Newton-Euler matrix

        ###W
        self.wbc = WholeBodyController(conf.robot_params[self.robot_name], self.real_robot, self.robot)

        self.force_th = conf.robot_params[self.robot_name].get('force_th', 0.)
        self.contact_th = conf.robot_params[self.robot_name].get('contact_th', 0.)

        self.W_vel_contacts_des = self.u.full_listOfArrays(4, 3)
        self.B_vel_contacts_des = self.u.full_listOfArrays(4, 3)

        # imu
        self.baseLinAccB = np.full(3, np.nan)
        self.baseLinAccW = np.full(3, np.nan)


        self.comPosB_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.comVelB_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.comPoseW_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.comTwistW_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.comPoseW_des_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.comTwistW_des_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.comVelW_leg_odom = np.full((3), np.nan)
        self.comVelW_leg_odom_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)


        self.basePoseW_des_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.baseTwistW_des_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.basePoseW_legOdom_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.baseTwistW_legOdom_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.tau_fb_log = np.full((self.robot.na, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.tau_des_log = np.full((self.robot.na, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)


        self.grForcesB_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.grForcesW_gt_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.grForcesW_des_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.grForcesW_wbc_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.W_contacts_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.W_contacts_des_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.B_contacts_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.B_contacts_des_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.B_vel_contacts_des_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.W_vel_contacts_des_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.contact_state_log = np.full((self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.baseLinAccW_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.baseLinAccB_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.baseLinTwistImuW_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.angVelB = np.zeros(3)

        # robot height is the height of the robot base frame in home configuration
        self.robot_height = 0.

        neutral_fb_jointstate = np.hstack([pin.neutral(self.robot.model)[0:7], self.u.mapToRos(conf.robot_params[self.robot_name]['q_0']) ])

        self.robot.forwardKinematics(neutral_fb_jointstate)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        for id in self.robot.getEndEffectorsFrameId:
            self.robot_height += self.robot.data.oMf[id].translation[2]
        self.robot_height /= -4.
        print(colored(f"Robot height correspontent to q0 configuration is {self.robot_height+0.02}","red")) #Includes the foot radius
        self.loop_time_log = np.full((conf.robot_params[self.robot_name]['buffer_size']), np.nan)

        half_lenght = self.robot.collision_model.geometryObjects[0].geometry.halfSide[0]
        half_width = self.robot.collision_model.geometryObjects[0].geometry.halfSide[1]
        half_height = self.robot.collision_model.geometryObjects[0].geometry.halfSide[2]
        self.bControlPoints = [ np.array([half_lenght, half_width, -half_height]),# lf_bottom
                                np.array([-half_lenght, half_width, -half_height])  ,# lh_bottom
                                np.array([half_lenght, -half_width, -half_height])  ,# rf_bottom
                                np.array([-half_lenght, -half_width, -half_height]) , # rf_bottom
                                np.array([half_lenght, half_width, half_height])  ,# lf_top
                                np.array([-half_lenght, half_width, half_height]) , # lh_top
                                np.array([half_lenght, -half_width, half_height]) , # rf_top
                                np.array([-half_lenght, -half_width, half_height])]  # rf_top

        self.kfe_idx = [self.robot.model.getFrameId(leg + '_kfe_joint') for leg in ['lf', 'lh', 'rf', 'rh']]
        #safety layer
        self.gracefulCollapseFlag = False
        self.alphaCollapse = 1.0
        self.collapseTime = 0.7

        self.pronto_contacts = np.zeros(4)
        self.controller_ready = False

    def logData(self):
        # full with new values
        self.comPosB_log[:, self.log_counter] = self.comPosB
        self.comVelB_log[:, self.log_counter] = self.comVelB
        self.comPoseW_log[:, self.log_counter] = self.comPoseW
        self.comTwistW_log[:, self.log_counter] = self.comTwistW
        self.comPoseW_des_log[:, self.log_counter] = self.comPoseW_des
        self.comTwistW_des_log[:, self.log_counter] = self.comTwistW_des
        self.basePoseW_log[:, self.log_counter] = self.basePoseW
        self.baseTwistW_log[:, self.log_counter] = self.baseTwistW
        self.basePoseW_des_log[:, self.log_counter] = self.basePoseW_des
        self.baseTwistW_des_log[:, self.log_counter] = self.baseTwistW_des
        self.basePoseW_legOdom_log[:, self.log_counter] = self.basePoseW_legOdom
        self.baseTwistW_legOdom_log[:, self.log_counter] = self.baseTwistW_legOdom
        self.q_des_log[:, self.log_counter] = self.q_des
        self.q_log[:, self.log_counter] = self.q
        self.qd_des_log[:, self.log_counter] = self.qd_des
        self.qd_log[:, self.log_counter] = self.qd
        self.tau_fb_log[:, self.log_counter] = self.tau_fb
        self.tau_ffwd_log[:, self.log_counter] = self.tau_ffwd

        self.tau_des = self.tau_ffwd + self.tau_fb
        self.tau_des_log[:, self.log_counter] = self.tau_des
        self.tau_log[:, self.log_counter] = self.tau
        self.grForcesW_log[:, self.log_counter] = self.grForcesW
        self.grForcesW_des_log[:, self.log_counter] = self.grForcesW_des
        self.grForcesW_wbc_log[:, self.log_counter] = self.grForcesW_wbc
        self.grForcesW_gt_log[:, self.log_counter] = self.grForcesW_gt
        self.grForcesB_log[:, self.log_counter] = self.grForcesB
        self.contact_state_log[:, self.log_counter] = self.contact_state

        self.baseLinAccW_log[:, self.log_counter] = self.baseLinAccW
        self.baseLinAccB_log[:, self.log_counter] = self.baseLinAccB

        self.comVelW_leg_odom_log[:, self.log_counter] = self.comVelW_leg_odom

        for leg in range(4):
            start = 3 * leg
            end = 3 * (leg+1)
            self.B_contacts_log[start:end, self.log_counter] = self.B_contacts[leg]
            self.B_contacts_des_log[start:end, self.log_counter] = self.B_contacts_des[leg]

            self.W_contacts_log[start:end, self.log_counter] = self.W_contacts[leg]
            self.W_contacts_des_log[start:end, self.log_counter] = self.W_contacts_des[leg]

            self.B_vel_contacts_des_log[start:end, self.log_counter] = self.B_vel_contacts_des[leg]
            self.W_vel_contacts_des_log[start:end, self.log_counter] = self.W_vel_contacts_des[leg]

        self.baseLinTwistImuW_log[:, self.log_counter] = self.imu_utils.baseLinTwistImuW

        ###W
        self.wrench_fbW_log[:, self.log_counter] = self.wrench_fbW
        self.wrench_ffW_log[:, self.log_counter] = self.wrench_ffW
        self.wrench_gW_log[:, self.log_counter] = self.wrench_gW
        self.wrench_desW_log[:, self.log_counter] = self.wrench_desW
###W

        self.time_log[self.log_counter] = self.time
        self.loop_time_log[self.log_counter] = self.loop_time

        self.log_counter += 1
        self.log_counter %= conf.robot_params[self.robot_name]['buffer_size']
    
    def check_faulty_ping(self, ip="192.168.1.1"):
        response = os.system("ping -c 1 " + ip)
        # and then check the response...
        if response == 0:
            pingstatus = f"Robot Network Active: pinging {ip} successful"
        else:
            pingstatus = f"Robot Network not Active, cannot ping {ip}, create a local network with gateway {ip}"
        print(colored(pingstatus, "red"))
        return response

    def startController(self, world_name=None, xacro_path=None,   use_ground_truth_contacts=True, additional_args=[]):

        if self.real_robot == False:
            self.use_ground_truth_contacts = use_ground_truth_contacts
        else:
            self.use_ground_truth_contacts = False
            if self.check_faulty_ping(conf.robot_params[self.robot_name]['ip']):
                sys.exit()

        self.start()                               # as a thread

        self.go0_conf = 'home'
        if additional_args is not None:
            for arg in additional_args:
                if 'go0_conf:=' in arg:
                    self.go0_conf = arg.replace('go0_conf:=', '')


        additional_args.append("load_force_sensors:="+str(not self.use_ground_truth_contacts).lower())
        if self.use_ground_truth_contacts and (world_name is None or not 'slow' in world_name):
            print('Cannot use ground truth contact with not slow world file')
            print('Set world file to slow.world')
            world_name = 'slow.world'

        self.startSimulator(world_name=world_name, additional_args=additional_args)            # run gazebo
        if world_name is None:
            self.world_name_str = ''
        else:
            self.world_name_str = world_name
        if 'camera' in self.world_name_str:
            # check if some old jpg are still in /tmp
            # this command prevent for Argument list too long in bash http://mywiki.wooledge.org/BashFAQ/095
            print(colored('Removing jpg files', 'blue'), flush=True)
            remove_jpg_cmd = 'for f in /tmp/camera_save/*; do rm "$f"; done'
            os.system(remove_jpg_cmd)
            print(colored('Jpg files removed', 'blue'), flush=True)
        self.loadModelAndPublishers(xacro_path)    # load robot and all the publishers
        #self.resetGravity(True)
        self.initVars()                            # overloaded method
        self.initSubscribers()
        self.rate = ros.Rate(1 / self.dt)
        print(colored("Started QuadrupedController", "blue"))

    def resetRobot(self, basePoseDes=np.array([0, 0, 0.3, 0., 0., 0.])):
        # this sets the position of the joints
        gazebo_interface.set_model_configuration_client(self.robot_name, '', self.joint_names, self.qj_0, '/gazebo')
        self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
        # this sets the position of the base
        if self.DEBUG == 'none' or self.DEBUG == 'pushup':
            self.freezeBase(False, basePoseW=basePoseDes)
        else:
            self.freezeBase(True, basePoseW=basePoseDes)

    def reset(self, basePoseW=None, baseTwistW=None, resetPid=False):
        self.q_des = conf.robot_params[self.robot_name]['q_0'].copy()
        self.qd_des = np.zeros(self.robot.na)
        self.tau_ffwd = np.zeros(self.robot.na)
        if resetPid:
            self.pid.setPDjoints(self.kp_j, self.kd_j, self.ki_j)
        if basePoseW is None:
            basePoseW = np.hstack([self.base_offset, np.zeros(3)])


        self.freezeBase(flag=True, basePoseW=basePoseW)
        ros.sleep(0.5)


        gazebo_interface.set_model_configuration_client(self.robot_name, '', self.joint_names, self.qj_0, '/gazebo')
        while np.linalg.norm(self.qd)>0.05 or np.linalg.norm(self.q-self.q_des)>0.05:
            self.updateKinematics()
            self.send_command(self.q_des, self.qd_des, self.tau_ffwd)
            if np.linalg.norm(self.u.linPart(self.basePoseW-basePoseW))>1:
                self.freezeBase(flag=True, basePoseW=basePoseW)
                ros.sleep(0.5)

        if baseTwistW is None:
            baseTwistW = np.zeros(6)

        self.freezeBase(flag=True, basePoseW=basePoseW, baseTwistW=baseTwistW)

        self.initVars() # reset logged values

        self.imu_utils.baseLinTwistImuW = self.u.linPart(self.baseTwistW).copy()




    def Hframe2World(self, poseH, dposeH=None, ddposeH=None):
        # returns variables from Hframe to World frame
        # dposeH is the rate (d/dt poseH) and ddposeH its time derivative
        # dposeH is mapped into twist and ddposeH into acceleration
        w_R_des_hf = pin.rpy.rpyToMatrix(0, 0, self.u.angPart(self.basePoseW)[2])

        poseW = np.empty(6)
        poseW[self.u.sp_crd['LX']:self.u.sp_crd['LX'] + 3] = w_R_des_hf @ self.u.linPart(poseH)
        poseW[self.u.sp_crd['AX']:self.u.sp_crd['AX'] + 3] = self.u.angPart(poseH)
        #poseW[self.u.sp_crd['AZ']] += np.pi/2#self.u.angPart(self.basePoseW)[2]

        if dposeH is not None:
            twistW = np.empty(6)

            twistW[self.u.sp_crd['LX']:self.u.sp_crd['LX'] + 3] =  w_R_des_hf @ self.u.linPart(dposeH)

            # map euler rates into omega
            Jomega = self.math_utils.Tomega(self.u.angPart(self.basePoseW))
            twistW[self.u.sp_crd['AX']:self.u.sp_crd['AX'] + 3] = Jomega @ (self.u.angPart(dposeH))

            if ddposeH is not None:
                accW = np.empty(6)
                accW[self.u.sp_crd['LX']:self.u.sp_crd['LX'] + 3] = w_R_des_hf @ self.u.linPart(ddposeH)
                accW[self.u.sp_crd['AX']:self.u.sp_crd['AX'] + 3] = self.u.angPart(ddposeH)

                # compute w_omega_dot =  Jomega* euler_rates_dot + Jomega_dot*euler_rates (Jomega already computed, see above)
                Jomega_dot = self.math_utils.Tomega_dot(self.u.angPart(self.basePoseW), self.u.angPart(self.baseTwistW))
                accW[self.u.sp_crd['AX']:self.u.sp_crd['AX'] + 3] = Jomega @ self.u.angPart(ddposeH) + \
                                                                    Jomega_dot @ self.u.angPart(dposeH)
                return poseW, twistW, accW

            return poseW, twistW

        return poseW


    def World2Hframe(self, poseW, twistW=None, accW=None):
        # inverse of the previous function
        # returns variables from World frame to Hframe
        # twist is mapped into dposeH and acceleration into ddposeH
        hf_R_des_w = pin.rpy.rpyToMatrix(0, 0, self.u.angPart(poseW)[2]).T

        poseH = np.empty(6)
        poseH[self.u.sp_crd['LX']:self.u.sp_crd['LX'] + 3] = hf_R_des_w @ self.u.linPart(poseW)
        poseH[self.u.sp_crd['AX']:self.u.sp_crd['AX'] + 3] = self.u.angPart(poseW)[0:3]
        # poseH[self.u.sp_crd['AZ']] = 0.

        if twistW is not None:
            dposeH = np.empty(6)
            dposeH[self.u.sp_crd['LX']:self.u.sp_crd['LX'] + 3] =  hf_R_des_w @ self.u.linPart(twistW)
            # map omega into euler rate
            Jomega_inv = self.math_utils.Tomega_inv(self.u.angPart(poseW))
            dposeH[self.u.sp_crd['AX']:self.u.sp_crd['AX'] + 3] = Jomega_inv @ self.u.angPart(twistW)

            if accW is not None:
                ddposeH = np.empty(6)
                ddposeH[self.u.sp_crd['LX']:self.u.sp_crd['LX'] + 3] = hf_R_des_w @ self.u.linPart(accW)
                # compute euler_rates_dot = Jomega_inv *( w_omega_dot - Jomega_dot*euler_rates) (Jomega_inv already computed, see above)
                Jomega_dot = self.math_utils.Tomega_dot(self.u.angPart(poseW), self.u.angPart(twistW))

                ddposeH[self.u.sp_crd['AX']:self.u.sp_crd['AX'] + 3] = Jomega_inv @ (self.u.angPart(accW) - Jomega_dot @ self.u.angPart(dposeH) )
                return poseH, dposeH, ddposeH

            return poseH, dposeH

        return poseH




    def Wcom2Wbase_des(self):
        # suppose comPose/Twist des and W_contacts_des are set
        # base ref in W
        b_R_w_des = pin.rpy.rpyToMatrix(self.u.angPart(self.comPoseW_des)).T
        omega_skew = pin.skew(self.u.angPart(self.comTwistW_des))

        self.basePoseW_des = self.comPoseW_des.copy()
        self.basePoseW_des[:3] -= b_R_w_des.T @ self.comPosB

        self.baseTwistW_des = self.comTwistW_des.copy()
        self.baseTwistW_des[:3] -= b_R_w_des.T @ (omega_skew @ self.comPosB + self.comVelB)

    def Wbase2Bcontact_des(self):
        b_R_w_des = pin.rpy.rpyToMatrix(self.u.angPart(self.basePoseW_des)).T
        omega_skew = pin.skew(self.u.angPart(self.baseTwistW_des))
        # feet ref in B
        for leg in range(4):
            self.B_contacts_des[leg] = b_R_w_des @ (self.W_contacts_des[leg] - self.u.linPart(self.basePoseW_des))
            self.B_vel_contacts_des[leg] = b_R_w_des @ (
                    omega_skew.T @ (self.W_contacts_des[leg] - self.u.linPart(self.basePoseW_des))
                    - self.u.linPart(self.baseTwistW_des))
            # self.B_vel_contacts_des[leg] = 5 * (self.B_contacts_des[leg]-self.B_contacts[leg])

    def Wbase2Joints_des(self):
        # before the first call, please set
        # q_des = q.copy()
        # qd_des[:] = 0.
        # suppose WbasePose/Twist des and W_contacts_des are set
        self.Wbase2Bcontact_des()

        # time = 0, 1, 2, 3, ...
        # time = 0 -> q_des = ik(),         qd_des = const.
        # time = 1 -> q_des += qd_des * dt, qd_des = const.
        # time = 2 -> q_des = const,        qd_des = diff_ik()
        # time = 3 -> q_des += qd_des * dt, qd_des = const.
        # ...

        if self.log_counter % 4 == 0:
            for leg in range(4):
                q_des_leg, isFeasible = self.IK.ik_leg(self.B_contacts_des[leg],
                                                       self.leg_names[leg],
                                                       self.legConfig[self.leg_names[leg]][0],
                                                       self.legConfig[self.leg_names[leg]][1])
                if isFeasible:
                    self.u.setLegJointState(leg, q_des_leg, self.q_des)

        elif self.log_counter % 4 == 2:
            for leg in range(4):
                qd_leg_des = self.IK.diff_ik_leg(q_des=self.q_des,
                                                 B_v_foot=self.B_vel_contacts_des[leg],
                                                 leg=self.leg_names[leg],
                                                 update=leg == 0)  # update Jacobians only with the first leg

                # qd_leg_des = self.J_inv[leg] @ self.B_vel_contacts_des[leg]

                self.u.setLegJointState(leg, qd_leg_des, self.qd_des)
        else:
            self.q_des += self.qd_des * self.dt

    def Wcom2Joints_des(self):
        # before the first call, please set
        # q_des = q.copy()
        # qd_des[:] = 0.
        # suppose WcomPose/Twist des and W_contacts_des are set
        # base ref in W
        self.Wcom2Wbase_des()
        self.Wbase2Joints_des()
        

    def support_poly(self, contacts):
        # Wcontacts: lf, rf, lh, rh
        sp = {}
        # CCW order
        sides_order = {'F': [1, 0], 'L': [0, 2], 'H': [2, 3], 'R': [3, 1]}
        for side in sides_order: # side is the key of sides_order
            p0 = contacts[sides_order[side][0]][0:2]
            p1 = contacts[sides_order[side][1]][0:2]
            m, q = self.line2points2D(p0, p1)
            sp['line'+side] = {'m': m, 'q': q, 'p0':p0, 'p1':p1}
        return sp

    @staticmethod
    def line2points2D(p0, p1):
        # line defined as y= mx+q
        m = (p1[1] - p0[1]) / (p1[0] - p0[0])
        q = p0[1] - m * p0[0]
        return m, q

    def send_command(self, q_des=None, qd_des=None, tau_ffwd=None, log_data_in_send_command = False):
        # q_des, qd_des, and tau_ffwd have dimension 12
        # and are ordered as on the robot

        if q_des is not None:
            self.q_des = q_des

        if qd_des is not None:
            self.qd_des = qd_des

        if tau_ffwd is not None:
            self.tau_ffwd = tau_ffwd

        self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)

        # if (self.APPLY_EXTERNAL_WRENCH and self.time > self.TIME_EXTERNAL_WRENCH):
        #     print("START APPLYING EXTERNAL WRENCH")
        #     self.applyForce(0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.05)
        #     self.APPLY_EXTERNAL_WRENCH = False

        # log variables
        if log_data_in_send_command:
            self.logData()
        self.rate.sleep()
        self.sync_check()
        self.time = np.round(self.time + self.dt, 4)#np.array([self.loop_time]), 3)


    def visualizeContacts(self, delete_markers=False):
        for legid in self.u.leg_map.keys():

            leg = self.u.leg_map[legid]
            if self.contact_state[leg]:
                self.ros_pub.add_arrow(self.W_contacts[leg],
                                       self.u.getLegJointState(leg, self.grForcesW/ (6*self.robot.robotMass)),
                                       "green")
                
                self.ros_pub.add_arrow(self.W_contacts[leg],
                                       self.u.getLegJointState(leg, self.grForcesW_des/ (6*self.robot.robotMass)),
                                       "blue")
                
                #self.ros_pub.add_marker(self.W_contacts[leg], radius=0.1)
            else:
                self.ros_pub.add_arrow(self.W_contacts[leg],
                                       np.zeros(3),
                                       "green", scale=0.0001)
                #self.ros_pub.add_marker(self.W_contacts[leg], radius=0.001)

            if (self.use_ground_truth_contacts):
                self.ros_pub.add_arrow(self.W_contacts[leg],
                                       self.u.getLegJointState(leg, self.grForcesW_gt / (6 * self.robot.robotMass)),
                                       "red")


        # self.ros_pub.add_polygon([self.B_contacts[0],
        #                           self.B_contacts[1],
        #                           self.B_contacts[3],
        #                           self.B_contacts[2],
        #                           self.B_contacts[0] ], "red", visual_frame="base_link")
        #
        # self.ros_pub.add_polygon([self.B_contacts_des[0],
        #                           self.B_contacts_des[1],
        #                           self.B_contacts_des[3],
        #                           self.B_contacts_des[2],
        #                           self.B_contacts_des[0]], "green", visual_frame="base_link")
        #

        self.ros_pub.publishVisual(delete_markers=delete_markers)

    def updateKinematics(self, update_legOdom=True, noise=None):
        if noise is not None:
            if 'qd' in noise:
                self.qd += noise['qd'].draw()
            if 'tau' in noise:
                self.tau += noise['tau'].draw()
        self.basePoseW_legOdom, self.baseTwistW_legOdom = self.leg_odom.base_in_world(contact_state=self.contact_state,
                                                                                      B_contacts=self.B_contacts,
                                                                                      b_R_w=self.b_R_w,
                                                                                      wJ=self.wJ,
                                                                                      ang_vel=self.u.angPart(self.baseTwistW),
                                                                                      qd=self.qd,
                                                                                      update_legOdom=update_legOdom)
        self.imu_utils.compute_lin_vel(self.baseLinAccW, self.loop_time)
        super(QuadrupedController, self).updateKinematics()

        #publish contact forces in the topic for pronto
        if not self.real_robot and self.state_estimation == 'pronto':
            from pronto_msgs.msg import QuadrupedForceTorqueSensors
            msg = QuadrupedForceTorqueSensors()
            msg.lf.force.z = self.u.getLegJointState(self.u.leg_map["LF"], self.grForcesW)[2] #z component
            msg.rf.force.z = self.u.getLegJointState(self.u.leg_map["RF"], self.grForcesW)[2]  # z component
            msg.lh.force.z = self.u.getLegJointState(self.u.leg_map["LH"], self.grForcesW)[2]  # z component
            msg.rh.force.z = self.u.getLegJointState(self.u.leg_map["RH"], self.grForcesW)[2] # z component
            self.pub_feet_forces.publish(msg)

    def checkBaseCollisions(self):
        # base control points
        for i, pt in enumerate(self.bControlPoints):
            wControlPoint = self.mapBaseToWorld(pt)
            if wControlPoint[2] < self.contact_th:
                return True
        return False

    def checkKFECollisions(self):
        # kfe collision
        for i, id in enumerate(self.kfe_idx):
            wKfe_pos = self.mapBaseToWorld(self.robot.data.oMf[id].translation)  # update kin computes quantity in base frame
            if wKfe_pos[2] < self.contact_th:
                return True
        return False

    def checkGroundCollisions(self):
        # retrun codes
        # -1 no collisions
        # base collisions
        # 0 lf_bottom
        # 1 lh_bottom
        # 2 rf_bottom
        # 3 rf_bottom
        # 4 lf_top
        # 5 lh_top
        # 6 rf_top
        # 7 rf_top
        # kfe collisions
        # 8 lf_kfe
        # 9 lf_kfe
        # 10 lf_kfe
        # 11 lf_kfe

        # return True/False

        # base control points
        for i, pt in enumerate(self.bControlPoints):
            wControlPoint = self.mapBaseToWorld(pt)
            if wControlPoint[2] < self.contact_th:
                #return i
                return True

        # kfe collision
        for i, id in enumerate(self.kfe_idx):
            wKfe_pos = self.mapBaseToWorld(self.robot.data.oMf[id].translation) # update kin computes quantity in base frame
            if wKfe_pos[2] < self.contact_th:
                #return self.bControlPoints+i
                return True

        # return -1
        return False

    def IMUBiasEstimation(self, sm, time):
        if sm.first_time:
            print(colored("[startupProcedure t: " + str(self.time[0]) + "s] Imu bias estimation", "blue"))
        #on loop
        #print(f"Estimating Bias {time}")
        self.updateKinematics()
        self.imu_utils.IMU_bias_estimation(self.b_R_w, self.baseLinAccB)
        self.tau_ffwd[:] = 0.
        self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
        self.imu_utils.counter+=1
        #event driven termination
        if self.imu_utils.counter >= self.imu_utils.timeout:
            print("Estimating Bias Accomplished → next state")
            sm.next(time)

    def goFoldConfig(self, sm , time):
        if sm.first_time:
            print(colored("[startupProcedure t: " + str(self.time[0]) + "s] Going to fold configuration", "blue"))
            # go from where you are to q_fold
            self.q_ref = np.zeros_like(self.q)
            # Going to fold config
            for i in range(12):
               if (i % 3) != 0:
                    self.q_ref[i] = conf.robot_params[self.robot_name]['q_fold'][i]
            self.q_init = self.q.copy()

        # on loop
        #print(f"Go fold...{time}")
        self.updateKinematics()
        self.tau_ffwd[:] = 0.
        alpha  = sm.timer.get_elapsed_time(time)/sm.get_state_duration()
        self.q_des = (1 - alpha) * self.q_init + alpha * self.q_ref
        self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)

        #termination
        if sm.timer.is_elapsed(time):
            print("Go fold Accomplished → next state")
            sm.next(time)

    def contactsAchieved(self):
        contacts_achieved = True
        for leg in range(4):
            if self.B_contacts[leg][2] > -0.04:
                contacts_achieved = False
            elif not self.contact_state[leg]:
                contacts_achieved = False
        return  contacts_achieved

    def searchingContacts(self, sm, time):
        if sm.first_time:
            print(colored("[startupProcedure t: " + str(self.time[0]) + "Searching contacts", "blue"))
            # sample feet position
            self.B_feet_vel = self.u.full_listOfArrays(4, 3, 0, 0.)
            neutral_fb_jointstate = np.hstack((pin.neutral(self.robot.model)[0:7], self.q))
            pin.forwardKinematics(self.robot.model, self.robot.data, neutral_fb_jointstate)
            pin.updateFramePlacements(self.robot.model, self.robot.data)
            for leg in range(4):
                foot = conf.robot_params[self.robot_name]['ee_frames'][leg]
                foot_id = self.robot.model.getFrameId(foot)
                self.B_contacts_des[leg] = self.robot.data.oMf[foot_id].translation.copy()
            # increase of the motion (m/s)
            if self.real_robot:
                self.delta_z = 0.005
            else:
                self.delta_z = 0.1
        #on loop
        #print(f"Searching contacts leg...{time}")
        h_R_w = self.b_R_w @ pin.rpy.rpyToMatrix(0, 0, self.u.angPart(self.basePoseW)[2])
        for leg in range(4):
            # update feet task to extend feet to acquire contact
            self.B_contacts_des[leg][2] -= self.delta_z * self.dt
            q_des_leg, isFeasible = self.IK.ik_leg(h_R_w.T @ self.B_contacts_des[leg],
                                                   self.leg_names[leg],
                                                   self.legConfig[self.leg_names[leg]][0],
                                                   self.legConfig[self.leg_names[leg]][1])
            self.u.setLegJointState(leg, q_des_leg, self.q_des)
        for leg in range(4):
            self.B_feet_vel[leg][2] = -self.delta_z
            qd_leg_des = self.IK.diff_ik_leg(q_des=self.q_des,
                                             B_v_foot=self.B_feet_vel[leg],
                                             leg=self.leg_names[leg],
                                             update=leg == 0)
            self.u.setLegJointState(leg, qd_leg_des, self.qd_des)
        self.tau_ffwd[:] = 0.
        self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
        #terminate
        if self.contactsAchieved():
            print("Searching contacts → next state")
            self.basePoseW_des = self.basePoseW.copy()
            self.baseTwistW_des[:] = 0
            self.comPoseW_des = self.comPoseW.copy()
            self.comTwistW_des[:] = 0
            self.q_des = self.q.copy()
            self.qd_des[:] = 0
            # base height
            base_height = 0.
            for leg in range(4):
                base_height -= self.B_contacts[leg][2]
            self.leg_odom.reset(np.hstack([0., 0., base_height / 4, self.quaternion, self.q]))
            sm.next(time)

    def applyGravityComp(self, sm, time):
        if sm.first_time:
            print(colored("[startupProcedure t: " + str(self.time[0]) + "s] appling gravity compensation", "blue"))

        #on loop
        #print(f"gravity comp...{time}")
        alpha = sm.timer.get_elapsed_time(time)/sm.get_state_duration()
        self.tau_ffwd, self.grForcesW_des = self.wbc.gravityCompensationBase(self.B_contacts,
                                                                             self.wJ,
                                                                             self.h_joints,
                                                                             self.basePoseW)
        #self.visualizeContacts()
        self.tau_ffwd *= alpha
        self.qd_des[:] = 0
        self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)

        #termination
        if sm.timer.is_elapsed(time):
            print("gravity comp Accomplished → next state")
            sm.next(time)

    def standUp(self, sm, time):
        if sm.first_time:
            print(colored(f"[startupProcedure to make RobotHeight {self.robot_height+0.02} t: " + str(self.time[0]) + "s] moving to desired height (" + str(np.around(self.robot_height+0.02, 3)) +" m)", "blue"))
            #compute desired feet position associated to desired robot height
            #1 sample B_contacts_des = actual
            self.B_contacts_sampled = self.u.full_listOfArrays(4, 3)
            for leg in range(4):
                self.B_contacts_sampled[leg] = self.B_contacts[leg].copy()
                self.B_contacts_des[leg] = self.B_contacts[leg].copy()

        #on loop
        #print(f"standUp leg...{time}")
        alpha = sm.timer.get_elapsed_time(time)/sm.get_state_duration()
        #update des feet positions Z component
        for leg in range(4):
            self.B_contacts_des[leg][2] = (1-alpha) *self.B_contacts_sampled[leg][2] + alpha * (-self.robot_height)

        #compute ik
        for leg in range(4):
            q_des_leg, isFeasible = self.IK.ik_leg(self.B_contacts_des[leg],
                                                   self.leg_names[leg],
                                                   self.legConfig[self.leg_names[leg]][0],
                                                   self.legConfig[self.leg_names[leg]][1])
            if isFeasible:
                self.u.setLegJointState(leg, q_des_leg, self.q_des)
        self.tau_ffwd, self.grForcesW_des = self.wbc.gravityCompensationBase(self.B_contacts,  self.wJ, self.h_joints,  self.basePoseW)
        #self.visualizeContacts()
        self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
        #termination
        if sm.timer.is_elapsed(time):
            print("standUp Accomplished → next state")
            sm.next(time)

    def startupProcedure(self):
        ros.sleep(.5)
        print(colored("Starting up", "blue"))
        if self.robot_name == 'hyq' or self.robot_name == 'solo':
            super(QuadrupedController, self).startupProcedure()
            return



        self.q_des = self.q.copy()
        self.pid = PidManager(self.joint_names)
        self.pid.setPDjoints(self.kp_j, self.kd_j, self.ki_j)

        if self.go0_conf == 'standUp':
            self._startup_from_stand_up()
        elif self.go0_conf == 'standDown':
            #deprecated
            #self._startup_from_stand_down()
            # this is needed to avoid initial instability in the real robot
            for i in range(10):
                self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
                ros.sleep(0.01)
            self.homing_sm = StateMachine(verbose=0)
            # IMU BIAS ESTIMATION
            if self.real_robot and (self.robot_name == 'go1' or self.robot_name == 'go2' or self.robot_name == 'aliengo'):
                self.homing_sm.add_state("IMUBiasEstimation", self.IMUBiasEstimation, 1.)
            self.homing_sm.add_state("goFold", self.goFoldConfig, 1.)
            self.homing_sm.add_state("searchingContacts", self.searchingContacts)
            self.homing_sm.add_state("applyGravityComp", self.applyGravityComp, 1.)
            self.homing_sm.add_state("standUp", self.standUp, 2.)
            self.homing_sm.start(start_time=self.time)
            try:
                while not ros.is_shutdown() and self.homing_sm.running:
                    self.updateKinematics()
                    self.homing_sm.step(self.time)
                    self.rate.sleep()
                    self.time = np.round(self.time + self.dt, 4)  # np.array([self.loop_time]), 3)
            except (ros.ROSInterruptException, ros.service.ServiceException):
                ros.signal_shutdown("killed")
                self.deregister_node()

    def _startup_from_stand_up(self):
        for i in range(10):
            self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
            ros.sleep(0.01)
        self.q_des = conf.robot_params[self.robot_name]['q_0']
        alpha = 0.
        try:
            print(colored(f"[startupProcedure to {self.q_des} t: " + str(self.time[0]) + "s] applying gravity compensation", "blue"))
            GCStartTime = self.time
            while not ros.is_shutdown():
                q_norm = np.linalg.norm(self.q - self.q_des)
                qd_norm = np.linalg.norm(self.qd - self.qd_des)
                if q_norm < 0.1 and qd_norm < 0.1 or self.time > 5:
                    break
                self.updateKinematics()
                # self.visualizeContacts()
                GCTime = self.time - GCStartTime
                if GCTime <= self.gravity_comp_duration:
                    if alpha < 1:
                        alpha = GCTime/self.gravity_comp_duration

                self.send_command(self.q_des, self.qd_des, alpha*p.wbc.gravityCompensationBase(self.B_contacts,
                                                            self.wJ,
                                                            self.h_joints,
                                                            self.basePoseW))

            # IMU BIAS ESTIMATION
            if self.real_robot and (self.robot_name == 'go1' or self.robot_name == 'go2' or self.robot_name == 'aliengo'):
                print(colored("[startupProcedure t: " + str(self.time[0]) + "s] Imu bias estimation", "blue"))
                # print('counter: ' + self.imu_utils.counter + ', timeout: ' + self.imu_utils.timeout)
                while self.imu_utils.counter < self.imu_utils.timeout:
                    self.updateKinematics()
                    self.imu_utils.IMU_bias_estimation(self.b_R_w, self.baseLinAccB)
                    self.tau_ffwd[:] = 0.
                    self.send_command(self.q_des, self.qd_des, self.tau_ffwd)


        except (ros.ROSInterruptException, ros.service.ServiceException):
            ros.signal_shutdown("killed")
            self.deregister_node()



    def _startup_from_stand_down(self):
        self.q_des = self.q.copy()
        self.pid = PidManager(self.joint_names)

        for i in range(10):
            self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
            ros.sleep(0.01)


        q_init = self.q.copy()
        q_ref = self.q.copy()
        for i in range(12):
            # modify HFEs & KFEs
            if (i%3) != 0:
                q_ref[i] =  conf.robot_params[self.robot_name]['q_fold'][i]
        # IMU BIAS ESTIMATION
        if self.real_robot and (self.robot_name == 'go1' or self.robot_name == 'go2' or self.robot_name == 'aliengo'):
            print(colored("[startupProcedure t: " + str(self.time[0]) + "s] Imu bias estimation", "blue"))
            # print('counter: ' + self.imu_utils.counter + ', timeout: ' + self.imu_utils.timeout)
            while self.imu_utils.counter < self.imu_utils.timeout:
                self.updateKinematics()
                self.imu_utils.IMU_bias_estimation(self.b_R_w, self.baseLinAccB)
                self.tau_ffwd[:] = 0.
                self.send_command(self.q_des, self.qd_des, self.tau_ffwd)

        self.pid.setPDjoints(self.kp_j, self.kd_j, self.ki_j)

        # Going to fold config
        print(colored("[startupProcedure t: " + str(self.time[0]) + "s] Going to fold configuration", "blue"))
        ref_timeout = int(self.imu_utils.timeout / 2)
        ref_counter = 0
        while ref_counter < ref_timeout:
            self.updateKinematics()
            self.tau_ffwd[:] = 0.
            sigma = ref_counter / ref_timeout
            self.q_des = (1 - sigma) * q_init + sigma * q_ref
            self.send_command(self.q_des, self.qd_des, self.tau_ffwd)
            ref_counter += 1


        # initial feet position
        B_feet_vel = self.u.full_listOfArrays(4, 3, 0, 0.)
        neutral_fb_jointstate = np.hstack((pin.neutral(self.robot.model)[0:7], self.q))
        pin.forwardKinematics(self.robot.model, self.robot.data, neutral_fb_jointstate)
        pin.updateFramePlacements(self.robot.model, self.robot.data)

        for leg in range(4):
            foot = conf.robot_params[self.robot_name]['ee_frames'][leg]
            foot_id = self.robot.model.getFrameId(foot)
            self.B_contacts_des[leg] = self.robot.data.oMf[foot_id].translation.copy()
        # desired final height
        neutral_fb_jointstate[7:] = self.q.copy()#conf.robot_params[self.robot_name]['q_0']

        # increase of the motion (m/s)
        if self.real_robot:
            delta_z = 0.005
        else:
            delta_z = 0.1
        update = [True, True, True, True]
        ########################
        # FINITE STATE MACHINE #
        ########################
        # state = -1: initialize pid
        # state = 0: not all the contacts are active, move the feet in order to activate all the contacts (use IK+PD)
        # state = 1: apply gravity compensation for 1 second
        # state = 2: apply PD + gravity compensation
        # state = 3: exit
        state = 0
        print(colored("[startupProcedure t: " + str(self.time[0]) + "s] searching contacts", "blue"))
        try:
            while not ros.is_shutdown() and state != 4:
                self.updateKinematics()
                # self.visualizeContacts()

                if state == 0:
                    switch_cond = True
                    for leg in range(4):
                        if self.B_contacts[leg][2] > -0.04:
                            switch_cond = False
                        elif not self.contact_state[leg]:
                            switch_cond = False
                        if switch_cond == False:
                            break

                    if not switch_cond:
                        h_R_w = self.b_R_w @ pin.rpy.rpyToMatrix(0, 0, self.u.angPart(self.basePoseW)[2])

                        for leg in range(4):
                            # update feet task
                            self.B_contacts_des[leg][2] -= delta_z * self.dt
                            q_des_leg, isFeasible = self.IK.ik_leg(h_R_w.T@ self.B_contacts_des[leg],
                                                                   self.leg_names[leg],
                                                                   self.legConfig[self.leg_names[leg]][0],
                                                                   self.legConfig[self.leg_names[leg]][1])

                            self.u.setLegJointState(leg, q_des_leg, self.q_des)


                        for leg in range(4):
                                B_feet_vel[leg][2] = -delta_z
                                qd_leg_des = self.IK.diff_ik_leg(q_des=self.q_des,
                                                                   B_v_foot=B_feet_vel[leg],
                                                                   leg=self.leg_names[leg],
                                                                   update=leg == 0)
                                self.u.setLegJointState(leg, qd_leg_des, self.qd_des)
                        # if self.log_counter != 0:
                        #     self.qd_des = (self.q_des - self.q_des_log[:, self.log_counter]) / self.dt
                        self.tau_ffwd[:] = 0.

                    else:
                        print(colored("[startupProcedure t: " + str(self.time[0]) + "s] appling gravity compensation", "blue"))
                        self.basePoseW_des = self.basePoseW.copy()
                        self.baseTwistW_des[:] = 0
                        self.comPoseW_des = self.comPoseW.copy()
                        self.comTwistW_des[:] = 0
                        self.q_des = self.q.copy()
                        self.qd_des[:] = 0
                        # base height
                        base_height = 0.
                        for leg in range(4):
                            base_height -= self.B_contacts[leg][2]
                        self.leg_odom.reset(np.hstack([0., 0., base_height/4, self.quaternion, self.q]))

                        state = 1
                        GCStartTime = self.time
                        alpha = 0.

                if state == 1:
                    GCTime = self.time-GCStartTime
                    self.qd_des[:] = 0
                    if GCTime <= self.gravity_comp_duration:
                        if alpha < 1:
                            alpha = GCTime

                        self.tau_ffwd, self.grForcesW_des = self.wbc.gravityCompensationBase(self.B_contacts,
                                                            self.wJ,
                                                            self.h_joints,
                                                            self.basePoseW)

                        self.visualizeContacts()
                        
                        self.tau_ffwd *= alpha
                    else:
                        print(colored(f"[startupProcedure to make RobotHeight {self.robot_height+0.02} t: " + str(self.time[0]) + "s] moving to desired height (" + str(np.around(self.robot_height+0.02, 3)) +" m)", "blue"))
                        HStarttime = self.time
                        # 5-th order polynomial
                        #start
                        final_comPose_des = self.comPoseW.copy()
                        final_comPose_des[2] = self.robot_height+0.02
                        pos, vel, acc = polynomialRef(self.comPoseW, final_comPose_des,
                                                      np.zeros(6), np.zeros(6),
                                                      np.zeros(6), np.zeros(6),
                                                      self.standup_period)
                        self.W_contacts_des = self.W_contacts.copy()
                        state = 2

                if state == 2:
                    if self.time - HStarttime < self.standup_period:
                        self.comPoseW_des = pos(self.time - HStarttime)
                        self.comTwistW_des = vel(self.time - HStarttime)
                        self.Wcom2Joints_des()
                        # self.wbc.gravityCompensation(self.W_contacts, self.wJ, self.h_joints, self.basePoseW, self.comPoseW)
                        self.tau_ffwd, self.grForcesW_des = self.wbc.gravityCompensationBase(self.B_contacts,
                                    self.wJ,
                                    self.h_joints,
                                    self.basePoseW)

                        self.visualizeContacts()

                    else:
                        print(colored("[startupProcedure t: " + str(self.time[0]) + "s] desired height reached", "blue"))
                        self.baseTwistW_des[:] = 0.
                        self.comTwistW_des[:] = 0.
                        state = 3
                        WTime = self.time

                if state == 3:
                    self.qd_des[:] = 0.
                    if all(np.abs(self.qd) < 0.025) and (self.time - WTime) >0.5:
                        print(colored("[startupProcedure t: " + str(self.time[0]) + "s] completed", "green"))
                        state = 4
                    else:
                        # enter
                        # if any of the joint velocities is larger than 0.02 or
                        # if the watchdog timer is not expired (0.5 sec)
                        # self.wbc.gravityCompensation(self.W_contacts, self.wJ, self.h_joints, self.basePoseW, self.comPoseW)
                        self.tau_ffwd, self.grForcesW_des = self.wbc.gravityCompensationBase(self.B_contacts,
                                    self.wJ,
                                    self.h_joints,
                                    self.basePoseW)

                        self.visualizeContacts()

                self.send_command(self.q_des, self.qd_des, self.tau_ffwd)

        except (ros.ROSInterruptException, ros.service.ServiceException):
            ros.signal_shutdown("killed")
            self.deregister_node()

    def _startup_from_stand_down(self):
        self.q_des = self.q.copy()
        self.pid = PidManager(self.joint_names)

        for i in range(10):
            self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
            ros.sleep(0.01)

        q_init = self.q.copy()
        q_ref = self.q.copy()
        for i in range(12):
            # modify HFEs & KFEs
            if (i % 3) != 0:
                q_ref[i] = conf.robot_params[self.robot_name]['q_fold'][i]
        # IMU BIAS ESTIMATION
        if self.real_robot and (self.robot_name == 'go1' or self.robot_name == 'go2' or self.robot_name == 'aliengo'):
            print(colored("[startupProcedure t: " + str(self.time[0]) + "s] Imu bias estimation", "blue"))
            # print('counter: ' + self.imu_utils.counter + ', timeout: ' + self.imu_utils.timeout)
            while self.imu_utils.counter < self.imu_utils.timeout:
                self.updateKinematics()
                self.imu_utils.IMU_bias_estimation(self.b_R_w, self.baseLinAccB)
                self.tau_ffwd[:] = 0.
                self.send_command(self.q_des, self.qd_des, self.tau_ffwd)

        self.pid.setPDjoints(self.kp_j, self.kd_j, self.ki_j)

        # Going to fold config
        print(colored("[startupProcedure t: " + str(self.time[0]) + "s] Going to fold configuration", "blue"))
        ref_timeout = int(self.imu_utils.timeout / 2)
        ref_counter = 0
        while ref_counter < ref_timeout:
            self.updateKinematics()
            self.tau_ffwd[:] = 0.
            sigma = ref_counter / ref_timeout
            self.q_des = (1 - sigma) * q_init + sigma * q_ref
            self.send_command(self.q_des, self.qd_des, self.tau_ffwd)
            ref_counter += 1

        # initial feet position
        B_feet_vel = self.u.full_listOfArrays(4, 3, 0, 0.)
        neutral_fb_jointstate = np.hstack((pin.neutral(self.robot.model)[0:7], self.q))
        pin.forwardKinematics(self.robot.model, self.robot.data, neutral_fb_jointstate)
        pin.updateFramePlacements(self.robot.model, self.robot.data)

        for leg in range(4):
            foot = conf.robot_params[self.robot_name]['ee_frames'][leg]
            foot_id = self.robot.model.getFrameId(foot)
            self.B_contacts_des[leg] = self.robot.data.oMf[foot_id].translation.copy()
        # desired final height
        neutral_fb_jointstate[7:] = self.q.copy()  # conf.robot_params[self.robot_name]['q_0']

        # increase of the motion (m/s)
        if self.real_robot:
            delta_z = 0.005
        else:
            delta_z = 0.1
        update = [True, True, True, True]
        ########################
        # FINITE STATE MACHINE #
        ########################
        # state = -1: initialize pid
        # state = 0: not all the contacts are active, move the feet in order to activate all the contacts (use IK+PD)
        # state = 1: apply gravity compensation for 1 second
        # state = 2: apply PD + gravity compensation
        # state = 3: exit
        state = 0
        print(colored("[startupProcedure t: " + str(self.time[0]) + "s] searching contacts", "blue"))
        try:
            while not ros.is_shutdown() and state != 4:
                self.updateKinematics()
                # self.visualizeContacts()

                if state == 0:
                    switch_cond = True
                    for leg in range(4):
                        if self.B_contacts[leg][2] > -0.04:
                            switch_cond = False
                        elif not self.contact_state[leg]:
                            switch_cond = False
                        if switch_cond == False:
                            break

                    if not switch_cond:
                        h_R_w = self.b_R_w @ pin.rpy.rpyToMatrix(0, 0, self.u.angPart(self.basePoseW)[2])

                        for leg in range(4):
                            # update feet task
                            self.B_contacts_des[leg][2] -= delta_z * self.dt
                            q_des_leg, isFeasible = self.IK.ik_leg(h_R_w.T @ self.B_contacts_des[leg],
                                                                   self.leg_names[leg],
                                                                   self.legConfig[self.leg_names[leg]][0],
                                                                   self.legConfig[self.leg_names[leg]][1])

                            self.u.setLegJointState(leg, q_des_leg, self.q_des)

                        for leg in range(4):
                            B_feet_vel[leg][2] = -delta_z
                            qd_leg_des = self.IK.diff_ik_leg(q_des=self.q_des,
                                                             B_v_foot=B_feet_vel[leg],
                                                             leg=self.leg_names[leg],
                                                             update=leg == 0)
                            self.u.setLegJointState(leg, qd_leg_des, self.qd_des)
                        # if self.log_counter != 0:
                        #     self.qd_des = (self.q_des - self.q_des_log[:, self.log_counter]) / self.dt
                        self.tau_ffwd[:] = 0.

                    else:
                        print(colored("[startupProcedure t: " + str(self.time[0]) + "s] appling gravity compensation",
                                      "blue"))
                        self.basePoseW_des = self.basePoseW.copy()
                        self.baseTwistW_des[:] = 0
                        self.comPoseW_des = self.comPoseW.copy()
                        self.comTwistW_des[:] = 0
                        self.q_des = self.q.copy()
                        self.qd_des[:] = 0
                        # base height
                        base_height = 0.
                        for leg in range(4):
                            base_height -= self.B_contacts[leg][2]
                        self.leg_odom.reset(np.hstack([0., 0., base_height / 4, self.quaternion, self.q]))

                        state = 1
                        GCStartTime = self.time
                        alpha = 0.

                if state == 1:
                    GCTime = self.time - GCStartTime
                    self.qd_des[:] = 0
                    if GCTime <= self.gravity_comp_duration:
                        if alpha < 1:
                            alpha = GCTime

                        self.tau_ffwd, self.grForcesW_des = self.wbc.gravityCompensationBase(self.B_contacts,
                                                                                             self.wJ,
                                                                                             self.h_joints,
                                                                                             self.basePoseW)

                        #self.visualizeContacts()

                        self.tau_ffwd *= alpha
                    else:
                        print(colored(f"[startupProcedure to make RobotHeight {self.robot_height + 0.02} t: " + str(
                            self.time[0]) + "s] moving to desired height (" + str(
                            np.around(self.robot_height + 0.02, 3)) + " m)", "blue"))
                        HStarttime = self.time
                        # 5-th order polynomial
                        # start
                        final_comPose_des = self.comPoseW.copy()
                        final_comPose_des[2] = self.robot_height + 0.02
                        pos, vel, acc = polynomialRef(self.comPoseW, final_comPose_des,
                                                      np.zeros(6), np.zeros(6),
                                                      np.zeros(6), np.zeros(6),
                                                      self.standup_period)
                        self.W_contacts_des = self.W_contacts.copy()
                        state = 2

                if state == 2:
                    if self.time - HStarttime < self.standup_period:
                        self.comPoseW_des = pos(self.time - HStarttime)
                        self.comTwistW_des = vel(self.time - HStarttime)
                        self.Wcom2Joints_des()
                        # self.wbc.gravityCompensation(self.W_contacts, self.wJ, self.h_joints, self.basePoseW, self.comPoseW)
                        self.tau_ffwd, self.grForcesW_des = self.wbc.gravityCompensationBase(self.B_contacts,
                                                                                             self.wJ,
                                                                                             self.h_joints,
                                                                                             self.basePoseW)

                        #self.visualizeContacts()

                    else:
                        print(
                            colored("[startupProcedure t: " + str(self.time[0]) + "s] desired height reached", "blue"))
                        self.baseTwistW_des[:] = 0.
                        self.comTwistW_des[:] = 0.
                        state = 3
                        WTime = self.time

                if state == 3:
                    self.qd_des[:] = 0.
                    if all(np.abs(self.qd) < 0.025) and (self.time - WTime) > 0.5:
                        print(colored("[startupProcedure t: " + str(self.time[0]) + "s] completed", "green"))
                        state = 4
                    else:
                        # enter
                        # if any of the joint velocities is larger than 0.02 or
                        # if the watchdog timer is not expired (0.5 sec)
                        # self.wbc.gravityCompensation(self.W_contacts, self.wJ, self.h_joints, self.basePoseW, self.comPoseW)
                        self.tau_ffwd, self.grForcesW_des = self.wbc.gravityCompensationBase(self.B_contacts,
                                                                                             self.wJ,
                                                                                             self.h_joints,
                                                                                             self.basePoseW)

                        #self.visualizeContacts()

                self.send_command(self.q_des, self.qd_des, self.tau_ffwd)

        except (ros.ROSInterruptException, ros.service.ServiceException):
            ros.signal_shutdown("killed")
            self.deregister_node()

    def saveData(self, path, filename='DATA.mat', EXTRADATA={},start=0, stop=-1, verbose = conf.verbose):
        DATA = {}
        DATA['comPosB_log'] = self.comPosB_log[:, start:stop]
        DATA['comVelB_log'] = self.comVelB_log[:, start:stop]
        DATA['comPoseW_log'] = self.comPoseW_log[:, start:stop]
        DATA['comTwistW_log'] = self.comTwistW_log[:, start:stop]
        DATA['comPoseW_des_log'] = self.comPoseW_des_log[:, start:stop]
        DATA['comTwistW_des_log'] = self.comTwistW_des_log[:, start:stop]
        DATA['basePoseW_log'] = self.basePoseW_log[:, start:stop]
        DATA['baseTwistW_log'] = self.baseTwistW_log[:, start:stop]
        DATA['basePoseW_des_log'] = self.basePoseW_des_log[:, start:stop]
        DATA['baseTwistW_des_log'] = self.baseTwistW_des_log[:, start:stop]

        DATA['basePoseW_legOdom_log'] = self.basePoseW_legOdom_log[:, start:stop]
        DATA['baseTwistW_legOdom_log'] = self.baseTwistW_legOdom_log[:, start:stop]
        DATA['q_des_log'] = self.q_des_log[:, start:stop]
        DATA['q_log'] = self.q_log[:, start:stop]
        DATA['qd_des_log'] = self.qd_des_log[:, start:stop]
        DATA['qd_log'] = self.qd_log[:, start:stop]
        DATA['tau_fb_log'] = self.tau_fb_log[:, start:stop]
        DATA['tau_ffwd_log'] = self.tau_ffwd_log[:, start:stop]
        DATA['tau_des_log'] = self.tau_des_log[:, start:stop]
        DATA['tau_log'] = self.tau_log[:, start:stop]

        DATA['grForcesW_log'] = self.grForcesW_log[:, start:stop]
        DATA['grForcesW_des_log'] = self.grForcesW_des_log[:, start:stop]
        DATA['grForcesW_wbc_log'] = self.grForcesW_wbc_log[:, start:stop]
        DATA['grForcesW_gt_log'] = self.grForcesW_gt_log[:, start:stop]
        DATA['grForcesB_log'] = self.grForcesB_log[:, start:stop]
        DATA['contact_state_log'] = self.contact_state_log[:, start:stop]

        DATA['baseLinAccW_log'] = self.baseLinAccW_log[:, start:stop]
        DATA['baseLinAccB_log'] = self.baseLinAccB_log[:, start:stop]

        DATA['comVelW_leg_odom_log'] = self.comVelW_leg_odom_log[:, start:stop]


        DATA['B_contacts_log'] = self.B_contacts_log[:, start:stop]
        DATA['B_contacts_des_log'] = self.B_contacts_des_log[:, start:stop]

        DATA['W_contacts_log'] = self.W_contacts_log[:, start:stop]
        DATA['W_contacts_des_log'] = self.W_contacts_des_log[:, start:stop]

        DATA['B_vel_contacts_des_log'] = self.B_vel_contacts_des_log[:, start:stop]
        DATA['W_vel_contacts_des_log'] = self.W_vel_contacts_des_log[:, start:stop]

        DATA['baseLinTwistImuW_log'] = self.baseLinTwistImuW_log[:, start:stop]


        DATA['wrench_fbW_log'] = self.wrench_fbW_log[:, start:stop]
        DATA['wrench_ffW_log'] = self.wrench_ffW_log[:, start:stop]
        DATA['wrench_gW_log'] = self.wrench_gW_log[:, start:stop]
        DATA['wrench_desW_log'] = self.wrench_desW_log[:, start:stop]


        DATA['kp_j'] = self.kp_j
        DATA['kd_j'] = self.kd_j
        DATA['ki_j'] = self.ki_j
        DATA['kp_lin'] = self.kp_lin
        DATA['kd_lin'] = self.kd_lin
        DATA['kp_ang'] = self.kp_ang
        DATA['kd_ang'] = self.kd_ang
        DATA['kp_wbc_j'] = self.kp_wbc_j
        DATA['kd_wbc_j'] = self.kd_wbc_j
        DATA['ki_wbc_j'] = self.ki_wbc_j

        DATA['time_log'] = self.time_log[start:stop]
        DATA['loop_time_log'] = self.loop_time_log[start:stop]
        DATA['log_counter'] = self.log_counter

        for key in EXTRADATA.keys():
            if key in DATA.keys():
                print("Key '" + key + "' found in EXTRADATA. Ignored", flush=True)
            else:
                DATA[key] = EXTRADATA[key]

        if filename[-4:] != ".mat":
            filename+=".mat"

        savemat(path+"/"+filename, DATA, do_compression=True)

        if verbose:
            print('Log data saved in '+ path+"/"+filename, flush=True)

    def get_current_frame_file(self):
        allfiles = [f for f in os.listdir('/tmp/camera_save') if os.path.isfile(os.path.join('/tmp/camera_save', f))]
        if len(allfiles) != 0:
            return max(allfiles)
        else:
            None

    def gracefulCollapse(self):
        self.alphaCollapse -= self.dt / self.collapseTime
        if self.alphaCollapse<=0:
            self.alphaCollapse = 0.0
            if self.state_estimation=='pronto':
                os.system(" rosnode kill /aliengo_joint_swapper")
                os.system(" rosnode kill /pronto_aliengo")
            ros.signal_shutdown("killed")
            self.deregister_node()
            return True
        else:
            self.pid.setPDjoints(self.alphaCollapse * self.kp_act, self.alphaCollapse * self.kd_act, self.alphaCollapse * self.ki_act)
            return False

    def saveVideo(self, path, start_file=None, filename='record', format='mkv',  fps=60, speedUpDown=1, remove_jpg=False):
        # only if camera_xxx.world has been used
        # for details on commands, check https://ffmpeg.org/ffmpeg.html
        if 'camera' not in self.world_name_str:
            print('Cannot create a video of a not camera world (world_name:'+self.world_name_str+')', flush=True)
            return
        #
        # # kill gazebo
        # os.system("killall rosmaster rviz gzserver gzclient")
        try:
            if start_file != None:
                # delete all the files before start_file
                allfiles = [f for f in os.listdir('/tmp/camera_save') if os.path.isfile(os.path.join('/tmp/camera_save', f))]
                for f in allfiles:
                    if f < start_file:
                        #parentheses are not supported by bash
                        f = f.replace('(', '\(')
                        f = f.replace(')', '\)')
                        cmd = 'rm /tmp/camera_save/' + f
                        os.system(cmd)

            if '.' in filename:
                filename=filename[:, filename.find('.')]
            if path[-1] != '/':
                path+='/'
            videoname = path + filename + '.' + format
            save_video_cmd = "ffmpeg -hide_banner -loglevel error -r "+str(fps)+" -pattern_type glob -i '/tmp/camera_save/default_camera_link_my_camera*.jpg' -c:v libx264 "+videoname
            ret = os.system(save_video_cmd)
            saved = ''
            if ret == 0:
                saved = ' saved'
            else:
                saved = ' did not saved'
            #print('Video '+videoname+saved, flush=True)


            if speedUpDown <= 0:
                print('speedUpDown must be greather than 0.0', flush=True)
            else:
                if speedUpDown != 1:
                    pts_multiplier = int(1 / speedUpDown)
                    videoname_speedUpDown = path + filename + str(speedUpDown).replace('.', '')+'x.'+format
                    speedUpDown_cmd = "ffmpeg -hide_banner -loglevel error -i "+videoname+" -filter:v 'setpts="+str(pts_multiplier)+"*PTS' "+videoname_speedUpDown
                    ret = os.system(speedUpDown_cmd)
                    saved = ''
                    if ret == 0:
                        saved= ' saved'
                    else:
                        saved = ' did not saved'
                    #print('Video ' + videoname_speedUpDown+saved, flush=True)

            if remove_jpg:
                remove_jpg_cmd = 'for f in /tmp/camera_save/*; do rm "$f"; done'
                os.system(remove_jpg_cmd)
                #print('Jpg files removed', flush=True)
        except:
            pass



if __name__ == '__main__':
    p = QuadrupedController('aliengo')
    world_name = 'fast.world'
    use_gui = False
    p.state_estimation = 'odometry' # 'odometry','imu', 'pronto', 'ground_truth' (only sim), 'mocap'
    rl_control = 'none' #'none', 'sensor_based' (Giulio), 'state_est_based' (Riccardo)
    # NOTE: in the RL controller, SE NN is used only if state estimation is not pronto
    rl_use_nn_se = p.state_estimation != 'pronto'
    # Which policy family RlVelocityController runs: 'legacy' is the original set of networks under
    # components/rl_velocity_controller/policies, 'velocity' is the single-network policy in
    # rl_quadruped/policies that estimates the base velocity inside the actor (and so ignores
    # rl_use_nn_se, having no variant that takes a measured base velocity).
    rl_policy = 'velocity'  # 'legacy', 'velocity'
    use_joy = False
    generate_reference = False
    p.SAVE_BAG = False  #
    if p.robot_name == 'go2':
        p.custom_locosim_launch_file = True


    if rl_control == 'state_est_based':
        if p.real_robot and (p.state_estimation != 'pronto' and p.state_estimation != 'pronto'):
            print(colored("RL is state_est based need to start state estimation!","red"))
            sys.exit()
        rl_controller = RlVelocityController(p.robot_name, p.dt, use_nn_se=rl_use_nn_se,
                                             debug=True, policy=rl_policy)
    if rl_control == 'sensor_based':
        rl_controller = LocomotionPolicyWrapper(use_state_est=True, dt = p.dt)

    try:
        #p.startController(world_name='slow.world')
        p.startController(world_name=world_name,
                          use_ground_truth_contacts=True,
                          additional_args=['gui:='+str(use_gui),
                                           'go0_conf:=standDown',
                                           'rviz:=true',
                                           *(['task_period:=0.002'] if p.real_robot else [])]) #change task period to 500Hz instead of 1000Hz for real robot
        if p.SAVE_BAG:
            from datetime import datetime, timezone

            now = datetime.now()
            format_date = now.strftime("%Y-%m-%d-%H-%M-%S")
            p.recorder = RosbagControlledRecorder(
                topics='/aliengo/joint_states /aliengo/imu /aliengo/feet_forces /state_estimator_pronto/pose '
                       '/state_estimator_pronto/twist /state_estimator_pronto/vel_raw  '
                       '/state_estimator_pronto/stance /value_function /qualisys/robot/pose /rl_ref_vel /tf /tf_static',
                        bag_name="saferl_" + format_date + ".bag", record_from_startup_=False)

            p.recorder.start_recording_srv()
        if use_joy:
            joy = JoyManager()
        p.startupProcedure()
        if p.state_estimation=='pronto':
            launchFileNode("mocap_qualisys", "qualisys.launch")
            launchFileNode("pronto_aliengo", "pronto_aliengo.launch", additional_args=['pronto_conf:='+p.pronto_config,
                                                                                       'use_sim_time:='+str(not p.real_robot)])
        if rl_control != 'none':
            p.pid.setPDjoints(rl_controller.kp, rl_controller.kd, np.full(12,0))
        p.counter = 0
        p.startTime = p.time

        if generate_reference:
            p.ref_gen = QuadrupedTasks(task='pushup', robot_conf=conf.robot_params[p.robot_name], gui=True, quadruped=p)
            p.ref_gen.startUp(p.time)
        counter = 0
        #to reduce simulation frequency
        #p.setSimSpeed(dt_sim=0.001, max_update_rate=300, iters=1500)
        while not ros.is_shutdown():
            p.updateKinematics()
            if p.gracefulCollapseFlag:
                if p.gracefulCollapse():
                    break
            if use_joy:
                axes, buttons = joy.get_commands()
                # use a scaling to make the joy input less reactive
                long_x = 0.2 * axes[0]
                long_y = 0.2 * axes[1]
                rot_z = 0.3 * axes[2]
                # safety layer
                if buttons[0] and not p.gracefulCollapseFlag:
                    print(colored("start Graceful collapse", "red"))
                    p.kp_act, p.kd_act, p.ki_act = p.pid.getPDjoints()
                    print(colored(f"Storing actual pd: {p.kp_act}, {p.kd_act},{p.ki_act}"), "red")
                    p.gracefulCollapseFlag = True
                if buttons[1]:
                    print(colored("Severe shutdown!", "red"))
                    if p.state_estimation == 'pronto':
                        os.system(" rosnode kill /aliengo_joint_swapper")
                        os.system(" rosnode kill /pronto_aliengo")
                    ros.signal_shutdown("killed")
                    p.deregister_node()
                    break
            if p.state_estimation == 'pronto':
                if np.all(p.pronto_contacts) and not p.controller_ready:
                    print(colored("ALL FEET ARE IN STANCE: It is possible to start RL controller","red"))
                    p.controller_ready = True
            else:
                p.controller_ready = True
            if rl_control != 'none' and (p.time > (p.startTime + 3.)) and p.controller_ready:
                if use_joy:
                    rl_controller.velocity_cmd = np.array([long_x, long_y, rot_z])
                else:
                    # send random vel every 3 sec btwn -0.4 and 0.4 m/s for x and y, and -0.4 and 0.4 rad/s for rotation
                    if p.time > (p.startTime + 3.0) and p.time % 3.0 == 0:
                        rl_controller.velocity_cmd = np.random.uniform(low=-0.4, high=0.4, size=(3,))
                        
                p.baseTwistW_des[:3] = p.b_R_w.T @ np.append(rl_controller.velocity_cmd[:2], 0.0)
                p.baseTwistW_des[5] = rl_controller.velocity_cmd[2]

                if rl_control == 'state_est_based':
                    # Compute observations for the policy
                    # Disable the lin_vel_b and use lin_acc_b observation if using NN SE
                    if rl_controller.use_nn_se and not rl_controller.debug:
                        lin_acc_b = p.baseLinAccB
                        lin_vel_b = None
                    elif rl_controller.use_nn_se and rl_controller.debug:
                        lin_acc_b = p.baseLinAccB
                        lin_vel_b = p.b_R_w.dot(p.baseTwistW[:3])
                    else:
                        lin_acc_b = None
                        lin_vel_b = p.b_R_w.dot(p.baseTwistW[:3])

                    ang_vel_b = p.b_R_w.dot(p.baseTwistW[3:6])
                    proj_gravity = p.b_R_w.dot(np.array([0,0,-1]))

                    p.rl_q_des = rl_controller.action(lin_acc_b, lin_vel_b, ang_vel_b, proj_gravity, p.q, p.qd, policy_type="default")

                if rl_control == 'sensor_based':
                    h_R_b = p.math_utils.eul2Rot(np.array([p.euler[0],p.euler[1],0.]))
                    p.rl_q_des =  rl_controller.compute_control(h_R_b=h_R_b,
                                                             joints_pos=p.q,
                                                             joints_vel=p.qd,
                                                             ref_base_lin_vel=np.array([rl_controller.velocity_cmd[0], rl_controller.velocity_cmd[1], 0.]),
                                                             ref_base_ang_vel=np.array([0., 0., rl_controller.velocity_cmd[2]]),
                                                             imu_linear_acceleration=p.baseLinAccB,
                                                             imu_angular_velocity= p.b_R_w @ p.baseTwistW[3:],
                                                             imu_orientation=np.array([p.quaternion[3],p.quaternion[0],p.quaternion[1],p.quaternion[2]]))
                #switch off wbc
                p.grForcesW_des = np.zeros((12))
                p.tau_ffwd = np.zeros(12)
                p.send_command(p.rl_q_des, np.zeros(12), np.zeros(12), log_data_in_send_command=True)
            else:
                if generate_reference:
                    p.q_des, p.qd_des, p.tau_ffwd, p.basePoseW_des, p.baseTwistW_des = p.ref_gen.generateReference(p.time)
                else:
                    p.tau_ffwd, p.grForcesW_des = p.wbc.gravityCompensationBase(p.B_contacts,
                                                                                p.wJ,
                                                                                p.h_joints,
                                                                                p.comPoseW)
                p.send_command(p.q_des, p.qd_des, p.alphaCollapse*p.tau_ffwd, log_data_in_send_command=True)

            p.visualizeContacts()
        
    except (ros.ROSInterruptException, ros.service.ServiceException):
        if p.SAVE_BAG:
            p.recorder.stop_recording_srv()
        ros.signal_shutdown("killed")
        p.deregister_node()
        
    if conf.plotting:
        plotJoint('position', time_log=p.time_log, q_log=p.q_log, q_des_log=p.q_des_log, sharex=True, sharey=False,
                  start=0, end=-1)
        plotFrame('position', time_log=p.time_log, des_Pose_log=p.basePoseW_des_log, Pose_log=p.basePoseW_log,
                  title='Base', frame='W', sharex=True, sharey=False, start=0, end=-1)
        plotFrame('velocity', time_log=p.time_log, des_Twist_log=p.baseTwistW_des_log, Twist_log=p.baseTwistW_log,
                  title='Base', frame='W', sharex=True, sharey=False, start=0, end=-1)
    if p.SAVE_BAG:
        p.recorder.stop_recording_srv()
