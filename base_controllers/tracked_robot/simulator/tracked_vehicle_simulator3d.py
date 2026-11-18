import numpy as np
from base_controllers.tracked_robot.simulator.tracked_vehicle import TrackedVehicle, VehicleParam
from base_controllers.tracked_robot.simulator.track import TrackParams
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from  base_controllers.tracked_robot.environment.trajectory import Trajectory, ModelsList
import base_controllers.tracked_robot.utils.maxxi_constants as constants
from base_controllers.utils.math_tools import computeOrientationError
from base_controllers.utils.math_tools import Math
from base_controllers.utils.rosbag_recorder import RosbagControlledRecorder
#to debug
from base_controllers.utils.common_functions import  launchFileGeneric
from base_controllers.utils.ros_publish import RosPub
import rospkg
import os
import pinocchio as pin
import rospy as ros
import tf
from termcolor import colored
from base_controllers.utils.math_tools import RK4_step, backward_euler_step, forward_euler_step, heun_step
from base_controllers.utils.rk45F_integrator import RK45F_step
from base_controllers.tracked_robot.simulator.terrain_manager import create_ramp_mesh
from base_controllers.utils.unit_test import UnitTest

class Ground3D():
    def __init__(self,
                 cohesion=0.0001040 * 1e6,
                 K=0.001,
                 shear_resistance_angle=20/180.*2*np.pi,
                 g=9.81,
                 friction_coefficient=0.1, # 0.1 is very low and on slopes it slips!
                 terrain_stiffness = 1e06,
                 terrain_damping = 1e04,
                 terrain_torsional_stiffness=1e04, #1e05 requires dt = 0.0005
                 terrain_torsional_damping=1e03): #1e04 requires dt = 0.0005
        self.cohesion = cohesion  # [Pa] #not used
        self.K = K  # [m]
        self.shear_resistance_angle = shear_resistance_angle  # [rad] #not used
        self.g = g
        self.friction_coefficient = friction_coefficient
        self.Kt_x = np.eye(3)*terrain_stiffness
        self.Dt_x = np.eye(3) * terrain_damping
        self.Kt_theta = np.eye(3) * terrain_torsional_stiffness
        self.Dt_theta = np.eye(3) * terrain_torsional_damping
        self.Kt_b = terrain_stiffness
        self.Dt_b = terrain_damping
        self.Kt_p = 5.5 #max percentual stiffness increase/(m/s)

class TrackedVehicleSimulator3D:
    def __init__(self,  dt=0.001, ground=None, USE_MESH = False, DEBUG=False, int_method='FORWARD_EULER',  enable_visuals=True, contact_distribution = False, terrain_manager=None):
        self.NO_SLIPPAGE = False
        self.USE_MESH = USE_MESH
        self.TYPE_OF_TERRAIN = 'slopes' #'slopes' /'ramp'
        self.DEBUG = DEBUG
        self.CONTACT_DISTRIBUTION = contact_distribution
        self.enable_visuals=enable_visuals
        self.int_method = int_method
        self.consider_robot_height = True
        self.SAVE_LOGS = False
        self.PROFILE = False
        self.unit_test = UnitTest()

        if  terrain_manager is None:
            print(colored("loading default terrain Manager", "red"))
            from base_controllers.tracked_robot.simulator.terrain_manager import TerrainManager
            self.terrain_manager = TerrainManager(rospkg.RosPack().get_path('tractor_description') + "/meshes/terrain.stl")

        self.dt = dt
        self.vehicle_param = VehicleParam()
        if self.CONTACT_DISTRIBUTION:
            self.track_param = TrackParams(parts_longitudinal=10, parts_lateral=4) #use small discretization otherwise is too slow
        else:
            self.track_param = TrackParams()

        if ground is not None:
            self.ground = ground
        else:
            self.ground = Ground3D()
        self.sigma_l = np.ones((self.track_param.parts_longitudinal, self.track_param.parts_lateral)) * self.vehicle_param.weight / (2 * self.track_param.A)
        self.sigma_r = np.ones((self.track_param.parts_longitudinal, self.track_param.parts_lateral)) * self.vehicle_param.weight / (2 * self.track_param.A)

        self.tracked_robot = TrackedVehicle(self.vehicle_param, self.track_param,  self)
        self.patch_pos_long_l, self.patch_pos_lat_l = self.tracked_robot.getLeftPatchesPositions()
        self.patch_pos_long_r, self.patch_pos_lat_r = self.tracked_robot.getRightPatchesPositions()

        self.w_patch_pos_l = np.zeros((self.track_param.parts_longitudinal*self.track_param.parts_lateral, 3))
        self.w_patch_pos_r = np.zeros((self.track_param.parts_longitudinal * self.track_param.parts_lateral,3))
        self.intersection_points_l =  np.zeros((self.track_param.parts_longitudinal * self.track_param.parts_lateral,3))
        self.intersection_points_r = np.zeros((self.track_param.parts_longitudinal * self.track_param.parts_lateral, 3))
        # this is to debug the baseline from which you cast the rays
        # self.w_patch_pos_l_baseline = np.zeros((self.track_param.parts_longitudinal * self.track_param.parts_lateral, 3))
        # self.w_patch_pos_r_baseline = np.zeros((self.track_param.parts_longitudinal * self.track_param.parts_lateral, 3))
        self.w_patch_vel_l = np.zeros((self.track_param.parts_longitudinal * self.track_param.parts_lateral, 3))
        self.w_patch_vel_r = np.zeros((self.track_param.parts_longitudinal * self.track_param.parts_lateral, 3))
        self.w_Fg_patch_l = np.zeros((self.track_param.parts_longitudinal * self.track_param.parts_lateral, 3))
        self.w_Fg_patch_r = np.zeros((self.track_param.parts_longitudinal * self.track_param.parts_lateral, 3))
        self.w_Mg_patch_l = np.zeros((self.track_param.parts_longitudinal * self.track_param.parts_lateral, 3))
        self.w_Mg_patch_r = np.zeros((self.track_param.parts_longitudinal * self.track_param.parts_lateral, 3))
        self.patch_stiffness_l = np.zeros((self.track_param.parts_longitudinal * self.track_param.parts_lateral))
        self.patch_stiffness_r = np.zeros((self.track_param.parts_longitudinal * self.track_param.parts_lateral))

        self.max_patch_stiffness_l = 3*self.ground.Kt_b
        self.max_patch_stiffness_r = 3*self.ground.Kt_b

        self.initLoggingVars(20.)
        self.math_utils = Math()

        self.q_des = np.zeros(2)
        self.qd_des = np.zeros(2)
        self.pose_des = np.zeros(3)
        self.orient_des = np.zeros(3)
        self.rk45F = RK45F_step(self.dynamics3D, adaptive_step=True)
        self.slow_down_factor = 1.0
        self.out_of_frequency_counter = 0

    def initLoggingVars(self, simulation_duration=20., pose_init = None):
        # these are just for the unit test
        self.t_end = simulation_duration  # [s]
        if pose_init is not None:
            self.pose_init = pose_init
        else:
            self.pose_init = [0., 0., 0.0, 0, 0, 0]  # position in WF, rpy angles
        self.twist_init = [0, 0.0, 0, 0, 0, 0]  # in WF
        self.number_of_steps = np.int32(self.t_end / self.dt)
        self.pose_des_log = np.full((6, self.number_of_steps), np.nan)
        self.pose_log = np.full((6, self.number_of_steps), np.nan)
        self.time_log = np.full((self.number_of_steps), np.nan)
        self.max_patch_stiffness_log = np.full((self.number_of_steps), np.nan)

    def setTerrainManager(self, terrain_manager):
        self.terrain_manager = terrain_manager

    def computeGroundForcesMoments(self,pose, twist, w_R_b, pg, terr_roll, terr_pitch, terr_yaw):
        pc = pose[:3]
        pc_d = twist[:3]
        w_omega = twist[3:]

        #projection of com on the ground level
        pcom_on_ground = pc -self.consider_robot_height*self.w_com_height_vector

        #force is present only if there is linear penetration
        # print("pg", pg)
        # print("pcom_",pcom_on_ground)
        if (self.terrain_normal.dot(pg - pcom_on_ground) > 0.0):
            Fk = self.ground.Kt_x.dot(pg - pcom_on_ground)
            #only if it is approaching
            Fd = np.zeros(3)
            if (self.terrain_normal.dot(pc_d) < 0.0):
                Fd = -self.ground.Dt_x.dot(pc_d)
            else:
                Fd = np.zeros(3)
            # project along terrain normal
            w_Fg =  self.terrain_normal*self.terrain_normal.dot(Fk + Fd)
        else:
            w_Fg = np.array([0.0, 0.0, 0.0])


        #moments due to penetration
        #compute projector on tx, ty contact plane
        N = np.eye(3) -  np.multiply.outer(self.terrain_normal.ravel(), self.terrain_normal.ravel())
        w_e_o =   computeOrientationError(w_R_b, self.w_R_terr)
        b_e_o = w_R_b.T.dot(w_e_o)
        #this projection is fundamental
        w_Mg = N.dot(self.ground.Kt_theta.dot(w_e_o) - self.ground.Dt_theta.dot(w_omega))

        #map to base frame
        b_Fg = w_R_b.T.dot(w_Fg)
        b_Mg = w_R_b.T.dot(w_Mg)

        return b_Fg, b_Mg, w_e_o[0],b_e_o[1]


    def computeDistributedGroundForcesMoments(self,pose, twist, w_R_b, terr_roll, terr_pitch, terr_yaw):

        # this is the direction patches are projected on ground to compute forces
        projection_direction =  np.array([0, 0, 1])
        #projection_direction = self.terrain_normal (does not do load transfer!)
        #projection_direction = w_R_b[:, 2] #w_base_z_axis  (does not do load transfer!)

        # compoute patches under tracks in wf
        patch_counter = 0
        # compute patch positions and twists in world frame
        for i in range(self.track_param.parts_longitudinal):
            for j in range(self.track_param.parts_lateral):
                b_rel_patch_position_l = np.array([self.patch_pos_long_l[i, j], self.patch_pos_lat_l[i, j], -self.consider_robot_height*self.vehicle_param.height])
                b_rel_patch_position_r = np.array([self.patch_pos_long_r[i, j], self.patch_pos_lat_r[i, j], -self.consider_robot_height*self.vehicle_param.height])
                self.w_patch_pos_l[patch_counter, :] = pose[:3] + w_R_b.dot(b_rel_patch_position_l)
                self.w_patch_pos_r[patch_counter, :] = pose[:3] + w_R_b.dot(b_rel_patch_position_r)

                # this is to debug the baseline from which you cast the rays
                # self.w_patch_pos_l_baseline[patch_counter, :] = pose[:3] + w_R_b.dot(np.array([self.patch_pos_long_l[i, j], self.patch_pos_lat_l[i, j], self.terrain_manager.baseline]))
                # self.w_patch_pos_r_baseline[patch_counter, :] = pose[:3] + w_R_b.dot(np.array([self.patch_pos_long_r[i, j], self.patch_pos_lat_r[i, j], self.terrain_manager.baseline]))
                self.w_patch_vel_l[patch_counter, :] = twist[:3] + np.cross(twist[3:],w_R_b.dot(b_rel_patch_position_l))
                self.w_patch_vel_r[patch_counter, :] = twist[:3] + np.cross(twist[3:],w_R_b.dot(b_rel_patch_position_r))
                patch_counter += 1

        #do the ray casting wrt to projection_direction
        if self.USE_MESH:
            self.intersection_points_l = self.terrain_manager.project_points_on_mesh(points=self.w_patch_pos_l, direction=projection_direction)
            self.intersection_points_r = self.terrain_manager.project_points_on_mesh(points=self.w_patch_pos_r, direction=projection_direction)
        else:
            patch_counter = 0
            for patch_pos_l, patch_pos_r in zip(self.w_patch_pos_l, self.w_patch_pos_r):
                self.intersection_points_l[patch_counter, :] = np.concatenate((patch_pos_l[:2], [self.computeZcomponent(patch_pos_l[0], patch_pos_l[0], terr_pitch)]))
                self.intersection_points_r[patch_counter, :] = np.concatenate((patch_pos_r[:2], [self.computeZcomponent(patch_pos_r[0], patch_pos_r[0], terr_pitch)]))
                patch_counter += 1

        if self.DEBUG:
            for point in self.w_patch_pos_l:
                self.ros_pub.add_marker(point, radius=0.02, color="white", alpha=0.5)
            for point in self.intersection_points_l:
                self.ros_pub.add_marker(point, radius=0.01, color="blue")
            for point in self.w_patch_pos_r:
                self.ros_pub.add_marker(point, radius=0.02, color="white", alpha=0.5)
            for point in self.intersection_points_r:
                self.ros_pub.add_marker(point, radius=0.01, color="blue")
            #this is for further debug, this is to debug the baseline from which you cast the rays
            # for point in self.w_patch_pos_l_baseline:
            #     self.ros_pub.add_arrow(point, w_base_z_axis, color="green")
            #     self.ros_pub.add_marker(point, radius=0.01, color="green")

        track_area = self.track_param.length * self.track_param.width

        # compute projector on tx, ty contact plane
        N = np.eye(3) - np.multiply.outer(projection_direction.ravel(), projection_direction.ravel())

        #compute ditributed forces for LEFT track
        #force is present only if there is linear penetration
        patch_counter = 0
        for patch_pos, terrain_pos, patch_vel in zip(self.w_patch_pos_l, self.intersection_points_l, self.w_patch_vel_l):
            rho = projection_direction.dot(terrain_pos - patch_pos)
            if (rho> 0.0 and np.all(np.isfinite(terrain_pos))):
                b_robot_vel_x = w_R_b.T.dot(twist[:3])[0]
                b_patch_pos_x = w_R_b.T.dot(self.w_patch_pos_l[patch_counter, :] - pose[:3])[0] # we do not care about the different Z components cause we take the X

                # change patch stiffness according to speed and pos of patch
                self.patch_stiffness_l[patch_counter] = self.ground.Kt_b * (1. + b_robot_vel_x *self.ground.Kt_p * (b_patch_pos_x + np.sign(b_robot_vel_x)* self.track_param.length/2)  )
                Fk = (self.patch_stiffness_l[patch_counter]*rho)*track_area

                #Fk = self.ground.Kt_x.dot(terrain_pos - patch_pos)*track_area
                #only if it is approaching
                #Fd = np.zeros(3)
                rho_dot = projection_direction.dot(patch_vel)
                if (rho_dot< 0.0):
                    Fd = (-self.ground.Dt_b*rho_dot)*track_area
                else:
                    Fd = 0. #np.zeros(3)
                self.w_Fg_patch_l[patch_counter, :] = projection_direction*(Fk + Fd)
                # compute moment
                self.w_Mg_patch_l[patch_counter, :] = N.dot(np.cross(patch_pos - pose[:3],self.w_Fg_patch_l[patch_counter, :] ))
            else:
                self.w_Fg_patch_l[patch_counter, :] = np.array([0.0, 0.0, 0.0])
                self.w_Mg_patch_l[patch_counter, :] = np.array([0.0, 0.0, 0.0])
            if self.enable_visuals:
                self.ros_pub.add_arrow(patch_pos, self.w_Fg_patch_l[patch_counter, :] / 50.,
                                       color=np.array([1.-self.patch_stiffness_l[patch_counter]/self.max_patch_stiffness_l, 0., self.patch_stiffness_l[patch_counter]/self.max_patch_stiffness_l])) #low stiffness is red
            patch_counter += 1

        # compute ditributed forces for RIGHT track
        patch_counter = 0
        for patch_pos, terrain_pos, patch_vel in zip(self.w_patch_pos_r, self.intersection_points_r, self.w_patch_vel_r):
            rho = projection_direction.dot(terrain_pos - patch_pos)
            if (rho > 0.0 and np.all(np.isfinite(terrain_pos))):
                b_robot_vel_x = w_R_b.T.dot(twist[:3])[0]
                b_patch_pos_x = w_R_b.T.dot(self.w_patch_pos_r[patch_counter, :] - pose[:3])[0]  # we do not care about the different Z components cause we take the X
                #change patch stiffness according to speed and pos of patch
                self.patch_stiffness_r[patch_counter] = self.ground.Kt_b * (1 + b_robot_vel_x *  self.ground.Kt_p * (b_patch_pos_x + np.sign(b_robot_vel_x) * self.track_param.length / 2) )

                Fk = (self.patch_stiffness_r[patch_counter] * rho)* track_area
                # Fk = self.ground.Kt_x.dot(terrain_pos - patch_pos)*track_area
                # only if it is approaching
                # Fd = np.zeros(3)
                rho_dot = projection_direction.dot(patch_vel)
                if (rho_dot < 0.0):
                    Fd = (-self.ground.Dt_b * rho_dot)* track_area
                else:
                    Fd = 0.  # np.zeros(3)
                self.w_Fg_patch_r[patch_counter, :] = projection_direction * (Fk + Fd)
                # compute moment
                self.w_Mg_patch_r[patch_counter, :] = N.dot(np.cross(patch_pos - pose[:3], self.w_Fg_patch_r[patch_counter, :]))
            else:
                self.w_Fg_patch_r[patch_counter, :] = np.array([0.0, 0.0, 0.0])
                self.w_Mg_patch_r[patch_counter, :] = np.array([0.0, 0.0, 0.0])



            if self.enable_visuals:
                self.ros_pub.add_arrow(patch_pos, self.w_Fg_patch_r[patch_counter, :] / 50.,
                                       color=np.array([1.-self.patch_stiffness_r[patch_counter]/self.max_patch_stiffness_r, 0., self.patch_stiffness_r[patch_counter]/self.max_patch_stiffness_r])) #low stiffness is red
            patch_counter += 1

        self.max_patch_stiffness_l = np.max(self.patch_stiffness_l)
        self.max_patch_stiffness_r = np.max(self.patch_stiffness_r)
        #to ckeck variation of stiffness wrt to track length
        # print(np.min(self.patch_stiffness_r))
        # print(np.max(self.patch_stiffness_r))


        # compute resultant of forces on tracks
        Fg_l_z = 0
        Fg_r_z = 0
        resultant_pos_l = np.zeros(3)
        resultant_pos_r = np.zeros(3)
        for patch_pos_l, fg_l, patch_pos_r, fg_r, in zip(self.w_patch_pos_l, self.w_Fg_patch_l, self.w_patch_pos_r, self.w_Fg_patch_r):
            resultant_pos_l[0] += patch_pos_l[0]*fg_l[2]
            resultant_pos_l[1] += patch_pos_l[1]*fg_l[2]
            resultant_pos_l[2] += patch_pos_l[2] * fg_l[2]
            Fg_l_z += fg_l[2]
            resultant_pos_r[0] += patch_pos_r[0] * fg_r[2]
            resultant_pos_r[1] += patch_pos_r[1] * fg_r[2]
            resultant_pos_r[2] += patch_pos_r[2] * fg_r[2]
            Fg_r_z += fg_r[2]
        resultant_pos_l/=Fg_l_z
        resultant_pos_r/= Fg_r_z
        if self.DEBUG:
            self.ros_pub.add_arrow(resultant_pos_l, np.array([0., 0., Fg_l_z]) / 1000., "red")
            self.ros_pub.add_arrow(resultant_pos_r, np.array([0., 0., Fg_r_z]) / 1000., "red")

        #integrate along all patches
        w_Fg = np.sum(self.w_Fg_patch_l, axis=0) + np.sum(self.w_Fg_patch_r, axis=0)

        w_Mg = np.sum(self.w_Mg_patch_l, axis=0) + np.sum(self.w_Mg_patch_r, axis=0)

        #map to base frame
        b_Fg = w_R_b.T.dot(w_Fg)
        b_Mg = w_R_b.T.dot(w_Mg)

        return b_Fg, b_Mg, 0, 0, self.w_Fg_patch_l, self.w_Fg_patch_r

    def dynamics3D(self, twist, w_R_b, b_Fg,b_Mg, Fx_l, Fy_l, M_long_l, M_lat_l, Fx_r, Fy_r, M_long_r, M_lat_r, vehicle_param):
        # calculate the acceleration with the vehicle dynamical model
        m = vehicle_param.mass
        bI = vehicle_param.bI

        #map everything to BF
        b_vc = w_R_b.T.dot(twist[:3])
        b_omega = w_R_b.T.dot(twist[3:])
        #print(b_vc)
        #compute traction forces and moments
        b_Ft = np.zeros(3)
        b_Ft[0] = Fx_l + Fx_r
        b_Ft[1] = Fy_l + Fy_r
        b_Mt = np.zeros(3)
        b_Mt[2] = M_long_l + M_lat_l + M_long_r + M_lat_r
        #compute gravity force
        b_Fgrav = w_R_b.T.dot(m*np.array([0.,0.,-self.ground.g]))
        #compute non linear terms
        NL_lin = m*(np.cross(b_omega,b_vc))
        NL_ang = np.cross(b_omega,  bI.dot(b_omega))

        #debug
        w_Ft = w_R_b.dot(b_Ft) #traction force
        w_Fg = w_R_b.dot(b_Fg) # terrain linear
        w_Mg = w_R_b.dot(b_Mg) # terrain moment

        if self.enable_visuals and self.DEBUG:
            # traction force
            self.ros_pub.add_arrow(self.pose[:3], w_Ft / 100., "red")
            #gravity force
            #w_Fgrav = w_R_b.dot(b_Fgrav)  # gravity force
            #self.ros_pub.add_arrow(self.pose[:3], w_Fgrav / 1000., "blue")
            #N = np.eye(3) - np.multiply.outer(self.terrain_normal.ravel(), self.terrain_normal.ravel())
            #self.ros_pub.add_arrow(self.pose[:3], N.dot(w_Fgrav) / 100., "blue")
            #ground normal force
            self.ros_pub.add_arrow(self.pose[:3], w_Fg / 1000., "green")
            #ground moment
            self.ros_pub.add_arrow(self.pose[:3], w_Mg / 100., "blue") #should be purely lateral flipping back forth on ramp with only pitch

        #rolling friction (not dependent on load/inclination) acts only on x direction
        b_R = np.array([-m*self.ground.g*self.track_param.c*np.sign(b_vc[0]), 0., 0.])

        b_vc_dot =  1/m*(b_Ft + b_R + b_Fgrav + b_Fg-NL_lin)
        b_omega_dot = np.linalg.inv(bI).dot(b_Mt + b_Mg-NL_ang)

        w_twist_dot = np.concatenate((w_R_b.dot(b_vc_dot), w_R_b.dot(b_omega_dot)))
        # print("b_Ft",b_Ft)
        # print("b_Fgrav",b_Fgrav)
        #print("b_Mg", b_Mg)
        return w_twist_dot

    def integrateTwist(self, pose, twist):
        rpy = pose[3:]
        # TOD replace with RK4
        pose[:3] += twist[:3] * self.dt
        rpy_dot = np.linalg.inv(self.math_utils.Tomega(rpy)).dot(twist[3:])
        pose[3:] +=rpy_dot* self.dt
        return pose

    def initSimulation(self,pose_init =np.zeros(6),  twist_init=np.zeros(6), ros_pub = None):
        self.pose = pose_init
        self.twist = twist_init
        if ros_pub is None:
            self.ros_pub = RosPub('tractor', only_visual=True)
        else:
            self.ros_pub =ros_pub

    def simulateOneStep(self,pg, terrain_roll, terrain_pitch, terrain_yaw, omega_left, omega_right, F_lg=None, F_rg=None):
        #compute terrain variables
        self.w_R_terr = self.math_utils.eul2Rot(np.array([terrain_roll, terrain_pitch, terrain_yaw]))
        self.terrain_normal = self.w_R_terr[:, 2]

        # compute base orientation
        w_R_b = self.math_utils.eul2Rot(self.pose[3:])
        #get states, they are the velocities in base frame b_v_c and the yaw rate
        b_v_c = w_R_b.T.dot(self.twist[:3])
        #project twist_ang_z onto w_z_b
        b_omega = w_R_b[2].dot(np.array([0.,0., self.twist[5]]))
        state2D = np.array([b_v_c[0],b_v_c[1], b_omega])
        self.w_com_height_vector = w_R_b[:, 2] * self.vehicle_param.height

        # compute ground peneration
        if self.CONTACT_DISTRIBUTION:
            b_Fg, b_Mg, b_eox, b_eoy, w_Fg_patch_l, w_Fg_patch_r = self.computeDistributedGroundForcesMoments(self.pose, self.twist, w_R_b,  terrain_roll, terrain_pitch, terrain_yaw)
            patch_counter = 0
            number_of_patches = self.track_param.parts_lateral*self.track_param.parts_longitudinal
            patch_area = self.track_param.A / number_of_patches

            # compute patch positions and twists in world frame
            for i in range(self.track_param.parts_longitudinal):
                for j in range(self.track_param.parts_lateral):
                    self.sigma_l[i,j] = w_R_b.T.dot(w_Fg_patch_l[patch_counter,:])[2]/patch_area
                    self.sigma_r[i, j] = w_R_b.T.dot(w_Fg_patch_r[patch_counter, :])[2] / patch_area
                    patch_counter+=1
        else:
            b_Fg, b_Mg, b_eox, b_eoy  = self.computeGroundForcesMoments(self.pose, self.twist, w_R_b, pg, terrain_roll, terrain_pitch, terrain_yaw)


        Fx_l, Fy_l, M_long_l, M_lat_l = self.tracked_robot.track_left.computeTerrainInteractions(state2D, omega_left, self.track_param,
                                                                                   self.sigma_l, self.ground, self.patch_pos_long_l, self.patch_pos_lat_l)
        Fx_r, Fy_r, M_long_r, M_lat_r= self.tracked_robot.track_right.computeTerrainInteractions(state2D, omega_right, self.track_param,
                                                                                   self.sigma_r, self.ground, self.patch_pos_long_r, self.patch_pos_lat_r)

        if F_lg is not None:
            Fx_l+=F_lg
        if F_rg is not None:
            Fx_r += F_rg

        if self.enable_visuals:
            # terrain normal
            self.ros_pub.add_arrow(self.pose[:3], self.terrain_normal * 0.5, color="white")
            # position of com
            self.ros_pub.add_marker(self.pose[:3], radius=0.1, color="white", alpha=1.)
            # position of com projection on ground
            self.ros_pub.add_marker(pg, radius=0.1, color="white", alpha=1.)


        if self.NO_SLIPPAGE:
            #extension of unicycle to 3d
            vel = constants.SPROCKET_RADIUS * (omega_left + omega_right) / 2
            omega = constants.SPROCKET_RADIUS / constants.TRACK_WIDTH * (omega_right - omega_left)
            self.twist[:3] = w_R_b.dot(np.array([vel,0,0]))
            self.twist[3:] = w_R_b.dot(np.array([0, 0, omega]))
        else:
            #update twist from dynamics
            if self.int_method=='FORWARD_EULER':
                self.twist = forward_euler_step(self.dynamics3D, y=self.twist, h=self.dt,  w_R_b=w_R_b, b_Fg=b_Fg, b_Mg=b_Mg,
                                  Fx_l=Fx_l, Fy_l=Fy_l, M_long_l=M_long_l, M_lat_l=M_lat_l, Fx_r=Fx_r, Fy_r=Fy_r,
                                  M_long_r=M_long_r, M_lat_r=M_lat_r, vehicle_param=self.vehicle_param)
            elif self.int_method=='HEUN':
                self.twist = heun_step(self.dynamics3D, y=self.twist, h=self.dt,  w_R_b=w_R_b, b_Fg=b_Fg, b_Mg=b_Mg,
                                  Fx_l=Fx_l, Fy_l=Fy_l, M_long_l=M_long_l, M_lat_l=M_lat_l, Fx_r=Fx_r, Fy_r=Fy_r,
                                  M_long_r=M_long_r, M_lat_r=M_lat_r, vehicle_param=self.vehicle_param)
            elif self.int_method=='RK4':
                self.twist = RK4_step(self.dynamics3D, y=self.twist, h=self.dt,  w_R_b=w_R_b, b_Fg=b_Fg, b_Mg=b_Mg,
                                  Fx_l=Fx_l, Fy_l=Fy_l, M_long_l=M_long_l, M_lat_l=M_lat_l, Fx_r=Fx_r, Fy_r=Fy_r,
                                  M_long_r=M_long_r, M_lat_r=M_lat_r, vehicle_param=self.vehicle_param)
            elif self.int_method=='BACKWARD_EULER':
                self.twist = backward_euler_step(self.dynamics3D, y=self.twist, h=self.dt,  w_R_b=w_R_b, b_Fg=b_Fg, b_Mg=b_Mg,
                                  Fx_l=Fx_l, Fy_l=Fy_l, M_long_l=M_long_l, M_lat_l=M_lat_l, Fx_r=Fx_r, Fy_r=Fy_r,
                                  M_long_r=M_long_r, M_lat_r=M_lat_r, vehicle_param=self.vehicle_param)
            elif self.int_method=='RK45F':
                self.twist, self.dt = self.rk45F.step(x_k=self.twist, d_t=self.dt, w_R_b=w_R_b, b_Fg=b_Fg, b_Mg=b_Mg,
                                  Fx_l=Fx_l, Fy_l=Fy_l, M_long_l=M_long_l, M_lat_l=M_lat_l, Fx_r=Fx_r, Fy_r=Fy_r,
                                  M_long_r=M_long_r, M_lat_r=M_lat_r, vehicle_param=self.vehicle_param)

            else:
                print("wrong integration method")
        self.pose  = self.integrateTwist(self.pose, self.twist)
        return b_eox, b_eoy

    def computeZcomponent(self, x, y, pitch):
        #important x should start from 0!
        return x * np.tan(-pitch)


    def simulate(self, hf_v, w_omega_z):
        self.time = 0.
        sim_counter = 0
        self.rate = ros.Rate(1 /  (self.dt))

        if self.PROFILE:
            from base_controllers.utils.profiler import Profiler
            profiler = Profiler(function_name=self.simulateOneStep)

        while self.time < self.t_end:
            if np.mod(self.time, 1) == 0:
                print(colored(f"TIME: {self.time}","red"))

            #get actual orient
            self.euler = self.pose[3:]
            self.w_R_b = self.math_utils.eul2Rot(self.euler)
            self.hf_R_b = self.math_utils.eul2Rot(np.array([self.euler[0],self.euler[1], 0.]))
            # get des pos
            des_x, des_y, des_theta, _, _, _, _, _ = self.traj.evalTraj(self.time)

            if self.USE_MESH:
                #base_x_axis = self.math_utils.eul2Rot(self.pose[3:])[:, 0]
                w_com_height_vector = self.w_R_b[:, 2] * self.vehicle_param.height
                pcom_on_ground = self.pose[:3] - self.consider_robot_height * w_com_height_vector
                # terrain yaw is determined by the robot orientation!
                terrain_yaw = self.pose[5]
                pg, terrain_roll, terrain_pitch = self.terrain_manager.project_on_mesh(point=pcom_on_ground[:2],  direction=np.array([0., 0., 1.]), base_yaw=terrain_yaw)
                self.pose_des,terrain_roll_des,terrain_pitch_des = self.terrain_manager.project_on_mesh(point=np.array([des_x, des_y]), direction=np.array([0., 0., 1.]), base_yaw=terrain_yaw)
                self.orient_des = np.array([0, 0, des_theta]) #first two elements are the expected values of b_eox, b_eoy
                # print("pose ", self.pose[:3])
                # print("pg ", pg)
                #for debug
                # print("pose ", self.pose[:3])
                # print("pg ", pg)
                if self.TYPE_OF_TERRAIN=='slopes':
                    self.ros_pub.add_mesh("tractor_description", "/meshes/terrain.stl", position=np.array([0., 0., 0.0]), color="red", alpha=1.0)
                elif self.TYPE_OF_TERRAIN=='ramp':
                    self.ros_pub.add_plane(pos=np.array([0, 0, -0.]), orient=np.array([0., self.RAMP_INCLINATION, 0]), color="white", alpha=0.8)
                else:
                    print("type of terrain not supported")

            else: #this is only used for the unit test and has the strong assumptio of being a pitch ramp followed with fwd motion
                terrain_roll = self.terrain_roll_vec[sim_counter]
                terrain_pitch = self.terrain_pitch_vec[sim_counter]
                terrain_yaw =  self.terrain_yaw_vec[sim_counter]
                w_com_height_vector = self.w_R_b[:, 2] * self.vehicle_param.height
                pcom_on_ground = self.pose[:3] - self.consider_robot_height*w_com_height_vector
                pg = np.array([pcom_on_ground[0], pcom_on_ground[1], self.computeZcomponent(pcom_on_ground[0], pcom_on_ground[1], terrain_pitch)])
                self.pose_des = np.array([des_x, des_y, self.computeZcomponent(des_x, des_y, terrain_pitch)])
                self.orient_des = np.array([0, 0, des_theta])

            # we use only velocity inputs because we are open loop, the des_x, des_y, des_theta are only for plotting
            #project hf_v onto hf_x_b
            b_v = self.hf_R_b[0].dot(np.array([hf_v[sim_counter], 0., 0.]))
            #project w_omega_z onto w_z_b
            b_omega = self.w_R_b[2].dot(np.array([0.,0., w_omega_z[sim_counter]]))

            # convert to wheels
            r = self.track_param.sprocket_radius
            B = self.vehicle_param.width
            omega_left = (b_v - b_omega * B / 2.0) / r
            omega_right = (b_v + b_omega * B / 2.0) / r

            #updates pose and twist
            b_eox, b_eoy = self.simulateOneStep(pg, terrain_roll, terrain_pitch, terrain_yaw, omega_left, omega_right)

            #log
            #robot_height is naturally incorporated in pose because is added at the beginning i pose_init, in the desired I should account for robot height mapped in WF
            self.pose_des_log[:3, sim_counter] = self.pose_des + self.consider_robot_height*self.w_com_height_vector
            self.pose_des_log[3:, sim_counter] = self.orient_des
            self.pose_log[:, sim_counter] = self.pose

            self.time_log[sim_counter] = self.time
            self.max_patch_stiffness_log[sim_counter] = self.max_patch_stiffness_l

            self.time = np.round(self.time + self.dt, 4)
            sim_counter +=1

            self.euler = self.pose[3:]
            self.quaternion = pin.Quaternion(self.w_R_b)
            #publish tf of base link
            self.broadcaster.sendTransform(self.pose[:3],
                                           self.quaternion,
                                           ros.Time.now(), '/base_link', '/world')
            if self.enable_visuals:
                # publish tf of horizontal frame on terrain to understand inclination
                self.broadcaster.sendTransform(pg,
                                               pin.Quaternion(self.w_R_b.dot(self.hf_R_b.T)),
                                               ros.Time.now(), '/horizontal_frame', '/world')

            self.ros_pub.publishVisual(delete_markers=False)

            self.rate.sleep()

            # check frequency of publishing
            # if self.USE_MESH: #not doing for unit test
            #     if hasattr(self, 'check_time'):
            #         loop_time = ros.Time.now().to_sec() - self.check_time  # actual publishing time interval
            #         ros_loop_time = self.slow_down_factor * self.dt  # ideal publishing time interval
            #         if loop_time > 1.20 * (ros_loop_time):
            #             loop_real_freq = 1 / loop_time  # actual publishing frequency
            #             freq_ros = 1 / ros_loop_time  # ideal publishing frequency
            #             print(colored(f"freq mismatch beyond 20%: loop is running at {loop_real_freq} Hz while it should run at {freq_ros} Hz, freq error is {(freq_ros - loop_real_freq) / freq_ros * 100} %", "red"))
            #             self.out_of_frequency_counter += 1
            #             if self.out_of_frequency_counter > 10:
            #                 original_slow_down_factor = self.slow_down_factor
            #                 self.slow_down_factor *= 2
            #                 self.rate = ros.Rate(1 / (self.slow_down_factor * self.dt))
            #                 print(colored(f"increasing slow_down_factor from {original_slow_down_factor} to {self.slow_down_factor}", "red"))
            #                 self.out_of_frequency_counter = 0
            #                 self.slowing_down_time = ros.Time.now().to_sec()
            #             if hasattr(self, 'slowing_down_time'):
            #                 time_from_last_slow_down = ros.Time.now().to_sec() - self.slowing_down_time
            #                 if time_from_last_slow_down > 2.:
            #                     self.slow_down_factor /= 2
            #     self.check_time = ros.Time.now().to_sec()


            if ros.is_shutdown():
                break

        if self.PROFILE:
            avg_time = profiler.get_total_time() / sim_counter
            print(colored(f"Average Time per Call: {avg_time:.6f} seconds", "red"))
        os.system("rosnode kill rviz")
        os.system("pkill rosmaster")

    def getRobotState(self):
        return self.pose, self.twist

    def setRampTerrain(self, ramp_inclination):
        self.TYPE_OF_TERRAIN = 'ramp'
        self.RAMP_INCLINATION = ramp_inclination
        ramp_mesh = create_ramp_mesh(length=350., width=350., inclination=p.RAMP_INCLINATION, origin=np.array([0, 0, 0]))
        self.terrain_manager.set_mesh(ramp_mesh)


if __name__ == '__main__':
    #unit test (very slippery friction is 0.1!)
    test = "unit_test"
    groundParams = Ground3D(friction_coefficient=0.1)
    p = TrackedVehicleSimulator3D(dt=0.001, ground=groundParams, USE_MESH=False, DEBUG=True, int_method='FORWARD_EULER')
    hf_v = np.linspace(0.4, 0.4, p.number_of_steps)
    w_omega_z = np.linspace(0.0, 0.0, p.number_of_steps)
    p.initial_ramp_inclination = 0 # pitch ramp
    p.final_ramp_inclination = -0.1

    #test = "normal_test"
    # groundParams = Ground3D(friction_coefficient=0.6, terrain_stiffness=1e05, terrain_damping=0.5e04)
    # p = TrackedVehicleSimulator3D(dt=0.001, ground=groundParams, USE_MESH=True, DEBUG=True, int_method='FORWARD_EULER', enable_visuals=True,contact_distribution=True)

    # PAPER: sloped test: distributed/non distributed
    # test = "sloped_test_chicane"
    # groundParams = Ground3D(friction_coefficient=0.7, terrain_stiffness=1e04, terrain_damping=0.1e04)
    # p = TrackedVehicleSimulator3D(dt=0.001, ground=groundParams, USE_MESH=True, DEBUG=True, int_method='FORWARD_EULER', enable_visuals=False, contact_distribution=False)
    # p.initLoggingVars(simulation_duration=20., pose_init=[19., 3., 0.0, 0, 0, 0]) # position in WF, rpy angles
    # hf_v = np.linspace(0.8, 0.8, p.number_of_steps)
    # w_omega_z = np.linspace(0.6, -0.6, p.number_of_steps)
    # p.SAVE_LOGS = True

    # PAPER: Stiffness variation
    # # to test the stiffness variation of the terrain we need a softer terrain and we need a flat location and high linear speed
    # test = "stiff_var_test"
    # groundParams = Ground3D(friction_coefficient=0.7, terrain_stiffness=0.5e04, terrain_damping=0.5e04)
    # p = TrackedVehicleSimulator3D(dt=0.001, ground=groundParams, USE_MESH=True, DEBUG=True, int_method='FORWARD_EULER', enable_visuals=False,contact_distribution=True)
    # p.initLoggingVars(simulation_duration=10., pose_init=[5., -5., 0.0, 0, 0, 0])
    # hf_v = np.linspace(0.01, 1.4, p.number_of_steps)
    # w_omega_z = np.linspace(0,0, p.number_of_steps)
    # p.SAVE_LOGS = True

    # PAPER: pitch ramp: distributed (not used)
    # test = "ramp_test_turning_left"
    # groundParams = Ground3D(friction_coefficient=0.7, terrain_stiffness=1e04, terrain_damping=0.1e04) #good setting for distributed contact
    # p = TrackedVehicleSimulator3D(dt=0.001, ground=groundParams, USE_MESH=True, DEBUG=True, int_method='FORWARD_EULER', enable_visuals=False, contact_distribution=True)
    # p.setRampTerrain(0.25)
    # p.initLoggingVars(simulation_duration=10.)
    # hf_v = np.linspace(0.4, 0.4, p.number_of_steps)
    # w_omega_z = np.linspace(0.4, 0.4, p.number_of_steps)
    # p.SAVE_LOGS = True

    #for rviz
    launchFileGeneric(rospkg.RosPack().get_path('tractor_description') + "/launch/rviz_nojoints.launch")
    p.broadcaster = tf.TransformBroadcaster()

    if p.USE_MESH:
        #start the robot on the mesh
        start_yaw = p.pose_init[5]
        start_position, start_roll, start_pitch = p.terrain_manager.project_on_mesh(point=p.pose_init[:2], direction=np.array([0., 0., 1.]), base_yaw=start_yaw)
        p.pose_init[:3] = start_position.copy()
        p.pose_init[3] = start_roll
        p.pose_init[4] = start_pitch
        p.pose_init[5] = start_yaw
        # init com self.vehicle_param.height above ground
        w_R_terr = p.math_utils.eul2Rot(np.array([start_roll, start_pitch, start_yaw]))
        p.pose_init[:3] += p.consider_robot_height*(w_R_terr[:, 2] * p.vehicle_param.height)
    else:#unit test
        p.terrain_roll_vec = np.linspace(0.0, 0.0, p.number_of_steps)
        p.terrain_pitch_vec = np.linspace(p.initial_ramp_inclination, p.final_ramp_inclination, p.number_of_steps)  # 0.1 if you set -0.3 the robot starts to slip backwards!
        p.terrain_yaw_vec = np.linspace(0.0, -0.0, p.number_of_steps)
        p.pose_init[:3] += p.consider_robot_height * np.array([0,0,1])* p.vehicle_param.height

    #PAPER: lateral ramp test
    #p.pose_init[5] = 1.57

    #the trajectory is in the WF
    p.traj = Trajectory(ModelsList.UNICYCLE, start_x=p.pose_init[0],
                        start_y= p.pose_init[1], start_theta=p.pose_init[5],
                        DT=p.dt, v=hf_v, omega=w_omega_z)

    # init vars
    p.initSimulation(p.pose_init, p.twist_init)
    if p.SAVE_LOGS:
        test_name =f"openLoop3DModel_{test}_Distr_{p.CONTACT_DISTRIBUTION}"
        p.recorder = RosbagControlledRecorder(bag_name=test_name+".bag")
        p.recorder.start_recording_srv()
    # simulation with internal while loop
    p.simulate(hf_v, w_omega_z)
    if p.SAVE_LOGS:
        p.recorder.stop_recording_srv()
        import scipy.io.matlab as mio
        mio.savemat(test_name+".mat", {'time': p.time_log, 'pose_des': p.pose_des_log, 'pose': p.pose_log, 'vdes':hf_v, 'Kmax': p.max_patch_stiffness_log})

    if not p.USE_MESH:#unit test
        errors = []
        #print(p.pose_log[:,-1])

        p.unit_test.check_assertion(p.pose_log[0, -1], 7.758488675688657, decimal=2, description="Pose X")
        p.unit_test.check_assertion(p.pose_log[1, -1],0. , decimal=2, description="Pose Y")
        if p.consider_robot_height:
            p.unit_test.check_assertion(p.pose_log[2, -1],     1.02914053317055, decimal=2, description="Pose Z")
        else:
            p.unit_test.check_assertion(p.pose_log[2, -1],    0.648124971, decimal=2,description="Pose Z")
        p.unit_test.check_assertion(p.pose_log[3, -1], 0., decimal=2, description="Pose Roll")
        p.unit_test.check_assertion(p.pose_log[4, -1], -0.099504975 , decimal=2,description="Pose Pitch")
        p.unit_test.check_assertion(p.pose_log[5, -1], 0., decimal=2, description="Pose Yaw")
        p.unit_test.report()

    # xy plot
    plt.figure()
    plt.plot(p.pose_log[0, :-1], p.pose_log[1, :-1], "-b", label="actual")
    plt.plot(p.pose_des_log[0, :-1], p.pose_des_log[1, :-1], "-r", label="desired")
    plt.legend()
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.axis("equal")
    plt.grid(True)

    fig = plt.figure()
    fig.suptitle("lin states", fontsize=20)
    plt.subplot(3, 1, 1)
    plt.ylabel("X")
    plt.plot(p.time_log[:-1], p.pose_log[0, :-1], linestyle='-',  lw=3, color='blue')
    plt.plot(p.time_log[:-1], p.pose_des_log[0, :-1], linestyle='-', lw=3, color='red')
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.ylabel("Y")
    plt.plot(p.time_log[:-1], p.pose_log[1, :-1], linestyle='-',  lw=3, color='blue')
    plt.plot(p.time_log[:-1], p.pose_des_log[1, :-1], linestyle='-', lw=3, color='red')
    plt.grid()
    plt.ylim([-10, 10])
    plt.subplot(3, 1, 3)
    plt.ylabel("Z")
    plt.plot(p.time_log[:-1], p.pose_log[2, :-1], linestyle='-',  lw=3, color='blue')
    plt.plot(p.time_log[:-1], p.pose_des_log[2, :-1], linestyle='-', lw=3, color='red')
    plt.xlabel("Time [s]")
    plt.grid()

    #TODO remove yaw and yaw_des such that we can compare pitch/roll
    fig = plt.figure()
    fig.suptitle("ang states", fontsize=20)
    plt.subplot(3, 1, 1)
    plt.ylabel("roll")
    plt.plot(p.time_log[:-1], p.pose_log[3, :-1], linestyle='-',  lw=3, color='blue')
    #plt.plot(p.time_log[:-1], p.pose_des_log[3, :-1], linestyle='-', lw=3, color='red')
    plt.grid()
    plt.ylim([-0.5,0.5])
    plt.subplot(3, 1, 2)
    plt.ylabel("pitch")
    plt.plot(p.time_log[:-1], p.pose_log[4, :-1], linestyle='-',  lw=3, color='blue')
    #plt.plot(p.time_log[:-1], p.pose_des_log[4, :-1], linestyle='-', lw=3, color='red')
    plt.grid()
    plt.ylim([-0.5,0.5])
    plt.subplot(3, 1, 3)
    plt.ylabel("yaw")
    plt.plot(p.time_log[:-1], p.pose_log[5, :-1], linestyle='-',  lw=3, color='blue')
    plt.plot(p.time_log[:-1], p.pose_des_log[5, :-1], linestyle='-', lw=3, color='red')
    plt.xlabel("Time [s]")
    plt.grid()
    plt.ylim([-5,5])
