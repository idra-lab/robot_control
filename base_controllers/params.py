# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np

robot_params = {}

robot_params['myrobot'] ={'dt': 0.001,
                        'kp': np.array([10.,   10.,    10.,  10.]),
                        'kd':  np.array([1.,    1.,    1.,   1.  ]),
                        'q_0':  np.array([0, 0, 0, 0]),
                        'joint_names': ['lf_shoulder_pan', 'lh_shoulder_pan',  'rf_shoulder_pan', 'rh_shoulder_pan'],
                        'ee_frames': ['lf_foot', 'lh_foot', 'rf_foot','rh_foot'],
                        'spawn_x': 0.0,
                        'spawn_y': 0.0,
                        'spawn_z': 1.0,
                        'buffer_size': 30001}

robot_params['ur5'] ={'dt': 0.001,
                       'kp': np.array([300, 300, 300,30,30,1]),
                       'kd':  np.array([20,20,20,5, 5,0.5]),
                       'ki':  np.array([.0,.0,.0,.0, .0,.0]),
                       'q_0':  np.array([ -0.32,-0.78, -2.56,-1.63, -1.57, 3.49]),
                       'joint_names': ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
                       'ee_frame': 'tool0',
                       'control_mode': 'point', # 'trajectory','point'
                       'real_robot': False,
                       'control_type': 'position', # 'position', 'torque'
                       'gripper_sim': False, # False: the gripper is treated as a Rigid Body
                       'gripper_type': 'soft_2', # 3 finger rigid gripper: 'hard',  2 finger soft gripper: 'soft_2',
                                                    # robotiq gripper: 'robotiq_2' (does not work in torque mode!)
                       'spawn_x' : 0.5,
                       'spawn_y' : 0.35,
                       'spawn_z' : 1.75,
                       'buffer_size': 5000}



robot_params['z1'] ={'dt': 0.001,
                        'kp': np.array([20.,   20.,    20.,  20., 20.,  20.]),
                        'kd':  np.array([0.5,    0.5,    0.5,   0.5, 0.5, 0.5]),
                        'q_0':  np.array([0, 0.5, -1, 0, 0, 0]),
                        'joint_names': ['joint1', 'joint2',  'joint3', 'joint4', 'joint5', 'joint6'],
                        'ee_frame': ['tool0'], #TODO rename ee_frame to ee_frames
                        'spawn_x': 0.0,
                        'spawn_y': 0.0,
                        'spawn_z': 0.0,
                        'buffer_size': 30001}

robot_params['tractor'] ={'dt': 0.01,
                        'kp': np.array([100.,   100.,    100.,  100.]),
                        'kd':  np.array([10.,    10.,    10.,   10.  ]),
                        'q_0':  np.array([0, 0, 0, 0]),
                        'joint_names': ['front_left_wheel_joint', 'front_right_wheel_joint'], # caster wheels are passive joints
                        'ee_frames': ['front_left_wheel', 'front_right_wheel'],
                        'spawn_x': -0.,
                        'spawn_y': 0.0,
                        'spawn_z': 1.5,
                        'buffer_size': 7000}

robot_params['limo0'] ={'dt': 0.01,
                        'kp': np.array([100.,   100.,    100.,  100.]),
                        'kd':  np.array([10.,    10.,    10.,   10.  ]),
                        'q_0':  np.array([0, 0, 0, 0]),
                        'joint_names': ['limo0/front_left_wheel', 'limo0/front_right_wheel', 'limo0/rear_left_wheel','limo0/rear_right_wheel',],
                        'ee_frames': ['limo0/front_left_wheel_link', 'limo0/front_right_wheel_link'],
                        'spawn_x': -0.,
                        'spawn_y': 0.0,
                        'spawn_z': 0.5,
                        'spawn_Y': 0.,
                        'real_robot': False,
                        'buffer_size': 15000}



robot_params['hyq'] = {'dt': 0.004,
                        'kp': np.array([400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400]),
                        'kd': np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]),
                        'q_0':  np.array([-0.2, 0.7, -1.4,  -0.2, -0.7, 1.4, -0.2, 0.7, -1.4, -0.2, -0.7, 1.4]),
                        'kp_wbc': np.array([15., 15., 15.]*4),#np.array([10., 10., 10.]*4),
                        'kd_wbc': np.array([1., 1., 1.]*4),#np.array([1., 1., 1.]*4),
                        'ki_wbc': np.array([0., 0., 0.]*4),#np.array([0.3, 0.3, 0.3]*4),
                        # virtual impedance wrench control
                        'kp_lin': np.array([2000., 2000., 2000.]),  # x y z
                        'kd_lin': np.array([200., 200., 200.]),
                        'kp_ang': np.array([1000., 1000., 1000.]),  # R P Y
                        'kd_ang': np.array([100., 100., 100.]),
                        'joint_names': ['lf_haa_joint', 'lf_hfe_joint', 'lf_kfe_joint',
                                        'lh_haa_joint', 'lh_hfe_joint', 'lh_kfe_joint',
                                        'rf_haa_joint', 'rf_hfe_joint', 'rf_kfe_joint',
                                        'rh_haa_joint', 'rh_hfe_joint', 'rh_kfe_joint'],
                        'ee_frames': ['lf_foot', 'lh_foot', 'rf_foot','rh_foot'],
                        'force_th': 50,
                        'spawn_x': 0.0,
                        'spawn_y': 0.0,
                        'spawn_z': 0.8,
                        'spawn_R': 0.0,
                        'spawn_P': 0.0,
                        'spawn_Y': np.pi/2,
                       'buffer_size': 30001}
robot_params['solo'] ={'dt': 0.002,
                       'kp': [5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.],
                       'kd': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                       'q_0':  np.array([0,  np.pi/4, -np.pi/2,    0, -np.pi/4,  np.pi/2, -0,  np.pi/4, -np.pi/2, -0, -np.pi/4,  np.pi/2]),
                       'q_fold': np.array([0,  1.57, -3.13, 0,    1.57, -3.13, 0, -1.57, 3.13, 0, -1.57, 3.13]),
                       'joint_names': ['lf_haa_joint', 'lf_hfe_joint', 'lf_kfe_joint',
                                       'lh_haa_joint', 'lh_hfe_joint', 'lh_kfe_joint',
                                       'rf_haa_joint', 'rf_hfe_joint', 'rf_kfe_joint',
                                       'rh_haa_joint', 'rh_hfe_joint', 'rh_kfe_joint'],
                       'ee_frames': ['lf_foot', 'lh_foot', 'rf_foot','rh_foot'],
                       'real_robot': False,
                       'force_th': 2,
                       'spawn_x': 0.0,
                       'spawn_y': 0.0,
                       'spawn_z': 0.3,
                        'buffer_size': 1501} # note the frames are all aligned with base for joints = 0
robot_params['solo_fw'] ={'dt': 0.002,
                        'kp': [5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 0, 0],
                        'kd': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0],
                        'joint_names':  ['lf_haa_joint', 'lf_hfe_joint', 'lf_kfe_joint',
                                         'lh_haa_joint', 'lh_hfe_joint', 'lh_kfe_joint',
                                         'rf_haa_joint', 'rf_hfe_joint', 'rf_kfe_joint',
                                         'rh_haa_joint', 'rh_hfe_joint', 'rh_kfe_joint',
                                         'z_left_wheel_joint', 'z_right_wheel_joint'],
                        'q_0':  np.array([0.,  np.pi/4, -np.pi/2,  0., -np.pi/4,  np.pi/2, -0.,  np.pi/4, -np.pi/2, 0., -np.pi/4,  np.pi/2, 0., 0.]),
                        'ee_frames': ['lf_foot', 'lh_foot', 'rf_foot','rh_foot'],
                        'real_robot': False,
                        'force_th': 4,
                        'spawn_x': 0.0,
                        'spawn_y': 0.0,
                        'spawn_z': 0.3,
                        'buffer_size': 1501} # note the frames are all aligned with base for joints = 0

robot_params['aliengo'] ={'dt': 0.002,
                          'kp': np.array([60., 90., 60.]*4),
                          'kd': np.array([10., 10., 10.]*4),
                          'ki': np.array([0., 0., 0.]*4),

                          'kp_nominal': np.array([21.5, 21.5, 21.5]*4),
                          'kd_nominal': np.array([3.5, 3.5, 3.5]*4),
                          'ki_nominal': np.array([0., 0., 0.]*4),
                          'kp_backup':  np.array([100., 100., 100.]*4),
                          'kd_backup':  np.array([3., 3., 3.]*4),
                          'ki_backup':  np.array([0., 0., 0.]*4),

                          # 'kp': np.array([25., 25., 25.]*4),
                          # 'kd': np.array([2., 2., 2.]*4),
                          # 'ki': np.array([0., 0., 0.]*4),

                          'kp_swing':  np.array([20., 30., 20.]*4),
                          'kd_swing':  np.array([1., 1., 1.]*4),
                          'ki_swing':  np.array([0., 0., 0.]*4),

                          'kp_land': np.array([60., 90., 60.]*4),
                          'kd_land': np.array([10., 10., 10.]*4),
                          'ki_land': np.array([0., 0., 0.]*4),

                          'kp_real': np.array([20., 30., 30.]*4),
                          'kd_real': np.array([1., 1., 1.]*4),
                          'ki_real': np.array([1., 1., 1.]*4),

                          'kp_real_swing':  np.array([15., 20., 20]*4),
                          'kd_real_swing':  np.array([1., 1., 1.]*4),
                          'ki_real_swing':  np.array([0., 0., 0.]*4),

                          'kp_real_land': np.array([15., 15., 15]*4),
                          'kd_real_land': np.array([1., 1., 1.]*4),
                          'ki_real_land': np.array([0., 0., 0.]*4),

                          # SIM ---------------------------

                          # joint pid + wbc (optional)
                          'kp_wbc': np.array([60., 60., 60.]*4),
                          'kd_wbc': np.array([10., 10., 10.]*4),
                          'ki_wbc': np.array([0., 0., 0.]*4),
                          # virtual impedance wrench control
                          'kp_lin': np.array([1000., 1000., 800.]),
                          'kd_lin': np.array([150., 150., 100.]),
                          'kp_ang': np.array([200., 200., 100.]),
                          'kd_ang': np.array([10., 20., 20.]),

                          # REAL---------------------------
                          # joint pid for landing contoller after td + wbc (optional)
                          'kp_wbc_real': np.array([10., 10., 10.]*4),
                          'kd_wbc_real': np.array([1., 1., 1.]*4),
                          'ki_wbc_real': np.array([0., 0., 0.]*4),

                        'kp_lin_real': np.array([1000., 1000., 1000.]),
                        'kd_lin_real': np.array([200., 150., 200.]),
                        'kp_ang_real': np.array([300., 300., 200.]),
                        'kd_ang_real': np.array([10., 20., 20.]),

                          #joint configuration
                          #default
                          #'q_0': np.array([0.2, 0.78, -1.7, 0.20, 0.78, -1.7, -0.20, 0.78, -1.7, -0.20, 0.78, -1.7]),  # this is to set the desired heigth for the startup phase
                          #orbit
                          'q_0': np.array([0.0951, 0.8303,-1.5419,
                                           0.0980, 0.9864, -1.4778,
                                           -0.0948, 0.8305,-1.5420,
                                           -0.0979,0.9864, -1.4779]),

                         'q_retraction': np.array([0.2867, 1.48, -2.6,
                                0.2867, 1.8, -2.48,
                                -0.2867, 1.48, -2.6,
                                -0.2867, 1.8, -2.48]),
                          # default
                          #'q_final': np.array([0.2, 1.1, -1.8, 0.2, 1.1, -1.8, -0.2, 1.1, -1.8, -0.2, 1.1, -1.8]),
                          # orbit
                          'q_final': np.array([0.1, 0.8, -1.6,  # lf
                                               0.1, 1.0, -1.6,  # lh
                                               -0.1, 0.8, -1.6,  # rf
                                               -0.1, 1.0, -1.6]),  # rh


                        #forward
                        'q_land_fwd':  np.array([0.1, 0.6, -1.8,  0.1, 0.6, -1.8, -0.1, 0.6, -1.8, -0.1, 0.6, -1.8]),
                        #backward
                        'q_land_bwd':  np.array([0.1, 1., -1.8,  0.1, 1., -1.8, -0.1, 1., -1.8, -0.1, 1., -1.8]),
                        'landing_duration': 1.4,
                        'com_z0_training' : 0.375,
                        'q_0_td': np.array([0.1789, 1.2234, -2.2329, 0.1867, 1.4733, -2.1055, -0.1784, 1.2230, -2.2327, -0.1861, 1.4733, -2.1053]),
                        'q_fold': np.array([0.2, 1.7, -2.7, 0.2,  1.7, -2.7, -0.2, 1.7,  -2.7, -0.2, 1.7, -2.7]),  #thjis is for the startup phase
                        'joint_names': ['lf_haa_joint', 'lf_hfe_joint', 'lf_kfe_joint',
                                       'lh_haa_joint', 'lh_hfe_joint', 'lh_kfe_joint',
                                       'rf_haa_joint', 'rf_hfe_joint', 'rf_kfe_joint',
                                       'rh_haa_joint', 'rh_hfe_joint', 'rh_kfe_joint'],
                        'ee_frames': ['lf_foot', 'lh_foot', 'rf_foot','rh_foot'],
                        'real_robot': False,
                        'force_th': 10.,
                        'spawn_x': 0.0,
                        'spawn_y': 0.0,
                        'spawn_z': 0.37,
                        'ip': "192.168.123.220",
                        'buffer_size': 50001} # note the frames are all aligned with base for joints = 0

robot_params['go1'] ={'dt': 0.002,
                      'buffer_size': 25001, # 120 seconds
                      # simulation gains

                      # default config
                      # 'kp': np.array([30., 30., 30.]*4),
                      # 'kd': np.array([3., 3., 3.]*4),
                      # 'ki': np.array([0., 0., 0.]*4),
                      #Orbit
                      'kp': np.array([50., 50., 50.] * 4),
                      'kd': np.array([3, 3, 3] * 4),
                      'ki': np.array([0., 0., 0.] * 4),

                      #default config
                      # 'kp_swing':  np.array([30., 30.,30]*4),
                      # 'kd_swing':  np.array([.2, .2, .24]*4),
                      # 'ki_swing':  np.array([2., 2., 2.]*4),
                      #orbit
                      'kp_swing': np.array([30., 30., 30.] * 4),
                      'kd_swing': np.array([0.3, 0.3, 0.3] * 4),
                      'ki_swing': np.array([0., 0., 0.] * 4),

                      'kp_land': np.array([30., 30., 30.] * 4),
                      'kd_land': np.array([3., 3., 3.] * 4),
                      'ki_land': np.array([0., 0., 0.] * 4),

                      # joint pid + wbc (optional)
                      'kp_wbc': np.array([15., 15., 15.]*4),#np.array([10., 10., 10.]*4),
                      'kd_wbc': np.array([1., 1., 1.]*4),#np.array([1., 1., 1.]*4),
                      'ki_wbc': np.array([0., 0., 0.]*4),#np.array([0.3, 0.3, 0.3]*4),
                      # virtual impedance wrench control
                      'kp_lin': np.array([800, 500., 900.]),  # x y z
                      'kd_lin': np.array([100, 100., 100.]),
                      'kp_ang': np.array([40., 40., 40.]),  # R P Y
                      'kd_ang': np.array([1.51, 1.51, 1.51]),
                      # real robot gains
                      # stand alone joint pid
                      'kp_real': np.array([30., 30.,30.]*4),
                      'kd_real': np.array([.15, .1, .15]*4),
                      'ki_real': np.array([2., 2., 2.]*4),
                      'kp_real_swing':  np.array([15., 7.,4.6]*4),
                      'kd_real_swing':  np.array([.2, .2, .24]*4),
                      'ki_real_swing':  np.array([2., 2., 2.]*4),
                      'kp_real_land': np.array([20., 10.,10]*4),
                      'kd_real_land': np.array([.2, .2, .24]*4),
                      'ki_real_land': np.array([2., 2., 2.]*4),
                       # joint pid + wbc (optional)
                      'kp_wbc_real': np.array([20., 30., 40.]*4),
                      'kd_wbc_real': np.array([.3, .3, .3]*4),
                      'ki_wbc_real': np.array([1.5, 1.5, 1.5]*4),
                      # virtual impedance wrench control
                      # 'kp_lin_real': np.array([300, 300., 300.]), # x y z
                      # 'kd_lin_real': np.array([30., 30., 30.]),
                      # 'kp_ang_real': np.array([50, 50., 50.]), # R P Y
                      # 'kd_ang_real': np.array([10., 10., 10.]),
                      'kp_lin_real': np.array([650., 300., 450.]), #np.array([300., 300., 400.]), # x y z
                      'kd_lin_real': np.array([40., 40., 40.]), #np.array([30., 20., 60.]),
                      'kp_ang_real': np.array([20., 35., 20.]), # #np.array([30., 50., 30.]), # R P Y
                      'kd_ang_real': np.array([1.5, 2.5, 1.5]), #np.array([2., 4., 2.]),
                      # joint configuration
                      #default configuration
                      #'q_0':  np.array([0.2, 0.78, -1.7,  0.2, 0.78, -1.7, -0.2, 0.78, -1.7, -0.2, 0.78, -1.7]),
                      # q0 orbit
                       'q_0':  np.array([0.1, 0.8, -1.6,#lf
                                         0.1, 1.0, -1.6,#lh
                                         -0.1, 0.8, -1.6,#rf
                                         -0.1, 1.0, -1.6]),#rh
                      'q_retraction': np.array([0.23, 1.38, -2.46, 0.23, 1.68, -2.31, -0.23, 1.38, -2.46, -0.23, 1.68, -2.31]),

                      #default configuration
                      #'q_final': np.array([0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, -0.1, 0.9, -1.8]),
                      #q final orbit
                      'q_final': np.array([0.1, 0.8, -1.6,#lf
                                         0.1, 1.0, -1.6,#lh
                                         -0.1, 0.8, -1.6,#rf
                                         -0.1, 1.0, -1.6]),#rh
                      #Forward
                      'q_land_fwd':  np.array([0.1, 0.75, -1.8,  0.1, 0.75, -1.8, -0.1, 0.75, -1.8, -0.1, 0.75, -1.8]),
                      #backward
                      'q_land_bwd':  np.array([0.1, 1., -1.8,  0.1, 1., -1.8, -0.1, 1., -1.8, -0.1, 1., -1.8]),




                      'q_fold': np.array([0.2, 1.7, -2.7, 0.2,  1.7, -2.7, -0.2, 1.7,  -2.7, -0.2, 1.7, -2.7]),
                      'q_0_td': np.array([0.1789, 1.2234, -2.2329, 0.1867, 1.4733, -2.1055, -0.1784, 1.2230, -2.2327, -0.1861, 1.4733, -2.1053]),
                      'landing_duration': 1.4,
                      'com_z0_training': 0.31,
                        # 'q_0_lo': np.array([0.3430, 1.5495, -2.6620, 0.3433, 1.9171, -2.4902, -0.3425, 1.5490, -2.6620, -0.3424, 1.9171, -2.4901]),
                      'joint_names': ['lf_haa_joint',  'lf_hfe_joint', 'lf_kfe_joint',
                                      'lh_haa_joint',  'lh_hfe_joint', 'lh_kfe_joint',
                                      'rf_haa_joint',  'rf_hfe_joint', 'rf_kfe_joint',
                                      'rh_haa_joint',  'rh_hfe_joint', 'rh_kfe_joint'],
                      # ee params
                      'ee_frames': ['lf_foot', 'lh_foot', 'rf_foot','rh_foot'],
                     #  'force_th': 18.,
                      'force_th': 7.,
                      'contact_th': 0.01, # tolerance understand when /knee base is touching the ground
                      # simulation spawn [m] and [rad]
                      'spawn_x': 0.0,
                      'spawn_y': 0.0,
                      'spawn_z': .27,
                      'spawn_R': 0.0,
                      'spawn_P': 0.0,
                      'spawn_Y': 0.0,
                      'ip': "192.168.123.15", #UDP node is running in the nano computer in the head
                      # use real robot or simulation
                      'real_robot': False} # note the frames are all aligned with base for joints = 0

robot_params['go2'] ={'dt': 0.002,
                      'buffer_size': 25001, # 120 seconds
                      # simulation gains

                      # default config
                      # 'kp': np.array([30., 30., 30.]*4),
                      # 'kd': np.array([3., 3., 3.]*4),
                      # 'ki': np.array([0., 0., 0.]*4),
                      #Orbit
                      'kp': np.array([60., 60., 60.] * 4),
                      'kd': np.array([0.8, 0.8, 0.8] * 4),
                      'ki': np.array([0., 0., 0.] * 4),

                      #default config
                      # 'kp_swing':  np.array([30., 30.,30]*4),
                      # 'kd_swing':  np.array([.2, .2, .24]*4),
                      # 'ki_swing':  np.array([2., 2., 2.]*4),
                      #orbit
                      'kp_swing': np.array([30., 30., 30.] * 4),
                      'kd_swing': np.array([0.4, 0.4, 0.4] * 4),
                      'ki_swing': np.array([0., 0., 0.] * 4),

                      'kp_land': np.array([30., 30., 30.] * 4),
                      'kd_land': np.array([0.8, 0.8, 0.8] * 4),
                      'ki_land': np.array([0., 0., 0.] * 4),

                      # joint pid + wbc (optional)
                      'kp_wbc': np.array([15., 15., 15.]*4),#np.array([10., 10., 10.]*4),
                      'kd_wbc': np.array([1., 1., 1.]*4),#np.array([1., 1., 1.]*4),
                      'ki_wbc': np.array([0., 0., 0.]*4),#np.array([0.3, 0.3, 0.3]*4),
                      # virtual impedance wrench control
                      'kp_lin': np.array([800, 500., 900.]),  # x y z
                      'kd_lin': np.array([100, 100., 100.]),
                      'kp_ang': np.array([40., 40., 40.]),  # R P Y
                      'kd_ang': np.array([1.51, 1.51, 1.51]),                      
                      # real robot gains
                      # stand alone joint pid
                      'kp_real': np.array([30., 30.,30.]*4),
                      'kd_real': np.array([1., 1., 1.]*4),
                      'ki_real': np.array([0., 0., 0.]*4),

                      'kp_real_swing':  np.array([30., 30, 30]*4),
                      'kd_real_swing':  np.array([1, 1, 1]*4),
                      'ki_real_swing':  np.array([0, 0, 0]*4),

                      'kp_real_land': np.array([20., 20.,20.]*4),
                      'kd_real_land': np.array([1, 1, 1]*4),
                      'ki_real_land': np.array([0, 0, 0]*4),
                       # joint pid + wbc (optional)
                      'kp_wbc_real': np.array([20., 30., 40.]*4),
                      'kd_wbc_real': np.array([.3, .3, .3]*4),
                      'ki_wbc_real': np.array([1.5, 1.5, 1.5]*4),
                      # virtual impedance wrench control
                      # 'kp_lin_real': np.array([300, 300., 300.]), # x y z
                      # 'kd_lin_real': np.array([30., 30., 30.]),
                      # 'kp_ang_real': np.array([50, 50., 50.]), # R P Y
                      # 'kd_ang_real': np.array([10., 10., 10.]),
                      'kp_lin_real': np.array([650., 300., 450.]), #np.array([300., 300., 400.]), # x y z
                      'kd_lin_real': np.array([40., 40., 40.]), #np.array([30., 20., 60.]),
                      'kp_ang_real': np.array([20., 35., 20.]), # #np.array([30., 50., 30.]), # R P Y
                      'kd_ang_real': np.array([1.5, 2.5, 1.5]), #np.array([2., 4., 2.]),
                      # joint configuration
                      #default configuration
                      #'q_0':  np.array([0.2, 0.78, -1.7,  0.2, 0.78, -1.7, -0.2, 0.78, -1.7, -0.2, 0.78, -1.7]),
                      # q0 orbit
                       'q_0':  np.array([0.1, 0.8, -1.5,#lf
                                         0.1, 1.0, -1.5,#lh
                                         -0.1, 0.8, -1.5,#rf
                                         -0.1, 1.0, -1.5]),#rh
                      'q_retraction': np.array([0.23, 1.38, -2.46, 0.23, 1.68, -2.31, -0.23, 1.38, -2.46, -0.23, 1.68, -2.31]),

                      #default configuration
                      #'q_final': np.array([0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, -0.1, 0.9, -1.8]),
                      #q final orbit
                      'q_final': np.array([0.1, 0.8, -1.5,#lf
                                         0.1, 1.0, -1.5,#lh
                                         -0.1, 0.8, -1.5,#rf
                                         -0.1, 1.0, -1.5]),#rh
                      #Forward
                      'q_land_fwd':  np.array([0.1, 0.75, -1.8,  0.1, 0.75, -1.8, -0.1, 0.75, -1.8, -0.1, 0.75, -1.8]),
                      #backward
                      'q_land_bwd':  np.array([0.1, 1., -1.8,  0.1, 1., -1.8, -0.1, 1., -1.8, -0.1, 1., -1.8]),




                      'q_fold': np.array([0.2, 1.7, -2.7, 0.2,  1.7, -2.7, -0.2, 1.7,  -2.7, -0.2, 1.7, -2.7]),
                      'q_0_td': np.array([0.1789, 1.2234, -2.2329, 0.1867, 1.4733, -2.1055, -0.1784, 1.2230, -2.2327, -0.1861, 1.4733, -2.1053]),
                      'landing_duration': 1.4,
                      'com_z0_training': 0.32,
                        # 'q_0_lo': np.array([0.3430, 1.5495, -2.6620, 0.3433, 1.9171, -2.4902, -0.3425, 1.5490, -2.6620, -0.3424, 1.9171, -2.4901]),
                      'joint_names': ['lf_haa_joint',  'lf_hfe_joint', 'lf_kfe_joint',
                                      'lh_haa_joint',  'lh_hfe_joint', 'lh_kfe_joint',
                                      'rf_haa_joint',  'rf_hfe_joint', 'rf_kfe_joint',
                                      'rh_haa_joint',  'rh_hfe_joint', 'rh_kfe_joint'],
                      # ee params
                      'ee_frames': ['lf_foot', 'lh_foot', 'rf_foot','rh_foot'],
                     #  'force_th': 18.,
                      'force_th': 7.,
                      'contact_th': 0.01, # tolerance understand when /knee base is touching the ground
                      # simulation spawn [m] and [rad]
                      'spawn_x': 0.0,
                      'spawn_y': 0.0,
                      'spawn_z': .32,
                      'spawn_R': 0.0,
                      'spawn_P': 0.0,
                      'spawn_Y': 0.0,
                      'ip': "192.168.123.161",
                      # use real robot or simulation
                      'real_robot': False} # note the frames are all aligned with base for joints = 0

robot_params['anymal_d'] ={'dt': 0.002,
                        'kp': np.array([200., 200., 200.]*4),
                        'kd': np.array([10., 10., 10.]*4),
                        'ki': np.array([0., 0., 0.]*4),
                           'kp_swing':  np.array([20., 30., 20.]*4),
                        'kd_swing':  np.array([1., 1., 1.]*4),
                        'ki_swing':  np.array([0., 0., 0.]*4),
                           'kp_land': np.array([60., 90., 60.]*4),
                        'kd_land': np.array([10., 10., 10.]*4),
                        'ki_land': np.array([0., 0., 0.]*4),
                           'kp_real': np.array([20., 30., 30.]*4),
                        'kd_real': np.array([1., 1., 1.]*4),
                        'ki_real': np.array([1., 1., 1.]*4),
                           'kp_real_swing':  np.array([15., 20., 20]*4),
                        'kd_real_swing':  np.array([1., 1., 1.]*4),
                        'ki_real_swing':  np.array([0., 0., 0.]*4),
                           'kp_real_land': np.array([15., 15., 15]*4),
                        'kd_real_land': np.array([1., 1., 1.]*4),
                        'ki_real_land': np.array([0., 0., 0.]*4),    # SIM ---------------------------    # joint pid + wbc (optional)
                        'kp_wbc': np.array([60., 60., 60.]*4),
                        'kd_wbc': np.array([10., 10., 10.]*4),
                        'ki_wbc': np.array([0., 0., 0.]*4),
                        # virtual impedance wrench control
                        'kp_lin': np.array([1000., 1000., 800.]),
                        'kd_lin': np.array([150., 150., 100.]),
                        'kp_ang': np.array([200., 200., 100.]),
                        'kd_ang': np.array([10., 20., 20.]),    # REAL---------------------------
                        # joint pid for landing contoller after td + wbc (optional)
                        'kp_wbc_real': np.array([10., 10., 10.]*4),
                        'kd_wbc_real': np.array([1., 1., 1.]*4),
                        'ki_wbc_real': np.array([0., 0., 0.]*4),    'kp_lin_real': np.array([1000., 1000., 1000.]),
                        'kd_lin_real': np.array([200., 150., 200.]),
                        'kp_ang_real': np.array([300., 300., 200.]),
                        'kd_ang_real': np.array([10., 20., 20.]),    #joint configuration
                        'q_0': np.array([ -0.1, 0.7, -1.,
                                          -0.1 , -0.7 , 1.,
                                          0.1, 0.7, -1.,
                                          0.1, -0.7, 1.]),
                        'q_retraction': np.array([0.2867, 1.48, -2.6,
                              0.2867, 1.8, -2.48,
                              -0.2867, 1.48, -2.6,
                              -0.2867, 1.8, -2.48]),
                        # default
                        #'q_final': np.array([0.2, 1.1, -1.8, 0.2, 1.1, -1.8, -0.2, 1.1, -1.8, -0.2, 1.1, -1.8]),
                        # orbit
                        'q_final': np.array([0.1, 0.8, -1.6,  # lf
                                              0.1, 1.0, -1.6,  # lh
                                              -0.1, 0.8, -1.6,  # rf
                                              -0.1, 1.0, -1.6]),  # rh    #forward
                        'q_land_fwd':  np.array([0.1, 0.6, -1.8,  0.1, 0.6, -1.8, -0.1, 0.6, -1.8, -0.1, 0.6, -1.8]),
                        #backward
                        'q_land_bwd':  np.array([0.1, 1., -1.8,  0.1, 1., -1.8, -0.1, 1., -1.8, -0.1, 1., -1.8]),
                        'landing_duration': 1.4,
                        'com_z0_training' : 0.375,
                        'q_0_td': np.array([0.1789, 1.2234, -2.2329, 0.1867, 1.4733, -2.1055, -0.1784, 1.2230, -2.2327, -0.1861, 1.4733, -2.1053]),
                        'q_fold': np.array([0.2, 1.7, -2.7, #lf
                                            0.2,  -1.7, 2.7, #lf
                                            -0.2, 1.7,  -2.7,
                                            -0.2, -1.7, 2.7]),  #thjis is for the startup phase
                       'joint_names': ['lf_haa_joint', 'lf_hfe_joint', 'lf_kfe_joint',
                                       'lh_haa_joint', 'lh_hfe_joint', 'lh_kfe_joint',
                                       'rf_haa_joint', 'rf_hfe_joint', 'rf_kfe_joint',
                                       'rh_haa_joint', 'rh_hfe_joint', 'rh_kfe_joint'],
                       # ee params
                       'ee_frames': ['lf_foot', 'lh_foot', 'rf_foot', 'rh_foot'],
                        'real_robot': False,
                        'force_th': 5.,
                        'spawn_x': 0.0,
                        'spawn_y': 0.0,
                        'spawn_z': 0.9,
                        'buffer_size': 50001
                    } # note the

robot_params['mantis'] ={'dt': 0.001,
                      'buffer_size': 5001, # 120 seconds
                      # simulation gains
                      'kp': 2*np.array([600., 300., 300., 600., 300., 300., 600., 300., 300., 600., 300., 300.]),
                      'kd': np.array([50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50.]),
                      'ki': np.array([0., 0., 0.]*4),
                       #star stance config
                      #'q_0': np.array([-1.57, 1.5, -1.5, -1.57, 1.5, -1.5,  1.57, 1.5, -1.5, 1.57, 1.5, -1.5]),
                      # quadruped stance config
                      'q_0': np.array([0.7, 0.75, -1.5, 0.7, 0.75, -1.5, -0.7, 0.75, -1.5, -0.7, 0.75, -1.5]),
                      'joint_names': ['lf_hy_joint',  'lf_hp_joint', 'lf_kp_joint',
                                      'lh_hy_joint',  'lh_hp_joint', 'lh_kp_joint',
                                      'rf_hy_joint',  'rf_hp_joint', 'rf_kp_joint',
                                      'rh_hy_joint',  'rh_hp_joint', 'rh_kp_joint'],
                      # ee params
                      'ee_frames': ['lf_foot', 'lh_foot', 'rf_foot','rh_foot'],
                      'force_th': 18.,
                      # simulation spawn [m] and [rad]
                      'spawn_x': 0.0,
                      'spawn_y': 0.0,
                      'spawn_z': 1.,
                      'spawn_R': 0.0,
                      'spawn_P': 0.0,
                      'spawn_Y': 0.0,
                      # use real robot or simulation
                      'real_robot': False} # note the frames are all aligned with base for joints = 0

robot_params['invpend'] ={'dt': 0.001,
                       'kp': np.array([0, 0 ]),
                       'kd':  np.array([0,0]),
                       'q_0':  np.array([ 0.0, 0.1]),
                       'joint_names': ['slider_to_cart', 'cart_to_pole'],
                       'ee_frame': 'lf_foot',
                       'spawn_x' : 0.0,
                       'spawn_y' : 0.0,
                       'spawn_z' : 0.0,
                       'buffer_size': 1000} # note the frames are all aligned with base for joints = 0

robot_params['inv'] ={'dt': 0.001,
                       'kp': np.array([100, 100, 100, 10, 10, 10 ]),
                       'kd':  np.array([10,10,10, 0.2,0.2,0.2]),
                       'q_0':  np.array([ 0.0, 0.0, 0.25, -0.24010055475883635,0.7092776153747403,-1.4185292429491714]),
                       'joint_names': ['base_x_joint', 'base_y_joint', 'base_z_joint', 'lf_haa_joint', 'lf_hfe_joint', 'lf_kfe_joint'],
                       'ee_frame': 'lf_foot',
                       'spawn_x' : 0.0,
                       'spawn_y' : 0.0,
                       'spawn_z' : 0.0,
                       'buffer_size': 1000} # note the frames are all aligned with base for joints = 0

robot_params['mir'] ={'dt': 0.01,
                       'q_0':  np.array([ 0.0, 0.0]),
                       'joint_names': ['left_wheel_joint', 'right_wheel_joint'],
                       'spawn_x' : 0.0,
                       'spawn_y' : 0.0,
                       'spawn_z' : 0.0,
                       'buffer_size': 12000} # note the frames are all aligned with base for joints = 0


robot_params['climbingrobot'] ={'dt': 0.001,
                       'kp': np.array([0 ,    0,    400,  100,    50,   50,    50, 30, 30]),
                       'kd':  np.array([0,    0,    10,   10,     10,   10,     4,   4, 4  ]),

                       'q_0':  np.array([ 0, 0, 8.0, 0,0,0, -1.57, 0.0, 0.0 ]),
                       'joint_names': ['mountain_wire_pitch', 'mountain_wire_roll',  'wire_base_prismatic',
                                       'wire_base_pitch', 'wire_base_roll','wire_base_yaw',
                                       'hip_pitch', 'hip_roll', 'knee'],
                       'ee_frame': 'foot',
                       'spawn_x' : 0.0,
                       'spawn_y' : 0.0,
                       'spawn_z' : 20.0,
                       'buffer_size': 10000} # note the frames are all aligned with base for joints = 0

robot_params['climbingrobot_slider'] ={'dt': 0.001,
                       'kp': np.array([400, 0 ,   0,    400,  100,    50,   50,    50, 30, 30]),
                       'kd':  np.array([10, 0,    0,    10,   10,     10,   10,     4,   4, 4 ]),

                       'q_0':  np.array([ 0.0, 0, 0, 8.0, 0,0,0, -1.57, 0.0, 0.0 ]),
                       'joint_names': ['slider', 'mountain_wire_pitch', 'mountain_wire_roll',  'wire_base_prismatic',
                                       'wire_base_pitch', 'wire_base_roll','wire_base_yaw',
                                       'hip_pitch', 'hip_roll', 'knee'],
                       'ee_frame': 'foot',
                       'spawn_x' : 0.0,
                       'spawn_y' : 0.0,
                       'spawn_z' : 20.0,
                       'buffer_size': 10000} # note the frames are all aligned with base for joints = 0

robot_params['climbingrobot2'] ={'dt': 0.001,
                       'kp': np.array([0 ,    0,    400,  40,    40,   40,
                                       0 ,    0,   400,  40,    40,   40,
                                       150, 130, 120]),
                       'kd':  np.array([20,    20,    100,   20,     20,   20,
                                        20,    20,    100,   20,     20,   20,
                                        10,   10, 10  ]),
                       # this corresposnds to p = [0.03, 2.5, -6] from matlab WF  which is located in anchor_pos1 and correspods to (rope joint starts from anchor disnace /2)
                       'q_0':  np.array([ 0.0,    1.17  ,  4.0000   ,      0. ,   -1.17 ,        0.,
                                          0.0,  -1.17,    4.0000  ,       0. ,  1.17,           0.,
                                          -1.57, 0.0, -0. ]), # negative knee value is extended
                       # 'q_0': np.array([0.0, 0, 4, 0, 0.0, 0,
                       #                    0.0, 0, 4, 0, 0, 0,
                       #                    -1.57, 0.0, 0.0]),
                      'Ko': np.array([500, 500, 500]),
                      'Do': np.array([20, 20, 20]),
                       'joint_names': ['mountain_wire_pitch_r', 'mountain_wire_roll_r',  'wire_base_prismatic_r',
                                       'wire_base_pitch_r', 'wire_base_roll_r','wire_base_yaw_r',
                                        'mountain_wire_pitch_l', 'mountain_wire_roll_l',  'wire_base_prismatic_l',
                                       'wire_base_pitch_l', 'wire_base_roll_l','wire_base_yaw_l',
                                       'hip_pitch', 'hip_roll', 'knee'],
                       'ee_frame': 'foot',
                       'spawn_x' : 0.2,
                       'spawn_y' : 0.0,
                       'spawn_z' : 20.0,
                       'spawn_2x': 0.2,
                       'spawn_2y': 5.0,
                       'spawn_2z': 20.0,
                       'wall_inclination': 0.0,
                       # ALPINE real robot geometry / rope zeroing.
                       # Same convention used by alpine_odometry.yaml and the real controller:
                       #   L_abs = home_offset + rope_sign * L_raw
                       #   L_raw_ref = rope_sign * (L_abs_ref - home_offset)
                       'anchor_left_xyz': np.array([0.45, 0.0, 2.50]),
                       'anchor_right_xyz': np.array([0.45, 0.65, 2.50]),
                       'body_origin_from_left_attachment_xyz': np.array([0.0, 0.0, -0.55]),
                       'right_attachment_from_left_body_xyz': np.array([0.0, -0.55, 0.0]),
                       'left_home_offset_m': 0.0,
                       'right_home_offset_m': 0.53,
                       'left_rope_sign': 1.0,
                       'right_rope_sign': -1.0,
                       'left_rope_axis': '-x',
                       'homing_test_delta_m': 0.5,
                       'real_robot': False,
                       'buffer_size': 30000} # note the frames are all aligned with base for joints = 0

robot_params['climbingrobot2landing'] ={'dt': 0.001,
                       'kp': np.array([0 ,    0,    1200,  40,    40,   40,
                                       0 ,    0,    1200,  40,    40,   40,
                                       50, 30, 80,
                                       30, 0., 30,0.
                                       ]),
                       'kd':  np.array([20,    20,    400,   20,     20,   20,
                                        20,    20,    400,   20,     20,   20,
                                        10,   10, 30,
                                        5, 0.01, 5 ,0.01
                                        ]),
                       # this corresposnds to p = [0.03, 2.5, -6] from matlab WF  which is located in anchor_pos1
                       'q_0':  np.array([ 0.0,    1.17  ,  4.0000   ,      0. ,   -1.17 ,        0.,
                                          0.0,  -1.17,    4.0000  ,       0. ,  1.17,           0.,
                                          -1.57, 0.0, 0.0,
                                          0., -0, 0.,0.
                                          ]),
                       'Ko': np.array([10, 10, 10]),
                       'Do': np.array([1, 1, 1]),
                       'joint_names': ['mountain_wire_pitch_r', 'mountain_wire_roll_r',  'wire_base_prismatic_r',
                                       'wire_base_pitch_r', 'wire_base_roll_r','wire_base_yaw_r',
                                        'mountain_wire_pitch_l', 'mountain_wire_roll_l',  'wire_base_prismatic_l',
                                       'wire_base_pitch_l', 'wire_base_roll_l','wire_base_yaw_l',
                                       'hip_pitch', 'hip_roll', 'knee',
                                       'hip_yaw_landing_l', 'wheel_joint_l', 'hip_yaw_landing_r', 'wheel_joint_r'],
                       'ee_frame': 'foot',
                       'spawn_x' : 0.2,
                       'spawn_y' : 0.0,
                       'spawn_z' : 20.0,
                       'spawn_2x': 0.2,
                       'spawn_2y': 5.0,
                       'spawn_2z': 20.0,
                       'wall_inclination':0.1,
                       'buffer_size': 30000} # note the frames are all aligned with base for joints = 0

#
robot_params['starbot'] ={'dt': 0.001,
                       'kp': 50.*np.array([50. ,   50.,    50.,  50.0 ,
                       			10. ,   10.,    10.,  10. ,
                                100. ,   100.,    100.,  100. ,
                                10. ,   10.,    10.,  10. ,
                       			0.05 ,   0.05,    0.05,  0.05]),
                       'kd':   1.0*np.array([2.,    2.,    2.,   2. ,
                       			1.,    1.,    1.,   1. ,
                                5.,    5.,   5.,   5. ,
                                1.,    1.,    1.,   1. ,
                       			1,   1,    1,  1 ]),

                       'q_0':  np.array([ 0, 0, 0, 0,
                       			0., 0., 0., 0.0,
                                0., 0., 0., 0.,
                                0., 0., 0., 0.0,
                                0., 0., 0., 0.0 ]),
                       'joint_names': ['lf_bs_joint',       'lh_bs_joint',  'rf_bs_joint', 'rh_bs_joint',
                       			      'lf_upper_leg_joint', 'lh_upper_leg_joint',  'rf_upper_leg_joint', 'rh_upper_leg_joint',
                       			      'lf_lower_leg_joint', 'lh_lower_leg_joint',  'rf_lower_leg_joint', 'rh_lower_leg_joint',
                                      'lf_pre_wheel_joint', 'lh_pre_wheel_joint',  'rf_pre_wheel_joint', 'rh_pre_wheel_joint',
                                      'lf_wheel_joint', 'lh_wheel_joint',  'rf_wheel_joint', 'rh_wheel_joint'],
                       'ee_frames': ['lf_wheel','lh_wheel','rf_wheel','rh_wheel'],
                       'force_th': 20.,
                       'spawn_x' : 0.0,
                       'spawn_y' : 0.0,
                       'spawn_z' : 1.0,
                       'buffer_size': 10000} # note the frames are all aligned with base for joints = 0

verbose = False
plotting = True

