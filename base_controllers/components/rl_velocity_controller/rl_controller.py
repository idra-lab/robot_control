"""Velocity policy runner for :mod:`quadruped_controller`.

Two policy families are supported behind one interface:

``policy="legacy"`` (the default, and byte-for-byte the original behaviour)
    The three- or four-network arrangement under ``policies/``: a 48-float observation for the
    actor, and a separate state-estimation network fed a 3-frame history, plus ``_safe`` variants
    of each.

``policy="velocity"``
    The single-network policy trained in ``safe_rl/tasks/manager_based/velocity``, which regresses
    the base velocity inside the actor and therefore needs no separate estimator.  It is run by
    :class:`~base_controllers.rl_quadruped.velocity_policy.VelocityPolicy`, so there is one
    implementation of that observation contract rather than two that can drift apart; this class
    only adapts it to the calling convention :mod:`quadruped_controller` already uses.

The default is unchanged, so existing callers keep the policies and the behaviour they had.
"""

import numpy as np
import onnxruntime as ort
import os
import json
from termcolor import colored
import rospy
from geometry_msgs.msg import Vector3

LEGACY = "legacy"
VELOCITY = "velocity"


class RlVelocityController():
    def __init__(self, robot_name: str, dt: float, use_nn_se: bool = True, freq: int = 50,
                 debug=False, policy: str = LEGACY, estimate_velocity: bool = True):
        self.debug = debug
        # policy='velocity' only: evaluate the actor's state-estimation head and publish what it
        # regresses on /<robot>/se_nn_base_lin_vel.  It is a diagnostic - the policy consumes its
        # own estimate inside the ONNX graph either way - so it can be switched off.  The legacy
        # family ignores this: there the estimator is a separate network the policy actually needs.
        self.estimate_velocity = bool(estimate_velocity)
        self.robot_name = robot_name
        self.policy_family = policy
        if policy not in (LEGACY, VELOCITY):
            raise ValueError(f"policy must be '{LEGACY}' or '{VELOCITY}', not '{policy}'")
        # Create a ros publisher for publishing the se_nn base linear velocity
        self.pub_se_nn_base_lin_vel = rospy.Publisher("/" + self.robot_name + "/se_nn_base_lin_vel", Vector3, queue_size=10)
        if self.debug:
            self.pub_gt_base_lin_vel = rospy.Publisher("/" + self.robot_name + "/gt_base_lin_vel", Vector3, queue_size=10)

        if policy == VELOCITY:
            self._initVelocityPolicy(dt, use_nn_se)
            return

        if use_nn_se:
            base_model_path = os.path.join(os.environ.get('LOCOSIM_DIR'),
                                           'robot_control',
                                           'base_controllers',
                                           'components',
                                           'rl_velocity_controller',
                                           'policies')
        else:
            base_model_path = os.path.join(os.environ.get('LOCOSIM_DIR'),
                                           'robot_control',
                                           'base_controllers',
                                           'components',
                                           'rl_velocity_controller',
                                           'policies', 'previous')

        config_path = os.path.join(base_model_path, f"{robot_name}.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Policy does not exist for {self.robot_name}!!!")

        with open(config_path, 'r') as f:
            self.cfg = json.load(f)

        self.model_path = os.path.join(base_model_path, f'{robot_name}.onnx')
        self.model = ort.InferenceSession(self.model_path)
        print(f'Policy for {robot_name} loaded')

        self.model_path_safe = os.path.join(base_model_path, f'{robot_name}_safe.onnx')
        self.model_safe = ort.InferenceSession(self.model_path_safe)
        print(f'Policy for {robot_name}_safe loaded')

        if use_nn_se:
            self.model_path_se = os.path.join(base_model_path, f'{robot_name}_se.onnx')
            self.model_se = ort.InferenceSession(self.model_path_se)
            print(f'Policy for {robot_name}_se loaded')

            self.model_path_se_safe = os.path.join(base_model_path, f'{robot_name}_se_safe.onnx')
            self.model_se_safe = ort.InferenceSession(self.model_path_se_safe)
            print(f'Policy for {robot_name}_se_safe loaded')

        self.q_def = self.cfg["q_def"]
        self.q_des = self.q_def
        self.action_scale = self.cfg["action_scale"]
        

        
        self.kp = np.full(12, self.cfg["kp"])
        self.kd = np.full(12, self.cfg["kd"])

        self.use_nn_se = use_nn_se

        self.prev_action = np.zeros(12)
        self.velocity_cmd = np.zeros(3)
        
        self.decimation = (1/dt)*(1/freq)
        self.decimation_counter = 0
        self.history_buffer = np.zeros((1, 3, 48))
        
    def action(self, base_lin_acc, base_lin_vel, base_ang_vel, pj_gravity, q, qd, policy_type='default'):

        if self.policy_family == VELOCITY:
            return self._velocityAction(base_lin_acc, base_lin_vel, base_ang_vel, pj_gravity,
                                        q, qd, policy_type)

        if (self.decimation_counter % self.decimation) == 0 :

            joint_pos_rel = q - self.q_def

            if self.use_nn_se:

                obs = np.concatenate([
                    base_lin_acc,
                    base_ang_vel,
                    pj_gravity,
                    self.velocity_cmd,
                    joint_pos_rel,
                    qd,
                    self.prev_action
                ]).astype(np.float32)[None]
                
                # update history buffer
                buf = self.history_buffer
                buf = np.concatenate([buf[:, 1:, :], obs[None]], axis=1)
                self.history_buffer = buf

                # run the se nn to get the se_nn base linear velocity
                se_nn_obs = self.history_buffer.flatten()[None].astype(np.float32)

                if policy_type=='default':
                    nn_base_lin_vel = self.model_se.run(None, {'obs': se_nn_obs})[0][0]
                elif policy_type=='safe':
                    nn_base_lin_vel = self.model_se_safe.run(None, {'obs': se_nn_obs})[0][0]

                # print("gt:", base_lin_vel, " nn:", nn_base_lin_vel, " err:", np.linalg.norm(base_lin_vel-nn_base_lin_vel))

                policy_base_lin_vel = nn_base_lin_vel

                # Publish the se_nn base linear velocity
                se_nn_base_lin_vel_msg = Vector3()
                se_nn_base_lin_vel_msg.x = nn_base_lin_vel[0]
                se_nn_base_lin_vel_msg.y = nn_base_lin_vel[1]
                se_nn_base_lin_vel_msg.z = nn_base_lin_vel[2]
                self.pub_se_nn_base_lin_vel.publish(se_nn_base_lin_vel_msg)

                if self.debug:
                    # Publish the gt base linear velocity for debug
                    gt_base_lin_vel_msg = Vector3()
                    gt_base_lin_vel_msg.x = base_lin_vel[0]
                    gt_base_lin_vel_msg.y = base_lin_vel[1]
                    gt_base_lin_vel_msg.z = base_lin_vel[2]
                    self.pub_gt_base_lin_vel.publish(gt_base_lin_vel_msg)

            else:

                policy_base_lin_vel = base_lin_vel
            
            obs = np.concatenate([
                policy_base_lin_vel,
                base_ang_vel,
                pj_gravity,
                self.velocity_cmd,
                joint_pos_rel,
                qd,
                self.prev_action
            ]).astype(np.float32)[None]

            if policy_type=='default':
                action = self.model.run(None, {'obs': obs})[0][0]
            elif policy_type=='safe':
                action = self.model_safe.run(None, {'obs': obs})[0][0]
            else:
                colored("Wrong policy type", "red")

            self.prev_action = action
            
            final_action = self.q_def + ( self.action_scale * action)
            self.q_des = final_action

        self.decimation_counter += 1
        
        return self.q_des
        

    # ---------------------------------------------------------------------------------------------
    # single-network velocity policy
    # ---------------------------------------------------------------------------------------------
    def _initVelocityPolicy(self, dt, use_nn_se):
        """Load the single-network policy and present it through this class's attributes."""
        # Imported here rather than at module scope so that a workspace without the newer package
        # still loads this module for the legacy path.
        from base_controllers.rl_quadruped.velocity_policy import VelocityPolicy

        if not use_nn_se:
            # Not a warning to be tidy: with use_nn_se False the caller passes base_lin_acc as None
            # and hands over ground-truth base velocity instead, and this policy consumes the
            # accelerometer and estimates the velocity itself.  There is no variant of it that
            # takes a measured base velocity, so the request cannot be honoured.
            print(colored("policy='velocity' estimates the base velocity inside the network; "
                          "use_nn_se=False is not available for it and is being ignored",
                          "yellow"))
        self.use_nn_se = True

        # measure_timing is off: nothing in this path reads the per-tick counters, so timing every
        # inference would only cost two perf_counter calls per tick.  Use rl_quadruped_controller
        # for the instrumented run.
        self.policy = VelocityPolicy(self.robot_name, dt=dt,
                                     enable_estimator=self.estimate_velocity,
                                     measure_timing=False)
        self.cfg = self.policy.cfg
        self.q_def = self.policy.q_default
        self.q_des = self.q_def.copy()
        self.action_scale = self.policy.action_scale
        self.kp = np.full(12, self.policy.kp)
        self.kd = np.full(12, self.policy.kd)
        self.prev_action = self.policy.prev_action
        self.velocity_cmd = np.zeros(3)
        self.decimation = self.policy.decimation
        self.decimation_counter = 0
        self._velocity_started = False
        print(colored(f"Single-network velocity policy for {self.robot_name} loaded: "
                      f"variants {self.policy.variants}, {self.policy.policy_rate:.0f} Hz, "
                      f"decimation {self.policy.decimation}", "green"))

    _VARIANT_OF = {"default": "normal", "safe": "safe"}

    def _velocityAction(self, base_lin_acc, base_lin_vel, base_ang_vel, pj_gravity, q, qd,
                        policy_type):
        """Run the single-network policy through the legacy ``action`` signature.

        ``base_lin_vel`` is deliberately unused: the actor regresses it internally, which is the
        whole point of this policy.  It is still published when ``debug`` is set, so the estimate
        and the truth can be compared on the same topics as before.
        """
        variant = self._VARIANT_OF.get(policy_type)
        if variant is None:
            print(colored(f"Wrong policy type '{policy_type}'", "red"))
            return self.q_des
        if base_lin_acc is None:
            raise ValueError("policy='velocity' needs base_lin_acc; it consumes the accelerometer "
                             "and estimates the base velocity from it")

        self.policy.select(variant)
        if not self._velocity_started:
            # The history has to be seeded before the first inference, and this class has no reset
            # hook of its own, so it happens on the first call.
            self.policy.reset(q, qd, base_lin_acc, base_ang_vel, pj_gravity,
                              velocity_cmd=self.velocity_cmd)
            self._velocity_started = True

        self.q_des = self.policy.step(q, qd, base_lin_acc, base_ang_vel, pj_gravity,
                                      self.velocity_cmd)
        self.prev_action = self.policy.prev_action
        self.decimation_counter = self.policy.inference_count

        if self.policy.has_estimator:
            estimate = self.policy.estimated_base_lin_vel
            message = Vector3()
            message.x, message.y, message.z = estimate
            # The same topic the legacy estimator published on, so anything plotting it still works.
            self.pub_se_nn_base_lin_vel.publish(message)
        if self.debug and base_lin_vel is not None:
            truth = Vector3()
            truth.x, truth.y, truth.z = base_lin_vel
            self.pub_gt_base_lin_vel.publish(truth)

        return self.q_des
