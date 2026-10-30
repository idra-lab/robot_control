import os

import numpy as np
import onnxruntime as ort

# The model and threshold per robot: models/VF_push_end_<robot>.onnx, trained on the push dataset of
# 2026-10-29 (value_function_report.md) and exported by export_onnx.py.  Aliengo at 0.5: falls from
# 48 % to 8 % on the same 105 pushes in Gazebo.
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_THRESHOLD = {"aliengo": 0.5, "go2": 0.5}
STATE = 30


def default_model(robot_name):
    return os.path.join(MODELS_DIR, f"VF_push_end_{robot_name}.onnx")


class ValueFunctionManager:
    """The safety filter.  V(x) is the chance that the walking policy recovers from the current state
    with no further push (train_value_function.py); the filter switches to the safe policy when it
    drops below the threshold and back when it has recovered.

    computeValueFnc returns (VF, V): VF True means the walking policy may run, False that the safe
    policy has to.  It switches to the safe policy after min_switch evaluations in a row below the
    threshold, and back after min_back in a row above it.  The counts are evaluations, so the times
    depend on the caller's rate: at 100 Hz the defaults are 0.01 s and 2 s.

    The value function runs as ONNX (export_onnx.py), on onnxruntime with one thread like the
    policies, in simulation and on the robot; training and its tools use JAX (network.py)."""

    def __init__(self, robot_name="aliengo", model_path=None, threshold=None, min_switch=1, min_back=200):
        model_path = default_model(robot_name) if model_path is None else model_path
        threshold = DEFAULT_THRESHOLD[robot_name] if threshold is None else threshold
        if not model_path.endswith(".onnx"):
            raise ValueError(f"{model_path}: the safety filter runs ONNX; export the model with export_onnx.py")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        self._obs = np.zeros((1, STATE), dtype=np.float32)
        print(f"Loaded value function {model_path}, threshold {threshold}")
        self.threshold = threshold
        self.min_switch = min_switch
        self.min_back = min_back
        self.backup_trigger_counter = 0
        self.reset()

    def reset(self):
        """Back to the walking policy, e.g. after the robot is reset."""
        self.VF = True
        self.count = 0
        self.count_back = 0

    def value(self, proj_gravity, body_ang_vel, joint_pos, joint_vel):
        """V(x) of the state, in the base frame: projected gravity, angular velocity, joint positions
        and velocities in the order of quadruped.q (the normalisation is inside the ONNX graph)."""
        # dimension of observation vector = 3+3+12+12=30
        self._obs[0, 0:3] = proj_gravity
        self._obs[0, 3:6] = body_ang_vel
        self._obs[0, 6:18] = joint_pos
        self._obs[0, 18:30] = joint_vel
        return float(self.session.run(None, {"obs": self._obs})[0][0])

    def computeValueFnc(self, proj_gravity, body_ang_vel, joint_pos, joint_vel, threshold=None, vf_additional_term=0.0):
        threshold = self.threshold if threshold is None else threshold
        V_safe = self.value(proj_gravity, body_ang_vel, joint_pos, joint_vel)
        if V_safe - vf_additional_term > threshold and self.VF:
            self.count = 0
            self.VF = True
        elif self.VF:
            self.count += 1
            if self.count >= self.min_switch:
                self.VF = False
                self.backup_trigger_counter += 1
                print(f"\033[91mV {V_safe:.3f} < {threshold}: safe policy (switch {self.backup_trigger_counter})\033[0m")
        elif V_safe - vf_additional_term > threshold and not self.VF:
            self.count_back += 1
            if self.count_back >= self.min_back:
                print(f"\033[92mV above {threshold} for {self.min_back} evaluations: walking policy\033[0m")
                self.VF = True
                self.count_back = 0
                self.count = 0
        elif not self.VF:
            self.count_back = 0

        return self.VF, V_safe
