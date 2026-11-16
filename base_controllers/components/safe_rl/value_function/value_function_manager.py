import json
import os

import numpy as np
import onnxruntime as ort

# The model per robot: models/VF_<name>_<robot>.onnx, exported by export_onnx.py, and next to it
# VF_<name>_<robot>.json with the threshold, margin and switching count calibrate_filter.py computed
# for it.  The default is the one with the fewest failures (leaving the safe set: falls, knees, joint
# limits) in the closed loop of 2026-11-05 (reports/v1/value_function_comparison.md): abs, ABS's reach-avoid
# value of the walking policy, on both robots (Aliengo 17.5 %, Go2 18.8 %, against 45 % and 47 %
# without the filter); mps, the paper's (the safe policy's chance of reaching rest), left 23 % and
# 31 %: the safe policies themselves often exceed the knees' velocity limit while recovering.
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_MODEL = {"aliengo": "abs", "go2": "abs"}
DEFAULT_THRESHOLD = {"aliengo": 0.5, "go2": 0.5}  # without a .json
STATE = 30


def default_model(robot_name, name=None):
    return os.path.join(MODELS_DIR, f"VF_{name or DEFAULT_MODEL.get(robot_name, 'mps')}_{robot_name}.onnx")


def model_settings(model_path):
    """The .json next to a model (calibrate_filter.py), {} without one."""
    path = os.path.splitext(model_path)[0] + ".json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


class ValueFunctionManager:
    """The safety filter, model predictive shielding with a value function (the paper, Sec. II-B,
    II-C).  For the paper's value function (mps), V(x) is the chance that the safe policy, started at
    x, brings the robot to rest without a fall; the walking policy runs while V(x) >= threshold +
    margin, the margin L of Eq. (5) covering the change of V before the next evaluation.  For the
    ABS baseline (abs), V(x) is about the walking policy itself, and the margin is 0.

    computeValueFnc returns (VF, V): VF True means the walking policy may run, False that the safe
    policy has to.  It switches to the safe policy after min_switch evaluations in a row below the
    threshold (plus margin), and back after min_back in a row above it (never with min_back inf, as
    in the paper's simulations).  The counts are evaluations, so the times depend on the caller's
    rate: at 100 Hz, min_switch 3 is 0.03 s and min_back 200 is 2 s.

    threshold, margin and min_switch default to the model's .json (calibrate_filter.py), else to
    DEFAULT_THRESHOLD, 0 and 1.  For the ABS baseline the model outputs the score -V_RA, so the same
    rule applies (ABS's own: recovery when V_RA >= -0.05, a threshold of 0.05).  The value
    function runs as ONNX (export_onnx.py), on onnxruntime with one thread like the policies, in
    simulation and on the robot; training and its tools use JAX (network.py)."""

    def __init__(self, robot_name="aliengo", model_path=None, threshold=None, margin=None, min_switch=None,
                 min_back=200):
        model_path = default_model(robot_name) if model_path is None else model_path
        if not model_path.endswith(".onnx"):
            raise ValueError(f"{model_path}: the safety filter runs ONNX; export the model with export_onnx.py")
        settings = model_settings(model_path)
        if threshold is None:
            threshold = settings.get("threshold", DEFAULT_THRESHOLD[robot_name])
        if margin is None:
            # the margin is the paper's (mps); the ABS baseline switches on the threshold alone
            margin = settings.get("margin", 0.0) if settings.get("method") != "abs" else 0.0
        if min_switch is None:
            min_switch = settings.get("min_switch", 1)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        self._obs = np.zeros((1, STATE), dtype=np.float32)
        print(f"Loaded value function {model_path}, threshold {threshold}, margin {margin}, min_switch {min_switch}")
        self.model_path = model_path
        self.threshold = threshold
        self.margin = margin
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

    def computeValueFnc(self, proj_gravity, body_ang_vel, joint_pos, joint_vel, threshold=None, vf_additional_term=None):
        threshold = self.threshold if threshold is None else threshold
        # the margin: the walking policy runs while V >= threshold + L
        vf_additional_term = self.margin if vf_additional_term is None else vf_additional_term
        V_safe = self.value(proj_gravity, body_ang_vel, joint_pos, joint_vel)
        if V_safe - vf_additional_term > threshold and self.VF:
            self.count = 0
            self.VF = True
        elif self.VF:
            self.count += 1
            if self.count >= self.min_switch:
                self.VF = False
                self.backup_trigger_counter += 1
                print(f"\033[91mV {V_safe:.3f} < {threshold} + {vf_additional_term:.3f}: safe policy "
                      f"(switch {self.backup_trigger_counter})\033[0m")
        elif V_safe - vf_additional_term > threshold and not self.VF:
            self.count_back += 1
            if self.count_back >= self.min_back:
                print(f"\033[92mV above {threshold} + {vf_additional_term:.3f} for {self.min_back} evaluations: "
                      f"walking policy\033[0m")
                self.VF = True
                self.count_back = 0
                self.count = 0
        elif not self.VF:
            self.count_back = 0

        return self.VF, V_safe
