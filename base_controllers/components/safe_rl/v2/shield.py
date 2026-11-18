"""
@author: riccardobussola

The v2 safety filter, on the policy's own observation (228 floats, rl_quadruped/velocity_policy.py)
at every inference, before the network runs (VelocityPolicy.on_observation): the decision takes
effect on this very policy step, so the filter adds no latency.

Two models, exported by train.py as ONNX with the normalisation inside, input "obs":

  mps   Q(o, delta): the chance that the safe policy recovers (reaches X_inv without leaving X_safe)
        when the walking policy acts for delta more policy steps and then hands over.  Input: the
        observation and delta / DELTA_MAX.  The rule is model predictive shielding: the walking
        policy acts now only if the safe policy can still recover after it, Q(o, 1) >= threshold;
        else the safe policy acts now.  delta > 1 adds a lookahead (a latency on the robot).
  paper V(o): the paper's value (Sec. II-B), the chance that the safe policy recovers when it takes
        over at o.  The walking policy acts while V(o_k) >= threshold, the threshold being Sec. II-C's
        V_thr + L (calibrate.py --paper).
  abs   the safety score -V_RA, ABS's reach-avoid value of the walking policy negated (He et al.,
        RSS 2024): the walking policy acts while -V_RA > threshold; ABS's rule V_RA >= -0.05
        switches, a threshold of 0.05.

Once switched, the safe policy keeps the robot (the protocol of the study).
"""

import json
import os

import numpy as np
import onnxruntime as ort

DELTA_MAX = 10  # policy steps (0.2 s at 50 Hz), the largest delta the mps model is trained on
# The deployed filters, VF_<abs|mps>_<robot>.onnx and .json: the study's final models
# (reports/v2/models), loaded by rl_quadruped_controller.py --safety-filter.
FILTERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "filters")


def default_model(robot_name, method):
    return os.path.join(FILTERS_DIR, f"VF_{method}_{robot_name}.onnx")


class Shield:
    """rule (mps only): "mps", switch when Q(o, delta) < threshold; "improve", switch when also
    Q(o, 0) - Q(o, DELTA_MAX) > eps, i.e. handing over now raises the chance of recovery by more than
    eps over handing over DELTA_MAX steps later.  Classic MPS switches when the next state is no longer
    recoverable, the current one being so; once both are below the threshold that premise is gone,
    and switching is only worth it if it improves the odds."""

    def __init__(self, model_path, threshold=None, delta=None, rule="mps", eps=0.05):
        with open(os.path.splitext(model_path)[0] + ".json") as f:
            self.info = json.load(f)
        self.method = self.info["method"]
        self.threshold = float(self.info["threshold"] if threshold is None else threshold)
        self.delta = int(self.info.get("delta", 1) if delta is None else delta)
        self.rule, self.eps = rule, float(eps)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
        obs_dim = int(self.info["obs_dim"])
        rows = 3 if self.method == "mps" and rule == "improve" else 1  # paper / paper_mc: V(o), no delta input
        self._x = np.zeros((rows, obs_dim + (1 if self.method == "mps" else 0)), dtype=np.float32)
        if self.method == "mps":
            self._x[:, -1] = np.array([self.delta, 0, DELTA_MAX][:rows]) / DELTA_MAX
        self.reset()

    def reset(self):
        self.walking = True
        self.value = np.nan

    def scores(self, obs):
        self._x[:, :len(obs)] = obs
        return self.session.run(None, {"obs": self._x})[0].ravel()

    def score(self, obs):
        return float(self.scores(obs)[0])

    def walk(self, obs):
        """Whether the walking policy may act on this observation; False once it has switched."""
        q = self.scores(obs)
        self.value = float(q[0])
        if self.walking and self.value < self.threshold:
            if self.rule != "improve" or q[1] - q[2] > self.eps:
                self.walking = False
        return self.walking
