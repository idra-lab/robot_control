"""
@author: riccardobussola

Exports a value function trained by train_value_function.py (.pkl, flax) to ONNX, the format the
safety filter runs (value_function_manager.py), in simulation and on the robot.

    python3 export_onnx.py <model.pkl> [<model.onnx>]        (default: next to the .pkl, .onnx)

The input normalisation is inside the graph: input "obs" (n x 30: projected gravity, angular
velocity, joint positions, joint velocities), output "value" (n).  The export fails if ONNX and JAX
disagree by more than TOLERANCE on states drawn around the training data.  It needs torch and the
onnx package (pip3 install onnx); train_value_function.py calls it after every training.
"""

import argparse
import os

import numpy as np

STATE = 30
TOLERANCE = 1e-4
LAYER_NORM_EPS = 1e-6   # flax's default; torch's and ONNX's is 1e-5
CHECK_STATES = 2000


def export_onnx(pkl_path, onnx_path=None):
    import torch
    import onnxruntime as ort
    from network import load_model, value

    onnx_path = onnx_path or os.path.splitext(pkl_path)[0] + ".onnx"
    model = load_model(pkl_path)
    params = model["model_params"]["params"]
    layers = []
    n_dense = len([k for k in params if k.startswith("Dense_")])
    for i in range(n_dense):
        kernel, bias = np.asarray(params[f"Dense_{i}"]["kernel"]), np.asarray(params[f"Dense_{i}"]["bias"])
        dense = torch.nn.Linear(*kernel.shape)
        dense.weight.data = torch.tensor(kernel.T, dtype=torch.float32)
        dense.bias.data = torch.tensor(bias, dtype=torch.float32)
        layers.append(dense)
        if f"LayerNorm_{i}" in params:
            norm = torch.nn.LayerNorm(kernel.shape[1], eps=LAYER_NORM_EPS)
            norm.weight.data = torch.tensor(np.asarray(params[f"LayerNorm_{i}"]["scale"]), dtype=torch.float32)
            norm.bias.data = torch.tensor(np.asarray(params[f"LayerNorm_{i}"]["bias"]), dtype=torch.float32)
            layers += [norm, torch.nn.ELU()]

    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.tensor(np.asarray(model["mean"]), dtype=torch.float32))
            self.register_buffer("std", torch.tensor(np.asarray(model["std"]), dtype=torch.float32))
            self.net = torch.nn.Sequential(*layers)

        def forward(self, obs):
            return self.net((obs - self.mean) / (self.std + 1e-8)).squeeze(-1)

    critic = Critic().eval()
    torch.onnx.export(critic, torch.zeros(1, STATE), onnx_path, input_names=["obs"], output_names=["value"],
                      dynamic_axes={"obs": {0: "batch"}, "value": {0: "batch"}}, opset_version=17)

    rng = np.random.default_rng(0)
    states = (np.asarray(model["mean"]) + np.asarray(model["std"]) * rng.standard_normal((CHECK_STATES, STATE)))
    states = states.astype(np.float32)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    diff = float(np.max(np.abs(session.run(None, {"obs": states})[0] - value(model, states))))
    if diff > TOLERANCE:
        os.remove(onnx_path)
        raise RuntimeError(f"ONNX and JAX disagree by {diff:.1e} > {TOLERANCE}: {onnx_path} not written")
    print(f"ONNX export {onnx_path}: max |V_onnx - V_jax| {diff:.1e} on {CHECK_STATES} states")
    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pkl")
    parser.add_argument("onnx", nargs="?")
    args = parser.parse_args()
    export_onnx(args.pkl, args.onnx)
