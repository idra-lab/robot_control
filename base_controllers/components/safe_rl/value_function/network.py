"""
@author: riccardobussola

The value function's network in flax, for the tools on the training side: train_value_function.py
saves its parameters (.pkl), export_onnx.py turns them into the ONNX the safety filter runs
(value_function_manager.py), and evaluate_value_function.py reads either.

Input: 30 values, projected gravity and angular velocity in the base frame, joint positions, joint
velocities (quadruped.q order), normalised with the training data's mean and std.  Output: V, the
chance that the walking policy recovers.
"""

import pickle

import flax.linen as nn
import jax.numpy as jnp
import numpy as np


class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        # Output initialized to 1 (probability of survival)
        x = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.ones)(x)
        return x.squeeze(-1)


def normalize_inputs(obss, mean, std):
    return (obss - mean) / (std + 1e-8)


def load_model(path):
    """A model saved by train_value_function.py: {'model_params', 'mean', 'std'}."""
    with open(path, "rb") as f:
        return pickle.load(f)


def value(model, states):
    """V of states (n x 30) with a loaded model."""
    states = normalize_inputs(jnp.asarray(states, jnp.float32), model["mean"], model["std"])
    return np.asarray(CriticNetwork().apply(model["model_params"], states))
