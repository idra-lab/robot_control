# ====================================
#         TD-Learning for Safety-V(x)
# ====================================
"""
    python3 train_value_function.py [--data <dataset.npy> ...] [--model <model.pkl>] [--epochs N]

Several datasets are pooled, e.g. runs with different seeds (saferl_controller_dataset.py).  Every
transition of every episode is a training sample; evaluate_value_function.py checks the result.
"""

import argparse
import os
import time
import copy
import math
import pickle
import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="TD learning of the safety value function")
parser.add_argument("--data", nargs="+", default=[os.path.join(BASE_DIR, "observation_datasets/observations_dataset_abs_test.npy")])
parser.add_argument("--model", default=os.path.join(BASE_DIR, "models/VF_abs_test.pkl"))
parser.add_argument("--epochs", type=int, default=1000)
args = parser.parse_args()
data_paths = [os.path.abspath(path) for path in args.data]
model_path = os.path.abspath(args.model)
os.chdir(BASE_DIR)   # force working directory to match script directory
print("Working directory set to:", BASE_DIR)

# ====================================
#      Environment Configuration
# ====================================
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "True"

load_parameters = False

# ====================================
#       Define Neural Network Entire model is in Flax, JAX's neural network library
# ====================================
class ValueNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Activation is ELU for smoother gradients.
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
        # x = nn.Dense(1)(x)
        return x.squeeze(-1)


# ====================================
#           Loss Function
# ====================================
""" def loss_fn(params, batch_states, batch_next_states, batch_fallen, batch_cp, target_params):
    # loss_fn computes the TD-target and returns MSE loss:
    loss = (V(s) - target)^2
    target = (1 - fallen) * [ (1 - cp)*V(next_s) + cp ] 
"""
def loss_fn(params, batch_states, batch_next_states, batch_fallen, batch_cp, target_params):
    #forward pass
    #eval value function function of the state
    V_s = model.apply(params, batch_states)
    #compute the value estimate V(x′) of the next state
    V_next = jax.lax.stop_gradient(model.apply(target_params, batch_next_states))
    indicators = 1.0 - batch_fallen #0 if fallen  I do not continue
    #TD target computation (batch_cp: probability of capture now)
    target = indicators * ((1-batch_cp)*V_next + batch_cp)
    # jax.debug.print("target: {x}", x=target)
    # target = indicators * V_next
    #compute mean squared error
    return jnp.mean(jnp.square(V_s - target))

@jax.jit
def train_step(state, batch_states, batch_next_states, batch_fallen, batch_cp, target_params):
    #loss = loss_fn(params, batch_states, batch_next_states, batch_fallen, batch_cp, target_params)
    loss, grads = jax.value_and_grad(loss_fn)(state.params, batch_states, batch_next_states, batch_fallen, batch_cp, target_params)
    # optimizer step to update params, Takes the gradients, Updates the parameters using the optimizer (params - learning_rate * Adam_update(grads))
    # replaces the old state with a new one containing the updated model (states.params)
    # Returns a new TrainState with updated values
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def update_target_params(target_params, params, tau):
    return jax.tree_util.tree_map(lambda t, p: tau * p + (1 - tau) * t, target_params, params)

# ====================================
#          Batch Sampling
# ====================================
def closest_power_of_2(n):
    return 2 ** (n.bit_length() - 1)

def get_batches(states, next_states, fallen, capt_p, batch_size, rng):
    """The transitions shuffled into minibatches: (num_batches, batch_size, ...) per variable."""
    batch_size = closest_power_of_2(batch_size)
    num_samples = states.shape[0]

    # Ensure at least one batch
    num_batches = max(1, num_samples // batch_size)

    rng, subkey = jax.random.split(rng)
    total_needed = num_batches * batch_size

    # If dataset is too small, repeat indices
    perm = jax.random.permutation(subkey, num_samples)
    if total_needed > num_samples:
        # Tile the permutation instead of causing slicing errors
        repeats = math.ceil(total_needed / num_samples)
        perm = jnp.tile(perm, repeats)

    indices = perm[:total_needed].reshape(num_batches, batch_size)
    batches = (states[indices], next_states[indices], fallen[indices], capt_p[indices])
    return batches, rng


# ====================================
#         State Normalization
# ====================================
def normalize_states(states):
    mean = np.mean(states, axis=0)
    std = np.std(states, axis=0)
    return (states - mean) / (std + 1e-8), mean, std

# ====================================
#            Load Dataset
# ====================================
input_dim = 30 # obst, obs_t+1, fallen, captured

def load_transitions(path):
    """The transitions of a dataset, one per row, and whether the episode of each fell.  A dataset
    row is an episode, zero padded to the longest one.  The padding is left out: filled in with the
    episode's last state, it weighed that state hundreds of times over every other."""
    data = np.load(path)
    print("Dataset loaded:", path, data.shape)
    valid = np.any(data[:, :, :input_dim] != 0, axis=2)
    episode_fell = np.any(data[:, :, -2] == 1, axis=1)
    return data[valid], np.repeat(episode_fell, valid.sum(axis=1))

datasets = [load_transitions(path) for path in data_paths]
data = np.concatenate([transitions for transitions, _ in datasets])
episode_fell = np.concatenate([fell for _, fell in datasets])
states = data[:, :input_dim]
next_states = data[:, input_dim:2 * input_dim]
fallen = data[:, -2]
capt_p = data[:, -1]
print(f"Transitions: {len(data)}, labelled fallen {int(fallen.sum())}, labelled success {int(capt_p.sum())}")

states, mean, std = normalize_states(states)
# the next states with the states' statistics: V sees both through the same normalisation
next_states = (next_states - mean) / (std + 1e-8)

# ====================================
#         Convert to JAX tensors
# ====================================
states = jnp.array(states, dtype=jnp.float32)
next_states = jnp.array(next_states, dtype=jnp.float32)
fallen = jnp.array(fallen, dtype=jnp.float32)
capt_p = jnp.array(capt_p, dtype=jnp.float32)

# ====================================
#    Initialize / Load Model Params
# ====================================
# instantiate the model
model = ValueNetwork()
if load_parameters:
    try:
        with open(model_path, 'rb') as f:
            loaded = pickle.load(f)
            params, mean, std = loaded['model_params'], loaded['mean'], loaded['std']
        print("Loaded model parameters.")
    except FileNotFoundError:
        print("Model file not found. Starting from scratch.")
else:
    key = jax.random.PRNGKey(42)  # seed random key for any operation involving randomness (weight initialization, random sampling, noise injection, etc.).
    # In Flax, models are defined as pure functions and do not carry parameters inside them.
    # Parameters are stored outside the model.
    params = model.init(key, jnp.ones((1, input_dim)))  # runs the model once with the given input,but instead of producing outputs, it creates the network’s weights.
    # params It becomes a dictionary-like structure containing all initialized weights:

target_params = copy.deepcopy(params)

# ====================================
#       Optimizer & Train State
# ====================================
learning_rate = 3e-4
tau = 0.01
optimizer = optax.adam(learning_rate)

class TrainState(train_state.TrainState):
    pass

state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

# ====================================
#           Training Loop
# ====================================
epochs = args.epochs
batch_size = 512
rng = jax.random.PRNGKey(int(time.time()))
losses, min_losses, max_losses = [], [], []

with tqdm(range(epochs), desc="Epoch") as pbar:
    #for each training epoch
    for epoch in pbar:
        start_time = time.time()
        rng, subkey = jax.random.split(rng)
        #Sample batches
        batches, rng = get_batches(states, next_states, fallen, capt_p, batch_size, subkey)

        #Compute loss + gradients (JIT-compiled)
        epoch_loss = 0
        batch_losses = []
        # batches is a tuple of four big tensors, one per variable:
        # batches = (
        #     all_batch_states,  # shape: (num_batches, batch_size, dim)
        #     all_batch_next_states,  # shape: (num_batches, batch_size, dim)
        #     all_batch_fallen,  # shape: (num_batches, batch_size)
        #     all_batch_cp  # shape: (num_batches, batch_size)
        # )
        #Loop over minibatches, over every batch inside the epoch.
        for batch_states, batch_next_states, batch_fallen, batch_cp in zip(*batches):
            #produces one batch at a time: batch_states: shape (batch_size, dim)...
            # Perform one gradient update
            state, loss = train_step(state, batch_states, batch_next_states, batch_fallen, batch_cp, target_params)
            #Update target network, target_params = tau * new_params + (1 - tau) * old_target_params, This is Polyak averaging (soft target update), which stabilizes TD learning.
            target_params = update_target_params(target_params, state.params, tau)
            #Accumulate total epoch loss
            epoch_loss += loss
            batch_losses.append(loss)
        #Compute average epoch loss across minibatches
        number_of_minibatches=batches[0].shape[0]
        epoch_avg = epoch_loss / number_of_minibatches
        # Store training statistics
        losses.append(epoch_avg)
        #minimum batch loss
        min_losses.append(np.min(batch_losses))
        #maximum batch loss
        max_losses.append(np.max(batch_losses))
        #Update progress bar
        pbar.set_postfix({"Loss": f"{epoch_avg:.2e}", "Hz": f"{1 / (time.time() - start_time):.2f}"})

# ====================================
#           Save Model
# ====================================
with open(model_path, 'wb') as f:
    pickle.dump({'model_params': state.params, 'mean': mean, 'std': std}, f)
print("Model saved at", model_path)

# ====================================
#       Value Estimate Statistics
# ====================================
# V(x) of every state, grouped by how its episode ended: the states of a falling episode, from the
# push to the fall, against those of the episodes that recovered (evaluate_value_function.py has
# the details, e.g. how long before a fall V drops)
v_est = np.asarray(model.apply(state.params, states))
print("\n===== Value Estimate Statistics =====")
for name, mask in (("Episodes that fell", episode_fell), ("Episodes that recovered", ~episode_fell)):
    print(f"{name:24s}: {mask.sum()} states, mean V(x) {v_est[mask].mean():.4f}, std {v_est[mask].std():.4f}, "
          f"V(x) < 0.5 in {np.mean(v_est[mask] < 0.5):.1%}")
print("=======================================")
