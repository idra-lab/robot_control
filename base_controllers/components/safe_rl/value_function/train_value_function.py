# ====================================
#         TD-Learning for Safety-V(x)
# ====================================
"""
    python3 train_value_function.py --method abs|mps [--data <dataset.npy> ...] [--val <dataset.npy> ...]
                                    [--model <model.pkl>] [--epochs N] [--x-inv static|polygon]
                                    [--obs-noise S] [--seed N]
                                    [--walking-data <abs dataset.npy> ...] [--walking-weight W]
                                    [--handover-weight W] [--val-every N]

TD(0) on the transitions of the datasets saferl_controller_dataset.py writes, V(x') from a target
network updated by an exponential moving average.  Several datasets are pooled, e.g. runs with
different seeds.

--method mps, the paper's value function, Eq. (4): V'(x) = (1 - I_fall) [(1 - I_CP) V(x') + I_CP],
V(x) the chance that the safe policy reaches the invariant set without leaving the safe set.

--method abs, the ABS baseline as published (He et al., RSS 2024, Eq. 17), on datasets of --method
abs (the walking policy's whole rollouts): the discounted reach-avoid target
    V'(s) = GAMMA_RA max{zeta(s), min{l(s), V(s')}} + (1 - GAMMA_RA) max{l(s), zeta(s)},
GAMMA_RA = 0.999999, with the margins the dataset records: zeta = +1 out of the safe set, -1 in it,
softened on the last RA_SOFTEN samples before a failure to -0.8, -0.6, ..., 1.0; l = tanh(log(e /
sigma)), the walking policy's tracking error against its command (dataset_manager.py).  The episode's
last state is terminal, V(s_T+1) = +inf, so V(s_T) = max{zeta(s_T), l(s_T)}.  V_RA <= 0: the
walking policy reaches its target (tracking again) without a failure.  The model is saved as the
safety score -V_RA (sign in the .pkl, applied in the ONNX export), so that the filter walks while
the score is above its threshold for both methods; ABS's own rule, recovery when V_RA >= -0.05, is a
threshold of 0.05 on the score.

--method mps details (datasets of --method mps, the safe policy from the switch
on): I_CP is the invariant set --x-inv, static equilibrium (default) or the capture point in the
support polygon; the episode ends at its first sample there.  An episode that ends in neither a fall
nor the set (a timeout) has no target for its last state and is left out.

--walking-data (mps): datasets of --method abs, whose rollouts start 1 s into the walking policy:
their samples before the push are undisturbed walking, the states the safe policy recovers from
every time it takes over there (0 failures in the walk handovers of 2026-11-06), yet an mps episode
holds one walking state, the one it hands over at.  Each minibatch adds a minibatch of them with
target 1, weight --walking-weight against the TD loss: supervised, as their next state is the walking
policy's, not the safe policy's.

--handover-weight W (mps): each episode's first transition, the state the safe policy takes over from
(a walking-policy state, the kind the filter evaluates), W times in the training set (an integer; 1
leaves it as recorded: about 1 transition in 140).  The held-out Brier score weighs those states the
same way, so the model kept is the one best on them as much as the training weighs them.

--obs-noise S adds Gaussian noise of S times SENSOR_NOISE to the states fed to V (not to the
targets): V of a measurement learns the value of the state it measures, as on the robot.

With --val, datasets kept out of the training: every 25 epochs V of their states is scored against
the outcome the target stands for (Brier score), and the model saved is the one that scored best: on
the training episodes V keeps improving long after it stops improving on unseen ones.

The model is saved twice: <model>.pkl (flax parameters, for training and its tools) and, through
export_onnx.py, <model>.onnx, the format the safety filter runs (needs torch and the onnx package).
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
parser.add_argument("--val", nargs="*", default=[])
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--method", choices=("abs", "mps"), required=True)
parser.add_argument("--x-inv", choices=("static", "polygon"), default="static", help="mps: the invariant set")
parser.add_argument("--obs-noise", type=float, default=0.0, help="state noise, times SENSOR_NOISE")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--walking-data", nargs="*", default=[], help="mps: abs datasets, their undisturbed walking "
                    "before the push anchored at V = 1")
parser.add_argument("--walking-weight", type=float, default=1.0)
parser.add_argument("--handover-weight", type=int, default=1, help="mps: the handover transitions this many times")
parser.add_argument("--val-every", type=int, default=25, help="epochs between held-out scores")
args = parser.parse_args()
data_paths = [os.path.abspath(path) for path in args.data]
walking_paths = [os.path.abspath(path) for path in args.walking_data]
val_paths = [os.path.abspath(path) for path in args.val]
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
def td_target(columns, V_next):
    """The TD target per transition from its columns: mps (I_fall, I_CP, -, -), Eq. (4) of the paper;
    abs (zeta_t, l_t, last, V of the terminal state), ABS's discounted reach-avoid target."""
    if args.method == "abs":
        zeta, l, last, terminal = columns[:, 0], columns[:, 1], columns[:, 2], columns[:, 3]
        V_next = jnp.where(last > 0, terminal, V_next)
        return GAMMA_RA * jnp.maximum(zeta, jnp.minimum(l, V_next)) + (1 - GAMMA_RA) * jnp.maximum(l, zeta)
    fallen, cp = columns[:, 0], columns[:, 1]
    return (1.0 - fallen) * ((1.0 - cp) * V_next + cp)


def loss_fn(params, batch_states, batch_next_states, batch_columns, target_params, noise, batch_walking,
            walking_noise):
    # loss = (V(s) - target)^2, V on the states with the sensor noise (none without --obs-noise); with
    # --walking-data, plus walking_weight (V(x_walking) - 1)^2
    V_s = model.apply(params, batch_states + noise)
    V_next = jax.lax.stop_gradient(model.apply(target_params, batch_next_states))
    target = td_target(batch_columns, V_next)
    loss = jnp.mean(jnp.square(V_s - target))
    if walking_paths:
        loss = loss + args.walking_weight * jnp.mean(jnp.square(model.apply(params, batch_walking + walking_noise) - 1.0))
    return loss

@jax.jit
def train_step(state, batch_states, batch_next_states, batch_columns, target_params, key, batch_walking):
    key, walking_key = jax.random.split(key)
    noise = noise_std * jax.random.normal(key, batch_states.shape)
    walking_noise = noise_std * jax.random.normal(walking_key, batch_walking.shape)
    loss, grads = jax.value_and_grad(loss_fn)(state.params, batch_states, batch_next_states, batch_columns,
                                              target_params, noise, batch_walking, walking_noise)
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

def get_batches(states, next_states, columns, batch_size, rng):
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
    batches = (states[indices], next_states[indices], columns[indices])
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
# sensor noise for --obs-noise, per state component: projected gravity (about 1 deg), angular velocity
# rad/s, joint positions rad, joint velocities rad/s
SENSOR_NOISE = np.concatenate([np.full(3, 0.02), np.full(3, 0.05), np.full(12, 0.005), np.full(12, 0.1)])
# ABS's reach-avoid training (--method abs)
GAMMA_RA = 0.999999
RA_SOFTEN = np.linspace(-0.8, 1.0, 10)  # zeta of the last 10 states up to a failure

def ra_episode(episode, ra):
    """ABS's targets for one episode: per transition (zeta_t, l_t, last, V of the terminal state) and
    the Monte Carlo reach-avoid value of each state along the episode (the held-out score)."""
    n = len(episode)
    zeta = np.concatenate([ra[:, 0], ra[-1:, 2]])  # the n + 1 states' failure margins
    l = np.concatenate([ra[:, 1], ra[-1:, 3]])
    if zeta[-1] > 0:  # ended in a failure: soften the last states
        k = min(len(RA_SOFTEN), n + 1)
        zeta[-k:] = RA_SOFTEN[-k:]
    terminal = max(zeta[-1], l[-1])
    value = np.empty(n + 1)
    value[-1] = terminal
    for j in range(n - 1, -1, -1):
        value[j] = GAMMA_RA * max(zeta[j], min(l[j], value[j + 1])) + (1 - GAMMA_RA) * max(l[j], zeta[j])
    columns = np.column_stack([zeta[:n], l[:n], np.arange(n) == n - 1, np.full(n, terminal)])
    return columns, value[:n]


def load_transitions(path):
    """The transitions of a dataset, one per row, labelled for --method, and per transition the
    outcome its target stands for (mps: 1 success, 0 failure; abs: the Monte Carlo reach-avoid value,
    as a safety score) and whether its episode failed.  Returns (states, next states, target columns,
    outcomes, failed).  A dataset row is an episode, zero padded to the longest one.  The padding is
    left out: filled in with the episode's last state, it weighed that state hundreds of times over
    every other."""
    data = np.load(path)
    xinv_path = os.path.splitext(path)[0] + "_xinv.npy"
    xinv = np.load(xinv_path) if os.path.exists(xinv_path) else None
    ra_path = os.path.splitext(path)[0] + "_ra.npy"
    ra = np.load(ra_path) if os.path.exists(ra_path) else None
    if args.method == "abs" and ra is None:
        raise SystemExit(f"{path}: no {ra_path}; record the data with saferl_controller_dataset.py --method abs")
    print("Dataset loaded:", path, data.shape)
    rows, columns, outcomes, fell, first, dropped = [], [], [], [], [], 0
    for e, episode in enumerate(data):
        valid = np.any(episode[:, :input_dim] != 0, axis=1)
        episode = episode[valid].copy()
        n = len(episode)
        if args.method == "abs":
            cols, value = ra_episode(episode, ra[e][valid])
            rows.append(episode)
            columns.append(cols)
            outcomes.append(-value)  # the safety score the model outputs
            fell.append(np.full(n, episode[-1, -2] == 1))
            first.append(np.zeros(n, dtype=bool))
            continue
        if args.method == "mps":
            if args.x_inv == "polygon":
                if xinv is None:
                    raise SystemExit(f"{path}: no {xinv_path}, --x-inv polygon needs it")
                episode[:, -1] = xinv[e, :n, 1]
            inside = np.nonzero(episode[:, -1] > 0)[0]
            if len(inside):
                episode = episode[:inside[0] + 1]
            if episode[-1, -2] == 0 and episode[-1, -1] == 0:
                dropped += 1  # a timeout
                continue
            outcome = np.full(len(episode), 1.0 - episode[-1, -2])
        rows.append(episode)
        # (I_fall, I_CP) and two unused columns
        columns.append(np.column_stack([episode[:, -2], episode[:, -1], np.zeros(len(episode)), np.zeros(len(episode))]))
        outcomes.append(outcome)
        fell.append(np.full(len(episode), episode[-1, -2] == 1))
        first.append(np.arange(len(episode)) == 0)  # the handover state
    if dropped:
        print(f"  {dropped} of {len(data)} episodes left out: they ended in neither a failure nor the invariant set")
    data = np.concatenate(rows)
    return (data[:, :input_dim], data[:, input_dim:2 * input_dim], np.concatenate(columns), np.concatenate(outcomes),
            np.concatenate(fell), np.concatenate(first))

def load_walking(path):
    """The undisturbed walking states of an abs dataset: its samples (50 Hz from 1 s, dataset_manager.py
    vf_from) whose transition ends at least a sample before the push starts."""
    data = np.load(path)
    episodes = np.atleast_1d(np.genfromtxt(os.path.splitext(path)[0] + "_episodes.csv", delimiter=",", names=True))
    walking = []
    for ep in episodes[episodes["row"] >= 0]:
        n = int(np.floor((ep["push_time"] - 1.0) / 0.02)) - 1
        rows = data[int(ep["row"]), :max(n, 0), :input_dim]
        walking.append(rows[np.any(rows != 0, axis=1)])
    return np.concatenate(walking)

datasets = [load_transitions(path) for path in data_paths]
states = np.concatenate([d[0] for d in datasets])
next_states = np.concatenate([d[1] for d in datasets])
columns = np.concatenate([d[2] for d in datasets])
episode_fell = np.concatenate([d[4] for d in datasets])
if args.method == "abs":
    print(f"Transitions: {len(states)}, failing states (zeta > 0) {int(np.sum(columns[:, 0] > 0))}, states in the "
          f"target (l <= 0) {int(np.sum(columns[:, 1] <= 0))}, episodes that failed {int(np.sum(columns[:, 2] * (columns[:, 3] > 0)))}")
else:
    print(f"Transitions: {len(states)}, labelled failed {int(columns[:, 0].sum())}, labelled success {int(columns[:, 1].sum())}")

states, mean, std = normalize_states(states)
if walking_paths:
    walking_states = np.concatenate([load_walking(path) for path in walking_paths])
    print(f"Walking states anchored at V = 1: {len(walking_states)}, weight {args.walking_weight}")
    walking_states = jnp.array((walking_states - mean) / (std + 1e-8), dtype=jnp.float32)
else:
    walking_states = jnp.zeros((1, input_dim), dtype=jnp.float32)
# the next states with the states' statistics: V sees both through the same normalisation
next_states = (next_states - mean) / (std + 1e-8)
if val_paths:
    val_sets = [load_transitions(path) for path in val_paths]
    val_states = jnp.array((np.concatenate([d[0] for d in val_sets]) - mean) / (std + 1e-8), dtype=jnp.float32)
    val_outcome = np.concatenate([d[3] for d in val_sets])
    val_first = np.concatenate([d[5] for d in val_sets])
    val_weight = np.where(val_first, args.handover_weight, 1.0)
# the state noise in the normalised space
noise_std = jnp.array(args.obs_noise * SENSOR_NOISE / (std + 1e-8), dtype=jnp.float32)

# --handover-weight: the handover transitions repeated (the normalisation above is the recorded data's)
handover = np.concatenate([d[5] for d in datasets])
if args.handover_weight > 1:
    keep = np.concatenate([np.arange(len(states)), np.repeat(np.flatnonzero(handover), args.handover_weight - 1)])
    states, next_states, columns = states[keep], next_states[keep], columns[keep]
    print(f"Handover transitions: {handover.sum()}, each {args.handover_weight} times: {len(states)} transitions")

# ====================================
#         Convert to JAX tensors
# ====================================
states = jnp.array(states, dtype=jnp.float32)
next_states = jnp.array(next_states, dtype=jnp.float32)
columns = jnp.array(columns, dtype=jnp.float32)
# the model's output: V for mps, the safety score -V_RA for abs (the network learns V_RA, the sign is
# applied after; see SIGN)
SIGN = -1.0 if args.method == "abs" else 1.0

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
    key = jax.random.PRNGKey(args.seed)  # seed random key for any operation involving randomness (weight initialization, random sampling, noise injection, etc.).
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
rng = jax.random.PRNGKey(args.seed)
losses, min_losses, max_losses = [], [], []
best_brier, best_params, best_epoch = np.inf, None, 0

with tqdm(range(epochs), desc="Epoch") as pbar:
    #for each training epoch
    for epoch in pbar:
        start_time = time.time()
        rng, subkey = jax.random.split(rng)
        #Sample batches
        batches, rng = get_batches(states, next_states, columns, batch_size, subkey)

        #Compute loss + gradients (JIT-compiled)
        epoch_loss = 0
        batch_losses = []
        # batches = (states (num_batches, batch_size, dim), next states (same), target columns
        # (num_batches, batch_size, 4)); one gradient update per minibatch
        for batch_states, batch_next_states, batch_columns in zip(*batches):
            rng, noise_key, walking_key = jax.random.split(rng, 3)
            batch_walking = walking_states[jax.random.randint(walking_key, (batch_states.shape[0],), 0, walking_states.shape[0])]
            state, loss = train_step(state, batch_states, batch_next_states, batch_columns, target_params, noise_key,
                                     batch_walking)
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
        # keep the model that scores best on the held-out episodes
        if val_paths and (epoch + 1) % args.val_every == 0:
            # mps: the Brier score of V against the outcome; abs: the squared error of the score against
            # the Monte Carlo reach-avoid value of the held-out episodes
            err = (SIGN * np.asarray(model.apply(state.params, val_states)) - val_outcome) ** 2
            brier = np.average(err, weights=val_weight)
            if brier < best_brier:
                best_brier, best_params, best_epoch = brier, state.params, epoch + 1
            pbar.write(f"epoch {epoch + 1}: held-out Brier score {brier:.4f} (best {best_brier:.4f}, epoch {best_epoch}); "
                       f"on the handover states {np.mean(err[val_first]) if val_first.any() else np.nan:.4f}")

if best_params is not None:
    print(f"Held-out Brier score best at epoch {best_epoch}: {best_brier:.4f}, that model is saved")
    state = state.replace(params=best_params)

# ====================================
#           Save Model
# ====================================
with open(model_path, 'wb') as f:
    pickle.dump({'model_params': state.params, 'mean': mean, 'std': std, 'method': args.method, 'sign': SIGN,
                 'gamma_ra': GAMMA_RA if args.method == "abs" else None,
                 'x_inv': args.x_inv if args.method == "mps" else None, 'obs_noise': args.obs_noise,
                 'walking_data': args.walking_data, 'walking_weight': args.walking_weight if walking_paths else None,
                 'handover_weight': args.handover_weight}, f)
print("Model saved at", model_path)
# the runtime format: the safety filter runs the ONNX export (value_function_manager.py)
from export_onnx import export_onnx
export_onnx(model_path)

# ====================================
#       Value Estimate Statistics
# ====================================
# V(x) of every state, grouped by how its episode ended: the states of a falling episode, up to the
# fall, against those of the episodes that recovered (evaluate_value_function.py has the details,
# e.g. how long before a fall V drops)
v_est = SIGN * np.asarray(model.apply(state.params, states[:len(episode_fell)]))  # the recorded transitions
level = 0.05 if args.method == "abs" else 0.5  # ABS's rule on the score; half the chance for mps
print("\n===== Value Estimate Statistics (the model's output: V, or the score -V_RA) =====")
for name, mask in (("Episodes that failed", episode_fell), ("Episodes that recovered", ~episode_fell)):
    print(f"{name:24s}: {mask.sum()} states, mean {v_est[mask].mean():.4f}, std {v_est[mask].std():.4f}, "
          f"below {level} in {np.mean(v_est[mask] < level):.1%}")
print("=======================================")
