"""
@author: riccardobussola

Trains a v2 safety value function on the records of saferl_controller_dataset.py --v2 (one
v2_dataset.npz and its episodes CSV per batch folder) and exports it for the filter (shield.py):

    python3 train.py --method mps|abs --data <batch folder> ... --out <model stem> [--epochs N]

Input: the policy's observation at an inference (228 floats, the 5-frame history of accelerometer,
gyroscope, gravity, joint positions and velocities and actions, and the command), what the policies
themselves act on.

mps, Q(o, delta), the delayed-switch value.  Q(o_k, delta) is the chance that the safe policy
recovers (reaches X_inv, static and polygon, without leaving X_safe) when the walking policy acts at
inferences k, ..., k + delta - 1 and the safe policy from k + delta on.  An episode of the mps data
hands over at inference h, so each of its walking observations o_s with h - s <= DELTA_MAX is a
sample of Q(., h - s), and o_h and every later (safe policy) observation a sample of Q(., 0), all
with the episode's outcome as the Monte Carlo target (cross-entropy).  Monte Carlo rather than TD(0):
the observation is not the full state, and bootstrapping on it is biased.  An episode that failed
before its handover labels its walking observations 0 at the delta of the scheduled handover.
Walking observations of a pushed episode count from the first inference after the push starts: the
push cannot be seen before, and the walking states would be taught that pushes come.  The filter's
rule, Q(o_k, 1) >= threshold, is model predictive shielding's: the walking policy acts now only if
the safe policy can still recover after it; the default threshold, 0.9, is a probability fixed in
advance (no tuning on the tests).

abs, ABS (He et al., RSS 2024, Eq. 17): the discounted reach-avoid value of the walking policy on
its whole rollouts, by TD with a target network, zeta +1 out of X_safe (softened over the last 10
states before a failure) and l = tanh(log(e / sigma)) the command tracking error (dataset_manager.py
ra_margins).  The export outputs -V_RA; ABS's rule V_RA >= -0.05 switches, a threshold of 0.05.

paper, the paper's V (Sec. II-B, Eq. 2-4): V(o), the chance that the safe policy recovers when it
takes over at o, trained by TD(0) on the safe policy's transitions only (from the handover on), with a
target network updated by an exponential moving average, the squared TD error, and V in [0, 1]
(sigmoid output).  Target 0 at a failure, 1 at X_inv, V'(o_{t+1}) otherwise.  Timeouts are left out
(no label).  The filter's rule is Sec. II-C's: walk while V(o_k) >= V_thr + L, the Lipschitz margin
L and V_thr from calibrate.py --paper.

paper_mc, the same V(o) and samples with Monte Carlo targets: cross-entropy against the episode's
outcome on every safe-policy observation, no bootstrapping (the observation is not the state).

15 % of the episodes are held out; the epoch with the best held-out loss is kept.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

DELTA_MAX = 10
GAMMA_RA = 0.999999
RA_SOFTEN = np.linspace(-0.8, 1.0, 10)
POLICY_DT = 0.02
T, SAFE, FALLEN, STATIC, POLYGON, ZETA, L, TERMINAL = range(8)

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=("mps", "abs", "paper", "paper_mc"), required=True)
parser.add_argument("--data", nargs="+", required=True, help="batch folders")
parser.add_argument("--out", required=True, help="model path without extension (.onnx, .json, .pt)")
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--batch", type=int, default=1024)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--threshold", type=float, default=None, help="default: mps 0.9, abs 0.05")
args = parser.parse_args()
rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)
torch.set_num_threads(min(8, os.cpu_count()))


def load_episodes(folders):
    """[(obs, meta, csv row)] of the kept episodes of every batch folder."""
    episodes = []
    for folder in folders:
        data = np.load(os.path.join(folder, "v2_dataset.npz"))
        csv = np.atleast_1d(np.genfromtxt(os.path.join(folder, "observations_dataset_abs_test_episodes.csv"),
                                          delimiter=",", names=True))
        for row in csv:
            # discarded 2 is also an mps episode that left the safe set after the push but before its
            # handover (the old recorder starts at the handover, so it saw no sample): its walking
            # observations are the "switched too late" samples, Q = 0 at the scheduled delta
            failed_before_handover = (row["discarded"] == 2 and row["fallen"] > 0 and row["end_time"] > row["push_time"]
                                      and row["overruns"] <= 5)
            if row["discarded"] != 0 and not failed_before_handover:
                continue
            idx = data["episode"] == row["episode"]
            if np.any(idx):
                episodes.append((data["obs"][idx], data["meta"][idx], row))
    return episodes


def mps_samples(episode):
    """(observations, delta, target, weight) of one mps episode, or None for a timeout."""
    obs, meta, row = episode
    end = meta[-1]
    if end[FALLEN] > 0:
        outcome = 0.0
    elif end[STATIC] > 0 and end[POLYGON] > 0:
        outcome = 1.0
    else:
        return None
    states, meta = obs[:-1], meta[:-1]  # the terminal row repeats the last observation
    pushed = row["kick"] > 0
    handover = row["backup_time"]  # s after the push start, the scheduled handover
    safe = meta[:, SAFE] > 0
    # deltas of the walking observations: inferences until the first one at or after the handover
    delta = np.ceil((handover - meta[:, T]) / POLICY_DT - 1e-3).astype(int)
    walking = ~safe & (delta >= 1) & (delta <= DELTA_MAX)
    if pushed:
        walking &= meta[:, T] > 0
    x = np.concatenate([states[walking], states[safe]])
    d = np.concatenate([delta[walking], np.zeros(np.sum(safe), dtype=int)])
    w = np.concatenate([np.full(np.sum(walking), 1.0), np.full(np.sum(safe), 0.0)])  # weights set later
    is_walk = np.concatenate([np.ones(np.sum(walking), bool), np.zeros(np.sum(safe), bool)])
    return x, d, np.full(len(x), outcome), is_walk


def abs_transitions(episode):
    """ABS's TD columns per state: (zeta, l, last, terminal value) and the Monte Carlo RA value."""
    obs, meta, _ = episode
    zeta = np.where(meta[:, FALLEN] > 0, 1.0, -1.0)
    l = meta[:, L].astype(float)
    if zeta[-1] > 0:
        k = min(len(RA_SOFTEN), len(zeta))
        zeta[-k:] = RA_SOFTEN[-k:]
    n = len(obs) - 1  # states; the terminal row closes the last one
    if n < 1:
        return None
    terminal = max(zeta[-1], l[-1])
    value = np.empty(n + 1)
    value[-1] = terminal
    for j in range(n - 1, -1, -1):
        value[j] = GAMMA_RA * max(zeta[j], min(l[j], value[j + 1])) + (1 - GAMMA_RA) * max(l[j], zeta[j])
    nxt = np.concatenate([obs[1:n], obs[n - 1:n]])  # the last state's next is unused (terminal)
    cols = np.column_stack([zeta[:n], l[:n], np.arange(n) == n - 1, np.full(n, terminal)])
    return obs[:n], nxt, cols, value[:n]


def paper_transitions(episode):
    """The safe policy's TD(0) transitions of one mps episode: (o_t, o_t+1, last, outcome), from the
    handover on; None for a timeout or an episode that never handed over."""
    obs, meta, row = episode
    end = meta[-1]
    if end[FALLEN] > 0:
        outcome = 0.0
    elif end[STATIC] > 0 and end[POLYGON] > 0:
        outcome = 1.0
    else:
        return None
    safe = np.nonzero(meta[:-1, SAFE] > 0)[0]
    if len(safe) == 0:
        return None
    x = obs[safe]
    nxt = np.concatenate([obs[safe[1:]], obs[safe[-1:]]])
    last, first = np.zeros(len(safe)), np.zeros(len(safe), bool)
    last[-1], first[0] = 1.0, True
    return x, nxt, last, np.full(len(safe), outcome), first


class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        layers, width = [], n_in
        for h in (512, 256, 128):
            layers += [nn.Linear(width, h), nn.LayerNorm(h), nn.ELU()]
            width = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


class Exported(nn.Module):
    """Normalisation, network and output map in one graph: input obs (n x 228, or 229 with delta /
    DELTA_MAX for mps), output the score the filter compares with its threshold."""
    def __init__(self, net, mean, std, method):
        super().__init__()
        self.net, self.method = net, method
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x):
        n = self.mean.shape[0]
        z = (x[:, :n] - self.mean) / self.std
        if self.method == "mps":
            return torch.sigmoid(self.net(torch.cat([z, x[:, n:]], dim=1)))
        if self.method in ("paper", "paper_mc"):
            return torch.sigmoid(self.net(z))
        return -self.net(z)


episodes = load_episodes(args.data)
order = rng.permutation(len(episodes))
n_val = max(1, int(0.15 * len(episodes)))
val_ids, train_ids = set(order[:n_val].tolist()), order[n_val:]
obs_dim = episodes[0][0].shape[1]
print(f"{len(episodes)} episodes ({n_val} held out), observation {obs_dim}")


def stack(ids, fn):
    parts = [p for p in (fn(episodes[i]) for i in ids) if p is not None]
    return [np.concatenate(c) for c in zip(*parts)], len(parts)


if args.method == "mps":
    (x, d, y, walk), kept = stack(train_ids, mps_samples)
    (vx, vd, vy, vwalk), vkept = stack(sorted(val_ids), mps_samples)
    print(f"train episodes kept {kept} (timeouts left out), walking samples {walk.sum()} "
          f"(failed {np.sum(walk & (y == 0))}), safe-policy samples {np.sum(~walk)}")
    mean, std = x.mean(0), x.std(0) + 1e-3
    # the walking samples, the ones the filter is queried on, weigh as much as all the safe policy's
    w = np.where(walk, 1.0 / max(walk.sum(), 1), 1.0 / max((~walk).sum(), 1)) * len(x) / 2

    def features(o, dd):
        return torch.as_tensor(np.column_stack([(o - mean) / std, dd / DELTA_MAX]), dtype=torch.float32)

    X, Y, W = features(x, d), torch.as_tensor(y, dtype=torch.float32), torch.as_tensor(w, dtype=torch.float32)
    VX, VY = features(vx[vwalk], vd[vwalk]), torch.as_tensor(vy[vwalk], dtype=torch.float32)
    net = Net(obs_dim + 1)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    best, best_state = np.inf, None
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), args.batch):
            b = perm[i:i + args.batch]
            loss = (bce(net(X[b]), Y[b]) * W[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(VX))
            val = float(bce(torch.logit(p.clamp(1e-6, 1 - 1e-6)), VY).mean())
            brier = float(((p - VY) ** 2).mean())
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}: held-out walking samples, cross-entropy {val:.4f}, Brier {brier:.4f}")
    net.load_state_dict(best_state)
    # held-out separation at delta = 1, the filter's query
    with torch.no_grad():
        one = vd[vwalk] == 1
        p1 = torch.sigmoid(net(VX[torch.as_tensor(one)])).numpy()
        y1 = vy[vwalk][one]
    if 0 < y1.sum() < len(y1):
        pos, neg = p1[y1 == 1], p1[y1 == 0]
        auc = float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
        print(f"held-out delta = 1: {len(y1)} samples, {int(len(y1) - y1.sum())} failed, AUC {auc:.3f}, "
              f"mean Q recovered {pos.mean():.3f}, failed {neg.mean():.3f}")
    threshold = 0.9 if args.threshold is None else args.threshold
elif args.method in ("paper", "paper_mc"):
    (x, nx, last, out, _), kept = stack(train_ids, paper_transitions)
    (vx, vnx, vlast, vout, vfirst), vkept = stack(sorted(val_ids), paper_transitions)
    print(f"train episodes {kept}, safe-policy transitions {len(x)}, of failed episodes {np.sum(out == 0)}")
    mean, std = x.mean(0), x.std(0) + 1e-3
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    X, NX, LAST, OUT_ = t((x - mean) / std), t((nx - mean) / std), t(last), t(out)
    VX, VOUT = t((vx - mean) / std), t(vout)
    net, target = Net(obs_dim), Net(obs_dim)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best, best_state = np.inf, None
    for epoch in range(args.epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), args.batch):
            b = perm[i:i + args.batch]
            if args.method == "paper_mc":
                loss = nn.functional.binary_cross_entropy_with_logits(net(X[b]), OUT_[b])
            else:
                with torch.no_grad():
                    y = torch.where(LAST[b] > 0, OUT_[b], torch.sigmoid(target(NX[b])))
                loss = ((torch.sigmoid(net(X[b])) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for p, q in zip(target.parameters(), net.parameters()):
                    p.mul_(0.99).add_(0.01 * q)
        with torch.no_grad():
            val = float(((torch.sigmoid(net(VX)) - VOUT) ** 2).mean())  # Brier against the episode outcome
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}: held-out Brier score against the Monte Carlo outcome {val:.4f}")
    net.load_state_dict(best_state)
    with torch.no_grad():
        p1 = torch.sigmoid(net(VX[torch.as_tensor(vfirst)])).numpy()
    y1 = vout[vfirst]
    if 0 < y1.sum() < len(y1):
        pos, neg = p1[y1 == 1], p1[y1 == 0]
        auc = float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
        print(f"held-out handovers: {len(y1)}, {int(len(y1) - y1.sum())} failed, AUC {auc:.3f}, "
              f"mean V recovered {pos.mean():.3f}, failed {neg.mean():.3f}")
        edges = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        k = np.clip(np.digitize(p1, edges) - 1, 0, 4)
        print("held-out handovers, reliability (V bin: n, observed recovery rate): " + ", ".join(
            f"{edges[i]:.1f}-{edges[i + 1]:.1f}: {np.sum(k == i)}, {y1[k == i].mean():.2f}" for i in range(5) if np.any(k == i)))
    threshold = 0.5 if args.threshold is None else args.threshold
else:
    (x, nx, cols, mc), kept = stack(train_ids, abs_transitions)
    (vx, _, _, vmc), _ = stack(sorted(val_ids), abs_transitions)
    print(f"train episodes {kept}, states {len(x)}, failed episodes' states {np.sum(cols[:, 0] > -1)}")
    mean, std = x.mean(0), x.std(0) + 1e-3
    X = torch.as_tensor((x - mean) / std, dtype=torch.float32)
    NX = torch.as_tensor((nx - mean) / std, dtype=torch.float32)
    C = torch.as_tensor(cols, dtype=torch.float32)
    VX, VMC = torch.as_tensor((vx - mean) / std, dtype=torch.float32), torch.as_tensor(vmc, dtype=torch.float32)
    net, target = Net(obs_dim), Net(obs_dim)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best, best_state = np.inf, None
    for epoch in range(args.epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), args.batch):
            b = perm[i:i + args.batch]
            with torch.no_grad():
                v_next = torch.where(C[b, 2] > 0, C[b, 3], target(NX[b]))
                zeta, l = C[b, 0], C[b, 1]
                y = GAMMA_RA * torch.maximum(zeta, torch.minimum(l, v_next)) + (1 - GAMMA_RA) * torch.maximum(l, zeta)
            loss = ((net(X[b]) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for p, q in zip(target.parameters(), net.parameters()):
                    p.mul_(0.99).add_(0.01 * q)
        with torch.no_grad():
            val = float(((net(VX) - VMC) ** 2).mean())
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}: held-out squared error against the Monte Carlo RA value {val:.4f}")
    net.load_state_dict(best_state)
    threshold = 0.05 if args.threshold is None else args.threshold

net.eval()
model = Exported(net, mean, std, args.method).eval()
n_in = obs_dim + (1 if args.method == "mps" else 0)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
torch.onnx.export(model, torch.zeros(1, n_in), args.out + ".onnx", input_names=["obs"], output_names=["value"],
                  dynamic_axes={"obs": {0: "n"}, "value": {0: "n"}}, opset_version=13)
torch.save({"state_dict": model.state_dict(), "method": args.method, "obs_dim": obs_dim}, args.out + ".pt")
with open(args.out + ".json", "w") as f:
    json.dump({"method": args.method, "obs_dim": int(obs_dim), "threshold": threshold, "delta": 1,
               "delta_max": DELTA_MAX, "held_out_loss": best, "data": args.data}, f, indent=2)
print(f"saved {args.out}.onnx (threshold {threshold})")
