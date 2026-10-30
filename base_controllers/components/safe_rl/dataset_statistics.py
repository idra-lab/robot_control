"""
@author: riccardobussola

Statistics of a value-function dataset written by DatasetManager (saferl_controller_dataset.py).

    python3 dataset_statistics.py <dataset.npy> [<dataset.npy> ...] [--robot go2] [--urdf robot.urdf] [--plot out.png]

Several datasets are pooled, e.g. runs with different seeds.  For each one the ``<name>_episodes.csv``
written next to it is read too; without it (older datasets) only the label and observation
statistics are reported.

* Labels: episodes fallen, captured or neither, and the samples of each.
* Commands: their range against the training range (+-0.5 m/s, rad/s), standing and turning shares.
* Pushes: the kick the robot falls from half the time (k50, m/s, logistic fit with a bootstrap 95 %
  interval), overall and per direction in the base frame; the fall rate per kick band; against the
  walking policy's push benchmark, whose push the dataset uses.
* Observations: joint positions against the URDF limits (a reading out of range or 2 pi off means
  states the real robot never shows), joint velocities, projected gravity.

Only numpy is needed; matplotlib for --plot.
"""

import argparse
import os
import xml.etree.ElementTree as ElementTree

import numpy as np

# Walking policy at zero command, push survived half the time, m/s (rl_quadruped/README.md,
# push_sweep.py).  The same 0.2 s force push as the dataset's; its fall rule (|roll| or |pitch| > 1 rad
# or base below half its height) differs from the dataset's (45 deg or base below 0.2 m), but no robot
# past 30 deg (Go2 15 deg) recovers, so both call the same episodes falls.
BENCHMARK_K50 = {"aliengo": {"forward": 1.65, "left": 1.90, "backward": 1.63, "right": 1.96},
                 "go2": {"forward": 1.25, "left": 1.46, "backward": 1.31, "right": 1.54}}
CMD_RANGE = 0.5  # training range of vx, vy, wz
STATE = 30  # projected gravity 3, base angular velocity 3, joint positions 12, joint velocities 12
JOINT_KINDS = ("haa", "hfe", "kfe")
BANDS = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf)


def load(path):
    data = np.load(path)
    csv = os.path.splitext(path)[0] + "_episodes.csv"
    episodes = None
    if os.path.exists(csv):
        episodes = np.genfromtxt(csv, delimiter=",", names=True)
        episodes = np.atleast_1d(episodes)
    return data, episodes


def fit_logistic(x, fell, ridge=1e-2):
    """P(fall) = 1 / (1 + exp(-(x - k50) / scale)) by Newton's method; the small ridge keeps it
    finite when the outcomes are perfectly separated.  Returns (k50, scale)."""
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    penalty = np.diag([0.0, ridge])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -50, 50)))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + penalty, X.T @ (fell - p) - penalty @ w)
        w += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return -w[0] / w[1], 1.0 / w[1]


def k50_with_interval(x, fell, rng, n_boot=1000):
    if len(x) < 4 or fell.all() or not fell.any():
        return np.nan, np.nan, np.nan, np.nan
    k50, scale = fit_logistic(x, fell)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if 0 < fell[i].sum() < len(i):
            boot.append(fit_logistic(x[i], fell[i])[0])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return k50, scale, lo, hi


def direction_of(kick_dir):
    """Direction of a push given in the base frame (rad, 0 forward): forward, left, backward or right."""
    angle = (np.degrees(kick_dir) + 180.0) % 360.0 - 180.0
    return np.select([np.abs(angle) <= 45, (angle > 45) & (angle <= 135), np.abs(angle) > 135],
                     ["forward", "left", "backward"], "right")


def urdf_limits(path):
    """(lower, upper) per joint kind, the union over the legs."""
    limits = {}
    for joint in ElementTree.parse(path).getroot().iter("joint"):
        limit = joint.find("limit")
        kind = next((k for k in JOINT_KINDS if k in joint.get("name", "")), None)
        if kind is None or limit is None or joint.get("type") != "revolute":
            continue
        lo, hi = float(limit.get("lower")), float(limit.get("upper"))
        old = limits.get(kind, (np.inf, -np.inf))
        limits[kind] = (min(old[0], lo), max(old[1], hi))
    return limits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--robot", default="aliengo", choices=sorted(BENCHMARK_K50))
    parser.add_argument("--urdf", help="robot URDF for the joint limits "
                        "(default $LOCOSIM_DIR/robot_urdf/generated_urdf/<robot>.urdf if present)")
    parser.add_argument("--plot", help="save fall rate and kick histogram to this file")
    args = parser.parse_args()
    benchmark = BENCHMARK_K50[args.robot]
    rng = np.random.default_rng(0)

    rows, episodes = [], []
    for path in args.datasets:
        data, eps = load(path)
        rows.append(data.reshape(-1, data.shape[-1]))
        if eps is not None:
            episodes.append(eps)
        print(f"{path}: {data.shape[0]} rows x {data.shape[1]} steps x {data.shape[2]}, "
              f"episodes file {'yes' if eps is not None else 'no'}")
    samples = np.concatenate(rows)
    samples = samples[np.any(samples != 0, axis=1)]          # drop the zero padding
    states = np.concatenate([samples[:, :STATE], samples[:, STATE:2 * STATE]])

    print(f"\nSamples: {len(samples)}, labelled fallen {int(samples[:, -2].sum())}, "
          f"success (captured or survived) {int(samples[:, -1].sum())}")

    if episodes:
        ep = np.concatenate(episodes)
        kept = ep["discarded"] == 0
        print(f"Episodes: {len(ep)}, discarded {int((~kept).sum())} (joint range "
              f"{int((ep['discarded'] == 1).sum())}, fell before the push {int((ep['discarded'] == 2).sum())}, "
              f"control ticks missed {int((ep['discarded'] == 3).sum())})")
        ep = ep[kept]
        fallen, captured, survived = ep["fallen"] > 0, ep["captured"] > 0, ep["survived"] > 0
        neither = ~fallen & ~captured & ~survived
        print(f"  fallen {fallen.sum()}, captured {captured.sum()}, survived {survived.sum()}"
              + (f", ended without a label {neither.sum()}" if neither.any() else ""))
        after_push = ep["end_time"] - ep["push_time"]
        for name, m in (("fallen", fallen), ("captured", captured), ("survived", survived)):
            if m.any():
                print(f"  {name:8s} episodes end {np.median(after_push[m]):.2f} s after the push (median)")

        cmd = np.column_stack([ep["cmd_vx"], ep["cmd_vy"], ep["cmd_wz"]])
        standing = np.all(cmd == 0, axis=1)
        turning = ~standing & np.all(cmd[:, :2] == 0, axis=1)
        outside = np.any(np.abs(cmd) > CMD_RANGE + 1e-6, axis=1)
        print(f"\nCommands: vx {cmd[:, 0].min():+.2f}..{cmd[:, 0].max():+.2f}, vy {cmd[:, 1].min():+.2f}.."
              f"{cmd[:, 1].max():+.2f}, wz {cmd[:, 2].min():+.2f}..{cmd[:, 2].max():+.2f}; standing "
              f"{standing.mean():.0%}, only turning {turning.mean():.0%}; outside the training range "
              f"+-{CMD_RANGE}: {outside.sum()}")

        kick = ep["kick"]
        direction = direction_of(ep["kick_dir"])
        print("\nPush: kick = body mass x velocity change of the push force, m/s; k50 = the kick fallen from "
              "half the time [95 %]")
        for name, m in (("all", np.ones(len(ep), bool)), ("standing", standing), ("walking", ~standing)):
            k50, scale, lo, hi = k50_with_interval(kick[m], fallen[m].astype(float), rng)
            print(f"  {name:8s} : k50 {k50:.2f} [{lo:.2f}, {hi:.2f}], scale {scale:.2f}, {m.sum():3d} episodes")
        for name, bench in benchmark.items():
            for label, m in (("", direction == name), (", standing", (direction == name) & standing)):
                k50, scale, lo, hi = k50_with_interval(kick[m], fallen[m].astype(float), rng)
                print(f"  {name + label:18s}: k50 {k50:.2f} [{lo:.2f}, {hi:.2f}], {m.sum():3d} episodes   "
                      f"(benchmark, standing: {bench:.2f})")
        print("  fall rate per kick band:")
        for lo_, hi_ in zip(BANDS[:-1], BANDS[1:]):
            m = (kick >= lo_) & (kick < hi_)
            if m.any():
                print(f"    {lo_:.1f}-{hi_:.1f} m/s: {m.sum():3d} episodes, fallen {fallen[m].mean():.0%}")

    urdf = args.urdf or os.path.join(os.environ.get("LOCOSIM_DIR", ""), "robot_urdf", "generated_urdf",
                                     args.robot + ".urdf")
    limits = urdf_limits(urdf) if os.path.exists(urdf) else {}
    q, qd = states[:, 6:18], states[:, 18:30]
    print("\nJoint positions, rad (min..max over the legs)" + ("" if limits else ", no URDF for the limits"))
    for k, kind in enumerate(JOINT_KINDS):
        lo, hi = q[:, k::3].min(), q[:, k::3].max()
        line = f"  {kind}: {lo:+.2f}..{hi:+.2f}"
        if kind in limits:
            out = ((q[:, k::3] < limits[kind][0] - 0.5) | (q[:, k::3] > limits[kind][1] + 0.5)).any(axis=1)
            line += f"   limits {limits[kind][0]:+.2f}..{limits[kind][1]:+.2f}, samples past them by > 0.5: {out.sum()}"
        print(line)
    print(f"Joint velocities: max |qd| {np.abs(qd).max():.1f} rad/s, 99th percentile {np.percentile(np.abs(qd), 99):.1f}")
    gz = states[:, 2]
    print(f"Projected gravity z: {gz.min():+.2f}..{gz.max():+.2f} (-1 upright), "
          f"{(gz > 0).mean():.1%} of states upside down")
    if not np.all(np.isfinite(samples)):
        print("WARNING: the dataset contains NaN or inf")

    if args.plot and episodes:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        centers, rates = [], []
        for lo_, hi_ in zip(np.arange(0, 4.5, 0.25), np.arange(0.25, 4.75, 0.25)):
            m = (kick >= lo_) & (kick < hi_)
            if m.sum() >= 2:
                centers.append((lo_ + hi_) / 2)
                rates.append(fallen[m].mean())
        ax1.plot(centers, rates, "o", label="dataset, per 0.25 m/s")
        k50, scale = fit_logistic(kick, fallen.astype(float))
        x = np.linspace(0, kick.max(), 200)
        ax1.plot(x, 1 / (1 + np.exp(-(x - k50) / scale)), label=f"fit, k50 {k50:.2f} m/s")
        ax1.axvspan(min(benchmark.values()), max(benchmark.values()), alpha=0.2, color="gray",
                    label="benchmark k50, standing")
        ax1.set_xlabel("kick, m/s")
        ax1.set_ylabel("fall rate")
        ax1.legend()
        bins = np.arange(0, kick.max() + 0.25, 0.25)
        # stacked by hand: the lab image's matplotlib cannot stack arrays of different lengths
        bottom = np.zeros(len(bins) - 1)
        for values, label in ((kick[fallen], "fallen"), (kick[captured], "captured"), (kick[survived], "survived")):
            counts = np.histogram(values, bins=bins)[0]
            ax2.bar(bins[:-1], counts, width=np.diff(bins), bottom=bottom, align="edge", label=label)
            bottom = bottom + counts
        ax2.set_xlabel("kick, m/s")
        ax2.set_ylabel("episodes")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"\nPlot: {args.plot}")


if __name__ == "__main__":
    main()
