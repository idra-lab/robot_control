"""
@author: riccardobussola

Check a value function trained by train_value_function.py, on datasets it was not trained on.

    python3 evaluate_value_function.py --model models/<name>.onnx --data <dataset.npy> ... [--plot out.png]

The model is the .onnx the safety filter runs or the .pkl (JAX) training saves.

Each dataset is read with the <name>_episodes.csv that saferl_controller_dataset.py writes next to
it.  V(x) is evaluated, as the safety filter does (value_function_manager.py), on the state of every
sample, from the push to the end of the episode.

* Walking: V of the recovered episodes' last second, and of the states before the push acts in a
  dataset recorded from the push start.  The filter switches to the backup policy below its
  threshold, so these have to stay above it.
* Separation: ROC AUC of V at a time after the push for telling the episodes that fall from those
  that recover, on the episodes still running then.
* As a filter, per threshold: the falls V warned of (below the threshold before the fall) and how
  long before, and the recovered episodes it would have switched in (false alarms), those after a
  light kick apart.
* Calibration: V 0.3 s after the push against the fraction of those episodes that recovered.
"""

import argparse
import os

import numpy as np
import onnxruntime as ort

STATE = 30  # projected gravity 3, base angular velocity 3, joint positions 12, joint velocities 12
SAMPLE_DT = 0.02  # s, the policy (and sample) period
AUC_TIMES = (0.1, 0.2, 0.3, 0.5)  # s after the push starts
THRESHOLDS = (0.5, 0.8, 0.9)
CALIBRATION_TIME = 0.3  # s after the push starts


def load(path):
    data = np.load(path)
    episodes = np.atleast_1d(np.genfromtxt(os.path.splitext(path)[0] + "_episodes.csv", delimiter=",",
                                           names=True))
    return data, episodes[episodes["discarded"] == 0]


def roc_auc(score, positive):
    """Probability that a positive (here: fell) scores below a negative, ties counted half."""
    pos, neg = score[positive], score[~positive]
    if not len(pos) or not len(neg):
        return np.nan
    below = (pos[:, None] < neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    return below


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="the value function: .onnx (what the filter runs) or .pkl")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--plot", help="save V over time, V against the kick and the calibration here")
    parser.add_argument("--light-kick", type=float, default=1.0,
                        help="m/s; below it hardly any episode falls (1 %% for both robots), so an alarm is false")
    args = parser.parse_args()

    if args.model.endswith(".onnx"):
        session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

        def value(states):
            return session.run(None, {"obs": np.asarray(states, dtype=np.float32)})[0]
    else:
        from network import load_model, value as jax_value
        model = load_model(args.model)

        def value(states):
            return jax_value(model, states)

    # per episode: times after the push start, V, fell, time of the fall after the push, kick
    runs = []
    for path in args.data:
        data, episodes = load(path)
        for ep in episodes:
            row = data[int(ep["row"])]
            row = row[np.any(row[:, :STATE] != 0, axis=1)]
            # the last sample is at the episode's end, the others every SAMPLE_DT before it (the
            # first at the push start or at its end, depending on how the dataset was recorded)
            t = ep["end_time"] - ep["push_time"] - SAMPLE_DT * np.arange(len(row))[::-1]
            runs.append({"t": t, "v": value(row[:, STATE:2 * STATE]), "fell": ep["fallen"] > 0,
                         "end": ep["end_time"] - ep["push_time"], "kick": ep["kick"]})
    fell = np.array([r["fell"] for r in runs])
    print(f"{len(runs)} episodes: fell {fell.sum()}, recovered {(~fell).sum()}")

    tail = np.concatenate([r["v"][r["t"] > r["end"] - 1.0] for r in runs if not r["fell"]])
    print(f"\nWalking: recovered episodes' last second, V {tail.mean():.3f} (min {tail.min():.3f}, "
          f"below {', '.join(f'{thr}: {np.mean(tail < thr):.1%}' for thr in THRESHOLDS)})")
    before = np.array([r["v"][0] for r in runs if r["t"][0] < SAMPLE_DT])
    if len(before):
        print(f"  the state before the push acts (datasets recorded from the push start), V {before.mean():.3f} "
              f"(min {before.min():.3f})")

    print("\nSeparation, ROC AUC of V (fell vs recovered) on the episodes still running:")
    for t_auc in AUC_TIMES:
        running = [(r["v"][np.searchsorted(r["t"], t_auc - 1e-6)], r["fell"]) for r in runs
                   if r["t"][-1] >= t_auc - 1e-6]
        v, f = np.array(running).T
        print(f"  {t_auc:.1f} s after the push: AUC {roc_auc(v, f.astype(bool)):.3f}, {len(v)} episodes")

    # below the light kick hardly any episode falls: an alarm there is false; above it a recovered
    # episode was a near miss, and switching to the backup policy is no mistake
    print(f"\nAs a filter, V below the threshold (false alarm: in a recovered episode, after a kick below "
          f"{args.light_kick} m/s or any):")
    light = np.array([r["kick"] < args.light_kick for r in runs]) & ~fell
    for thr in THRESHOLDS:
        leads, alarmed = [], []
        for r in runs:
            # the last sample is the fall itself
            below = np.nonzero(r["v"][:-1] < thr)[0] if r["fell"] else np.nonzero(r["v"] < thr)[0]
            if r["fell"] and len(below):
                leads.append(r["end"] - r["t"][below[0]])
            alarmed.append(len(below) > 0)
        leads, alarmed = np.array(leads), np.array(alarmed)
        warned = (f"before the fall by median {np.median(leads):.2f} s (10th percentile {np.percentile(leads, 10):.2f} s)"
                  if len(leads) else "")
        print(f"  {thr}: falls warned {len(leads)}/{fell.sum()} {warned}; false alarms {alarmed[light].sum()}/{light.sum()} "
              f"light kicks, {alarmed[~fell].sum()}/{(~fell).sum()} any")

    print(f"\nCalibration, V {CALIBRATION_TIME} s after the push against the episodes that recovered:")
    pairs = np.array([(r["v"][np.searchsorted(r["t"], CALIBRATION_TIME - 1e-6)], not r["fell"]) for r in runs
                      if r["t"][-1] >= CALIBRATION_TIME - 1e-6])
    edges = np.array([-np.inf, 0.2, 0.4, 0.6, 0.8, 0.9, np.inf])
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pairs[:, 0] >= lo) & (pairs[:, 0] < hi)
        if m.any():
            print(f"  V {max(lo, 0):.1f}-{min(hi, 1):.1f}: {m.sum():3d} episodes, mean V {pairs[m, 0].mean():.2f}, "
                  f"recovered {pairs[m, 1].mean():.0%}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"fell": "#c2410c", "recovered": "#1d4ed8"}
        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        grid = np.arange(0, 3.21, SAMPLE_DT)
        for name, group in (("fell", fell), ("recovered", ~fell)):
            # the episodes still running at each time
            curves = np.array([np.interp(grid, r["t"], r["v"], left=np.nan, right=np.nan) for r, g in zip(runs, group) if g])
            enough = np.sum(np.isfinite(curves), axis=0) >= 5
            ax[0, 0].plot(grid[enough], np.nanmedian(curves[:, enough], axis=0), color=colors[name], lw=2,
                          label=f"{name}, median")
            ax[0, 0].fill_between(grid[enough], *np.nanpercentile(curves[:, enough], [10, 90], axis=0),
                                  color=colors[name], alpha=0.15, lw=0, label=f"{name}, 10-90 %")
        ax[0, 0].axvspan(0, 0.2, color="gray", alpha=0.15, lw=0)
        ax[0, 0].set(xlabel="time after the push starts, s (gray: push)", ylabel="V", ylim=(-0.05, 1.1),
                     xlim=(0, 1.5), title="episodes still running")
        ax[0, 0].legend(loc="lower right", fontsize=8)
        for r in runs:
            if r["fell"]:
                ax[0, 1].plot(r["t"][:-1] - r["end"], r["v"][:-1], color=colors["fell"], alpha=0.15, lw=1)
        ax[0, 1].set(xlabel="time to the fall, s", ylabel="V", ylim=(-0.05, 1.1), xlim=(-1.5, 0),
                     title="falling episodes, each a line")
        at = [(r["kick"], r["v"][np.searchsorted(r["t"], CALIBRATION_TIME - 1e-6)], r["fell"]) for r in runs
              if r["t"][-1] >= CALIBRATION_TIME - 1e-6]
        k, v, f = np.array(at).T
        for name, m in (("fell", f > 0), ("recovered", f == 0)):
            ax[1, 0].scatter(k[m], v[m], s=12, color=colors[name], label=name)
        ax[1, 0].set(xlabel="kick, m/s", ylabel=f"V {CALIBRATION_TIME} s after the push", ylim=(-0.05, 1.1),
                     title="V against the push")
        ax[1, 0].legend(fontsize=8)
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(pairs[:, 0], bins) - 1, 0, 9)
        mids = [(pairs[idx == b, 0].mean(), pairs[idx == b, 1].mean(), (idx == b).sum()) for b in range(10)
                if (idx == b).sum() >= 3]
        mv, rec, n = np.array(mids).T
        ax[1, 1].plot([0, 1], [0, 1], color="gray", lw=1, ls="--")
        ax[1, 1].scatter(mv, rec, s=10 + 2 * n, color=colors["recovered"])
        ax[1, 1].set(xlabel=f"V {CALIBRATION_TIME} s after the push (bins of 0.1, size: episodes)",
                     ylabel="fraction recovered", xlim=(0, 1.05), ylim=(0, 1.05), title="calibration")
        for a in ax.flat:
            a.grid(color="#e5e7eb", lw=0.6)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"\nPlot: {args.plot}")


if __name__ == "__main__":
    main()
