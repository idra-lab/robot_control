"""
@author: riccardobussola

The falls against needless switches trade-off of the closed loop, one panel per robot: each value
function at its thresholds, joined in order of threshold, the walking rule's starred, the paper's
percentile rule's circled; the walking policy alone and the safe policy alone as references.

    python3 components/safe_rl/plot_tradeoff.py [--robots aliengo go2] [--out report_figures/tradeoff.png]

Reads the closed-loop arms run_pipeline.sh writes (observation_datasets/<robot>/closed_loop/), the
same episodes per robot; needless switches are counted on the episodes the walking policy alone
recovered from, as in compare_filter.py.
"""

import argparse
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "value_function", "observation_datasets")
# per value function: its arms in order of threshold, with the threshold rule they use
FAMILIES = {
    "ABS (horizon 0.2 s)": ("#c2410c", ["abs_t0.5", "abs_pct", "abs", "abs_t0.99"]),
    "ABS, no horizon": ("#f59e0b", ["absfull_pct", "absfull"]),
    "ours (MPS)": ("#1d4ed8", ["mps_t0.5", "mps_pct", "mps_t0.9", "mps", "mps_t0.95"]),
    "ours, early handovers": ("#0891b2", ["mps_early_pct", "mps_early", "mps_early_t0.9"]),
}


# the arms at the walking rule's threshold (the model's .json)
WALKING_RULE = ("abs", "absfull", "mps", "mps_early")


def load(robot, arm):
    files = sorted(glob.glob(os.path.join(OUT, robot, "closed_loop", arm, "batch*", "*_episodes.csv")),
                   key=lambda f: int(f.split("batch")[-1].split(os.sep)[0]))
    return np.concatenate([np.atleast_1d(np.genfromtxt(f, delimiter=",", names=True)) for f in files]) if files else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robots", nargs="+", default=["aliengo", "go2"])
    parser.add_argument("--out", default=os.path.join(HERE, "report_figures", "tradeoff.png"))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(args.robots), figsize=(5.2 * len(args.robots), 4.2), squeeze=False)
    for ax, robot in zip(axes[0], args.robots):
        alone = load(robot, "alone")
        arms = {a: load(robot, a) for f in FAMILIES.values() for a in f[1]}
        arms["safe"] = load(robot, "safe")
        # the episodes kept in every arm (not discarded)
        kept = alone["discarded"] == 0
        for ep in arms.values():
            if ep is not None:
                kept &= ep["discarded"] == 0
        recovered = (alone["fallen"] == 0) & kept

        def point(ep):
            switched = np.isfinite(ep["switch_time"]) | np.isfinite(ep["backup_time"])
            return np.sum(switched & recovered) / recovered.sum(), np.mean(ep["fallen"][kept] > 0)

        ax.scatter(0, np.mean(alone["fallen"][kept] > 0), color="#6b7280", s=40, zorder=3)
        ax.annotate("walking policy alone", (0, np.mean(alone["fallen"][kept] > 0)), xytext=(6, -3),
                    textcoords="offset points", fontsize=8, color="#374151")
        if arms["safe"] is not None:
            x, y = point(arms["safe"])
            ax.scatter(x, y, color="#6b7280", marker="s", s=40, zorder=3)
            ax.annotate("safe policy alone", (x, y), xytext=(-6, 8), textcoords="offset points", fontsize=8,
                        color="#374151", ha="right")
        for label, (color, names) in FAMILIES.items():
            pts = [(name, *point(arms[name])) for name in names if arms.get(name) is not None]
            if not pts:
                continue
            pts.sort(key=lambda p: p[1])
            ax.plot([p[1] for p in pts], [p[2] for p in pts], color=color, lw=1.4, label=label, zorder=2)
            for name, x, y in pts:
                walking = name in WALKING_RULE
                pct = name.endswith("_pct")
                ax.scatter(x, y, color=color, s=110 if walking else 28, marker="*" if walking else "o",
                           edgecolors="black" if pct else color, linewidths=1.2 if pct else 0, zorder=3)
        ax.set_title(robot.capitalize(), fontsize=11)
        ax.set(xlabel=f"needless switches (of the {recovered.sum()} pushes the walking policy survives alone)",
               ylabel="pushes that ended in a fall", xlim=(-0.03, 1.05), ylim=(0, None))
        ax.grid(color="#e5e7eb", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([], [], marker="*", ls="", color="#374151", ms=10),
                Line2D([], [], marker="o", ls="", color="white", markeredgecolor="black", ms=6)]
    labels += ["walking rule", "percentile rule (paper)"]
    axes[0][-1].legend(handles, labels, fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Plot: {args.out}")


if __name__ == "__main__":
    main()
