"""
@author: riccardobussola

The failures against needless switches trade-off of the closed loop, one panel per robot: each arm of
the two value functions (ABS's reach-avoid value, ours), the walking rule's starred, ABS's own rule a
diamond, the paper's percentile rule circled; the walking policy alone and the safe policy alone as
references.  A failure is leaving the safe set (a fall, a knee on the ground, a joint past its limit).

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
# per value function, its arms
FAMILIES = {
    "ABS (reach-avoid value)": ("#c2410c", ["abs", "abs_walk"]),
    "ours (MPS)": ("#1d4ed8", ["mps_pct", "mps_k1", "mps_L0", "mps"]),
    "ours, ablations": ("#0891b2", ["mps_early", "mps_polygon"]),
}
LABELS = {"abs": "ABS rule", "abs_walk": "walking rule", "mps_pct": "percentile", "mps_k1": "1 value",
          "mps_L0": "no margin", "mps": "ours", "mps_early": "early handovers", "mps_polygon": "polygon X_inv"}


# label offsets (points, alignment) where the default, up and right, collides with a neighbour
OFFSETS = {("aliengo", "mps_polygon"): (-6, 6, "right"), ("aliengo", "mps_L0"): (-6, -10, "right"),
           ("go2", "mps_pct"): (-6, 4, "right"), ("go2", "mps_L0"): (0, 9, "center"), ("go2", "mps"): (9, -1, "left")}

# the arms at the walking rule's threshold
WALKING_RULE = ("abs_walk", "mps", "mps_early", "mps_polygon")


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
            for i, (name, x, y) in enumerate(pts):
                walking = name in WALKING_RULE
                pct = name.endswith("_pct")
                marker = "*" if walking else "D" if name == "abs" else "o"
                ax.scatter(x, y, color=color, s=110 if walking else 36, marker=marker, label=label if i == 0 else None,
                           edgecolors="black" if pct else color, linewidths=1.2 if pct else 0, zorder=3)
                dx, dy, ha = OFFSETS.get((robot, name), (5, 3, "left"))
                ax.annotate(LABELS.get(name, name), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                            color="#374151", ha=ha)
        ax.set_title(robot.capitalize(), fontsize=11)
        ax.set(xlabel=f"needless switches (of the {recovered.sum()} pushes the walking policy survives alone)",
               ylabel="pushes that ended in a failure", xlim=(-0.03, 1.05), ylim=(0, None))
        ax.grid(color="#e5e7eb", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([], [], marker="*", ls="", color="#374151", ms=10),
                Line2D([], [], marker="o", ls="", color="white", markeredgecolor="black", ms=6)]
    labels += ["walking rule", "percentile rule (paper)"]
    handles.insert(len(handles) - 2, Line2D([], [], marker="D", ls="", color="#374151", ms=6))
    labels.insert(len(labels) - 2, "ABS's rule")
    axes[0][-1].legend(handles, labels, fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Plot: {args.out}")


if __name__ == "__main__":
    main()
