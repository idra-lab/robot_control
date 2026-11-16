"""
@author: riccardobussola

The failures against needless switches trade-off of the closed loop, one panel per robot: each arm of
the two value functions (ABS's reach-avoid value, ours), the walking rule's starred, ABS's own rule a
diamond, the paper's percentile rule circled; the walking policy alone and the safe policy alone as
references.  A failure is leaving the safe set (a fall, a knee on the ground, a joint past its limit).

    python3 components/safe_rl/plot_tradeoff.py [--robots aliengo go2] [--out report_figures/tradeoff.png]
                                                [--data value_function/observation_datasets]

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


# label placements tried, in order of preference: offset from the point (points) and alignment
PLACEMENTS = ((6, 3, "left"), (-6, 3, "right"), (6, -10, "left"), (-6, -10, "right"), (0, 8, "center"),
              (0, -14, "center"), (10, -4, "left"), (-10, -4, "right"))


def place_labels(ax, items):
    """Annotate (text, x, y, fontsize) items, each at the placement whose text box overlaps the least the
    markers and the labels placed before it."""
    fig = ax.figure
    px = fig.dpi / 72.0
    to_disp = ax.transData.transform
    markers = [to_disp((x, y)) for _, x, y, _ in items]
    boxes = [(mx - 5, my - 5, mx + 5, my + 5) for mx, my in markers]

    def overlap(a, b):
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

    placed = []
    for k, (text, x, y, size) in enumerate(items):
        mx, my = markers[k]
        w, h = 0.55 * size * len(text) * px, size * px
        best = None
        for dx, dy, ha in PLACEMENTS:
            x0 = mx + dx * px - {"left": 0, "right": w, "center": w / 2}[ha]
            box = (x0, my + dy * px, x0 + w, my + dy * px + h)
            score = sum(overlap(box, b) for i, b in enumerate(boxes) if i != k) + sum(overlap(box, b) for b in placed)
            if best is None or score < best[0] - 1e-6:
                best = (score, dx, dy, ha, box)
        placed.append(best[4])
        ax.annotate(text, (x, y), xytext=(best[1], best[2]), textcoords="offset points", fontsize=size,
                    color="#374151", ha=best[3])

# the arms at the walking rule's threshold
WALKING_RULE = ("abs_walk", "mps", "mps_early", "mps_polygon")


def load(robot, arm, out=OUT):
    files = sorted(glob.glob(os.path.join(out, robot, "closed_loop", arm, "batch*", "*_episodes.csv")),
                   key=lambda f: int(f.split("batch")[-1].split(os.sep)[0]))
    return np.concatenate([np.atleast_1d(np.genfromtxt(f, delimiter=",", names=True)) for f in files]) if files else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robots", nargs="+", default=["aliengo", "go2"])
    parser.add_argument("--out", default=os.path.join(HERE, "report_figures", "tradeoff.png"))
    parser.add_argument("--data", default=OUT, help="the observation datasets (OUT of run_pipeline.sh)")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(args.robots), figsize=(5.2 * len(args.robots), 4.2), squeeze=False)
    all_items = []
    for ax, robot in zip(axes[0], args.robots):
        alone = load(robot, "alone", args.data)
        arms = {a: load(robot, a, args.data) for f in FAMILIES.values() for a in f[1]}
        arms["safe"] = load(robot, "safe", args.data)
        # the episodes kept in every arm (not discarded)
        kept = alone["discarded"] == 0
        for ep in arms.values():
            if ep is not None:
                kept &= ep["discarded"] == 0
        recovered = (alone["fallen"] == 0) & kept

        def point(ep):
            switched = np.isfinite(ep["switch_time"]) | np.isfinite(ep["backup_time"])
            return np.sum(switched & recovered) / recovered.sum(), np.mean(ep["fallen"][kept] > 0)

        items = [("walking policy alone", 0.0, np.mean(alone["fallen"][kept] > 0), 8)]
        ax.scatter(0, items[0][2], color="#6b7280", s=40, zorder=3)
        if arms["safe"] is not None:
            x, y = point(arms["safe"])
            ax.scatter(x, y, color="#6b7280", marker="s", s=40, zorder=3)
            items.append(("safe policy alone", x, y, 8))
        for label, (color, names) in FAMILIES.items():
            pts = [(name, *point(arms[name])) for name in names if arms.get(name) is not None]
            for i, (name, x, y) in enumerate(pts):
                walking = name in WALKING_RULE
                pct = name.endswith("_pct")
                marker = "*" if walking else "D" if name == "abs" else "o"
                ax.scatter(x, y, color=color, s=110 if walking else 36, marker=marker, label=label if i == 0 else None,
                           edgecolors="black" if pct else color, linewidths=1.2 if pct else 0, zorder=3)
                items.append((LABELS.get(name, name), x, y, 7))
        ax.set_title(robot.capitalize(), fontsize=11)
        ax.set(xlabel=f"needless switches (of the {recovered.sum()} pushes the walking policy survives alone)",
               ylabel="pushes that ended in a failure", xlim=(-0.03, 1.05),
               ylim=(0, 1.08 * max(y for _, _, y, _ in items)))
        all_items.append(items)
        ax.grid(color="#e5e7eb", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    for ax, robot_items in zip(axes[0], all_items):
        place_labels(ax, robot_items)
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
