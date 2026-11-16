"""
@author: riccardobussola

The v2 closed-loop tests of a robot (run_v2.sh, stage test) in tables and figures:

    python3 summarize.py --robot go2 --data <OUT>/go2/test --out <FIGURES>

Per push family and arm, over the pushes every arm ran without missed control ticks (paired by seed
and episode): failed (left the safe set, the episode ends there), its kinds, switched (the safe policy
took over), needless (switched in a push the walking policy alone survived), the median switch time
after the push start, and the exact McNemar p of ours against ABS.  Writes <out>/<robot>_v2.md,
<robot>_v2_kick.png (failures against the kick) and <robot>_v2_tradeoff.png.
"""

import argparse
import glob
import os
from math import comb

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--robot", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
ARM_NAMES = {"walk": "Walking policy alone", "safe": "Safe policy alone", "abs": "ABS, its rule (0.05)",
             "abs_03": "ABS, 0.3", "mps": "Ours, Q(o, 1) >= 0.9", "mps_05": "Ours, Q(o, 1) >= 0.5",
             "mps_wr": "Ours, walking rule", "mps_95": "Ours, Q(o, 1) >= 0.95", "mps_99": "Ours, Q(o, 1) >= 0.99",
             "paper": "Paper V(o) >= V_thr + L", "paper_wr": "Paper V(o), walking rule",
             "paper_mc": "Paper V(o), Monte Carlo, >= 0.8 + L"}
FAMILY_NAMES = {"train": "training pushes (0.2 s)", "impulse": "impulse (0.05 s)", "long": "long (0.6 s)",
                "offset": "off-centre (0.25 m)", "down": "pressing down (45 deg)"}


def load_arm(folder):
    rows = {}
    for path in sorted(glob.glob(os.path.join(folder, "batch*", "observations_dataset_abs_test_episodes.csv"))):
        batch = int(os.path.basename(os.path.dirname(path))[5:])
        for r in np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True)):
            rows[(batch, int(r["episode"]))] = r
    return rows


def mcnemar(a, b):
    """Exact two-sided McNemar p of paired failures a, b (boolean arrays)."""
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


os.makedirs(args.out, exist_ok=True)
families = [f for f in FAMILY_NAMES if os.path.isdir(os.path.join(args.data, f))]
lines = [f"# v2 closed loop, {args.robot}", "",
         "% of the pushes; needless: of the pushes the walking policy alone survived; switch s: median, "
         "after the push start; p: exact McNemar against ABS at its rule.", ""]
per_kick, tradeoff = {}, {}
for family in families:
    arms = [a for a in ARM_NAMES if os.path.isdir(os.path.join(args.data, family, a))]
    data = {a: load_arm(os.path.join(args.data, family, a)) for a in arms}
    keys = sorted(set.intersection(*[{k for k, r in d.items() if r["discarded"] == 0} for d in data.values()]))
    if not keys:
        continue
    col = lambda a, c: np.array([data[a][k][c] for k in keys], dtype=float)
    failed = {a: col(a, "fallen") > 0 for a in arms}
    walk_ok = ~failed["walk"] if "walk" in arms else np.zeros(len(keys), bool)
    lines += [f"## {args.robot}: {FAMILY_NAMES[family]}, {len(keys)} pushes", "",
              "| | failed | fall | base/hip < 0.1 m | knee | joint pos | joint vel | switched | needless | switch s | p vs ABS |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        f = failed[a]
        sw = ~np.isnan(col(a, "switch_time")) if a != "walk" else np.zeros(len(keys), bool)
        kinds = [np.mean(f & (col(a, c) > 0)) * 100 for c in ("fall_rule", "paper_fall", "knee_contact", "jl_pos", "jl_vel")]
        need = 100 * np.sum(sw & walk_ok) / max(np.sum(walk_ok), 1)
        st = np.nanmedian(col(a, "switch_time")[sw]) if sw.any() and a != "safe" else np.nan
        p = mcnemar(f, failed["abs"]) if "abs" in arms and a != "abs" else np.nan
        lines.append(f"| {ARM_NAMES[a]} | **{100 * f.mean():.1f}** | " + " | ".join(f"{k:.1f}" for k in kinds)
                     + f" | {100 * sw.mean():.0f} | {need:.0f} | {st:.2f} | " + ("–" if np.isnan(p) else f"{p:.2g}") + " |")
        kick = col(a, "kick")
        per_kick[(family, a)] = {k: 100 * f[kick == k].mean() for k in np.unique(kick)}
        tradeoff[(family, a)] = (need, 100 * f.mean())
    lines.append("")

with open(os.path.join(args.out, f"{args.robot}_v2.md"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"walk": "#888888", "safe": "#444444", "abs": "#d1495b", "abs_03": "#edae49",
              "mps": "#00798c", "mps_05": "#66a182", "mps_wr": "#2e4057", "mps_95": "#1b5e6b", "mps_99": "#0b3c49",
              "paper": "#8e44ad", "paper_wr": "#c39bd3", "paper_mc": "#5b2c6f"}
    fig, axes = plt.subplots(1, len(families), figsize=(3.2 * len(families), 3), sharey=True, squeeze=False)
    for ax, family in zip(axes[0], families):
        for a in ARM_NAMES:
            if (family, a) in per_kick:
                k = sorted(per_kick[(family, a)])
                ax.plot(k, [per_kick[(family, a)][x] for x in k], "o-", color=colors[a], label=ARM_NAMES[a], ms=3)
        ax.set_title(FAMILY_NAMES[family], fontsize=9)
        ax.set_xlabel("kick, m/s")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("failed, %")
    axes[0][-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{args.robot}_v2_kick.png"), dpi=150)
    fig, axes = plt.subplots(1, len(families), figsize=(3.2 * len(families), 3), sharey=True, squeeze=False)
    for ax, family in zip(axes[0], families):
        for a in ARM_NAMES:
            if (family, a) in tradeoff:
                x, y = tradeoff[(family, a)]
                ax.plot(x, y, "o", color=colors[a], label=ARM_NAMES[a])
        ax.set_title(FAMILY_NAMES[family], fontsize=9)
        ax.set_xlabel("needless switches, %")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("failed, %")
    axes[0][-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{args.robot}_v2_tradeoff.png"), dpi=150)
except ImportError:
    pass
