"""
@author: riccardobussola

What the safety filter changes: the same episodes run by several arms (the walking policy alone, the
safe policy alone, the filter with each value function, threshold and margin), side by side.

    python3 compare_filter.py --arm alone <dataset.npy> ... --arm abs <dataset.npy> ... [--arm ...]
                              [--reference alone] [--table out.md] [--plot out.png]

Each arm is one or more runs of saferl_controller_dataset.py with the same --seed and --episodes, in
the same order: episode by episode the arms then share the command and the push.  Only the
<name>_episodes.csv next to each dataset is read.  Episodes discarded in any arm are left out of all.

Per arm, in % of the episodes (the paper's Table I and more):
* fell: our fall rule (dataset_manager.py, 45 deg or base below 0.2 m); paper fall: base or a hip
  joint below 0.1 m; knees: a knee joint below 3.5 cm; joint limits: a joint position or velocity past
  its URDF limit; total: a paper fall, knee or joint limit violation.
* switched: the safe policy ran (the filter chose it, before or after the push, or the arm is the
  safe policy alone); walking: the filter switched before the push, while walking undisturbed (a
  false trigger: with no switching back, the robot then meets the push on the safe policy); needless: of the episodes the reference
  arm (the walking policy alone) recovered from, those it switched in; stopped: of the episodes it
  switched in, those where the robot reached static equilibrium (the paper's X_inv).
* prevented and caused: falls (our rule) the reference had and the arm did not, and the reverse.
The fall rate per kick (or kick band) goes in --plot.  --tradeoff plots each arm's falls against its
needless switches, with 95 % Wilson intervals, the trade-off a filter's threshold sets.  --versus ARM
compares each arm with ARM episode by episode (falls, our rule): the episodes only one of the two
fell in, and the exact McNemar test on them (two-sided binomial p).
"""

import argparse
import os

import numpy as np
from math import comb

BANDS = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0)


def load(paths):
    episodes = [np.atleast_1d(np.genfromtxt(os.path.splitext(p)[0] + "_episodes.csv", delimiter=",", names=True))
                for p in paths]
    return np.concatenate(episodes)


def kick_groups(kick):
    """(label, mask) per kick when the kicks are a few values (the evaluation), else per band."""
    values = np.unique(np.round(kick, 3))
    if len(values) <= 8:
        return [(f"{k:.2f}", np.isclose(kick, k, atol=1e-3)) for k in values]
    return [(f"{lo:.1f}-{hi:.1f}", (kick >= lo) & (kick < hi)) for lo, hi in zip(BANDS[:-1], BANDS[1:])
            if np.any((kick >= lo) & (kick < hi))]


def switched_in(ep):
    """Episodes the safe policy ran in: the filter switched (at any time), or it was scheduled (the safe
    policy alone, --backup-delay)."""
    return np.isfinite(ep["switch_time"]) | np.isfinite(ep["backup_time"])


def wilson(k, n, z=1.96):
    """95 % Wilson interval of a proportion k / n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar(b, c):
    """Exact two-sided McNemar p: b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", nargs="+", action="append", required=True, metavar=("NAME", "DATASET"),
                        help="an arm: its name and its datasets")
    parser.add_argument("--reference", help="the arm the others are compared with (default the first)")
    parser.add_argument("--table", help="also write the table here, markdown")
    parser.add_argument("--plot", help="save the fall rate per kick here")
    parser.add_argument("--tradeoff", help="save falls against needless switches here")
    parser.add_argument("--versus", help="compare each arm with this one, episode by episode")
    args = parser.parse_args()
    names = [arm[0] for arm in args.arm]
    arms = {arm[0]: load(arm[1:]) for arm in args.arm}
    reference = args.reference or names[0]
    ref = arms[reference]
    for name, ep in arms.items():
        if len(ep) != len(ref) or not (np.allclose(ep["kick"], ref["kick"]) and np.allclose(ep["push_time"], ref["push_time"])):
            raise SystemExit(f"{name} and {reference} are not the same episodes: same --seed and --episodes, same order")
    # 2: fell before the push, 3: the loop missed control ticks
    kept = np.all([~np.isin(ep["discarded"], (2, 3)) for ep in arms.values()], axis=0)
    arms = {name: ep[kept] for name, ep in arms.items()}
    ref = arms[reference]
    fell_ref = ref["fallen"] > 0
    kick = ref["kick"]
    print(f"{kept.sum()} episodes in every arm ({len(kept) - kept.sum()} left out); reference {reference}, "
          f"which recovered from {np.sum(~fell_ref)}")

    rows = []
    for name in names:
        ep = arms[name]
        fell = ep["fallen"] > 0
        paper = ep["paper_fall"] > 0
        knees = ep["knee_contact"] > 0
        limits = (ep["jl_pos"] > 0) | (ep["jl_vel"] > 0)
        switched = switched_in(ep)
        stopped = np.isfinite(ep["static_time"])
        rows.append({
            "arm": name, "fell": fell.mean(), "paper fall": paper.mean(), "knees": knees.mean(),
            "joint limits": limits.mean(), "total": (paper | knees | limits).mean(), "switched": switched.mean(),
            "walking": np.mean(ep["switch_time"] < 0),
            "needless": np.sum(switched & ~fell_ref) / max(1, np.sum(~fell_ref)),
            "stopped": np.sum(switched & stopped) / max(1, switched.sum()) if switched.any() else np.nan,
            "prevented": int(np.sum(fell_ref & ~fell)), "caused": int(np.sum(~fell_ref & fell)),
            "switch s": float(np.median(ep["switch_time"][filtered])) if (filtered := np.isfinite(ep["switch_time"])).any()
            else np.nan})

    percent = ("fell", "paper fall", "knees", "joint limits", "total", "switched", "walking", "needless", "stopped")
    header = ["arm"] + list(percent) + ["prevented", "caused", "switch s"]
    cell = lambda r, c: (r[c] if c == "arm" else "-" if isinstance(r[c], float) and np.isnan(r[c])
                         else f"{100 * r[c]:.1f}" if c in percent else f"{r[c]:.2f}" if c == "switch s" else str(r[c]))
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(r, c) for c in header) + " |" for r in rows]
    table = "\n".join(lines)
    print("\n% of the episodes; walking: switched before the push; needless: of those the reference recovered "
          "from; stopped: of those switched in; switch s: median time of the filter's switch after the push start\n")
    print(table)
    if args.table:
        with open(args.table, "w") as f:
            f.write(f"{kept.sum()} episodes, the same in every arm; reference {reference}.\n\n{table}\n")

    if args.versus:
        base = arms[args.versus]["fallen"] > 0
        print(f"\nEpisode by episode against {args.versus} (falls): fell only with the arm / only with "
              f"{args.versus}, exact McNemar p")
        lines = [f"| arm | fell only with the arm | fell only with {args.versus} | p |", "|---|---|---|---|"]
        for name in names:
            if name == args.versus:
                continue
            fell = arms[name]["fallen"] > 0
            b, c = int(np.sum(fell & ~base)), int(np.sum(~fell & base))
            print(f"  {name:24s} {b:3d} / {c:3d}, p {mcnemar(b, c):.3g}")
            lines.append(f"| {name} | {b} | {c} | {mcnemar(b, c):.3g} |")
        if args.table:
            with open(args.table, "a") as f:
                f.write(f"\nEpisode by episode against {args.versus} (falls):\n\n" + "\n".join(lines) + "\n")

    if args.tradeoff:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        recovered = ~fell_ref
        families = {"abs": "#c2410c", "mps": "#1d4ed8"}
        for name in names:
            ep = arms[name]
            fell = ep["fallen"] > 0
            switched = switched_in(ep)
            x, y = np.sum(switched & recovered) / max(1, recovered.sum()), fell.mean()
            xlo, xhi = wilson(np.sum(switched & recovered), recovered.sum())
            ylo, yhi = wilson(fell.sum(), len(fell))
            color = next((c for prefix, c in families.items() if name.startswith(prefix)), "#6b7280")
            ax.errorbar(x, y, xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]], fmt="o", ms=5, color=color,
                        ecolor=color, elinewidth=0.8, capsize=2, alpha=0.9)
            ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, color="#374151")
        ax.set(xlabel=f"needless switches (of the {recovered.sum()} episodes {reference} recovered from)",
               ylabel="episodes that fell", xlim=(-0.03, 1.03), ylim=(0, None))
        ax.grid(color="#e5e7eb", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.tradeoff, dpi=130)
        print(f"Trade-off plot: {args.tradeoff}")

    groups = kick_groups(kick)
    print("\nfall rate per kick, m/s (episodes): " + ", ".join(f"{label} ({m.sum()})" for label, m in groups))
    for name in names:
        fell = arms[name]["fallen"] > 0
        print(f"  {name:24s} " + "  ".join(f"{fell[m].mean():4.0%}" for _, m in groups))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        palette = ["#9ca3af", "#6b7280", "#c2410c", "#f59e0b", "#1d4ed8", "#0891b2", "#7c3aed", "#16a34a",
                   "#db2777", "#475569", "#a16207", "#0f766e"]
        fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(groups) * len(names) / 3), 3.6))
        x = np.arange(len(groups))
        width = 0.85 / len(names)
        for i, name in enumerate(names):
            fell = arms[name]["fallen"] > 0
            rates = [fell[m].mean() for _, m in groups]
            ax.bar(x - 0.425 + width * (i + 0.5), rates, width * 0.92, color=palette[i % len(palette)],
                   label=f"{name} ({fell.mean():.0%})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{label}\n({m.sum()})" for label, m in groups])
        ax.set(xlabel="kick, m/s (episodes)", ylabel="episodes that fell", ylim=(0, 1.05))
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"\nPlot: {args.plot}")


if __name__ == "__main__":
    main()
