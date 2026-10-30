"""
@author: riccardobussola

Checks the fall rule and the capture hold on episodes run until the robot is on the ground.

    python3 saferl_controller_dataset.py --robot go2 --fall-check ...   (or run_batches.sh ... --fall-check)
    python3 components/safe_rl/fall_rule.py <dataset.npy> ... [--plot fall_rule.png]

With --fall-check an episode ends with the robot on the ground (inclination over 85 deg or base
below 0.1 m) or 3 s after the push, capture off.  It fell if it ended on the ground or with its base
below the rule's height (a robot lying on its belly, upright).  Only the <name>_episodes.csv next to
each dataset is read.

* The gap: the largest inclination and the lowest base of the recoveries, the smallest largest
  inclination of the falls.  The fall rule (ROBOT_SETTINGS in dataset_manager.py) belongs in it.
* Per inclination threshold, the recoveries that crossed it: they would be labelled falls.
* Per capture hold, the episodes captured (capture point within 5 cm that long, from the push end,
  before the robot first tilted past the rule) that fell afterwards: false successes.
"""

import argparse
import os

import numpy as np

HOLDS = (0.0, 0.1, 0.2, 0.3, 0.5)  # s; 0 is a single tick


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--inclination", type=float, default=45.0, help="the fall rule's inclination, deg")
    parser.add_argument("--height", type=float, default=0.2, help="the fall rule's base height, m")
    parser.add_argument("--plot", help="save the histogram of the largest inclination here")
    parser.add_argument("--title", default="", help="prefix of the plot title, e.g. the robot")
    args = parser.parse_args()
    ep = np.concatenate([np.atleast_1d(np.genfromtxt(os.path.splitext(p)[0] + "_episodes.csv", delimiter=",",
                                                     names=True)) for p in args.datasets])
    ep = ep[~np.isin(ep["discarded"], (2, 3))]
    fell = (ep["fallen"] > 0) | (ep["final_height"] < args.height)
    rec, fal = ep[~fell], ep[fell]
    print(f"{len(ep)} episodes run to the ground: fell {fell.sum()}, recovered {(~fell).sum()} "
          f"(on the belly, upright: {np.sum(fell & (ep['fallen'] == 0))})")
    print(f"recoveries: largest inclination {rec['max_incl'].max():.1f} deg, lowest base {rec['min_height'].min():.3f} m")
    print(f"falls: smallest largest inclination {fal['max_incl'].min():.1f} deg")
    print(f"the rule, {args.inclination:.0f} deg or base below {args.height} m: "
          f"{'in the gap' if rec['max_incl'].max() < args.inclination < fal['max_incl'].min() and rec['min_height'].min() > args.height else 'NOT in the gap'}")
    print("recoveries that crossed an inclination (they would be labelled falls):")
    for th in (20, 25, 30, 35, 40, 45, 50, 60):
        print(f"  {th:3d} deg: {np.sum(rec['max_incl'] > th)}")
    print("captured, then fell (capture point within 5 cm for the hold, before the robot tilted past the rule):")
    for hold in HOLDS:
        captured = ep["cp_hold_max"] >= hold - 1e-9
        print(f"  {'one tick' if hold == 0 else f'{hold:.1f} s':8s}: captured {captured.sum():3d}, then fell {np.sum(captured & fell)}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3.2))
        bins = np.arange(0, 92, 2)
        ax.hist(rec["max_incl"], bins=bins, color="#1d4ed8", label=f"recovered ({len(rec)})")
        ax.hist(fal["max_incl"], bins=bins, color="#c2410c", label=f"fell to the ground ({len(fal)})")
        lo, hi = rec["max_incl"].max(), fal["max_incl"].min()
        top = ax.get_ylim()[1] * 1.1
        ax.set_ylim(0, top)
        ax.axvspan(lo, hi, color="#9ca3af", alpha=0.18, lw=0)
        ax.axvline(args.inclination, color="#374151", lw=1, ls="--")
        ax.text(args.inclination + 0.8, top * 0.9, f"rule {args.inclination:.0f}°", fontsize=9, color="#374151")
        # the label in the wider side of the gap, clear of the rule's line
        x_label = (args.inclination + hi) / 2 if hi - args.inclination > args.inclination - lo else (lo + args.inclination) / 2
        ax.text(x_label, top * 0.4, f"no episode\nbetween\n{lo:.0f}° and {hi:.0f}°", ha="center", fontsize=9,
                color="#374151")
        ax.set(xlabel="largest inclination after the push, deg", ylabel="episodes",
               title=f"{args.title}no robot tilted past {lo:.0f}° recovered ({len(ep)} pushed episodes, run to the ground)")
        ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.95), frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"\nPlot: {args.plot}")


if __name__ == "__main__":
    main()
