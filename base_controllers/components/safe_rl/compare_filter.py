"""
@author: riccardobussola

What the safety filter changes: the same episodes with the walking policy alone and with the filter.

    python3 compare_filter.py --alone <dataset.npy> ... --filtered <dataset.npy> ... [--plot out.png]

Each set is one or more runs of saferl_controller_dataset.py, the filtered ones with --filter and the
same --seed and --episodes as the runs alone, in the same order: episode by episode they then share
the command and the push.  Only the <name>_episodes.csv next to each dataset is read.

* Falls alone and filtered; falls prevented (fell alone, not filtered) and caused (the reverse).
* Switches to the safe policy: in episodes the walking policy recovered from alone (needless), and
  of those after a kick below 1 m/s; before the push (while walking); the time after the push.
* Falls left and caused, and needless switches, by push direction (base frame), and how long before
  a fall left the filter had switched.
* The fall rate per kick band.
"""

import argparse
import os

import numpy as np

BANDS = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0)


def direction_of(kick_dir):
    """forward, left, backward or right, for a push direction in the base frame (rad, 0 forward)."""
    angle = (np.degrees(kick_dir) + 180.0) % 360.0 - 180.0
    return np.select([np.abs(angle) <= 45, (angle > 45) & (angle <= 135), np.abs(angle) > 135],
                     ["forward", "left", "backward"], "right")


def load(paths):
    episodes = [np.atleast_1d(np.genfromtxt(os.path.splitext(p)[0] + "_episodes.csv", delimiter=",", names=True))
                for p in paths]
    return np.concatenate(episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alone", nargs="+", required=True)
    parser.add_argument("--filtered", nargs="+", required=True)
    parser.add_argument("--plot", help="save the fall rate per kick band here")
    parser.add_argument("--light-kick", type=float, default=1.0,
                        help="m/s; the walking policy falls from about 1 %% of the pushes below")
    args = parser.parse_args()
    alone, filtered = load(args.alone), load(args.filtered)
    if len(alone) != len(filtered) or not (np.allclose(alone["kick"], filtered["kick"])
                                           and np.allclose(alone["push_time"], filtered["push_time"])):
        raise SystemExit("the two sets are not the same episodes: same --seed and --episodes, same order")
    # 2: fell before the push, 3: the loop missed control ticks
    kept = ~np.isin(alone["discarded"], (2, 3)) & ~np.isin(filtered["discarded"], (2, 3))
    alone, filtered = alone[kept], filtered[kept]
    kick = alone["kick"]
    fell_alone, fell = alone["fallen"] > 0, filtered["fallen"] > 0
    switch = filtered["switch_time"]
    switched = np.isfinite(switch)
    light = kick < args.light_kick

    print(f"{len(kick)} episodes")
    print(f"  fell: alone {fell_alone.sum()} ({fell_alone.mean():.0%}), filtered {fell.sum()} ({fell.mean():.0%})")
    print(f"  falls prevented {np.sum(fell_alone & ~fell)}, caused {np.sum(~fell_alone & fell)}")
    print(f"  switches to the safe policy: {switched.sum()}; needless (recovered alone) {np.sum(switched & ~fell_alone)}"
          f"/{np.sum(~fell_alone)}, of them after a kick below {args.light_kick} m/s {np.sum(switched & ~fell_alone & light)}"
          f"/{np.sum(~fell_alone & light)}; before the push {np.sum(switch < 0)}")
    if switched.any():
        print(f"  switch after the push start: median {np.median(switch[switched]):.2f} s "
              f"(10-90 % {np.percentile(switch[switched], 10):.2f}-{np.percentile(switch[switched], 90):.2f} s)")
    direction = direction_of(alone["kick_dir"])
    by_direction = lambda m: ", ".join(f"{d} {np.sum(m & (direction == d))}" for d in ("forward", "left", "backward", "right"))
    if fell.any():
        lead = (filtered["end_time"] - filtered["push_time"] - switch)[fell & switched]
        print(f"  falls left: {fell.sum()}, {np.sum(fell & switched)} after a switch"
              + (f" {lead.min():.2f}-{lead.max():.2f} s before the fall" if len(lead) else "")
              + f"; kicks {np.array2string(np.sort(kick[fell]), precision=2)}; by direction {by_direction(fell)}")
    caused = ~fell_alone & fell
    if caused.any():
        print(f"  falls caused: " + ", ".join(f"{k:.2f} m/s {d}" for k, d in zip(kick[caused], direction[caused])))
    print(f"  needless switches by direction: {by_direction(switched & ~fell_alone)}")
    print("  fall rate per kick band, alone -> filtered:")
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        m = (kick >= lo) & (kick < hi)
        if m.any():
            print(f"    {lo:.1f}-{hi:.1f} m/s ({m.sum():3d}): {fell_alone[m].mean():4.0%} -> {fell[m].mean():4.0%}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.2))
        bands = [(lo, hi) for lo, hi in zip(BANDS[:-1], BANDS[1:]) if np.any((kick >= lo) & (kick < hi))]
        x = np.arange(len(bands))
        for offset, outcome, color, label in ((-0.2, fell_alone, "#9ca3af", "walking policy alone"),
                                              (0.2, fell, "#1d4ed8", "with the safety filter")):
            rates = [outcome[(kick >= lo) & (kick < hi)].mean() for lo, hi in bands]
            ax.bar(x + offset, rates, 0.37, color=color, label=label)
            for xi, r in zip(x + offset, rates):
                ax.text(xi, r + 0.02, f"{r:.0%}", ha="center", fontsize=7, color="#374151")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lo:.1f}-{hi:.1f}\n({np.sum((kick >= lo) & (kick < hi))})" for lo, hi in bands])
        ax.set(xlabel="kick, m/s (episodes)", ylabel="episodes that fell", ylim=(0, 1.1))
        ax.set_title(f"Falls {fell_alone.mean():.0%} alone, {fell.mean():.0%} with the filter, same {len(kick)} pushes",
                     fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"\nPlot: {args.plot}")


if __name__ == "__main__":
    main()
