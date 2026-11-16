"""
@author: riccardobussola

Several value functions on the same trajectories: the closed-loop arms run with --watch log, per
control tick of the filter (100 Hz), the value of each watched model and the decision its own rule
(threshold, margin, min_switch from its .json) would take, whatever policy is acting.

    python3 components/safe_rl/plot_vf_traces.py [--robots aliengo go2] [--data OUT] [--out FIGURES]
                                                 [--arms watch_alone watch_abs watch_mps] [--examples 6]

Reads OUT/<robot>/closed_loop/<arm>/batch*/observations_dataset_abs_test_vf_trace.npz and the
_episodes.csv next to it.  Writes, per robot and arm, into FIGURES:
* <robot>_<arm>_vf_traces.png: per model, V from 0.3 s before the push start, median and 10-90 %
  band of the pushes that ended in a failure and of those that did not, with the model's switching
  level (threshold + margin);
* <robot>_<arm>_vf_examples.png: a few pushes, one column each, every model's V in its row, where its
  rule would switch and where the episode failed;
* <robot>_vf_traces.md: when each model's rule would switch (from the walking-alone arm: the same
  trajectories for all), and, with a filter acting, what the other models said at its switch.
"""

import argparse
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "value_function", "observation_datasets", "motor")
FIGURES = os.path.join(HERE, "report_figures_motor")
COLORS = {"abs": "#c2410c", "mps": "#1d4ed8", "mps_start": "#0891b2", "mps_walk": "#7c3aed", "mps_start_walk": "#047857"}
LABELS = {"abs": "ABS (score -V_RA)", "mps": "ours, warm-up from the push end", "mps_start": "ours, warm-up from the push start",
          "mps_walk": "ours + walking anchor, push end", "mps_start_walk": "ours + walking anchor, push start"}
# the arms where the walking policy acts alone (the watched models only log)
ALONE = ("watch_alone", "watch_walk")
FAILED, RECOVERED = "#b91c1c", "#6b7280"
T_FROM, T_TO, DT = -0.3, 1.2, 0.01


def load(robot, arm, data):
    """(episodes, traces, meta): the kept episodes' rows (with batch), per episode its trace array,
    and the watched models' names and levels."""
    rows, traces, meta = [], [], None
    for folder in sorted(glob.glob(os.path.join(data, robot, "closed_loop", arm, "batch*", "")),
                         key=lambda f: int(f.rstrip(os.sep).split("batch")[-1])):
        z = np.load(os.path.join(folder, "observations_dataset_abs_test_vf_trace.npz"))
        ep = np.atleast_1d(np.genfromtxt(os.path.join(folder, "observations_dataset_abs_test_episodes.csv"),
                                         delimiter=",", names=True))
        meta = {k: z[k] for k in ("names", "threshold", "margin", "min_switch", "columns")}
        d = z["data"]
        for e in ep:
            if e["discarded"] != 0:
                continue
            rows.append(e)
            traces.append(d[d[:, 0] == e["episode"]])
    return np.array(rows), traces, meta


def switch_time(trace, k):
    """When model k's rule would first leave the walking policy (s after the push start), nan if never."""
    walk = trace[:, 4 + 2 * k]
    return float(trace[np.argmax(walk == 0), 1]) if np.any(walk == 0) else np.nan


def acting_switch(trace):
    return float(trace[np.argmax(trace[:, 2] == 1), 1]) if np.any(trace[:, 2] == 1) else np.nan


def on_grid(trace, column):
    """The column on the common time grid, nan outside the episode."""
    grid = np.arange(T_FROM, T_TO + 1e-9, DT)
    out = np.full(len(grid), np.nan)
    idx = np.round((trace[:, 1] - T_FROM) / DT).astype(int)
    ok = (idx >= 0) & (idx < len(grid))
    out[idx[ok]] = trace[ok, column]
    return grid, out


def style(ax):
    ax.grid(color="#e5e7eb", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axvspan(0, 0.2, color="#f3f4f6", zorder=0)


def plot_bands(plt, robot, arm, rows, traces, meta, out):
    names = list(meta["names"])
    fell = rows["fallen"] > 0
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 3.4), squeeze=False)
    for k, (ax, name) in enumerate(zip(axes[0], names)):
        for mask, color, label in ((fell, FAILED, "ended in a failure"), (~fell, RECOVERED, "did not")):
            if not mask.any():
                continue
            grid, _ = on_grid(traces[0], 3 + 2 * k)
            values = np.array([on_grid(traces[i], 3 + 2 * k)[1] for i in np.flatnonzero(mask)])
            enough = np.sum(np.isfinite(values), axis=0) >= 5
            with np.errstate(all="ignore"):
                med = np.where(enough, np.nanmedian(values, axis=0), np.nan)
                lo = np.where(enough, np.nanpercentile(values, 10, axis=0), np.nan)
                hi = np.where(enough, np.nanpercentile(values, 90, axis=0), np.nan)
            ax.fill_between(grid, lo, hi, color=color, alpha=0.15, lw=0)
            ax.plot(grid, med, color=color, lw=2, label=f"{label} ({mask.sum()})")
        level = meta["threshold"][k] + meta["margin"][k]
        ax.axhline(level, color="#111827", lw=1, ls="--", label="switching level (threshold + margin)")
        ax.annotate(f"{level:.2f}", (T_TO, level), xytext=(-2, 3), textcoords="offset points", ha="right",
                    fontsize=7, color="#374151")
        ax.set_title(LABELS.get(name, name), fontsize=10, color=COLORS.get(name, "#374151"))
        ax.set(xlabel="s after the push start (shaded: the push)", xlim=(T_FROM, T_TO))
        style(ax)
    axes[0][0].set_ylabel("V (median, 10-90 %)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center", ncol=3)
    fig.suptitle(f"{robot.capitalize()}, {arm.replace('watch_', '')}: value functions on the same pushes", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    path = os.path.join(out, f"{robot}_{arm}_vf_traces.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def pick_examples(rows, traces, meta, n):
    """Pushes that tell the models apart: failures where the two first models' rules switch furthest
    apart (or one never), then pushes survived where some model's rule switched."""
    names = list(meta["names"])
    fell = rows["fallen"] > 0
    s = np.array([[switch_time(t, k) for k in range(len(names))] for t in traces])
    gap = np.where(np.isfinite(s[:, 0]) & np.isfinite(s[:, 1]), np.abs(s[:, 1] - s[:, 0]),
                   np.where(np.isfinite(s[:, 0]) ^ np.isfinite(s[:, 1]), 1.0, -1.0))
    failed = [i for i in np.argsort(-gap) if fell[i] and gap[i] >= 0][: (n + 1) // 2]
    needless = [i for i in np.argsort(-gap) if not fell[i] and gap[i] >= 0][: n - len(failed)]
    return failed + needless, s


def plot_examples(plt, robot, arm, rows, traces, meta, out, n):
    names = list(meta["names"])
    picks, s = pick_examples(rows, traces, meta, n)
    if not picks:
        return None
    fig, axes = plt.subplots(len(names), len(picks), figsize=(2.6 * len(picks), 1.9 * len(names)), sharex=True,
                             squeeze=False)
    for c, i in enumerate(picks):
        trace, ep = traces[i], rows[i]
        failure = ep["end_time"] - ep["push_time"] if ep["fallen"] > 0 else np.nan
        acting = acting_switch(trace)
        for k, name in enumerate(names):
            ax = axes[k][c]
            sel = (trace[:, 1] >= T_FROM) & (trace[:, 1] <= T_TO)
            ax.plot(trace[sel, 1], trace[sel, 3 + 2 * k], color=COLORS.get(name, "#374151"), lw=1.6)
            ax.axhline(meta["threshold"][k] + meta["margin"][k], color="#111827", lw=0.8, ls="--")
            if np.isfinite(s[i, k]):
                ax.axvline(s[i, k], color=COLORS.get(name, "#374151"), lw=1, ls=":")
            if np.isfinite(acting):
                ax.axvline(acting, color="#111827", lw=1.2)
            if np.isfinite(failure):
                ax.axvline(failure, color=FAILED, lw=1.2)
            style(ax)
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel(LABELS.get(name, name).split(",")[0] + ("" if "," not in LABELS.get(name, "") else
                              "\n" + LABELS[name].split(", ")[1]), fontsize=7)
        axes[0][c].set_title(f"kick {ep['kick']:.1f} m/s, {'failed' if ep['fallen'] > 0 else 'survived'}", fontsize=8)
        axes[-1][c].set_xlabel("s after the push start", fontsize=7)
    fig.suptitle(f"{robot.capitalize()}, {arm.replace('watch_', '')}: dotted, where each model's rule would switch; "
                 f"black, the acting filter's switch;\nred, the failure; dashed, the switching level; shaded, the push",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(out, f"{robot}_{arm}_vf_examples.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def tables(robot, data, arms):
    lines = [f"# {robot.capitalize()}: value functions on the same trajectories", ""]
    for alone in [a for a in arms if a in ALONE]:
        rows, traces, meta = load(robot, alone, data)
        names = list(meta["names"])
        fell = rows["fallen"] > 0
        failure = np.where(fell, rows["end_time"] - rows["push_time"], np.nan)
        s = np.array([[switch_time(t, k) for k in range(len(names))] for t in traces])
        lines += [f"Walking policy alone, {len(rows)} pushes ({fell.sum()} ended in a failure): where each model's rule "
                  f"would switch (s after the push start); lead: the failure time minus the switch.", "",
                  "| model | level | of the failures, would switch | before the failure | median switch | median lead "
                  "| of the pushes survived, would switch | before the push |", "|---|---|---|---|---|---|---|---|"]
        for k, name in enumerate(names):
            sw = np.isfinite(s[:, k])
            before = sw & fell & (s[:, k] < failure)
            lines.append(f"| {name} | {meta['threshold'][k] + meta['margin'][k]:.3f} (k {meta['min_switch'][k]}) "
                         f"| {100 * np.mean(sw[fell]):.1f} % | {100 * np.mean(before[fell]):.1f} % "
                         f"| {np.nanmedian(s[fell & sw, k]):.3f} | {np.nanmedian((failure - s[:, k])[before]):.3f} "
                         f"| {100 * np.mean(sw[~fell]):.1f} % | {int(np.sum(sw & (s[:, k] < 0)))} |")
        if len(names) > 1:
            both = fell & np.isfinite(s[:, 0]) & np.isfinite(s[:, 1])
            d = s[both, 1] - s[both, 0]
            lines += ["", f"On the {both.sum()} failures both {names[0]} and {names[1]} would switch in: {names[1]} "
                      f"later in {np.sum(d > 0)}, earlier in {np.sum(d < 0)}, median {names[1]} - {names[0]} "
                      f"{np.median(d):.3f} s (10-90 %: {np.percentile(d, 10):.3f} to {np.percentile(d, 90):.3f})."]
    for arm in arms:
        if arm in ALONE:
            continue
        rows, traces, meta = load(robot, arm, data)
        names = list(meta["names"])
        fell = rows["fallen"] > 0
        acting = np.array([acting_switch(t) for t in traces])
        switched = np.isfinite(acting)
        at = np.full((len(rows), len(names)), np.nan)
        for i in np.flatnonzero(switched):
            t = traces[i]
            j = np.argmax(t[:, 2] == 1)
            at[i] = t[j, 3::2][: len(names)]
        lines += ["", f"{arm.replace('watch_', '')} acting, {len(rows)} pushes, switched in {switched.sum()} "
                  f"({np.sum(switched & fell)} of them failed): each model's V at the acting filter's switch (median), "
                  f"and the share below its level.", "",
                  "| model | level | V at the switch, then failed | below the level | V at the switch, then recovered "
                  "| below the level |", "|---|---|---|---|---|---|"]
        for k, name in enumerate(names):
            level = meta["threshold"][k] + meta["margin"][k]
            f, r = switched & fell, switched & ~fell
            lines.append(f"| {name} | {level:.3f} | {np.nanmedian(at[f, k]) if f.any() else np.nan:.3f} "
                         f"| {100 * np.mean(at[f, k] < level) if f.any() else np.nan:.0f} % "
                         f"| {np.nanmedian(at[r, k]) if r.any() else np.nan:.3f} "
                         f"| {100 * np.mean(at[r, k] < level) if r.any() else np.nan:.0f} % |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robots", nargs="+", default=["aliengo", "go2"])
    parser.add_argument("--data", default=OUT)
    parser.add_argument("--out", default=FIGURES)
    parser.add_argument("--arms", nargs="+", default=["watch_alone", "watch_abs", "watch_mps"])
    parser.add_argument("--examples", type=int, default=6)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(args.out, exist_ok=True)
    for robot in args.robots:
        arms = [a for a in args.arms if glob.glob(os.path.join(args.data, robot, "closed_loop", a, "batch*"))]
        for arm in arms:
            rows, traces, meta = load(robot, arm, args.data)
            print(plot_bands(plt, robot, arm, rows, traces, meta, args.out))
            print(plot_examples(plt, robot, arm, rows, traces, meta, args.out, args.examples))
        text = tables(robot, args.data, arms)
        tag = "" if args.arms == parser.get_default("arms") else "_" + "_".join(args.arms)
        path = os.path.join(args.out, f"{robot}{tag}_vf_traces.md")
        with open(path, "w") as f:
            f.write(text)
        print(text)


if __name__ == "__main__":
    main()
