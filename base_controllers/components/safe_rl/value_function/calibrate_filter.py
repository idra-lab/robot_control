"""
@author: riccardobussola

The safety filter's threshold, margin and switching count for a value function (the paper, Sec.
II-B, II-C), written next to the model as <model>.json, which ValueFunctionManager and the
controllers read.  The filter walks while the model's output (V, or ABS's score -V_RA) stays above
threshold + margin, and switches after min_switch evaluations in a row below it.

    python3 calibrate_filter.py --model models/VF_mps_aliengo.onnx --threshold-data <dataset.npy> ...
                                --margin-data <dataset.npy> ... [--rule walking|percentile|risk|abs]
                                [--consecutive 3] [--write]

Both sets of datasets are runs of saferl_controller_dataset.py with the walking policy alone,
recorded with --record-all --sample-rate 100 (every state, at the filter's rate).  --threshold-data
are runs of the evaluation's protocol, as long as the paper's (--episode-length 20, a push of the
evaluation's kicks in each): undisturbed walking before the push, then the push and its outcome.

* Margin, Eq. (5): L = max |V(o_t) - V(o_t+h)| / ||o_t - o_t+h|| * max ||o_t - o_t+h||, on states h apart
  from runs the walking policy recovers from (--margin-data: light pushes).  h is the time from the
  first value below the level to the safe policy acting: the --consecutive - 1 further evaluations
  the filter waits (0.01 s each) and --action-delay (one policy step, 0.02 s, and the actuators'
  delay, up to 0.01 s).  As in the paper the --trim % largest state changes are left out first, then
  the --trim % largest ratios of the rest; the states are compared in the model's input
  normalisation, so the norm does not depend on units.  ABS uses no margin.
* Threshold, --rule:
  walking (default): the highest threshold at which the filter never switches on the undisturbed
    walking states of --threshold-data: with --consecutive k, the filter switches only after k values
    in a row below its level, so the level is the lowest, over the walking states, of the largest of
    k consecutive values; threshold = that level - L.  A single-sample dip of V does not count.
  walking_robust: the walking rule without its outlier episodes: from --filter-from s on (the
    filter is off before), per episode the lowest of the k-consecutive maxima, and the level the
    --walking-tolerance % lowest episodes go below; a walking false switch in that share of the
    episodes, the rest of the walking never below.  Also for a single value (walking_robust_k1).
  percentile: the paper's statistical rule, V below 99 % of the values over all the states, walking
    and pushed (--percentile 1).  It controls a rate per state, not per unit of time, and falls on
    the steep drop of V before a failure, so it moves with the share of failing states in the data.
  risk: 1 - --risk, V being the probability that the safe policy succeeds (mps).
  abs: ABS's own rule, recovery when V_RA >= -0.05: a threshold of 0.05 on the score -V_RA, on a single
    value (min_switch 1).
  The thresholds of all rules, and of more percentiles (--sweep), go in the .json for reference.
"""

import argparse
import json
import os
import pickle

import numpy as np
import onnxruntime as ort

STATE = 30
SAMPLE_DT = 0.01  # s, the calibration runs' and the filter's period
ABS_THRESHOLD = 0.05  # ABS: recovery when V_RA >= -0.05


def load_episodes(paths, times=False):
    """Per episode, its states at 0.01 s (n + 1, the transitions' states and the last next state) and
    which of them came before the push (and, with times, the states' times)."""
    episodes = []
    for path in paths:
        data = np.load(path)
        rows = np.atleast_1d(np.genfromtxt(os.path.splitext(path)[0] + "_episodes.csv", delimiter=",", names=True))
        for ep in rows[rows["discarded"] == 0]:
            row = data[int(ep["row"])]
            row = row[np.any(row[:, :STATE] != 0, axis=1)]
            states = np.vstack([row[:, :STATE], row[-1:, STATE:2 * STATE]])
            t = ep["end_time"] - SAMPLE_DT * np.arange(len(states))[::-1]
            episodes.append((states, t < ep["push_time"], t) if times else (states, t < ep["push_time"]))
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="the .onnx; the .pkl next to it gives the normalisation")
    parser.add_argument("--threshold-data", nargs="+", required=True)
    parser.add_argument("--margin-data", nargs="+", required=True)
    parser.add_argument("--rule", choices=("walking", "walking_robust", "percentile", "risk", "abs"), default=None,
                        help="default: abs for the ABS baseline, walking otherwise")
    parser.add_argument("--consecutive", type=int, default=3, help="evaluations in a row below the level to switch")
    parser.add_argument("--action-delay", type=float, default=0.03, help="s from the decision to the safe policy acting")
    parser.add_argument("--percentile", type=float, default=1.0)
    parser.add_argument("--risk", type=float, default=0.1)
    parser.add_argument("--walking-tolerance", type=float, default=2.5,
                        help="walking_robust: %% of the walking episodes allowed to switch (the outliers)")
    parser.add_argument("--filter-from", type=float, default=1.0, help="walking_robust: s the filter starts at")
    parser.add_argument("--trim", type=float, default=5.0)
    parser.add_argument("--sweep", type=float, nargs="*", default=[0.5, 2.0], help="more percentiles, for reference")
    parser.add_argument("--write", action="store_true", help="write <model>.json")
    args = parser.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    value = lambda x: session.run(None, {"obs": np.asarray(x, dtype=np.float32)})[0]
    with open(os.path.splitext(args.model)[0] + ".pkl", "rb") as f:
        trained = pickle.load(f)
    method = trained.get("method")
    rule = args.rule or ("abs" if method == "abs" else "walking")
    spread = np.asarray(trained["std"]) + 1e-8
    k = args.consecutive

    # -- margin, over the time from the first value below the level to the safe policy acting
    margin_episodes = load_episodes(args.margin_data)

    def eq5(consecutive):
        h = max(1, int(round(((consecutive - 1) * SAMPLE_DT + args.action_delay) / SAMPLE_DT)))
        pairs = [(s[:-h], s[h:]) for s, _ in margin_episodes if len(s) > h]
        s0 = np.concatenate([a for a, _ in pairs])
        s1 = np.concatenate([b for _, b in pairs])
        step = np.linalg.norm((s1 - s0) / spread, axis=1)
        dv = np.abs(value(s1) - value(s0))
        keep = step <= np.percentile(step, 100 - args.trim)
        ratio = dv[keep] / np.maximum(step[keep], 1e-9)
        ratio_keep = ratio <= np.percentile(ratio, 100 - args.trim)
        return h, step, dv, float(ratio[ratio_keep].max()), float(step[keep][ratio_keep].max())

    h, step, dv, lipschitz_v, lipschitz_f = eq5(k)
    margin_eq5 = lipschitz_v * lipschitz_f
    margin = 0.0 if method == "abs" else margin_eq5
    # the same for a filter switching on a single value (the ablation of --consecutive)
    _, _, _, lv1, lf1 = eq5(1)
    margin_k1 = 0.0 if method == "abs" else lv1 * lf1
    print(f"Margin, Eq. (5): {len(step)} pairs {h * SAMPLE_DT:.2f} s apart ({k - 1} more evaluations + "
          f"{args.action_delay:.2f} s to act), {args.trim:g} % of the largest steps then of the largest ratios left "
          f"out: L_V {lipschitz_v:.4f}, L_f {lipschitz_f:.3f}, L {margin_eq5:.4f} (max |dV| {dv.max():.4f}, 99th "
          f"percentile {np.percentile(dv, 99):.4f}); used {margin:.4f}")

    # -- threshold
    episodes = load_episodes(args.threshold_data, times=True)
    values = [value(s) for s, _, _ in episodes]
    v = np.concatenate(values)
    walking_v = [vals[before] for vals, (_, before, _) in zip(values, episodes) if before.sum() >= k]
    # walking_robust: per episode, the walking while the filter is on, its lowest k-consecutive maximum
    # and lowest value; the --walking-tolerance % lowest episodes are the outliers
    filtered = [vals[before & (t >= args.filter_from)] for vals, (_, before, t) in zip(values, episodes)]
    filtered = [w for w in filtered if len(w) >= k]
    episode_rolling = np.array([np.lib.stride_tricks.sliding_window_view(w, k).max(axis=1).min() for w in filtered])
    episode_min = np.array([w.min() for w in filtered])
    # the largest of k consecutive values, per walking stretch: the filter switches only below all k
    rolling = np.concatenate([np.lib.stride_tricks.sliding_window_view(w, k).max(axis=1) for w in walking_v])
    walking = np.concatenate(walking_v)
    rules = {"walking": float(rolling.min()) - margin, "percentile": float(np.percentile(v, args.percentile)),
             "risk": 1.0 - args.risk, "abs": ABS_THRESHOLD, "walking_k1": float(walking.min()) - margin_k1,
             "walking_robust": float(np.percentile(episode_rolling, args.walking_tolerance)) - margin,
             "walking_robust_k1": float(np.percentile(episode_min, args.walking_tolerance)) - margin_k1}
    threshold = rules[rule]
    sweep = {f"{p:g}": round(float(np.percentile(v, p)), 4) for p in args.sweep}
    print(f"Threshold data: {len(v)} states, {len(walking)} of them undisturbed walking; walking: lowest value "
          f"{walking.min():.4f}, lowest of {k} in a row {rolling.min():.4f}, 0.1st percentile {np.percentile(walking, 0.1):.4f}")
    print("  rules: " + ", ".join(f"{name} {t:.4f}" for name, t in rules.items())
          + "; percentiles " + ", ".join(f"{p}th {t:.4f}" for p, t in sweep.items()))
    for name, lowest, m in (("walking_robust", episode_rolling, margin), ("walking_robust_k1", episode_min, margin_k1)):
        print(f"  {name}: {len(lowest)} walking episodes from {args.filter_from:g} s, {np.sum(lowest < rules[name] + m)} "
              f"of them below the level {rules[name] + m:.4f}; the 5 lowest " + ", ".join(f"{x:.3f}" for x in np.sort(lowest)[:5]))
    print(f"Threshold, rule {rule}: {threshold:.4f}, switching below {threshold + margin:.4f} for "
          f"{1 if rule == 'abs' else k} evaluation(s)")
    level = threshold + margin
    print(f"  walking undisturbed: values below the level {np.mean(walking < level):.3%}, "
          f"{k} in a row below it {np.mean(rolling < level):.3%} of the windows; all states {np.mean(v < level):.2%}")

    if args.write:
        out = os.path.splitext(args.model)[0] + ".json"
        # ABS's rule switches on a single value, as published; the other rules after k in a row
        settings = {"method": method, "threshold": round(threshold, 4), "margin": round(margin, 4),
                    "min_switch": 1 if rule == "abs" else k,
                    "rule": rule, "rule_thresholds": {name: round(t, 4) for name, t in rules.items()},
                    "margin_eq5": round(margin_eq5, 4), "margin_horizon": round(h * SAMPLE_DT, 3),
                    "margin_k1": round(margin_k1, 4),
                    "percentile": args.percentile, "risk": args.risk, "thresholds": sweep,
                    "walking_lowest": float(walking.min()), "walking_lowest_consecutive": float(rolling.min()),
                    "walking_tolerance": args.walking_tolerance, "filter_from": args.filter_from,
                    "trim": args.trim, "lipschitz_v": lipschitz_v, "lipschitz_f": lipschitz_f,
                    "threshold_states": int(len(v)), "margin_pairs": int(len(step)),
                    "walking_windows_below": float(np.mean(rolling < level))}
        with open(out, "w") as f:
            json.dump(settings, f, indent=1)
        print(f"Written {out}")


if __name__ == "__main__":
    main()
