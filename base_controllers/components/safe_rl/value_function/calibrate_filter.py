"""
@author: riccardobussola

The safety filter's threshold and margin for a value function (the paper, Sec. II-B, II-C), written
next to the model as <model>.json, which ValueFunctionManager and the controllers read.

    python3 calibrate_filter.py --model models/VF_mps_aliengo.onnx --threshold-data <dataset.npy> ...
                                --margin-data <dataset.npy> ... [--rule walking|percentile|risk] [--write]

Both sets of datasets are runs of saferl_controller_dataset.py with the walking policy alone,
recorded with --record-all --sample-rate 100 (every state, at the filter's rate).  --threshold-data
are runs of the evaluation's protocol, as long as the paper's (--episode-length 20, a push of the
evaluation's kicks in each): undisturbed walking before the push, then the push and its outcome.

* Threshold, --rule:
  walking (default): the highest threshold at which the filter never switches on the undisturbed
    walking states, threshold + L = their lowest V.  The filter then never interrupts normal walking
    (none of the calibration's, about 11 min of walking at 100 Hz) and acts as early as it can after
    a disturbance.
  percentile: the paper's statistical rule, V below 99 % of the values over all the states, walking
    and pushed (--percentile 1).  It is fragile: the percentile falls on the steep drop of V in the
    falls, so it moves with the share of falling states in the data (0.5th, 1st, 2nd percentiles
    0.38, 0.77, 0.92 for one model), and it can sit above walking values or far below them.
  risk: 1 - --risk; V is the probability that the safe policy recovers, so the filter accepts at
    most that chance of the safe policy failing once it takes over.  Above walking values it switches
    during normal walking.
  The thresholds of all three, and of more percentiles (--sweep), go in the .json for reference.
* Margin, Eq. (5): L = max_t |V(o_t) - V(o_t+1)| / ||o_t - o_t+1|| * max_t ||o_t - o_t+1||, on consecutive
  states 0.01 s apart, from runs the walking policy recovers from (--margin-data: light pushes).  As
  in the paper, the --trim % largest state changes are left out first, then the --trim % largest
  ratios of the rest.  The states are compared in the model's input normalisation (each component
  divided by its spread in the training data), so the norm does not depend on units.  The walking
  policy runs while V >= threshold + L.

The ABS baseline switches on the threshold alone, its margin is 0 (Eq. (5) is written for reference).
It also prints how often the filter would switch while walking undisturbed (the states before the
push in --threshold-data), at the threshold and at the threshold plus margin.
"""

import argparse
import json
import os

import numpy as np
import onnxruntime as ort

STATE = 30


def load_pairs(paths):
    """Consecutive (o_t, o_t+1) of every recorded sample, and per pair whether it came before the push."""
    states, next_states, before = [], [], []
    for path in paths:
        data = np.load(path)
        episodes = np.atleast_1d(np.genfromtxt(os.path.splitext(path)[0] + "_episodes.csv", delimiter=",",
                                               names=True))
        for ep in episodes[episodes["discarded"] == 0]:
            row = data[int(ep["row"])]
            row = row[np.any(row[:, :STATE] != 0, axis=1)]
            # samples 0.01 s apart, the last at the episode's end
            t = ep["end_time"] - 0.01 * np.arange(len(row))[::-1]
            states.append(row[:, :STATE])
            next_states.append(row[:, STATE:2 * STATE])
            before.append(t < ep["push_time"])
    return np.concatenate(states), np.concatenate(next_states), np.concatenate(before)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="the .onnx; the .pkl next to it gives the normalisation")
    parser.add_argument("--threshold-data", nargs="+", required=True)
    parser.add_argument("--margin-data", nargs="+", required=True)
    parser.add_argument("--percentile", type=float, default=1.0)
    parser.add_argument("--trim", type=float, default=5.0)
    parser.add_argument("--sweep", type=float, nargs="*", default=[0.5, 2.0],
                        help="more percentiles, whose thresholds go in the .json as thresholds (an ablation)")
    parser.add_argument("--rule", choices=("walking", "percentile", "risk"), default="walking")
    parser.add_argument("--risk", type=float, default=0.1, help="--rule risk: threshold 1 - RISK")
    parser.add_argument("--write", action="store_true", help="write <model>.json")
    args = parser.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    value = lambda x: session.run(None, {"obs": np.asarray(x, dtype=np.float32)})[0]
    import pickle
    with open(os.path.splitext(args.model)[0] + ".pkl", "rb") as f:
        trained = pickle.load(f)
    method = trained.get("method")
    spread = np.asarray(trained["std"]) + 1e-8

    states, _, before = load_pairs(args.threshold_data)
    v = value(states)
    walking = v[before]
    percentile_threshold = float(np.percentile(v, args.percentile))
    sweep = {f"{p:g}": round(float(np.percentile(v, p)), 4) for p in args.sweep}
    print(f"Threshold data: {len(v)} states of the walking policy, {before.mean():.0%} undisturbed walking; "
          f"V median {np.median(v):.3f}; walking V lowest {walking.min():.4f}, 0.1st percentile "
          f"{np.percentile(walking, 0.1):.4f}, 1st {np.percentile(walking, 1):.4f}")
    print(f"  percentile rule: {args.percentile:g}th percentile of V {percentile_threshold:.4f}; other percentiles: "
          + ", ".join(f"{p}th {t:.4f}" for p, t in sweep.items()))

    s0, s1, _ = load_pairs(args.margin_data)
    step = np.linalg.norm((s1 - s0) / spread, axis=1)
    dv = np.abs(value(s1) - value(s0))
    keep = step <= np.percentile(step, 100 - args.trim)
    ratio = dv[keep] / np.maximum(step[keep], 1e-9)
    ratio_keep = ratio <= np.percentile(ratio, 100 - args.trim)
    lipschitz_v = float(ratio[ratio_keep].max())
    lipschitz_f = float(step[keep][ratio_keep].max())
    margin = lipschitz_v * lipschitz_f
    print(f"Margin, Eq. (5): {len(step)} pairs 0.01 s apart, {args.trim:g} % of the largest steps then of the "
          f"largest ratios left out: L_V {lipschitz_v:.4f}, L_f {lipschitz_f:.3f}, L {margin:.4f} "
          f"(max |dV| over the pairs {dv.max():.4f}, 99th percentile {np.percentile(dv, 99):.4f})")

    if method == "abs":
        margin_used = 0.0
    else:
        margin_used = margin
    walking_threshold = float(walking.min()) - margin_used
    risk_threshold = 1.0 - args.risk
    threshold = {"walking": walking_threshold, "percentile": percentile_threshold, "risk": risk_threshold}[args.rule]
    print(f"Threshold, rule {args.rule}: {threshold:.4f} (walking {walking_threshold:.4f}, percentile "
          f"{percentile_threshold:.4f}, risk {risk_threshold:.4f}); margin used {margin_used:.4f}")
    for name, level in (("threshold", threshold), ("threshold + L", threshold + margin_used)):
        print(f"  walking undisturbed, V below the {name} ({level:.4f}): {np.mean(walking < level):.2%} of "
              f"{len(walking)} states; all states {np.mean(v < level):.2%}")

    if args.write:
        out = os.path.splitext(args.model)[0] + ".json"
        settings = {"method": method, "threshold": round(threshold, 4), "margin": round(margin_used, 4),
                    "rule": args.rule, "margin_eq5": round(margin, 4),
                    "rule_thresholds": {"walking": round(walking_threshold, 4), "percentile": round(percentile_threshold, 4),
                                        "risk": round(risk_threshold, 4)},
                    "percentile": args.percentile, "risk": args.risk, "thresholds": sweep,
                    "walking_lowest": float(walking.min()), "trim": args.trim, "lipschitz_v": lipschitz_v,
                    "lipschitz_f": lipschitz_f, "threshold_states": int(len(v)), "margin_pairs": int(len(step)),
                    "walking_below_threshold": float(np.mean(walking < threshold)),
                    "walking_below_threshold_margin": float(np.mean(walking < threshold + margin_used))}
        with open(out, "w") as f:
            json.dump(settings, f, indent=1)
        print(f"Written {out}")


if __name__ == "__main__":
    main()
