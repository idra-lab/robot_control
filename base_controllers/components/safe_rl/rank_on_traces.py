"""
@author: riccardobussola

Value functions ranked on the same trajectories, offline: any models (.onnx) on the states a
walking-alone closed-loop arm recorded (saferl_controller_dataset.py, 50 Hz from 1 s), each
model's level set so that it would switch needlessly in the same share of the pushes the walking
policy survives, then when it would switch on the pushes that end in a failure.  A threshold only
moves the cut, it does not reorder the states: at an equal needless rate, the model that switches
earlier on the failures separates them better.

    python3 components/safe_rl/rank_on_traces.py --robot go2 --models VF_abs_go2.onnx ... \\
        [--arm watch_walk] [--data OUT] [--needless 0.15 0.3] [--table out.md]

The rule here is a switch on a single value below the level, from the push start.  Median switch and
lead, and the share with at least 0.3 s of lead, are over all the failures, a missed one counted as
never switching (no lead).  The first
column also gives the level that ABS's own rule (score 0.05) would need for the same needless
rate.  "walking": the share of the episodes whose undisturbed walking before the push already goes
below that level (a false switch the closed loop would pay).
"""

import argparse
import glob
import os

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "value_function", "observation_datasets", "motor")
STATE = 30
DT = 0.02  # s, the recording's period
FIRST = 0.98  # s, the time of sample 0's state (vf_from 1 s, one period before: the recorded obs_t)
ABS_THRESHOLD = 0.05


def load(robot, arm, data):
    """Per kept episode: its states, their times after the push start, and the episode's row."""
    episodes = []
    for folder in sorted(glob.glob(os.path.join(data, robot, "closed_loop", arm, "batch*", "")),
                         key=lambda f: int(f.rstrip(os.sep).split("batch")[-1])):
        base = os.path.join(folder, "observations_dataset_abs_test")
        states = np.load(base + ".npy")
        rows = np.atleast_1d(np.genfromtxt(base + "_episodes.csv", delimiter=",", names=True))
        for ep in rows[(rows["discarded"] == 0) & (rows["row"] >= 0)]:
            x = states[int(ep["row"]), :, :STATE]
            x = x[np.any(x != 0, axis=1)]
            episodes.append((x.astype(np.float32), FIRST + DT * np.arange(len(x)) - ep["push_time"], ep))
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--models", nargs="+", required=True, help=".onnx files; the first is the reference")
    parser.add_argument("--arm", default="watch_walk", help="a walking-alone closed-loop arm")
    parser.add_argument("--data", default=OUT)
    parser.add_argument("--needless", type=float, nargs="*", default=[0.15, 0.30],
                        help="more needless rates to compare at, besides the reference's own rule")
    parser.add_argument("--table", default=None, help="also write the table (markdown) here")
    args = parser.parse_args()

    episodes = load(args.robot, args.arm, args.data)
    fell = np.array([e[2]["fallen"] > 0 for e in episodes])
    failure = np.array([e[2]["end_time"] - e[2]["push_time"] for e in episodes])
    values = {}
    for path in args.models:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        values[path] = [session.run(None, {"obs": x})[0].ravel() for x, _, _ in episodes]

    def lowest_after(v):
        return np.array([vi[t >= 0].min() for vi, (_, t, _) in zip(v, episodes)])

    def rank(v, level):
        switch = np.array([t[np.argmax((t >= 0) & (vi < level))] if np.any((t >= 0) & (vi < level)) else np.nan
                           for vi, (_, t, _) in zip(v, episodes)])
        walking = np.array([np.any((t < 0) & (vi < level)) for vi, (_, t, _) in zip(v, episodes)])
        caught = np.isfinite(switch) & fell & (switch < failure)
        # over all the failures: a missed one never switches (no lead)
        switch_all = np.where(caught, switch, np.inf)[fell]
        lead_all = np.where(caught, failure - switch, 0.0)[fell]
        return {"needless": np.mean(np.isfinite(switch[~fell])), "caught": caught.sum() / fell.sum(),
                "switch": np.median(switch_all), "lead": np.median(lead_all),
                "lead03": np.mean(lead_all >= 0.3), "walking": walking.mean()}

    reference = values[args.models[0]]
    own = rank(reference, ABS_THRESHOLD)["needless"]
    rates = [own] + list(args.needless)
    lines = [f"# {args.robot.capitalize()}: ranking on the same {len(episodes)} pushes ({fell.sum()} failed, "
             f"{(~fell).sum()} survived, arm {args.arm})", "",
             f"Each model's level set to switch needlessly in the share of the survived pushes given; a single value "
             f"below it from the push start.  The first rate is {os.path.basename(args.models[0])} at the score "
             f"{ABS_THRESHOLD}.", ""]
    for rate in rates:
        lines += [f"## needless {100 * rate:.1f} %", "",
                  "| model | level | failures caught | median switch (s) | median lead (s) | lead >= 0.3 s | walking |",
                  "|---|---|---|---|---|---|---|"]
        for path in args.models:
            lows = lowest_after(values[path])
            level = np.quantile(lows[~fell], rate) + 1e-9
            r = rank(values[path], level)
            lines.append(f"| {os.path.splitext(os.path.basename(path))[0]} | {level:.3f} | {100 * r['caught']:.1f} % "
                         f"| {r['switch']:.3f} | {r['lead']:.3f} | {100 * r['lead03']:.0f} % | {100 * r['walking']:.1f} % |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.table:
        with open(args.table, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
