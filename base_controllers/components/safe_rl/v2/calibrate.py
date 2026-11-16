"""
@author: riccardobussola

The walking rule for a v2 mps model, from training data only: the threshold under which the filter
would switch during undisturbed walking in at most --rate of the walking-policy rollouts of the abs
dataset (their observations before the push, from 1 s).  Prints the threshold and the walk-switch
failures of the mps data (the safe policy handed a walking robot); --write stores it in the model's
.json as threshold_walking (the arm mps_wr of run_v2.sh), the default threshold left as it is.

    python3 calibrate.py --model <VF_mps_robot.onnx> --abs-data <batch folder> ... [--mps-data ...] [--rate 0.01] [--write]
"""
import argparse, glob, json, os
import numpy as np
from shield import Shield

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--abs-data", nargs="+", required=True)
parser.add_argument("--mps-data", nargs="*", default=[])
parser.add_argument("--rate", type=float, default=0.01)
parser.add_argument("--write", action="store_true")
args = parser.parse_args()
shield = Shield(args.model)
minima = []
for folder in args.abs_data:
    d = np.load(os.path.join(folder, "v2_dataset.npz"))
    for e in np.unique(d["episode"]):
        idx = (d["episode"] == e) & (d["meta"][:, 0] < 0) & (d["meta"][:, 7] == 0)
        if idx.sum():
            minima.append(min(shield.score(o) for o in d["obs"][idx]))
minima = np.array(minima)
threshold = float(np.quantile(minima, args.rate))
print(f"{len(minima)} walking rollouts; lowest Q(o, {shield.delta}) per rollout: percentiles 0/1/5/50 "
      f"{np.round(np.quantile(minima, [0, 0.01, 0.05, 0.5]), 3)}; share of rollouts below 0.9: {np.mean(minima < 0.9):.2f}")
print(f"walking-rule threshold (switch in at most {args.rate:.0%} of undisturbed walking): {threshold:.3f}")
if args.mps_data:
    rows = np.concatenate([np.atleast_1d(np.genfromtxt(os.path.join(f, "observations_dataset_abs_test_episodes.csv"),
                                                       delimiter=",", names=True)) for f in args.mps_data])
    w = rows[(rows["kick"] == 0) & (rows["discarded"] == 0) & (rows["timeout"] == 0)]
    print(f"handovers during undisturbed walking in the mps data: {len(w)}, failed {int(w['fallen'].sum())} "
          f"(knee {int(w['knee_contact'].sum())}, fall rule {int(w['fall_rule'].sum())})")
if args.write:
    path = os.path.splitext(args.model)[0] + ".json"
    info = json.load(open(path))
    info.update(threshold_walking=threshold, threshold_walking_rate=args.rate)
    json.dump(info, open(path, "w"), indent=2)
    print(f"written to {path}")
