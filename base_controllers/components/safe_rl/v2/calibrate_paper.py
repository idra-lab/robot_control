"""
@author: riccardobussola

Sec. II-C of the paper for the paper's V(o) (train.py --method paper), from the walking-policy
rollouts of the abs dataset (training data only), restricted to the rollouts the walking policy
survives ("pushes that do not require switching to the backup"):

  V_thr  the value lower than 99 % of the V values over every state of those rollouts
  L      max_t |V(o_t) - V(o_t+1)| / ||o_t - o_t+1||  x  max_t ||o_t - o_t+1||, over consecutive
         policy steps (raw observation), after removing the 5 % largest ||o_t - o_t+1|| and then
         the 5 % largest ratios of what remains (the paper's trimming)

The filter walks while V(o_k) >= V_thr + L (threshold_paper).  Also the walking rule
(threshold_walking: the 1st percentile of each rollout's lowest V before its push, as calibrate.py),
under which the margin is absorbed: calibrating V_thr + L on walking data gives the same cut whatever L.

    python3 calibrate_paper.py --model models/VF_paper_<robot>.onnx --abs-data data/<robot>/abs/batch*/ [--write]
"""
import argparse, json, os
import numpy as np
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--abs-data", nargs="+", required=True)
parser.add_argument("--write", action="store_true")
args = parser.parse_args()
sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
V = lambda o: sess.run(None, {"obs": o.astype(np.float32)})[0].ravel()

values, ratios, steps, minima = [], [], [], []
for folder in args.abs_data:
    d = np.load(os.path.join(folder, "v2_dataset.npz"))
    rows = np.atleast_1d(np.genfromtxt(os.path.join(folder, "observations_dataset_abs_test_episodes.csv"),
                                       delimiter=",", names=True))
    for row in rows:
        if row["discarded"] != 0:
            continue
        idx = (d["episode"] == row["episode"]) & (d["meta"][:, 7] == 0)
        obs, t = d["obs"][idx], d["meta"][idx, 0]
        if len(obs) < 2:
            continue
        v = V(obs)
        pre = t < 0
        if pre.any():
            minima.append(v[pre].min())
        if row["fallen"] > 0:
            continue
        values.append(v)
        do = np.linalg.norm(np.diff(obs, axis=0), axis=1)
        steps.append(do)
        ratios.append(np.abs(np.diff(v)) / np.maximum(do, 1e-9))
values, ratios, steps = map(np.concatenate, (values, ratios, steps))
v_thr = float(np.quantile(values, 0.01))
keep = steps <= np.quantile(steps, 0.95)
r, s = ratios[keep], steps[keep]
keep = r <= np.quantile(r, 0.95)
L_V, L_f = float(r[keep].max()), float(s[keep].max())
L = L_V * L_f
walking = float(np.quantile(minima, 0.01))
print(f"surviving walking rollouts: {len(values)} states; V percentiles 1/5/50: {np.round(np.quantile(values, [0.01, 0.05, 0.5]), 3)}")
print(f"V_thr {v_thr:.3f}, L_V {L_V:.4f}, L_f {L_f:.2f}, L {L:.3f}: threshold_paper {v_thr + L:.3f}")
print(f"walking rule (1st percentile of {len(minima)} rollouts' lowest pre-push V): {walking:.3f}; "
      f"rollouts whose pre-push V ever falls below threshold_paper: {np.mean(np.array(minima) < v_thr + L):.0%}")
minima = np.array(minima)
for p in (0.5, 0.6, 0.7, 0.8, 0.9):
    print(f"fixed probability p = {p}: walk while V >= p + L = {p + L:.3f}; normal-walking rollouts that would "
          f"switch before the push: {np.mean(minima < p + L):.1%}")
if args.write:
    path = os.path.splitext(args.model)[0] + ".json"
    info = json.load(open(path))
    info.update(v_thr=v_thr, lipschitz_L=L, L_V=L_V, L_f=L_f, threshold_paper=v_thr + L, threshold_walking=walking)
    json.dump(info, open(path, "w"), indent=2)
    print(f"written to {path}")
