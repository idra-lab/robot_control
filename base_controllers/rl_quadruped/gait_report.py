"""Measure the gait a *deployed* policy actually produces, in safe_rl's own terms.

The counterpart of ``safe_rl/scripts/rsl_rl/eval_gait.py``: same quantities, same sections, same
order, computed from a recording made by :mod:`run_sim` instead of from an Isaac rollout.  Run
eval_gait.py on the checkpoint in Isaac and this on the deployment recording, and the two reports
can be read side by side - which is the only way to answer "it works in Isaac and not here" with
something better than an impression.

    python3 run_sim.py --record gz.npz
    python3 gait_report.py gz.npz --phase fwd0.3

Quantiles are p05 / p50 / p95, as in eval_gait.  Contact is "loaded" above CONTACT_THRESHOLD
newtons, matching the 2 N force threshold of the training env's contact sensor.
"""

import argparse
import os

import numpy as np

CONTACT_THRESHOLD = 2.0          # N, the training env's ContactSensorCfg.force_threshold
FOOT_ORDER = ["FL", "RL", "FR", "RR"]      # the policy's leg order
BODY_WEIGHT = 24.94 * 9.81       # N, aliengo


def _quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "no samples"
    p05, p50, p95 = np.quantile(values, [0.05, 0.5, 0.95])
    return f"p05 {p05:8.3f}   p50 {p50:8.3f}   p95 {p95:8.3f}   (n={values.size})"


def _episodes(loaded, time):
    """Completed air and stance durations for one foot, from its boolean contact trace."""
    air, stance = [], []
    if loaded.size < 3:
        return air, stance
    edges = np.flatnonzero(np.diff(loaded.astype(int)) != 0) + 1
    for start, end in zip(edges[:-1], edges[1:]):
        duration = time[end] - time[start]
        (stance if loaded[start] else air).append(duration)
    return air, stance


def report(path, phase=None, label=None):
    data = np.load(os.path.expanduser(path), allow_pickle=True)
    mask = np.ones(len(data["sim"]), dtype=bool)
    if phase is not None:
        mask &= data["phase"] == phase
    index = np.flatnonzero(mask)
    if index.size < 50:
        print(f"{path}: phase '{phase}' has too few samples ({index.size})")
        return
    mask[index[:index.size // 4]] = False            # drop the transient into the phase

    time = data["sim"][mask]
    grf = data["grf"][mask]
    pose, twist, angvel = data["pose"][mask], data["twist"][mask], data["angvel"][mask]
    action, command = data["action"][mask], data["cmd"][mask]
    foot_z = data["foot_z"][mask] if "foot_z" in data else np.full_like(grf, np.nan)

    loaded = grf > CONTACT_THRESHOLD

    print("\n" + "=" * 78)
    print(f"GAIT REPORT  --  {label or os.path.basename(path)}"
          + (f"   phase '{phase}'" if phase else ""))
    print("=" * 78)

    air_all, stance_all, per_foot_air = [], [], []
    for foot in range(grf.shape[1]):
        air, stance = _episodes(loaded[:, foot], time)
        per_foot_air.append(air)
        air_all += air
        stance_all += stance

    print("\n-- air time (s), one sample per completed swing")
    print(f"   {_quantiles(air_all)}")
    if air_all:
        print(f"   fraction of swings >= 0.30 s: "
              f"{np.mean(np.asarray(air_all) >= 0.30):.3f}")
    print("\n-- air time per foot (s)")
    for foot, name in enumerate(FOOT_ORDER):
        print(f"   {name + '_foot':<10} {_quantiles(per_foot_air[foot])}")

    print("\n-- contact-pattern synchrony (fraction of steps two feet share a contact state)")
    same = lambda a, b: float(np.mean(loaded[:, a] == loaded[:, b]))
    fl, rl, fr, rr = 0, 1, 2, 3
    print(f"   diagonal  FL/RR {same(fl, rr):.3f}   FR/RL {same(fr, rl):.3f}   (trot: near 1.0)")
    print(f"   lateral   FL/FR {same(fl, fr):.3f}   RL/RR {same(rl, rr):.3f}   (trot: near 0.0)")
    print(f"   fore/aft  FL/RL {same(fl, rl):.3f}   FR/RR {same(fr, rr):.3f}   (trot: near 0.0)")

    print("\n-- base attitude while walking")
    print(f"   pitch (deg):        {_quantiles(np.degrees(pose[:, 4]))}")
    print(f"   roll  (deg):        {_quantiles(np.degrees(pose[:, 3]))}")
    print(f"   pitch rate (rad/s): {_quantiles(angvel[:, 1])}")
    print(f"   roll  rate (rad/s): {_quantiles(angvel[:, 0])}")

    print("\n-- stance time (s), one sample per completed stance")
    print(f"   {_quantiles(stance_all)}")
    print(f"\n-- duty factor (fraction of time a foot is loaded): {loaded.mean():.3f}")

    if np.isfinite(foot_z).any():
        peaks = []
        for foot in range(grf.shape[1]):
            airborne = ~loaded[:, foot]
            edges = np.flatnonzero(np.diff(airborne.astype(int)) != 0) + 1
            for start, end in zip(edges[:-1], edges[1:]):
                if airborne[start]:
                    peaks.append(np.nanmax(foot_z[start:end, foot]))
        print("\n-- swing peak, sole clearance above ground (m), one sample per completed swing")
        print(f"   {_quantiles(peaks)}")

    touchdown = []
    for foot in range(grf.shape[1]):
        rising = np.flatnonzero((~loaded[:-1, foot]) & loaded[1:, foot]) + 1
        for start in rising:
            touchdown.append(grf[start:start + 20, foot].max() / BODY_WEIGHT)
    print("\n-- peak foot force at touchdown (body weights)")
    print(f"   {_quantiles(touchdown)}")

    # The action is held between inference ticks, so rate and jerk are taken on the policy's own
    # steps: differencing the held signal would report zeros for nine ticks out of ten.
    changed = np.flatnonzero(np.any(np.diff(action, axis=0) != 0, axis=1)) + 1
    policy_action = action[changed] if changed.size else action[::10]
    delta = np.diff(policy_action, axis=0)
    print("\n-- action rate  ||a_t - a_t-1||^2")
    print(f"   {_quantiles(np.sum(delta ** 2, axis=1))}")
    jerk = policy_action[2:] - 2.0 * policy_action[1:-1] + policy_action[:-2]
    print("\n-- action jerk  ||a_t - 2 a_t-1 + a_t-2||^2")
    print(f"   {_quantiles(np.sum(jerk ** 2, axis=1))}")

    yaw = pose[:, 5]
    cos, sin = np.cos(yaw), np.sin(yaw)
    vx = cos * twist[:, 0] + sin * twist[:, 1]
    vy = -sin * twist[:, 0] + cos * twist[:, 1]
    print("\n-- tracking error")
    print(f"   lin xy (m/s):  {_quantiles(np.hypot(vx - command[:, 0], vy - command[:, 1]))}")
    print(f"   yaw  (rad/s):  {_quantiles(np.abs(angvel[:, 2] - command[:, 2]))}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="+", help="npz files written by run_sim.py --record")
    parser.add_argument("--phase", default="fwd0.3",
                        help="phase to measure; the default matches eval_gait's pinned command")
    args = parser.parse_args()
    for path in args.recordings:
        report(path, phase=args.phase)


if __name__ == "__main__":
    main()
