# Value function: paper vs implementation

2026-11-03 · riccardobussola · paper `Safe_Locomotion_using_MP_Shielding.pdf` (draft, Sec. II-B, II-C,
III, IV) against `robot_control` master f5daa78 (`components/safe_rl`, `rl_quadruped`). No code was
changed.

## Summary

The training target and the TD loss follow the paper: Eq. (2)–(4), TD(0) with a target network
updated by an exponential moving average. The data behind them does not. The paper's V is the
probability that the **backup** policy avoids a fall and reaches X_inv when it is started from x
(Sec. II-B). It explicitly contrasts this with ABS, whose V comes from rollouts of the nominal
policy. Our dataset runs the **nominal** (walking) policy for the whole episode, so our V is the
ABS-style value: the chance that the walking policy recovers. Our README and report define V that
way. This was inherited from master, where the dataset had been configured for the ABS baseline,
and our fixes kept it.

Two other differences change what V means. A success label is also given for surviving 3 s after
the push. X_inv, as implemented, is "the base moves slower than about 0.25 m/s for 0.5 s", which
is not static equilibrium. The switching rule has neither the Lipschitz margin nor the paper's
threshold-selection procedure. The closed-loop numbers in our report (falls 48 % → 8 % Aliengo,
45 % → 20 % Go2) therefore measure an ABS-style filter, under a protocol that differs from Sec. III.
They are not comparable with Table I.

## Discrepancies

### Major: they change what V estimates

**1. Which policy V is learned for.**
- Paper (II-B): the data comes from π_nom with random commands and a perturbation, then the
  rollout *switches to π_bkp*. V(x) is the likelihood of avoiding a fall and reaching X_inv "when
  π_bkp is activated starting from x". The introduction states this is what differs from ABS, whose
  V comes from nominal rollouts and "does not guarantee recovery from the backup policy".
- Implementation: `DatasetManager.backup_delay = None` by default
  ([dataset_manager.py:63](../../dataset_manager.py#L63),
  [:240](../../dataset_manager.py#L240)), so the walking policy runs
  for the whole episode and every transition is a π_nom transition. The README and report define V
  as "the chance that the walking policy recovers".
- Origin: Jessica's commits 0814e22 "Add option to use previous nominal policy for dataset" and
  8656053 "Update creation of dataset for ABS" (May 2026) replaced the switch to the safe policy at
  `warmup_time` with `if True:  # nominal policy`. Master 7b46959 was in that state, and our series
  kept it. Before 0814e22 (e.g. 775aad6) the dataset switched to π_bkp after the push, as in the
  paper.
- Consequence: the filter switches when π_nom is predicted to fail, with no estimate of whether
  π_bkp can still recover from that state. This is the ABS failure mode the paper describes, and it
  shows in our closed loop. On Go2, all 21 falls left with the filter came after a switch
  0.18–0.37 s before the fall (`report_figures/go2_summary.txt`): the filter switched, but too late
  for the safe policy. Our models correspond to the paper's "RL/RL (ABS)" rows, not to "RL/RL".
- Existing knob: `backup_delay` (attribute only, no CLI flag). For example, `0.2` hands over to
  π_bkp at the end of the push, which is where recording starts.

**2. Success label: surviving counts as success.**
- Paper (Eq. 3–4): the target is 1 only when x_t ∈ X_inv (I_CP). An episode that neither falls nor
  reaches X_inv has no terminal label.
- Implementation: `success = capture or survived`
  ([dataset_manager.py:164](../../dataset_manager.py#L164)).
  "Survived" means not fallen 3 s after the push (`post_push_time`). In the training data more
  episodes end as survived than as captured: Aliengo 143 vs 99, Go2 131 vs 97 of 420. Under π_nom
  walking at a command, X_inv is rarely reached, which is why this label was needed. With π_bkp
  (item 1) it would not be.

**3. X_inv as implemented.**
- Paper: X_inv is the set of static equilibrium states. For the RL design (II-D, III-B), trunk and
  joint angular velocities must be below 0.1 rad/s and trunk linear velocity below 0.1 m/s. For the
  One-NN scheme (III-C), the CP must lie inside the scaled support polygon with four feet in contact.
- Implementation: the CP is compared with the base's own ground projection, `cp_local = cp - base`
  ([dataset_manager.py:112](../../dataset_manager.py#L112)). That
  difference is v·√(h/g), so "CP within 5 cm" only means a base horizontal speed below about
  0.25–0.3 m/s (h ≈ 0.3–0.37 m), held for 0.5 s. Feet, the support polygon, joint velocities and
  angular velocity are not checked. A robot walking or turning slowly is labelled captured:
  74 of 99 Aliengo and 76 of 97 Go2 captures had a nonzero command, with linear commands up to
  0.31 and 0.47 m/s. This definition was inherited from master; only the radius (0.02 → 0.05 m)
  and the 0.5 s hold changed.

**4. Shielding rule and threshold.**
- Paper (II-C, Eq. 5): with sensor feedback, keep π_nom while V(o_t) ≥ V_thr + L, where
  L = L_V·L_f is estimated from 1,000 nominal runs with 12 push magnitudes (with the top 5 %
  trimmed). The threshold is chosen statistically, as a value below 99 % of the V values over
  10,000 nominal runs.
- Implementation: switch on V(o_t) < 0.5, with `vf_additional_term = 0` everywhere
  ([value_function_manager.py:64](../../value_function/value_function_manager.py#L64)),
  no estimate of L, and no prediction of the next state. The threshold 0.5 was chosen from the
  evaluation and the closed loop (warned falls against false alarms), not by the percentile rule.
  The paper's Gazebo tests also omitted the margin ("the additional term was not considered"), so
  this matches those tests but not the method in II-C.

### Medium: protocol and definitions

**5. X_safe (fall rule).**
- Paper: a fall is the trunk or any hip below 0.1 m (III-A). X_safe also excludes joint-limit
  violations, and the results count knee contacts and limit violations as failures.
- Implementation: fallen when inclination exceeds 45° or the base is below 0.2 m
  ([dataset_manager.py:11](../../dataset_manager.py#L11)). This fires
  earlier than the paper's rule; `fall_rule.py` shows no recovery past 29° (Aliengo). Joint-limit
  violations are not labelled unsafe: episodes more than 0.5 rad outside the range are dropped
  ([:391](../../dataset_manager.py#L391)), none in the current data,
  and smaller violations are labelled normally. Knee contact is not checked.

**6. Perturbation.**
- Paper: II-B says "base velocity perturbations" for the training data; the evaluation (III-A)
  uses a 150–250 N force for 0.2 s.
- Implementation: the dataset uses a 0.2 s force of mass·kick/0.2. Kicks are 0–3 m/s for Aliengo
  and 0–2.5 m/s for Go2. 150–250 N corresponds to kicks of 1.2–2.0 m/s on Aliengo (24.9 kg), which
  is covered, and 1.9–3.1 m/s on Go2 (16.1 kg), whose upper part is not. The paper's text describes
  the old base-twist push, which in Gazebo also zeroed the joint velocities.

**7. Evaluation protocol.**
- Paper (III-A, III-D): 21 force magnitudes × 100 directions × 10 instants, 20 s runs, and π_bkp
  kept until the end once triggered. Failures are falls, knee contacts and joint-limit violations.
- Implementation (`compare_filter.py`, `run_pipeline.sh`): random kick and direction 2–3 s after
  the start, episodes end 3 s after the push, the filter switches back after 2 s above threshold,
  and only falls are counted (item 5's rule).

**8. Switching back on the robot.**
- Paper (III-D): after π_bkp has held V above the threshold for a set number of iterations, the
  robot returns to π_nom, and V is ignored for 1 s of recovery time.
- Implementation: the robot returns after 200 evaluations (2 s), and V is checked again
  immediately ([rl_quadruped_controller.py:944](../../../../rl_quadruped/rl_quadruped_controller.py#L944)).
  The 1 s delay applies only when the operator starts RL. The paper's robot setting was a 0.8
  threshold with 1 consecutive value; ours is 0.5 with 1.

**9. Observation source.**
- The value function's inputs (projected gravity, angular velocity, q, q̇; no history) are
  measurable quantities, as in the sensor-based scheme of II-C. In the dataset, however, the first
  two come from Gazebo ground truth (`state_estimation = 'ground_truth'`,
  [saferl_controller_dataset.py:149](../../../../saferl_controller_dataset.py#L149)), with no
  IMU noise or bias. On the robot they come from the IMU. Table I's "Sensor" rows show how large
  that gap can be.

### Minor

10. **Indicator indexing.** Eq. (3)–(4) attach I_fall and I_CP to x_t. Our transitions
    (x_t, x_{t+1}, flags) carry the flags of x_{t+1}, so the target is
    (1 − I_fall(x_{t+1}))[(1 − I_CP(x_{t+1}))·V'(x_{t+1}) + I_CP(x_{t+1})]. This is the usual
    terminal handling and equivalent in effect. Only the paper's notation differs.
11. **Network.** The paper has ELU "followed by a normalization layer" in each layer; ours is
    Dense → LayerNorm → ELU. The paper states V ∈ [0, 1]; our output is linear (bias initialised
    to 1) and is not bounded.
12. **Rates.** Policy and TD transitions run at 50 Hz (Δt = 0.02 s, `policy_rate` 50), and V is
    evaluated at 100 Hz. The paper's low-frequency loop is 100 Hz. This matters for L_f if Eq. (5)
    is implemented.
13. **Not in the paper.** Held-out Brier early stopping, padding excluded from training, a fixed
    dataset mean/std normalisation, the 0.5 s capture hold, and torque noise of 10 Nm std per
    joint per 2 ms tick from the push on. The paper gives no noise magnitude.
14. **The ABS 10-sample split.** Our commit 4831aff removed it as a bug. It came from 8656053
    "Update creation of dataset for ABS": the last 10 samples (0.2 s) before a fall became their
    own row, and the rest of the episode a row padded with success labels. That may have been a
    deliberate finite-horizon labelling for the ABS baseline. If the paper's ABS rows were produced
    with it, our pipeline does not reproduce that baseline.

## What matches

- The TD(0) target, Eq. (4): `(1 − fallen)·((1 − cp)·V'(x') + cp)` with V' from the target
  network, an EMA with τ = 0.01 per batch, and a squared loss, Eq. (2).
- The architecture: 512–256–128, ELU, with a normalisation layer.
- Random velocity commands and additive Gaussian torque noise during data collection.
- The two-policy scheme of III-D: the walking policy and a separate safe policy at zero command.
- A minimum of 1 consecutive value below the threshold to switch.
- V evaluated on the current measurements, without state estimation.

## To discuss with Jessica

- Which dataset produced the paper's "RL/RL" and "RL/RL (ABS)" value functions (the pre-0814e22
  backup switch, the `if True` nominal version, the 10-sample split), so that each can be
  reproduced.
- Whether the paper's VF should be retrained on π_bkp rollouts (item 1). If so, which X_inv to use
  (item 3), and whether "survived" should still count as success (item 2).
- The threshold procedure and Eq. (5): whether L was ever estimated for the RL/RL model, and its
  value.
- Sec. II-F (policy training) is still assigned to me in the draft.
