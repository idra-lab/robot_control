# Why our MPS value function loses to ABS, and what would fix it

2026-11-12 · riccardobussola · based on `paper/original/root.tex`, `value_function_comparison.md`
(runs of 2026-11-05/10), `dataset_manager.py`, `train_value_function.py`, `value_function_manager.py`,
and safe_rl `safe_env_cfg.py`. No code was changed and nothing was rerun.

## Short answer

The losses don't come from a bug in the pipeline. They come from five assumptions behind the
MPS guarantee that our setup breaks. Each one hurts our value function more than ABS's:

1. **No guarantee covers pushes.** MPS certifies safety for a known model. A 0.2 s push of unknown
   size isn't in that model, so the method offers no guarantee for exactly the event we test.
   The most MPS can promise is that the robot stays in the backup policy's viability kernel.
   It can't be safer than its backup. ABS can be, because it keeps the walking policy on the pushes
   the walking policy absorbs better than the safe one.
2. **V is not a function of the state.** It sees 30 dimensions (gravity, angular velocity, q, q̇).
   It has no base linear velocity, no previous action and no history. A push mostly changes the base
   linear velocity, so V can't see it until the joints react. TD(0) on aliased inputs is biased.
3. **V is queried where it has no data.** Our labels start at the handover, 0–0.5 s after the push
   ends, and there is one handover per episode (≈440 per robot). The filter queries V during the
   push and in the next 50 ms. ABS labels every state of its own rollouts (≈100k samples), which is
   exactly the distribution it is queried on.
4. **The switch logic doesn't implement the MPS check.** MPS evaluates the backup from the *predicted
   next state*. We evaluate V on the current measurement, wait for 3 values in a row, then pay the
   actuator delay. The Lipschitz margin that should cover this is estimated on light pushes, so it
   comes out at 0.004–0.04, right where V moves fastest.
5. **The backup was never trained on what it is handed.** The safe policy trains from its own
   resets plus instantaneous ±1.5 m/s velocity kicks. It is never handed a walking policy's
   mid-gait state with the walking policy's last action in its buffer. Hence the knee speed
   overshoots 0.13 s after every handover, and the 17–31 % failures of "safe alone".

Our results show these effects directly. Ours switches **later** than ABS (0.23–0.29 s vs
0.19–0.21 s after push start) **and** more needlessly (Aliengo 54 % vs 36 %). That means V is a worse
classifier of the pushes, not a more permissive one. Ours fails about as often as "safe alone",
which is the most a backup-only shield can do. ABS beats "safe alone" on both robots, which is
possible only because it uses the nominal policy's own robustness.

## 1. What MPS actually guarantees

Classic MPS (Li 2020, Bastani 2021) gives forward invariance of X_safe under these assumptions:

| Assumption | Our setup |
|---|---|
| A1 Known dynamics, or disturbances bounded and accounted for robustly | Push of unknown force, direction and timing; the training data has none during the handover |
| A2 The certificate is exact, from the predicted next state f(x_t, π_nom) | Learned V, evaluated on o_t, not on x̂_{t+1} |
| A3 The backup acts at the next step | 3 evaluations, then a policy step, then actuator delay ≈ 50 ms |
| A4 Full state | 30-dim observation, no base velocity |

When A1 fails, the only true statement is this: if a disturbance takes the state out of
Viab(π_bkp), no backup-only shield can recover it. Taking a statement like that and treating it as
"MPS has a feasibility guarantee on recovery" is the paper's main overclaim (Intro: "allows us to
guarantee safety without any assumption on the backup policy"). The edited draft already softens
it. The guarantee holds only conditionally:

> If the push ends with the state in Viab(π_bkp), and V correctly certified the last state before
> the switch, the robot recovers.

ABS doesn't claim this guarantee, but in practice it can also win the pushes where π_nom
survives and π_bkp wouldn't. Our numbers confirm it:

| % failed (motor curve, §6.2) | Walking alone | Safe alone | ABS | Ours |
|---|---|---|---|---|
| Aliengo | 47.5 | 17.1 | **12.5** | 20.0 (1 value: 14.6) |
| Go2 | 46.7 | 29.2 | **20.0** | 32.5 (1 value: 27.5) |

Ours sits at about "safe alone". ABS is below both single policies.

## 2. Why V(o) without base velocity breaks the method

- A push is a change of momentum, mostly base linear velocity. V's input doesn't include it, so
  during the push and right after it, the observation looks like walking until the legs react.
  The capture point, which `x_inv` and the paper both rely on, needs exactly the variable V lacks.
- In a partially observed system, V(o) learns E[success | o] **under the training distribution of
  the hidden state**. Our training distribution (handover after the push ends) is not the query
  distribution (during the push), so V is optimistic during the push. That is why the switch is late.
- This also explains the `mps_early` ablation that failed. Adding in-push handovers made walking
  states look dangerous, because the observation can't tell "a push just started" from "walking".
  The conclusion in §2 of the comparison ("handing over during the push is not observable, so
  don't") treats a symptom. The right fix is to make the push observable, not to remove the data.
- TD(0) bootstraps V(o_{t+1}) as if o were Markov. With aliasing that adds bias. A Monte Carlo
  target, or TD(λ) with high λ, is the safer choice here, and the episodes are short.
- ABS has the same inputs but suffers less. Its labels come from the query distribution, so its
  E[· | o] is the right conditional expectation at deployment. It answers "is something bad
  happening to the walking robot?", which is observable from the early joint and attitude reaction.

**Fix:** give V what the policies already get: `lin_vel_b` (ground truth in sim, the
estimator on the robot) and the previous action. Add a short history (5–10 frames at 50 Hz) or a
momentum-observer residual (measured minus predicted base acceleration), which detects an external
force within a few ms. Then in-push handovers become learnable instead of harmful.

## 3. Data: one label per episode on the distribution that matters

| | Episodes | Samples on the query distribution (nominal, around the push) |
|---|---|---|
| ABS | 560 | every sample of every rollout: 101,047 on the Go2 |
| Ours | ≈440–490 | 1 per episode (the handover), ≈300 of them after a real push |

All other samples of ours are backup-policy states, which the filter never queries.
`--handover-weight` repeats those ≈300 samples but doesn't add information. Real-time Gazebo
is the bottleneck: 6 h per robot.

**Fix: the concurrent simulator, done offline and in parallel.** Generate the MPS data in
Isaac Lab, where the policies were trained, not in Gazebo:

1. Run N parallel envs with π_nom, random commands, and pushes from the evaluation's distribution
   (a force over 0.2 s, not a velocity reset).
2. At random times t_k, **including during the push**, clone the full sim state (root state,
   joints, policy action buffer, the remaining push force) into K branches. Each branch switches to
   π_bkp after a delay δ_j ∈ [0, 0.3] s.
3. Label x_{t_k} with the branch outcomes.

This is the original "simulate the backup from every state" idea, made cheap. With 4096 envs it
gives millions of nominal-state labels per hour instead of ≈300. It also gives Monte Carlo success
probabilities per state (K branches) rather than single Bernoulli draws.

## 4. The switch rule: certify the state you will actually hand over

The filter keeps π_nom while V(o_t) ≥ thr + L, then switches after 3 values in a row. The backup
starts acting ≈50 ms later (h in §1 of the comparison). The Lipschitz margin L is supposed to cover
those 50 ms, but:

- it is an empirical maximum, not a bound
- it is estimated on light-push walking, where V hardly changes (L = 0.004 on the Aliengo)
- ‖o_{t+1} − o_t‖ during a heavy push is far outside that calibration set.

**Fix: learn the value of a delayed switch directly.** Let

  Q_Δ(x_t) = P(π_bkp recovers | π_nom runs for Δ more, under the real disturbance, then switch).

Our data already supports this. For a handover at t_h, every earlier nominal state x_s is a
labelled sample of Q_{t_h − s}(x_s). Train with Δ as an input, or fixed at the latency plus a small
horizon. This:

- replaces the Lipschitz margin with the quantity the margin approximates
- is the MPS check "from the next state, including what happens in between"
- turns every nominal state of an episode into training data, not only the handover.

With the branching data of §3 it comes for free: δ_j is Δ.

## 5. The threshold throws away the probabilistic meaning

- The walking rule sets the threshold just below V on undisturbed walking. On the Aliengo that is
  0.986: decisions happen in the saturated tail of an unbounded linear output (AUC 0.88). On the
  Go2 one calibration episode set it at 0.136.
- If V is calibrated on the query distribution (§2–3), the threshold should be what V means: a
  required success probability, e.g. switch when Q_Δ < 0.9. This is the selling point of a
  probabilistic VF over ABS's score, and the current calibration doesn't use it.
- Bound the output with a sigmoid and train with cross-entropy on Monte Carlo branch outcomes.
  Its calibration on pushed states is then directly measurable (reliability diagram on the test pushes).

## 6. The backup policy sets the ceiling

In MPS the backup's viability kernel *is* the shield's safe set, so this has the most leverage.
In `safe_env_cfg.py`:

- Resets come from standing or near-standing poses (plus a random base velocity), and pushes are
  instantaneous velocity resets (`push_by_setting_velocity_tracked`, ±1.5 m/s). The evaluation uses
  a 0.2 s force, with kicks up to 2.0 m/s on the Aliengo.
- The policy is never started from the walking policy's state: mid-gait, legs in swing, the walking
  policy's last action in its observation. Every handover is out of its distribution. That fits
  the overshoots at a median 0.13 s after the switch.
- Nothing in the reward or the termination conditions keeps joint speed within the limits (the
  knee overshoots).

**Fix:** train or fine-tune π_bkp on the handover distribution. Spawn it by running π_nom with
force pushes and switching at random delays, including during the push, with π_nom's action in
the buffer. Add a joint-velocity limit term. This enlarges Viab(π_bkp) where the shield uses it,
and nothing else in the method changes.

## 7. Show the guarantee where it matters

The current benchmark favours ABS by construction. ABS's V is trained on the walking policy under
the same push distribution it is tested on, and our backup is weak. MPS's real structural
advantages are elsewhere, and the paper doesn't test them:

- **V doesn't depend on the nominal policy.** Swap π_nom (the MPC, a faster gait, commands up to
  1 m/s) without retraining: ABS's V is stale, ours is not, if V conditions on state rather than on
  the nominal's habits.
- **Out-of-distribution disturbances.** Longer or weaker-but-sustained pushes (a load or a pull),
  pushes during turning, terrain. ABS learned "what a push looks like"; ours learns "can the backup
  recover from here".
- **Interpretable risk level.** For a calibrated Q_Δ, the failure rate after a switch should track
  1 − thr. Plot it.

## 8. Optional: a formulation that can't do worse than either baseline

If the goal is simply to beat ABS, make the switch time an optimal-stopping problem over both
policies:

  V_stop(x) = P(π_bkp recovers from x)      (ours)
  V_cont(x) = E[ max(V_stop(x'), V_cont(x')) ]  along π_nom, with 0 at failure and 1 once the push is absorbed

Switch when V_stop(x_{t+Δ}) ≥ V_cont(x_t). The data is both datasets we already collect. If both
values are exact, the result is at least as safe as "never switch", "always switch" and the MPS rule.
It keeps the MPS reading ("switch while the backup can still save it"), but stops switching on pushes
the walking policy handles better (our 33–54 % needless switches). It is still MPS in the sense of
certifying the backup. It adds the ABS-like information as a second value, instead of choosing one.

## Suggested order

| # | Change | Cost | Expected effect |
|---|---|---|---|
| 1 | Add `lin_vel_b`, the last action and a short history to V's input. Retrain on the current data, add in-push handovers | small | Earlier, sharper switches; `mps_early` stops hurting |
| 2 | Q_Δ relabeling of existing episodes; drop the Lipschitz margin; probabilistic threshold | small | Removes latency and miscalibration |
| 3 | Isaac Lab branching data generator (§3) | medium | 3–4 orders of magnitude more labels where they matter |
| 4 | Fine-tune π_bkp on the handover distribution with a joint-speed term | medium | Raises the ceiling: "safe alone" is the bound for ours |
| 5 | Experiments of §7, optionally §8 | medium | The comparison where MPS has a structural advantage |

Steps 1–2 can be tested on the current Gazebo pipeline (`run_pipeline.sh`, `VFS=mps`) before
investing in 3–4. If ours still sits at "safe alone" after them, the backup is the limit and step 4
is the one that matters.
