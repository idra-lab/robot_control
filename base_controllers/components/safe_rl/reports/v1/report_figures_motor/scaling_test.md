<!-- @author: riccardobussola -->
# More handover data for our value function: offline gain, no closed-loop gain

**Question.** Our value function (MPS) and ABS's use the same 30 inputs. Ours learns from about 245
pushed handovers per robot, against about 100k labelled states for ABS. Does our value function
catch up with ABS if we collect ten times more handovers?

**Answer: no.** With 3800 training episodes, our value function switches earlier than ABS on
recorded trajectories. In the closed loop that earlier switch does not lower the failure rate:
- Aliengo ties with ABS.
- Go2 gets worse than ABS, and worse than the 350-episode model.

## Setup

- **Data.** The 0–0.3 s push-start warm-up protocol (`--switch-from push_start --switch-window 0 0.3`,
  `--motor-limit`), 3600 new episodes per robot. Seeds are seed+600 onwards, in
  `observation_datasets/motor/<robot>/mps_start_w03_big/`.
- **Models.** Trained on nested sets of 350, 1000, 2000 and 3800 episodes.
  - Each set is trained with each episode's handover transition weighted 1× and 50×
    (`train_value_function.py --handover-weight`).
  - All of them use the walking anchor and the same 220 held-out episodes.
  - Names: `models_motor/VF_big_n<N>_w<W>_<robot>`.
  - The 350-episode, weight-1 model reproduces the existing `mps_start_w03_walk` model exactly.
- **Offline ranking** (`rank_on_traces.py`). Each model runs on the 238 recorded walking-alone
  pushes (arm `watch_walk`).
  - Each model's level is set so that it switches needlessly as often as ABS does with its own
    rule: Aliengo 29 %, Go2 13 %.
  - It switches on the first single value below that level after the push.
  - The score is the share of the walking policy's failures caught with at least 0.3 s of lead. A
    missed failure counts as no lead.
- **Closed loop.** The two 3800-episode models are run with the walking rule and its variants. They
  use the same 240 pushes and the same motor limit as every other arm (`run_pipeline.sh`, table
  `big` in `summarize.sh`).

## Offline: earlier switches as the data grows

Share of the failures caught with at least 0.3 s of lead, at ABS's needless rate. The full tables
are in `aliengo_rank_scaling.md` and `go2_rank_scaling.md`.

| training episodes | Aliengo w1 | Aliengo w50 | Go2 w1 | Go2 w50 |
|---|---|---|---|---|
| ABS | 55 % | | 57 % | |
| 350 | 46 % | 42 % | 41 % | 28 % |
| 1000 | 50 % | 47 % | 16 % | 16 % |
| 2000 | 50 % | 50 % | 26 % | 28 % |
| 3800 | **60 %** | 57 % | 34 % | 46 % |

- **Aliengo:** the score rises steadily with the data. At 3800 episodes it beats ABS at every
  needless rate. Median switch is 0.154 s after the push, against ABS's 0.194 s.
- **Go2:** the score is not monotonic. At 3800 episodes with weight 50, it matches ABS at looser
  levels: 53 % against 59 % at 15 % needless, and 67 % against 68 % at 30 %.

## Closed loop: the gain does not transfer

Share of the pushes that ended in a failure, out of 236 (Aliengo) and 240 (Go2). "Needless" is the
share of switches in pushes the walking policy survives alone. "Caused" counts the failures in pushes
the walking policy survives alone. p is McNemar against `abs` on the paired episodes.

| arm | Aliengo failed | needless | caused | p | Go2 failed | needless | caused | p |
|---|---|---|---|---|---|---|---|---|
| walking alone | 46.6 % | – | – | | 46.7 % | – | – | |
| ABS, its rule | 12.3 % | 35.7 % | 1 | | 20.0 % | 21.9 % | 6 | |
| ABS, walking rule | 11.9 % | 51.6 % | 2 | 1 | **15.0 %** | 34.4 % | 5 | 0.036 |
| ours, 350 episodes (w03) | 12.3 % | 77.8 % | 1 | 1 | 20.0 % | 50.8 % | 9 | 1 |
| ours, 3800 w1 | 12.3 % | 50.0 % | 1 | 1 | 30.4 % | 45.3 % | 15 | 2e-4 |
| ours, 3800 w1, level 0.9 | **11.4 %** | 69.0 % | 1 | 0.86 | 20.8 % | 93.0 % | 15 | 0.89 |
| ours, 3800 w50, robust | 17.4 % | 34.1 % | 3 | 0.11 | 24.2 % | 42.2 % | 10 | 0.13 |

On Aliengo, the weight-50 model's plain walking-rule level is 0.087, set by one outlier walking
episode. That arm hardly switches (39 % failed), so the table shows its robust-rule arm. All the
variants are in `aliengo_loop_big.md` and `go2_loop_big.md`.

- **Aliengo:** the 3800-episode model ties with ABS (12.3 % against 12.3 %) at a higher needless rate
  (50 % against 36 %). It is no better than the 350-episode model, and none of the differences is
  significant.
- **Go2:** the 3800-episode model fails in 30 % of the pushes, significantly worse than ABS (20 %)
  and than the 350-episode model (20 %).

## Why the offline ranking did not predict the closed loop

On Go2, take the 68 rescuable pushes: the walking policy fails alone, and the safe policy survives
when it takes over at the push start.

| | ABS | ours, 350 episodes | ours, 3800 w1 |
|---|---|---|---|
| rescuable pushes saved | 56 | 55 | **40** |
| failures in pushes the walking policy survives | 6 | 9 | 15 |

The 3800-episode model fails in 35 pushes where ABS survives:
- It switched in all 35, at almost the same moment as ABS. The difference ranges from −0.27 s to
  +0.41 s, and it switched earlier in 14 of the 30 pushes where both switched.
- So switch timing doesn't explain these failures. The safe policy fails after a switch at
  practically the same time.

The offline score only measures how early a model switches on the walking policy's failures. It
does not see the state at the handover, or whether the safe policy can recover from it. That second
part decides the closed loop.

On Aliengo, ABS switches 0.04 s later than our 3800-episode model on the recorded trajectories,
and it still saves as many pushes. The two value functions answer different questions:
- ABS switches when the walking policy is in trouble.
- Ours switches when it predicts the safe policy will succeed.

With 3800 episodes, that prediction does not separate the handovers the safe policy recovers from
the ones it does not, at least on Go2.

## Conclusions

1. Label count is not the bottleneck. Ten times more handovers moves the offline switch time to
   ABS's level but leaves the closed loop unchanged on Aliengo and worse on Go2.
2. The 50× handover weight does not help in the closed loop.
   - On Aliengo it is worse (17 % at the robust rule, against 12 % with weight 1), and one outlier
     walking episode sets its walking-rule level.
   - On Go2 it is about the same as weight 1 (24–30 %).
3. Ranking on switch time alone is not a reliable proxy for the closed loop. A future offline score
   should replay the handover: switch to the safe policy at the model's switch time and record
   whether it recovers.
4. The best arms per robot remain:
   - Aliengo: ABS's walking rule, or our level 0.9, around 11–12 %. Nothing is significantly
     better than anything else.
   - Go2: ABS's walking rule, 15 %, the only significant improvement over ABS's own rule.
