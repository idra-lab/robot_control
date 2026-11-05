# Safe RL: value function and safety filter

@author: riccardobussola

The walking policy can be pushed over. The safety filter, model predictive shielding with a learned
value function, evaluates V(x) 100 times a second and hands the robot to the safe policy (the
zero-command recovery policy) when V drops below a threshold plus a margin; once V has been back
above it for 2 s, the walking policy takes over again (the controllers; in the paper's simulations
the safe policy keeps the robot).

Two value functions come out of the same pipeline:

| | `mps`: the paper's value function | `abs`: the ABS baseline (He et al., RSS 2024) |
|---|---|---|
| Data | the walking policy, a push, a warm-up from the push end, then the **safe policy** from the switch on | the **walking policy**'s whole rollouts |
| V(x) | the chance that the safe policy, started at x, brings the robot to rest without leaving the safe set | ABS's discounted reach-avoid value: does the walking policy track its command again before a failure (V_RA <= 0) |
| Switch | V < threshold + L for 3 evaluations, L the margin of Eq. (5), the threshold from the walking rule | ABS's rule, V_RA >= -0.05 (the model outputs the score -V_RA, so a threshold of 0.05) |

Leaving the safe set is a fall (45°, base below 0.2 m, or the base or a hip below 0.1 m), a knee on
the ground or a joint past its limit, the same in the data, in the training and in the
evaluation.

[value_function_comparison.md](value_function_comparison.md) has the results on the Aliengo and
the Go2, the ablations and what was decided; `paper/` (local only) holds the paper and the proposed
changes. The pipeline runs for the Aliengo and the Go2 (`--robot`, policies in
`rl_quadruped/policies/<robot>_velocity*`).

## Quick start

```bash
# the robot controller with the safety filter (lab container, rl_quadruped/)
python3 rl_quadruped_controller.py --robot go2 --safety-filter                 # Gazebo
./run_rl_controller.sh --robot go2 --real --input joy --safety-filter          # robot

# watch the filter in Gazebo: pushes, V in rviz and rqt_plot (lab container, base_controllers/)
python3 saferl_controller.py --robot go2 --plot

# the whole study for a robot: data, training, calibration, evaluation, closed loop, figures
# (host, base_controllers/)
components/safe_rl/run_pipeline.sh go2
```

The filter loads the robot's default model, `value_function/models/VF_abs_<robot>.onnx` (ABS's
reach-avoid value, the one that left the safe set least in closed loop on both robots, see the
comparison), with its threshold, margin and switching count from the
`.json` next to it; nothing besides the lab image is needed to run it. `--filter-model` runs
another one.

## Files

Paths from `robot_control/base_controllers`, where every command below runs: the scripts marked
*host* on the host (they start their own containers), the others in the lab container.

| File | What it does |
|---|---|
| `components/safe_rl/run_pipeline.sh` | *host*: the whole study for one or more robots, fall-rule check to figures |
| `saferl_controller_dataset.py` | push episodes in Gazebo: the datasets (`--method abs/mps`), the calibration runs, the closed loop (`--filter`), the fall-rule check (`--fall-check`) |
| `components/safe_rl/dataset_manager.py` | one episode (reset, command, push, switch, labels, failures), a batch and its files; per-robot settings in `ROBOT_SETTINGS` |
| `components/safe_rl/run_batches.sh` | *host*: batches of `saferl_controller_dataset.py` in parallel containers |
| `components/safe_rl/run_in_image.sh` | *host*: a command in a throwaway container of the lab image (training, calibration, evaluation, summary) |
| `components/safe_rl/fall_rule.py` | from a `--fall-check` run: where the fall rule and the capture hold can sit |
| `components/safe_rl/dataset_statistics.py` | a dataset's outcomes, fall rate against the kick (k50), commands, joint ranges |
| `components/safe_rl/value_function/train_value_function.py` | TD(0) training (JAX), `--method abs` (with the ABS horizon) or `mps`; saves `.pkl` and its `.onnx` export |
| `components/safe_rl/value_function/calibrate_filter.py` | a model's threshold (walking, percentile or risk rule) and margin L (Eq. 5), into `<model>.json` |
| `components/safe_rl/value_function/network.py` | the network in flax, for training, export and evaluation |
| `components/safe_rl/value_function/export_onnx.py` | `.pkl` to `.onnx`, the format the filter runs; fails if ONNX and JAX disagree |
| `components/safe_rl/value_function/evaluate_value_function.py` | V on held-out episodes: separation, warning time, false alarms, Brier score, calibration |
| `components/safe_rl/compare_filter.py` | the same pushes run by several arms: falls, the paper's failures, switches, paired tests, the falls against needless switches plot |
| `components/safe_rl/summarize.sh` | a robot's figures and numbers: `report_figures/<robot>_*` and `<robot>_summary.txt` |
| `components/safe_rl/value_function/value_function_manager.py` | the filter: V, threshold, margin and the switch, for any controller |
| `saferl_controller.py` | Gazebo test to watch: walking, pushes, V in rviz and `rqt_plot` |
| `rl_quadruped/rl_quadruped_controller.py --safety-filter` | the robot controller with the filter (off by default) |
| `vf_rosbag.py` | V of a recorded bag |
| `components/safe_rl/value_function/models/VF_<name>_<robot>.{pkl,onnx,json}` | the models: `abs`, `mps`, `mps_early`, on the Aliengo `mps_polygon`; the default per robot in `DEFAULT_MODEL` |
| `components/safe_rl/report_figures/` | the figures and the numbers behind them |

## Where to run

Everything runs in the locosim image (`mfocchi/trento_lab_framework:locosim`), from
`$LOCOSIM_DIR/robot_control/base_controllers`. The *host* scripts start their own containers of
it; each Gazebo needs its own ROS master, so parallel simulations need separate containers. rviz
and the Gazebo GUI need the lab container with the display, and the NVIDIA OpenGL libraries in it:
`--gpus all` alone gives the container compute only, so add `--env="NVIDIA_DRIVER_CAPABILITIES=all"`
to its `docker run` (the `lab_locosim` alias).

Training uses JAX on the CPU (the image has jax 0.4.13, flax and optax) and exports ONNX with
torch; it also needs tqdm and the onnx package, `pip3 install tqdm onnx` once per container
(`run_in_image.sh` does it). The filter itself runs the ONNX on onnxruntime, in simulation and on
the robot, without JAX.

## Run it all

From the host:

```bash
cd ~/trento_lab_home/ros_ws/src/locosim/robot_control/base_controllers
components/safe_rl/run_pipeline.sh              # aliengo and go2
components/safe_rl/run_pipeline.sh go2          # one robot
STAGES="calibrate loop summary" components/safe_rl/run_pipeline.sh aliengo     # some stages
```

The stages, in order (`run_pipeline.sh` has the details):

1. **data:** the fall-rule check (episodes run to the ground); per method 6 training batches of 70
   episodes (the 6th stops the training) and 2 test batches; `mps_early` (handovers from the push
   start, an ablation).
2. **calib-data:** the walking policy alone, every state at 100 Hz: 80 episodes of 20 s with one
   push of the evaluation's kicks (the threshold) and 240 with light pushes (the margin).
3. **train:** `VF_abs` (ABS's reach-avoid value), `VF_mps`, `VF_mps_early`, and on the Aliengo
   `VF_mps_polygon` (X_inv by the capture point); a model whose `.onnx` exists is kept (delete it to retrain).
4. **calibrate:** each model's threshold, margin and switching count into its `.json`.
5. **evaluate:** each model on its method's test batches.
6. **loop:** the same 240 pushes per arm (the walking policy alone, the safe policy alone, ABS at its
   rule and at the walking rule, ours at the walking rule, at the paper's percentile rule, without
   the margin, on a single value, switching back; the ablations), every episode 7 s long unless the
   robot leaves the safe set.
7. **summary:** `summarize.sh`, the tables and figures in `report_figures/`.

A stage skips the batches already complete, so an interrupted run resumes; delete a folder to redo
it. About 3 h for the Aliengo (with its ablations) and 2.5 h for the Go2 on 24 cores, 8 simulations
at a time in real time; `CONCURRENT=1` runs a second robot alongside (16 simulations, the two
sharing each arm's wait). Seeds are fixed per robot (`SEED`, Aliengo 1000, Go2 2000, plus
an offset per run), so a rerun sees the same commands and pushes; Gazebo is not deterministic, so
its numbers move a little. Environment variables change the sizes (`FALL_EPISODES`, `EPISODES`,
`THR_EPISODES`, `MARGIN_EPISODES`, `LOOP_EPISODES`, `LOOP_LENGTH`, `EPOCHS`) and the outputs
(`OUT`, `MODELS`, `FIGURES`), e.g. a quick trial that leaves the study's runs alone:

```bash
OUT=components/safe_rl/trial MODELS=components/safe_rl/trial FIGURES=components/safe_rl/trial \
FALL_EPISODES=3 EPISODES=10 THR_EPISODES=2 MARGIN_EPISODES=4 LOOP_EPISODES=5 EPOCHS=50 \
components/safe_rl/run_pipeline.sh go2
```

The steps below run the same tools one by one. The commands use

```bash
ROBOT=go2        # or aliengo
O=components/safe_rl/value_function/observation_datasets/$ROBOT
M=components/safe_rl/value_function/models
npy() { ls $1/batch*/observations_dataset_abs_test.npy; }
```

## 1. Check the fall rule

A new robot first needs its fall rule: an inclination and a base height past which it never
recovers. Run episodes until the robot is on the ground, from the host:

```bash
components/safe_rl/run_batches.sh 2000 8 30 $O/fall_check --robot $ROBOT --fall-check    # host
python3 components/safe_rl/fall_rule.py $(npy $O/fall_check) --plot fall_rule.png
```

`--fall-check` ends an episode only with the robot on the ground (inclination over 85° or base below
0.1 m) or 3 s after the push, and never at capture. `fall_rule.py` prints the gap between the
recoveries and the falls, whether the rule sits in it, and, per capture hold, the captured robots
that fell afterwards. The rule and the kick range go into `ROBOT_SETTINGS` in `dataset_manager.py`.
With the torque noise of 2026-11-04 the Aliengo recovered once from 51° (the rule is 45°; 60° would
sit in the gap), the Go2 never past 23°, and every fall went past 85° (Go2 50°).

## 2. Generate push episodes

```bash
python3 saferl_controller_dataset.py --robot $ROBOT --method mps --episodes 70 --seed 2201 --no-rviz --out $O/mps/episodes/batch1
```

An episode:

- **Reset.** With gravity off, the PD drives the robot into the policy's default posture (0, 0.9,
  −1.8); it is set down at its standing height and waits until still.
- **Command.** vx, vy and wz uniform in ±0.5, the training range of both robots; 10 % of episodes
  standing, 10 % only turning.
- **Push.** 2–3 s after the start (`--push-window`): a horizontal force on the base, robot mass ×
  kick / 0.2 s for 0.2 s, the push of `rl_quadruped/push_sweep.py`. The kick is uniform in 0–3 m/s
  (Aliengo) or 0–2.5 m/s (Go2), or drawn from `--kicks`; the direction is uniform.
- **Torque noise** (`--noise-model`): `ou` (default), an Ornstein-Uhlenbeck process per joint, std
  `--noise` × the joint's torque limit (0.04) and time constant `--noise-tau` (0.05 s), throughout;
  `white`, `--noise` Nm per tick from the push on (the datasets of 2026-10-29: with it the robot never
  comes to rest); `none`.
- **Safe set.** An episode ends, labelled fallen, when the robot leaves it: the fall rule
  (inclination over 45° or base below 0.2 m), the base or a hip joint below 0.1 m, a knee joint below
  3.5 cm, or a joint position past its URDF limit by more than 0.01 rad or a velocity past its limit
  (`failures` in `dataset_manager.py`; `--fall-check` keeps the fall rule alone).
- **`--method abs`:** the walking policy throughout, recorded whole from 1 s, and the episode ends
  leaving the safe set or 3 s after the push. Per sample ABS's reach-avoid margins: zeta, +1 out of
  the safe set and −1 in it, and l = tanh(log(e / 0.2 m/s)), e the walking policy's tracking error
  (base velocity averaged over 0.5 s against the command, plus 0.2 × the yaw-rate error), so l ≤ 0
  when it tracks its command again.
- **`--method mps`:** the walking policy for a warm-up uniform in `--switch-window` (0–0.5 s) after the
  push end (`--switch-from push_start` counts from its start, the `mps_early` ablation: the push's
  onset is not in the state, so those handovers drag down V of walking states too); in 30 % of the
  episodes (`--walk-switch`) there is no
  push and the switch comes during undisturbed walking. Samples from the switch; the episode ends
  leaving the safe set, with the robot in both invariant sets, or 8 s after the switch (a timeout,
  left out of the training). Success is static equilibrium: the robot on its feet (all four in
  contact), upright (under 10°) and at height (over 0.75 × its standing height), with trunk and joint
  angular velocities below 0.1 rad/s and trunk linear velocity below 0.1 m/s averaged over 0.5 s (at
  any instant the safe policy keeps correcting and the noise shakes the joints), all held 0.3 s. The
  other invariant set, the capture point in the support polygon scaled by 0.8 with the same posture,
  is saved for the ablation.

Each episode is seeded by (seed, its index), so runs with the same `--seed` see the same commands,
pushes and switches. Gazebo runs in real time (`--speed`), about 6 s per episode; an episode where
the Python loop missed over 1 % of the 500 Hz control ticks is discarded.

From the host, batches in parallel: `run_batches.sh <first seed> <batches> <episodes> <out folder>
[arguments]`. Batch i gets seed first + i − 1 and writes `<out folder>/batch<i>/` and its log:

```bash
components/safe_rl/run_batches.sh 2200 6 70 $O/mps/episodes --robot $ROBOT --method mps
components/safe_rl/run_batches.sh 2206 2 70 $O/mps/test --robot $ROBOT --method mps
```

Each batch folder holds three files:

- `observations_dataset_abs_test.npy`: episodes × samples × 62, zero padded. Columns 0–29 are the
  state: projected gravity and angular velocity in the base frame, joint positions, joint
  velocities. Columns 30–59 are the next state, 60 is fallen, 61 is success.
- `observations_dataset_abs_test_xinv.npy`: episodes × samples × 2, whether the robot has been in
  static equilibrium and in the support-polygon set.
- `observations_dataset_abs_test_episodes.csv`: one row per episode, `EPISODE_COLUMNS` in
  `dataset_manager.py`: command, push, `backup_time` (the scheduled switch), outcome (`fallen`,
  `captured`, `survived`, `static`, `polygon`, `timeout`), `static_time`, `switch_time` (the filter's
  switch, before the push if negative), the paper's failures (`paper_fall`: base or hip below 0.1 m;
  `knee_contact`: a knee below 3.5 cm; `jl_pos`, `jl_vel`: a joint past its limit), `row` in the npy,
  `discarded` (1: joint range, 2: fell before the push or no sample recorded, 3: control ticks missed).

## 3. Look at the dataset

```bash
python3 components/safe_rl/dataset_statistics.py --robot $ROBOT $(npy $O/mps/episodes) --plot stats.png
```

## 4. Train

```bash
python3 components/safe_rl/value_function/train_value_function.py --method mps \
    --data $(npy $O/mps/episodes | grep -v batch6) --val $(npy $O/mps/episodes | grep batch6) \
    --model $M/VF_mps_$ROBOT.pkl --epochs 600
```

TD(0) on every transition, shuffled, with a target network (τ = 0.01). `--method mps`: the paper's
target, (1 − fail) × ((1 − success) × V(next state) + success); `--x-inv polygon` uses the
support-polygon set; an episode ending in neither a failure nor the set is left out. `--method abs`:
ABS's discounted reach-avoid target, γ (max(ζ, min(l, V(next))) + (1 − γ) max(l, ζ)), γ = 0.999999,
ζ softened to −0.8 … 1.0 over the last 10 states before a failure, and the last state terminal
(V = max(ζ, l)); the model is saved as the score −V_RA, higher safer like ours. `--obs-noise S` adds
sensor noise to the states V sees. Every 25 epochs the `--val` states are scored against what the
target stands for (mps: the outcome, Brier score; abs: the Monte Carlo reach-avoid value), and the
best epoch is saved. The model is saved as `.pkl` (flax)
and `.onnx` (`export_onnx.py`, normalisation inside the graph, checked against JAX to 1e-4).

## 5. Calibrate: threshold and margin

The calibration runs, the walking policy alone with every state recorded at 100 Hz:

```bash
components/safe_rl/run_batches.sh 2300 8 10 $O/calibration/threshold --robot $ROBOT --record-all \
    --sample-rate 100 --episode-length 20 --push-window 2 17 --kicks 1.0 1.2 1.4 1.6 1.8        # host
components/safe_rl/run_batches.sh 2400 8 30 $O/calibration/margin --robot $ROBOT --record-all \
    --sample-rate 100 --kicks $(python3 -c "import numpy as np; print(*np.round(np.linspace(0.05, 0.75, 12), 3))")
python3 components/safe_rl/value_function/calibrate_filter.py --model $M/VF_mps_$ROBOT.onnx --write \
    --threshold-data $(npy $O/calibration/threshold) --margin-data $(npy $O/calibration/margin)
```

- **Margin**, Eq. (5) of the paper: L = max |ΔV| / ‖Δo‖ × max ‖Δo‖ over states as far apart as the
  time from the first value below the level to the safe policy acting (the `--consecutive` − 1
  further evaluations, 0.01 s each, and `--action-delay`, one policy step plus the actuators' delay,
  0.03 s), after leaving out the 5 % largest steps and then the 5 % largest ratios; the states are
  compared in the model's input normalisation. The ABS baseline uses no margin.
- **Threshold**, `--rule`: `walking` (ours' default), the highest threshold at which the filter never
  switches on the undisturbed walking states of the calibration runs: it switches only after
  `--consecutive` (3) values in a row below its level, so the level is the lowest, over the walking
  states, of the largest of 3 consecutive values, and a single-sample dip of V does not set it;
  `abs` (ABS's default), ABS's own rule, V_RA ≥ −0.05, on a single value; `percentile`, the paper's
  rule (the 1st percentile of V over all the states), which controls a rate per state rather than
  per unit of time and moves with the share of failing states, kept for reference; `risk`,
  1 − `--risk`. All of them, and the walking rule on a single value (`walking_k1`, with its margin),
  go in the `.json`, with `min_switch`.

## 6. Evaluate

```bash
python3 components/safe_rl/value_function/evaluate_value_function.py --model $M/VF_mps_$ROBOT.onnx \
    --data $(npy $O/mps/test) --light-kick 0.75 --plot eval.png
```

V while walking, the ROC AUC of V for falls against recoveries at times after the push and after
the first sample, the Brier score, per threshold the falls warned and how early and the false
alarms, and the calibration. ABS models go on the `abs` test batches (the walking policy's
outcome), `mps` models on the `mps` ones (the safe policy's).

## 7. Closed loop

```bash
L="--robot $ROBOT --kicks 1.0 1.2 1.4 1.6 1.8 --episode-length 7"
components/safe_rl/run_batches.sh 2500 8 30 $O/closed_loop/alone $L                                        # host
components/safe_rl/run_batches.sh 2500 8 30 $O/closed_loop/mps $L --filter $M/VF_mps_$ROBOT.onnx --no-switch-back
python3 components/safe_rl/compare_filter.py --arm alone $(npy $O/closed_loop/alone) --arm mps $(npy $O/closed_loop/mps) \
    --versus alone --table loop.md --plot loop.png --tradeoff tradeoff.png
```

`--filter` runs the filter as the controllers do: `ValueFunctionManager` at 100 Hz from 1 s after
the start, threshold and margin from the model's `.json` (`--threshold`, `--margin` override them);
`--no-switch-back` keeps the safe policy once switched (the paper's simulations); `--backup-delay 0`
runs the safe policy alone from the push start. `compare_filter.py` pairs the episodes of the arms
and reports, per arm, the falls, the paper's failures, the switches (and those during undisturbed
walking, before the push), the needless ones, the robots brought to rest, falls prevented and
caused, and McNemar tests against `--versus`.

## 8. Watch it in Gazebo

In the lab container with the display:

```bash
python3 saferl_controller.py --robot $ROBOT --plot
```

The robot walks at 0.3 m/s and is pushed every 4 s: 1, 1.5, 2, 2.5 and 3 m/s, each to the left,
right, front and back. rviz shows V and the active policy over the robot and the push as an arrow;
`rqt_plot` draws `/safety_filter/value`, `/safety_filter/threshold` (threshold plus margin, the level
the filter switches at) and `/safety_filter/backup`. A robot that falls is reset and the run goes
on; at the end a table lists each push.

| Argument | Default | Effect |
|---|---|---|
| `--robot` | `aliengo` | `aliengo` or `go2` |
| `--model` | the robot's default | value function |
| `--threshold`, `--margin` | the model's `.json` | the filter switches below threshold + margin |
| `--no-filter` | off | show V, never switch |
| `--kicks` | `1,1.5,2,2.5,3` | m/s, each in the 4 directions in turn |
| `--cmd` | `0.3,0,0` | walking command vx, vy, wz |
| `--push-period` | 4 | s between pushes |
| `--speed` | 1 | × real time |
| `--plot` | off | start `rqt_plot` on the filter topics |
| `--gui` / `--no-rviz` | off / rviz on | Gazebo GUI, rviz |

## 9. Use the filter in a controller

```python
from base_controllers.components.safe_rl.value_function.value_function_manager import ValueFunctionManager

vf = ValueFunctionManager(robot_name="go2")     # VF_abs_go2.onnx, threshold, margin, min_switch from its .json
...
# at 100 Hz, inputs in the base frame, joints in the order of quadruped.q
walking, V = vf.computeValueFnc(proj_gravity_b, ang_vel_b, q, qd)
if walking:
    rl_controller.velocity_cmd = command
    q_des = rl_controller.action(..., policy_type="default")
else:
    rl_controller.velocity_cmd = np.zeros(3)
    q_des = rl_controller.action(..., policy_type="safe")
```

It switches to the safe policy after `min_switch` evaluations in a row below threshold + margin
(the model's `.json`: 3, ABS 1), and back after `min_back` in a row above it (200: 2 s at 100 Hz;
`np.inf`: never). `vf.reset()` returns to the walking policy. The manager runs the `.onnx` on onnxruntime with one thread and
needs no JAX: 0.015 ms per evaluation on average.

`rl_quadruped/rl_quadruped_controller.py` has the filter behind `--safety-filter` (off by default;
`--filter-threshold`, `--filter-margin`, `--filter-model`): `RL` hands over to `SAFE_STOP` and back,
the filter keeps checking after it hands back, and a safe stop the operator asks for stays the
operator's. Its README (Safety filter) has the details.

## Reproducing the report

`run_pipeline.sh` reruns everything the report shows; `report_figures/<robot>_summary.txt` holds
the numbers, written by `summarize.sh`, which reruns the analysis alone on existing runs:
`components/safe_rl/run_in_image.sh bash components/safe_rl/summarize.sh <robot>` on the host. The
study's runs are in `observation_datasets/<robot>/`.
