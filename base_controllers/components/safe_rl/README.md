# Safe RL: value function and safety filter

@author: riccardobussola

The walking policy can be pushed over. The safety filter evaluates a value function V(x), 100 times
a second: the chance that the walking policy recovers from the robot's current state with no
further push. When V drops below 0.5 it hands the robot to the safe policy, which stands still at
zero command; once V has been back above 0.5 for 2 s, the walking policy takes over again. In
Gazebo, on the same 105 pushes per robot, it cut falls from 48 % to 8 % on the Aliengo and from 45 %
to 20 % on the Go2.

[value_function_report.md](value_function_report.md) has the results and what was fixed to get
them. The pipeline runs for the Aliengo and the Go2 (`--robot`, policies in
`rl_quadruped/policies/<robot>_velocity*`).

## Quick start

```bash
# the robot controller with the safety filter (lab container, rl_quadruped/)
python3 rl_quadruped_controller.py --robot go2 --safety-filter                 # Gazebo
./run_rl_controller.sh --robot go2 --real --input joy --safety-filter          # robot

# watch the filter in Gazebo: pushes, V in rviz and rqt_plot (lab container, base_controllers/)
python3 saferl_controller.py --robot go2 --plot

# redo everything for a robot: data, training (JAX, exported to ONNX), evaluation, closed loop,
# figures (host, base_controllers/)
components/safe_rl/run_pipeline.sh go2
```

The filter loads `value_function/models/VF_push_end_<robot>.onnx`; nothing besides the lab image is
needed to run it. Training also needs `pip3 install tqdm onnx` (`run_pipeline.sh` does it).

## Files

Paths from `robot_control/base_controllers`, where every command below runs: the scripts marked
*host* on the host (they start their own containers), the others in the lab container.

| File | What it does |
|---|---|
| `components/safe_rl/run_pipeline.sh` | *host*: the whole pipeline for one or more robots, fall-rule check to figures |
| `saferl_controller_dataset.py` | push episodes in Gazebo: the dataset; with `--filter` the closed-loop test, with `--fall-check` the fall-rule check |
| `components/safe_rl/dataset_manager.py` | one episode (reset, command, push, labels), a batch and its files; per-robot settings in `ROBOT_SETTINGS` |
| `components/safe_rl/run_batches.sh` | *host*: batches of `saferl_controller_dataset.py` in parallel containers |
| `components/safe_rl/run_in_image.sh` | *host*: a command in a throwaway container of the lab image (training, evaluation, summary) |
| `components/safe_rl/fall_rule.py` | from a `--fall-check` run: where the fall rule and the capture hold can sit |
| `components/safe_rl/dataset_statistics.py` | a dataset's outcomes, fall rate against the kick (k50), commands, joint ranges |
| `components/safe_rl/value_function/train_value_function.py` | TD training of V (JAX), stopped on held-out batches; saves `.pkl` and its `.onnx` export |
| `components/safe_rl/value_function/network.py` | the network in flax, for training, export and evaluation |
| `components/safe_rl/value_function/export_onnx.py` | `.pkl` to `.onnx`, the format the filter runs; fails if ONNX and JAX disagree |
| `components/safe_rl/value_function/evaluate_value_function.py` | V on held-out episodes: separation, warning time, false alarms, calibration |
| `components/safe_rl/compare_filter.py` | the same episodes with and without the filter |
| `components/safe_rl/summarize.sh` | a robot's figures and numbers: `report_figures/<robot>_*.png` and `<robot>_summary.txt` |
| `components/safe_rl/value_function/value_function_manager.py` | the filter: V and the switch, for any controller; per-robot thresholds in `DEFAULT_THRESHOLD` |
| `saferl_controller.py` | Gazebo test to watch: walking, pushes, V in rviz and `rqt_plot` |
| `rl_quadruped/rl_quadruped_controller.py --safety-filter` | the robot controller with the filter (off by default) |
| `vf_rosbag.py` | V of a recorded bag |
| `components/safe_rl/value_function/models/VF_push_end_<robot>.onnx` | the model the filter runs; `.pkl` next to it is the training's |
| `components/safe_rl/value_function/observation_datasets/<robot>/` | the report's runs: `fall_check/`, `episodes/`, `test/`, `closed_loop/` |
| `components/safe_rl/report_figures/` | the report's figures and the numbers behind them |

## Where to run

Everything runs in the locosim image (`mfocchi/trento_lab_framework:locosim`), from
`$LOCOSIM_DIR/robot_control/base_controllers`. The *host* scripts start their own containers of
it; each Gazebo needs its own ROS master, so parallel simulations need separate containers. rviz
and the Gazebo GUI need the lab container with the display, and the NVIDIA OpenGL libraries in it:
`--gpus all` alone gives the container compute only, so add `--env="NVIDIA_DRIVER_CAPABILITIES=all"`
to its `docker run` (the `lab_locosim` alias). Without them rviz draws on the CPU; with the Go2's
meshes (25 MB) it cannot keep up and slows the simulation down.

Training uses JAX on the CPU (the image has jax 0.4.13, flax and optax) and exports ONNX with
torch; it also needs tqdm and the onnx package, `pip3 install tqdm onnx` once per container
(`run_in_image.sh` does it). The filter itself runs the ONNX on onnxruntime, in simulation and on
the robot, without JAX.

## Run it all

From the host:

```bash
cd ~/trento_lab_home/ros_ws/src/locosim/robot_control/base_controllers
components/safe_rl/run_pipeline.sh              # aliengo and go2, about 40 minutes on 24 cores
components/safe_rl/run_pipeline.sh go2          # one robot
```

For each robot, in turn:

1. **Fall-rule check:** 8 batches of 30 episodes run until the robot is on the ground
   (`--fall-check`), into `observation_datasets/<robot>/fall_check/`.
2. **Dataset:** 6 batches of 70 episodes for training, the 6th to stop it, into `episodes/`; 2
   batches recorded from the push start for the test, into `test/`.
3. **Training:** `models/VF_push_end_<robot>.pkl` and its export `.onnx`, the model the filter
   runs, overwritten; the log is `observation_datasets/<robot>/train.log`.
4. **Closed loop:** 3 batches of 35 episodes with the walking policy alone, and the same episodes
   with the filter at 0.5 and at 0.8, into `closed_loop/alone/`, `filter_0.5/`, `filter_0.8/`.
5. **Summary:** `summarize.sh`, the figures and numbers in `report_figures/`.

The seeds are fixed per robot (Aliengo 401, 501, 701; Go2 421, 521, 711 for the three runs), so a
rerun sees the same commands and pushes; Gazebo is not deterministic, so its numbers move a little.
Environment variables change the sizes (`FALL_EPISODES`, `EPISODES`, `LOOP_EPISODES`, `EPOCHS`,
`THRESHOLDS`) and the outputs (`OUT`, `MODELS`, `FIGURES`), e.g. a quick trial that leaves the
report's runs alone:

```bash
OUT=components/safe_rl/trial MODELS=components/safe_rl/trial FIGURES=components/safe_rl/trial \
FALL_EPISODES=3 EPISODES=10 LOOP_EPISODES=5 EPOCHS=50 components/safe_rl/run_pipeline.sh go2
```

The steps below run the same tools one by one. The commands use

```bash
ROBOT=go2        # or aliengo
O=components/safe_rl/value_function/observation_datasets/$ROBOT
```

## 1. Check the fall rule

A new robot first needs its fall rule: an inclination and a base height past which it never
recovers. Run episodes until the robot is on the ground, from the host:

```bash
components/safe_rl/run_batches.sh 421 8 30 $O/fall_check --robot $ROBOT --fall-check    # host
python3 components/safe_rl/fall_rule.py $O/fall_check/batch*/observations_dataset_abs_test.npy --plot fall_rule.png
```

`--fall-check` ends an episode only with the robot on the ground (inclination over 85° or base below
0.1 m) or 3 s after the push, and never at capture. `fall_rule.py` prints the gap between the
recoveries (their largest inclination, their lowest base) and the falls, whether the rule
(`--inclination`, `--height`) sits in it, and, per capture hold, the captured robots that fell
afterwards. The rule and the kick range go into `ROBOT_SETTINGS` in `dataset_manager.py`.

## 2. Generate push episodes

```bash
python3 saferl_controller_dataset.py --robot $ROBOT --episodes 70 --seed 501 --no-rviz --out $O/episodes/batch1
```

An episode:

- **Reset.** With gravity off, the PD drives the robot into the policy's default posture (0, 0.9,
  −1.8); it is set down at its standing height and waits until still.
- **Command.** vx, vy and wz uniform in ±0.5, the training range of both robots; 10 % of episodes
  standing, 10 % only turning.
- **Push.** 2–3 s after the start: a horizontal force on the base, robot mass × kick / 0.2 s for
  0.2 s, the push of `rl_quadruped/push_sweep.py`. The kick is uniform in 0–3 m/s (Aliengo) or
  0–2.5 m/s (Go2), so that about 45 % of the episodes fall; the direction is uniform. Torque noise
  (10 Nm std per joint and tick) starts with the push.
- **Samples.** At 50 Hz from the end of the push (`--from-push-start`: from its start, for test
  data with the walking state just before the push): state, next state, fallen, success.
- **End.** A fall (inclination over 45° or base below 0.2 m) is labelled fallen. Capture (capture
  point within 5 cm for 0.5 s), or 3 s survived after the push, is labelled success.

Each episode is seeded by (seed, its index), so runs with the same `--seed` see the same commands
and pushes. Other settings are `DatasetManager` attributes (`push_window`, `cmd_range`, ...).

Gazebo runs in real time (`--speed`, default 1), about 6 s per episode. The policy runs on control
ticks (500 Hz), so the Python loop has to keep up: it takes about 0.6 ms of the 2 ms period. An
episode where the loop missed over 1 % of the ticks is discarded; if many are, lower `--speed`. The
same 280 Aliengo episodes at 0.2× and in real time fell 117 and 123 times, with the fall rate per
kick band within 3 points.

From the host, batches in parallel: `run_batches.sh <first seed> <batches> <episodes> <out folder>
[arguments]`. Batch i gets seed first + i − 1 and writes `<out folder>/batch<i>/` and its log
`batch<i>.log`; the arguments go to `saferl_controller_dataset.py`:

```bash
components/safe_rl/run_batches.sh 501 6 70 $O/episodes --robot $ROBOT
components/safe_rl/run_batches.sh 507 2 70 $O/test --robot $ROBOT --from-push-start
```

Each batch folder holds two files:

- `observations_dataset_abs_test.npy`: episodes × samples × 62, zero padded. Columns 0–29 are the
  state: projected gravity and angular velocity in the base frame, joint positions, joint
  velocities. Columns 30–59 are the next state, 60 is fallen, 61 is success.
- `observations_dataset_abs_test_episodes.csv`: one row per episode. Command; push time, kick and
  direction (base frame); base velocity at the push; outcome; `end_time`; `switch_time` (when the
  filter switched, after the push start); `overruns` (control ticks the loop missed); `max_incl`,
  `min_height`, `final_height` (after the push); `cp_hold_max` (longest capture-point hold, −1:
  never); `n_samples`; `row` in the npy; `discarded` (1: joint range, 2: fell before the push, 3:
  over 1 % of the control ticks missed).

## 3. Look at the dataset

```bash
python3 components/safe_rl/dataset_statistics.py --robot $ROBOT $O/episodes/batch*/observations_dataset_abs_test.npy --plot stats.png
```

Check that no command is out of the ±0.5 training range, that few episodes were discarded, and
that the kick the robot falls from half the time (k50) is near the report's (Aliengo 1.76 m/s,
Go2 1.35 m/s). k50
moves by about ±0.2 m/s from one batch of 70 to the next, so pool batches.

## 4. Train

```bash
python3 components/safe_rl/value_function/train_value_function.py \
    --data $O/episodes/batch{1,2,3,4,5}/observations_dataset_abs_test.npy \
    --val $O/episodes/batch6/observations_dataset_abs_test.npy \
    --model components/safe_rl/value_function/models/VF_my_run_$ROBOT.pkl --epochs 600
```

TD learning with the target (1 − fallen) × ((1 − success) × V(next state) + success), on every
transition, shuffled. Every 25 epochs V of the `--val` states is scored against how their episode
ended (Brier score). The best epoch is saved, not the last; trained on, the network memorises
episodes and false alarms grow. It takes 2–5 minutes on the CPU.

The model is saved twice: `<model>.pkl`, the flax parameters, and `<model>.onnx`, written by
`export_onnx.py` with the input normalisation inside the graph (input `obs`, n × 30; output `value`,
n). The export compares the two on 2,000 states and fails above 1e-4; the report's models differ by
under 1e-6. A `.pkl` alone is exported with
`python3 components/safe_rl/value_function/export_onnx.py <model.pkl>`.

## 5. Evaluate

```bash
python3 components/safe_rl/value_function/evaluate_value_function.py --model components/safe_rl/value_function/models/VF_my_run_$ROBOT.onnx \
    --data $O/test/batch{1,2}/observations_dataset_abs_test.npy --light-kick 0.75 --plot eval.png
```

`--model` takes the `.onnx`, what the filter runs, or the `.pkl` (JAX), e.g. an older model. Use
batches the model was neither trained nor stopped on, recorded from the push start so that the
walking states before the push are in. `--light-kick` is the kick below which the walking policy
falls about 1 % of the time, so an alarm there is false: 1 m/s for the Aliengo, 0.75 m/s for the
Go2. It prints V while walking, the ROC AUC of V for falls against recoveries at times after the
push, per threshold the falls warned and how early and the false alarms, and the calibration.

The report's models gave, on the test batches:

| Check (140 test episodes) | Aliengo | Go2 |
|---|---|---|
| V of the walking state before the push, mean (lowest) | 0.995 (0.975) | 0.998 (0.973) |
| ROC AUC, falls against recoveries, 0.2 s after the push starts | 0.96 | 0.98 |
| Falls warned at V < 0.5, median lead | 52 / 52, 0.41 s | 66 / 66, 0.32 s |
| False alarms at V < 0.5: light kick / any recovered episode | 0 / 52, 9 / 88 | 4 / 32, 10 / 74 |

## 6. Closed loop: the same pushes with and without the filter

From the host, then `compare_filter.py`:

```bash
components/safe_rl/run_batches.sh 711 3 35 $O/closed_loop/alone --robot $ROBOT                          # host
components/safe_rl/run_batches.sh 711 3 35 $O/closed_loop/filter_0.5 --robot $ROBOT --filter --threshold 0.5
python3 components/safe_rl/compare_filter.py --light-kick 0.75 \
    --alone $O/closed_loop/alone/batch*/observations_dataset_abs_test.npy \
    --filtered $O/closed_loop/filter_0.5/batch*/observations_dataset_abs_test.npy --plot closed_loop.png
```

`--filter` runs the filter as the controllers do: `ValueFunctionManager` at 100 Hz from 1 s after
the start. It loads `models/VF_push_end_<robot>.onnx` by default; `--filter <model.onnx>` and
`--threshold` change that. `compare_filter.py` pairs the episodes and reports the falls prevented
and caused, the needless switches, when the filter switched, the falls left and caused by push
direction, and the fall rate per kick band.

| 105 episodes, threshold 0.5 | Aliengo | Go2 |
|---|---|---|
| Fell: alone → with the filter | 48 % → 8 % | 45 % → 20 % |
| Falls prevented / caused | 42 / 0 | 27 / 1 |
| Needless switches: light kick / any recovered episode | 0 / 34, 9 / 55 | 1 / 30, 13 / 58 |
| Switches while walking, before the push | 0 | 0 |

## 7. Watch it in Gazebo

In the lab container with the display:

```bash
python3 saferl_controller.py --robot $ROBOT --plot
```

The robot walks at 0.3 m/s and is pushed every 4 s: 1, 1.5, 2, 2.5 and 3 m/s, each to the left,
right, front and back. In rviz:

- **Over the robot:** V and the active policy, green "walking" or red "SAFE POLICY", and a sphere
  that turns from green to red as V drops.
- **The push:** a blue arrow.

`rqt_plot` draws `/safety_filter/value`, `/safety_filter/threshold` and `/safety_filter/backup` (1
while the safe policy runs). A robot that falls is reset, and the run goes on. At the end a table
lists each push: the lowest V, when the safe policy took over, and whether the robot fell. It runs
in real time, so the 20 pushes take under 2 minutes; the markers go to rviz at 30 Hz. The last
line counts the control ticks the loop missed; over 1 %, lower `--speed`.

| Argument | Default | Effect |
|---|---|---|
| `--robot` | `aliengo` | `aliengo` or `go2` |
| `--threshold` | per robot, 0.5 | switch below this V |
| `--no-filter` | off | show V, never switch: the falls the filter prevents |
| `--kicks` | `1,1.5,2,2.5,3` | m/s, each in the 4 directions in turn |
| `--cmd` | `0.3,0,0` | walking command vx, vy, wz |
| `--push-period` | 4 | s between pushes |
| `--speed` | 1 | × real time |
| `--model` | `VF_push_end_<robot>.onnx` | value function |
| `--plot` | off | start `rqt_plot` on the filter topics |
| `--gui` / `--no-rviz` | off / rviz on | Gazebo GUI, rviz |

## 8. Use the filter in a controller

```python
from base_controllers.components.safe_rl.value_function.value_function_manager import ValueFunctionManager

vf = ValueFunctionManager(robot_name="go2")     # models/VF_push_end_go2.onnx, threshold 0.5
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

It switches to the safe policy after `min_switch` evaluations in a row below the threshold (1),
and back after `min_back` in a row above it (200: 2 s at 100 Hz). `vf.reset()` returns to the
walking policy, e.g. after a reset. `saferl_controller.py` is a complete example.

The manager runs the `.onnx` on onnxruntime with one thread, as the policies run, and needs no JAX.
One evaluation takes 0.015 ms on average (0.021 ms at the 99th percentile, 0.12 ms at worst) in a
process of 48 MB with one thread; under JAX it took 0.044 ms (0.44 ms at worst), 159 MB and 50
threads.

`rl_quadruped/rl_quadruped_controller.py`, the controller that runs the robots, has the filter
behind `--safety-filter` (off by default; `--filter-threshold`, `--filter-model`): `RL` hands over
to `SAFE_STOP` and back, and a safe stop the operator asks for stays the operator's. Its README
(Safety filter) has the details; `rl_quadruped/push_sweep.py trial --safety-filter` pushes it with
the filter on.

## Reproducing the report

`run_pipeline.sh` reruns everything the report shows; `report_figures/<robot>_summary.txt` holds
the numbers, written by `summarize.sh`, which reruns the analysis alone on existing runs:
`bash components/safe_rl/summarize.sh <robot>` in the lab container, or
`components/safe_rl/run_in_image.sh bash components/safe_rl/summarize.sh <robot>` on the host. The
report's runs are in `observation_datasets/<robot>/`.
