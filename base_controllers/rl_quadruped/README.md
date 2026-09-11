# RL quadruped controller (Aliengo + Isaac Lab velocity policy)

Runs the single-network velocity policy trained in
`safe_rl/tasks/manager_based/velocity` (gym id `Rl-Velocity-Aliengo`) on the Aliengo, in Gazebo or on
hardware, behind a state machine that handles calibration, standing up, walking, standing down and
emergency collapse.

Two actors from that project are deployed together: the walking policy, and the zero-command
[safe stop](#safe-stop-a-second-policy-not-a-collapse) trained in the same environment to hold
station and reject disturbances. They share one observation history and hand over mid-stride.

## Files

Everything this controller owns lives in one directory,
`robot_control/base_controllers/rl_quadruped/`:

| File | Role |
|---|---|
| `rl_quadruped_controller.py` | controller, state machine, real-time loop, CLI entry point |
| `rl_controller_config.py` | **the only configuration this controller reads** — robot description, gains, timing, limits, telemetry, real-time |
| `commandInterface.py` | keyboard and Xbox-joystick operator input |
| `velocity_policy.py` | ONNX policy wrapper: observation assembly, history, decimation, the estimation head |
| `training_model.py` | reads a URDF as physics bodies and compares it against the deployed robot |
| `run_rl_controller.sh` | launcher that applies real-time priority and optional core pinning |
| `README.md` | this document |
| `policies/aliengo_velocity.json` | the deployment contract, shared by **both** networks below: obs layout, `q_default`, gains, rate, and the `onnx_variants` map that names them |
| `policies/aliengo_velocity_normal.onnx` | the walking policy (copied from `safe_rl/normal.onnx`) |
| `policies/aliengo_velocity_safe.onnx` | the zero-command safe-stop policy (copied from `safe_rl/safe.onnx`) |
| `policies/aliengo_velocity_model.urdf` | the robot model the policy was trained on, vendored from `safe_rl` so the contract is checkable |

It depends on the framework but adds nothing to it: `BaseController`, `IMU_utils`, `PidManager`,
`getRobotModelFloating` and the `ros_impedance_controller` messages are imported from their usual
places. `params.py` is untouched **and unused** — see [Configuration](#configuration).

Nothing in here depends on anything outside it except the framework. The one import in the other
direction is deliberate: `components/rl_velocity_controller/rl_controller.py` can run this policy
too, and it does so by importing `velocity_policy` from here rather than carrying a second copy of
the observation contract — see [Running this policy from the old
controller](#running-this-policy-from-the-old-controller).

The one change made outside this directory is to `aliengo_description/urdfs/const.xacro`, which
carried a physically impossible calf inertia — see
[The model the policy was trained on](#the-model-the-policy-was-trained-on).

## Running

Inside the container (`lab_locosim`, or `dock-other` for a second shell):

```bash
cd $LOCOSIM_DIR/robot_control/base_controllers/rl_quadruped

python3 rl_quadruped_controller.py                    # Gazebo, keyboard
python3 rl_quadruped_controller.py --input joy        # Gazebo, Xbox pad
python3 rl_quadruped_controller.py --gui              # Gazebo with the GUI
python3 rl_quadruped_controller.py --real --input joy # real robot
```

### Command-line arguments

| Argument | Default | Effect |
|---|---|---|
| `--robot NAME` | `aliengo` | Robot to run. Needs an entry in `ROBOTS` of `rl_controller_config.py`; an unknown robot, an unknown config key, or a missing required field is refused rather than run on silent defaults. |
| `--policy NAME` | robot name | Policy basename under `policies/`, without the `_velocity` suffix — `--policy aliengo` loads `aliengo_velocity.json`, and every ONNX file its `onnx_variants` names. Use it to A/B two exported policy sets. |
| `--input {keyboard,joy,none}` | `keyboard` | Operator input. `keyboard` needs a real tty, so start it from `lab_locosim`/`dock-other`, not from a pipe. `joy` restarts the ROS `joy_node` and reads `/joy`. `none` accepts no input — the machine waits in `FOLD` until something posts events programmatically. |
| `--real` | off | Run on hardware. Pings the robot's IP first and exits if it does not answer; folds in the `real` sub-dict of the robot config (slower motions, softer collapse, tighter command limits); enables joint-limit clipping and real-time tuning; subscribes to the split hardware IMU topics instead of Gazebo's single one; and skips the simulation spawn fix. |
| `--world NAME` | config `world_name` (`fast.world`) | Gazebo world, resolved inside `ros_impedance_controller/worlds/`. Ignored with `--real`. |
| `--rviz` / `--no-rviz` | config `use_rviz` (**on**) | Start rviz with the simulator. On by default: it is the intended way to look at the robot, and it renders from the published TF and markers rather than driving the physics window. `--no-rviz` takes one process out of contention, for timing work or a loaded real-robot control PC. See [rviz and TF](#rviz-and-the-world-transform) — the controller has to publish `world -> base_link` itself. |
| `--gui` | off | Show the Gazebo GUI. Off by default because the GUI competes with the control loop for the machine; rviz is off either way. |
| `--skip-calibration` | off | Skip the accelerometer-bias measurement entirely. The bias stays at zero, and since the policy consumes raw specific force that offset goes straight into the observation it acts on — so this is a bring-up switch, not a shortcut for a run you intend to trust. Also `skip_calibration` in the config. |
| `--bag [PREFIX]` | off | Record a rosbag of the joint topics and all telemetry topics for the whole run, closing it cleanly on exit. Optional filename prefix, default `rl_<robot>`. |
| `--no-telemetry` | off | Stop publishing the telemetry topics. Only useful for isolating whether publishing costs you timing — nothing is logged in memory either way. |
| `--no-estimator` | config `estimate_base_velocity` (**on**) | Stop evaluating the actor's state-estimation head. The `estimated_base_lin_vel` topic is not advertised and the estimate stays at zero. **The policy is unaffected** — it consumes its own estimate inside the ONNX graph either way; what goes away is only the second, external evaluation that exists so the number can be seen. Worth ~110 µs of every 20 ms inference tick. |
| `--no-inference-timing` | config `measure_inference_timing` (**on**) | Stop timing the inference ticks: no start-up benchmark, no `policy/infer_ms` or `policy/infer_over_budget` topics, and no timing section in the shutdown summary. Pure instrumentation — turn it off once the timing question is settled. |
| `--realtime` / `--no-realtime` | config `realtime` (on with `--real`) | Force the real-time tuning on or off: `SCHED_FIFO`, the GC policy, and pinning. Turn it off when profiling, when the container is not privileged, or to check whether an oddity is caused by the tuning itself. |
| `--pin-cpu` | config `pin_cpu` (off) | Pin the control thread to one core. Only worth it once `SCHED_FIFO` is granted and the core is isolated — pinning alone can make timing *worse* by putting the loop on a contended core. Prefer `run_rl_controller.sh`, which pins the whole process including start-up. |

`--help` prints the same list.

### Real-time priority and core isolation

The controller asks for `SCHED_FIFO` itself, but the launcher does it from outside, which also covers
start-up and works where the in-process call is denied:

```bash
./run_rl_controller.sh --real --input joy          # RT priority
CPU=3 ./run_rl_controller.sh --real --input joy    # RT priority, pinned to core 3
PRIO=90 ./run_rl_controller.sh --real              # different priority
```

Pinning only helps if nothing else may run on that core. To reserve core 3, add to the **host** kernel
command line (`/etc/default/grub`) and reboot:

```
isolcpus=3 nohz_full=3 rcu_nocbs=3 irqaffinity=0-2
```

`isolcpus` keeps the scheduler off it, `nohz_full` stops the periodic timer tick, `rcu_nocbs` moves
RCU callbacks away, and `irqaffinity` steers device interrupts to the other cores. Then run with
`CPU=3`. Check it took:

```bash
cat /sys/devices/system/cpu/isolated
chrt -p    $(pgrep -f rl_quadruped_controller)
taskset -p $(pgrep -f rl_quadruped_controller)
```

The locosim container already runs `--privileged`, which is what makes `SCHED_FIFO` available;
without it the controller warns and continues at normal priority.

## Operator controls

| Action | Keyboard | Xbox pad |
|---|---|---|
| Calibrate IMU bias | `c` | Y |
| Stand up | `u` | A |
| Start the RL policy | `r` | X |
| **Safe stop** (stay standing) | `f` | RB |
| One random push *(sim only)* | `p` | — |
| Burst of random pushes *(sim only)* | `P` | — |
| Push force − / + | `-` / `+` | — |
| Stand down | `d` | B |
| **Emergency damping** | `SPACE` | BACK, or LB+RB together |
| Quit (collapses first) | `q` | START |
| Forward / backward | `w` / `s` | left stick Y |
| Left / right | `a` / `z` | left stick X |
| Turn | `j` / `l` | right stick X |
| **Set an exact velocity** | `v` then `vx[,vy[,wz]]` + Enter | — |
| Forward speed presets | `1`…`5` | — |
| Zero the velocity command | `x` | release the sticks |

Three ways to set a velocity from the keyboard:

* **Step it** with the direction keys, by `key_lin_step` / `key_ang_step` (0.1 by default). The
  command is held between presses, so the robot keeps walking while you are not touching anything.
* **Jump to a preset** with the number keys — `key_speed_presets`, forward velocity in m/s.
* **Type it exactly** — press `v`, then a comma-separated command, then Enter:

  ```
  v0.35        -> vx = 0.35
  v0.25,-0.1   -> vx = 0.25, vy = -0.1
  v0.2,0,0.4   -> vx = 0.2, wz = 0.4
  ```

  The separator is a comma, never a space, because **`SPACE` keeps its emergency meaning at all
  times** — including halfway through typing a velocity, which cancels the entry and collapses the
  robot. `Esc` cancels, backspace edits, and the entry is abandoned after 8 s of no keystrokes so a
  forgotten half-typed command cannot swallow the action keys. Out-of-range values are clamped to
  the configured limits and the clamp is reported; unparseable input is rejected and the previous
  command is kept.

Every change is echoed with the **full resulting command** and where it came from, so a stepped,
preset and typed command all read the same way and there is never any doubt what the robot was
asked for:

```
velocity command  vx=+0.25 m/s  vy=+0.00 m/s  wz=+0.00 rad/s   [step w]
velocity command  vx=+0.30 m/s  vy=-0.10 m/s  wz=+0.40 rad/s   [typed]
velocity command  vx=+0.50 m/s  vy=+0.00 m/s  wz=+0.00 rad/s   [step w, clamped to 0.50 m/s]
```

Both backends clamp to `max_lin_vel_cmd` / `max_ang_vel_cmd` — 0.5 m/s and 0.5 rad/s in simulation,
0.3 / 0.4 on hardware, against the ±0.5 range the policy was trained on.

## State machine

```
INIT ──▶ CALIBRATE ──▶ FOLD ──▶ STAND_UP ──▶ STANDING
                         ▲                          │
                         │                          ▼
                         └── STAND_DOWN ◀───── RL ⇄ SAFE_STOP

any state ──[emergency]──▶ DAMPING ──▶ DONE
```

`INIT`, `CALIBRATE` and `FOLD` run automatically at start-up and the machine then rests in `FOLD`;
everything after that waits for a command. Commands that do not apply to the current state are
refused with a message rather than queued.

There is no separate "holding the fold" state: it held exactly the posture `FOLD` arrives at, so it
was two names for one behaviour and one more transition to read in a log.

* **INIT** pushes the joint PD gains and freezes the *measured* posture, so enabling the gains moves
  nothing. There is no integral term in any state — `ki` is zero everywhere by construction.
* **CALIBRATE** averages `accelerometer - b_R_w · g0` to get the accelerometer bias, over a
  **continuously still** window. It runs **before any commanded motion**, on the posture the robot
  was placed in — see [Calibration](#calibration-first-and-why-it-has-to-be-still).
* **FOLD** eases into the resting posture `q_fold` on the soft gains once the bias is known, then
  **stays there waiting for a command** — see
  [Start-up](#start-up-the-spawn-drop-and-the-vibration-it-caused) for why the gains and `q_fold`
  itself matter. It is the robot's resting state and where a stand-down returns to; arriving from a
  stand-down, whose last waypoint *is* `q_fold`, it skips the move rather than replaying an
  interpolation to a posture already held.
* **STAND_UP / STAND_DOWN** interpolate through joint waypoints with a quintic profile, so every
  waypoint is entered and left at zero velocity *and* zero acceleration. Stand-up goes
  `current → q_fold → q_stand`; tucking the feet under the body first is what makes it work from a
  sprawled pose. `q_stand` defaults to the policy's own training posture `[0, 0.9, -1.8] × 4`.
* **RL** runs the walking policy at 50 Hz on top of the faster loop. Gains switch to the values
  the policy was trained with (kp 35, kd 0.5) and **the policy has the robot from its first tick**:
  the operator's command is live immediately and the joint target the network returns is applied as
  it comes — see [Immediate hand-over](#immediate-hand-over-no-ramps-no-blends).
* **SAFE_STOP** runs a *second network* with the command held at zero — see below. Not the
  emergency path: the robot stays standing. It is also the only reliable way to bring this robot to
  rest; see [Stopping](#stopping-why-the-robot-sometimes-oscillates-and-what-actually-fixes-it).
* **DAMPING** is the emergency path: `kp = 0`, `kd = 10`, so joint torque is `-kd·qd`. The robot
  sinks under its own weight with the energy bled off instead of dropping, and any feed-forward
  torque is ramped out inside the loop. `/set_pids` is called once, not per tick. After it settles,
  gains are released entirely.

Damping also triggers by itself if roll or pitch exceeds `max_roll_pitch` (1.0 rad, or
`safe_max_roll_pitch` = 1.3 rad during the safe stop) or any joint exceeds 25 rad/s.

### Safe stop: a second policy, not a collapse

There are two exported actors, and the contract names both in `onnx_variants`:

| variant | trained by | command range | reached by |
|---|---|---|---|
| `normal` | `UnitreeAliengoEnvCfg` (`Rl-Velocity-Aliengo`) | ±0.5 m/s, ±0.5 rad/s | `r` / X |
| `safe` | `UnitreeAliengoEnvSafeEnvCfg` | **pinned to zero**, with ±0.25 rad reset tilts, ±1 m/s reset velocities and a push every 2–3 s | `f` / RB |

`safe` is a standstill that *actively rejects disturbances*: the same environment, the same
observation, the same actuator model, trained only ever to stand still and to recover from states
the walking policy never sees. That makes it the right answer to a command that has gone wrong, an
unexpected push, or an operator who simply wants the machine to hold where it is on bad ground —
all cases where damping, which gives up on the robot and lets it sink, is far too blunt.

So the safe stop has **its own command, separate from the emergency**, and both remain available:

```
r ─▶ RL ──[f]──▶ SAFE_STOP ──[r]──▶ RL          resume walking
                     ├───[d]──▶ STAND_DOWN      put it down normally
                     └─[SPACE]─▶ DAMPING        still one key away, from any state
```

Three properties are worth knowing:

* **The hand-over does not reset anything.** Both variants consume the identical 228-float
  observation, so the controller keeps *one* buffer and switches which network reads it. History
  blocks, action block and decimation counter carry straight over, and the switch can happen
  mid-stride in either direction. Resetting would tell a recovery policy that the robot has been
  standing still — exactly wrong at the moment you reach for it. You can see it in the log:
  `inference_count` goes 202 → 203 across the switch, where a reset would zero it.
* **The command is held at zero, not merely ignored.** Zero is the only command this variant ever
  saw; anything else is off-distribution. The operator's stick is also cleared on entry, so
  resuming later cannot start with a leftover command. Measured with the stick pushed to
  `[0.4, 0, -0.3]` throughout: the command actually fed to the policy stayed at exactly `0.000000`.
* **The attitude envelope is wider there** (`safe_max_roll_pitch`). Applying the walking limit
  during a recovery would trip damping on precisely the excursions the safe stop exists to catch —
  the operator would ask for a recovery and get a collapse.

The variant currently driving the joints is published on `<ns>/policy_variant`, latched, so a bag
or a plot always says which network produced an action.

A missing or misnamed variant is refused **at start-up**, not when the operator asks for it: that
request is made when something is already going wrong.

## Policy contract

The observation is 228 floats in exactly the declaration order of the training `PolicyCfg`:

```
[  0:  3)  velocity_commands       (vx, vy, wz)            current step only
[  3: 18)  imu_lin_acc             3 × 5 history
[ 18: 33)  imu_ang_vel             3 × 5 history
[ 33: 48)  imu_projected_gravity   3 × 5 history
[ 48:108)  joint_pos_rel          12 × 5 history
[108:168)  joint_vel_rel          12 × 5 history
[168:228)  actions                12 × 5 history
```

Each history block is oldest-frame-first, matching `isaaclab.utils.buffers.CircularBuffer.buffer`.
Action output is `q_des = q_default + 0.2 · action`.

**Both variants share this contract exactly** — same observation, same action space, same
`q_default` and `action_scale`, since `UnitreeAliengoEnvSafeEnvCfg` changes only the command range,
the reset distribution and the push schedule. The controller checks the input and output widths of
every variant against the json at start-up, because that agreement is the entire basis for keeping
one observation buffer and switching networks under it.

That is also why there is **one** `aliengo_velocity.json` and not one per network. A second contract
file would either duplicate this one — and then be free to drift from it, quietly breaking the
hand-over — or imply the two networks disagree about something, which is exactly what must not be
true. The json sits above both `.onnx` files and names them in `onnx_variants`; the files are called
`..._normal.onnx` and `..._safe.onnx` so that neither looks like the one the json belongs to.

Three details that had to line up and do:

* **The observation normalizer is inside the ONNX graph** (`Sub`/`Div` on the `obs` input), so raw
  observations are fed. Do not normalize before the call.
* **`imu_lin_acc` is specific force, gravity included.** Confirmed in IsaacLab's own source, where
  the Imu sensor computes
  `lin_acc_w = (lin_vel_w - prev_lin_vel_w) / dt + gravity_bias_w` with `ImuCfg.gravity_bias`
  defaulting to `(0, 0, 9.81)` — so the training signal is a finite-differenced velocity plus
  gravity, i.e. what a real accelerometer reads. Only the sensor bias is removed on deployment;
  gravity is *not* subtracted. Measured in Gazebo: `|acc| = 9.81` exactly while resting.
* **Joint order needs no permutation.** The task declares its joints as `locosim_joint_names`
  (`FL, RL, FR, RR` × `hip, thigh, calf`), which is element-for-element
  `conf.robot_params['aliengo']['joint_names']` (`lf, lh, rf, rh` × `haa, hfe, kfe`).

Sanity check that the whole chain is consistent: the policy's zero-command output puts the base at
0.327 m above the ground, against the 0.32 m `base_height_l2` target it was trained on.

**How far each convention is actually verified**, since two of these were read out of IsaacLab and
one could not be:

| term | verified by |
|---|---|
| `imu_lin_acc` | IsaacLab source (the `lin_acc_w` expression above) *and* measurement |
| `imu_ang_vel` | IsaacLab source (`imu_ang_vel` returns `data.ang_vel_b`, the sensor frame) *and* measurement against ground truth, r = 0.997 / 0.999 / 1.000 |
| `imu_projected_gravity` | **measurement only** — 1.6e-4 RMS against ground truth, plus the base-height cross-check above |

`imu_projected_gravity` is not defined in any IsaacLab checkout on this machine, so `safe_rl` builds
against a newer one that is not available here. The convention used on deployment — unit
`b_R_w @ (0, 0, -1)`, so a level base gives `(0, 0, -1)` — is therefore inferred from measurement
rather than confirmed from source. It is consistent with everything measured, but if the policy is
ever retrained against a version that changed this term (normalised or not, sign, sensor vs base
frame), this is the one line of the contract that would not announce the change.

### The joint controller underneath

`ros_impedance_controller` computes, in its default branch:

```cpp
des_joint_efforts_pids_(i) = kp[i]*(q_des - q) + kd[i]*(qd_des - qd) + integral_action;
joint_states_[i].setCommand(des_joint_efforts_(i) + des_joint_efforts_pids_(i));
```

so with `ki = 0` — which this controller always sets — it is exactly
`tau = kp(q_des − q) + kd(qd_des − qd) + tau_ffwd`, the same PD actuator Isaac used in training. The
policy's gains therefore carry over directly. There is no torque saturation in the controller itself;
the URDF effort limits are enforced below it.

Its *other* branch, selected by the global `/pid_discrete_implementation` parameter, replaces the
velocity term with a filtered derivative of the position error — a different controller, which the
policy has never seen. That would not fail loudly on its own, so the controller checks the parameter
at start-up and says which law is active.

### Reading the policy's own velocity estimate

The actor regresses the base linear velocity internally and feeds it to its own backbone; the value
never leaves the ONNX graph, so `session.run` cannot return it. `VelocityPolicy` therefore evaluates
the estimation head itself, from weights read out of the same policy file, and publishes it on
`estimated_base_lin_vel` (base frame).

It is the identical computation, not an approximation — reproducing the *whole* actor this way agrees
with onnxruntime to 6e-05 on random observations, and the head alone to 7e-07 against a reference
implementation. Cost is ~110 µs on an inference tick, i.e. every 20 ms, through preallocated buffers
— measured as the difference in per-tick cost with the head on and off, which is most of what a tick
costs, the graph itself running in ~40-60 µs.

It is a **diagnostic, not part of the control path**, so it can be switched off with
`--no-estimator` (config `estimate_base_velocity`). Off, the head is never even read out of the
policy file, `has_estimator` is `False`, the topic is not advertised, and the joint targets are
bit-identical — verified. `publish_estimated_velocity` then has nothing to publish and is moot.

Measured against Gazebo ground truth over a walk:

| command | estimated (base frame) | ground truth | error |
|---|---|---|---|
| `vx = +0.3` | +0.296 | +0.284 | 0.013 m/s |
| `wz = +0.4` | −0.003 | +0.003 | 0.007 m/s |
| `vx = −0.2` | −0.197 | −0.190 | 0.007 m/s |
| zero | −0.006 | −0.001 | 0.008 m/s |

RMS error 0.049 m/s over the whole walk, x-axis correlation 0.993. This is the most useful single
signal for judging whether the policy is reading the robot correctly: if it disagrees with reality,
the policy is acting on a wrong belief about its own motion, and no amount of gain tuning will fix
that.

## The model the policy was trained on

A position-control policy is a feedback law tuned against particular link inertias. Its output is a
joint target; a fixed PD (`kp` 35, `kd` 0.5) turns that into torque, so each leg's closed-loop
response is set by those gains against that leg's inertia. Run the same policy on a robot whose
legs differ by a factor and the response it learned to expect is not the response it gets. That is
not a footnote on a quadruped: standing still is held by feedback, so a mis-tuned leg answers a
stop command with a residual limit cycle instead of a standstill.

So the training model is part of the policy's deployment contract, and it is vendored next to the
policy (`aliengo_velocity_model.urdf`, copied from `safe_rl/.../aliengo_description/urdf/`) and
checked at start-up.

### The comparison is between physics bodies, not URDF links

Both engines merge fixed-joint groups before integrating anything, and report the merged body. On
this robot that matters a lot:

* locosim fuses the foot into the calf and the IMU into the trunk — and hangs `trunk` off a
  massless `floating_base` link, which Gazebo then calls `base_link`;
* the Isaac asset additionally has a rotor body fixed to each leg link, so *its* hips carry the
  thigh rotor, its thighs the calf rotor, and its trunk all four hip rotors.

Comparing raw URDF links would compare quantities neither simulator ever uses.
[`training_model.py`](training_model.py) therefore groups links by
fixed-joint connectivity and combines them with the parallel-axis theorem. That fusion is validated
against Gazebo itself: fusing the locosim URDF here and asking
`/gazebo/get_link_properties` for the same bodies agree to **3.4e-7** on mass, inertia and centre of
mass across all 13 bodies.

### What they disagreed about, and the fix

**`aliengo_description/urdfs/const.xacro` has been corrected to Unitree's own values**, so the
tables below are history — the check now reports a match on all 13 bodies. What was wrong:

| body (sim / training) | mass old/Unitree | ixx | iyy | izz |
|---|---|---|---|---|
| `lf_lowerleg` / `FL_calf` (calf + foot) | 0.267 / 0.267 kg | **2.570** | **2.563** | 1.029 |
| `base_link` / `trunk` | 12.042 / 12.229 kg | 0.942 | **2.475** | **2.309** |
| `lf_hip` / `FL_hip` | 1.993 / **2.139** kg | 1.135 | 1.188 | 1.291 |
| `lf_upperleg` / `FL_thigh` | 0.639 / **0.771** kg | 1.016 | 1.235 | **0.248** |

Ratios are old locosim / Unitree; all four legs were identical. Total mass **23.638 kg against
24.937 kg**, i.e. the locosim robot was 1.3 kg (5.2 %) lighter — 1.696 kg of actuator rotor mass
that the Isaac asset models and locosim did not, less 0.397 kg of extra trunk.

### One of them was provably wrong

The calf is not a matter of taste. A rigid body's inertia about its centre of mass cannot exceed
`m · d²`, where `d` is the greatest distance from the CoM to any point of the body — that bound is
attained only when the entire mass sits at that single farthest point. For the locosim calf
(`lowerleg_length` 0.25 m, mass 0.207 kg, CoM at z = −0.1425, so d = 0.1425 m):

```
I_max = 0.207 * 0.1425^2 = 0.004204 kg m^2
locosim  lowerleg_ixx = 0.006341   ->  1.51x the physical maximum   IMPOSSIBLE
isaac    FL_calf ixx  = 0.002129   ->  0.51x the maximum            plausible
```

The locosim value implies a radius of gyration 1.23× the distance to the farthest point of the
link. No mass distribution achieves that. `aliengo_description/urdfs/const.xacro` has
`lowerleg_ixx`/`lowerleg_iyy` about 3× too large, and since the foot is a separate link it cannot
be explained by the foot either.

The trunk is a judgement call rather than a proof, but points the same way: a uniform box of the
trunk's outer dimensions (0.647 × 0.150 × 0.112 m) at 12.041 kg would have `iyy` 0.433, and
locosim's 0.640 is 1.48× that — mass at the extremities — while Isaac's 0.247 is 0.57× — mass
concentrated centrally, which is what a battery in the middle of the trunk looks like, with the hip
motors accounted for as their own bodies.

This was a bug in the shared `aliengo_description`, and it is now fixed there: `const.xacro`
carries Unitree's inertials, the two joint origins that disagreed (`leg_offset_x` 0.2399 → 0.2407,
`upperleg_offset` 0.083 → 0.0868) and `KFE_velocity_max` 16 → 15.89. **It changes the model for
every locosim controller on this robot**, which is the point — the old calf value was not a
different opinion, it was impossible.

Two conventions had to be handled to transfer the values without changing the file's structure or
any joint name:

* Unitree models each actuator rotor as a separate link on a fixed joint. Since a physics engine
  merges those before integrating, they are **folded into their parent** by the parallel-axis
  theorem rather than added as links: `hip` carries its thigh rotor, `upperleg` its calf rotor, and
  `trunk` all four hip rotors — 1.696 kg the model did not previously have at all.
* `const.xacro` stores one canonical leg and `leg.xacro` mirrors it with sign flips on
  `com_y`, `ixy`, `ixz`, `iyz`. That Unitree's four legs follow *exactly* that pattern was checked
  rather than assumed: all four agree to 0.0 on mass, CoM and every inertia component, so one
  canonical set is faithful.

`hip_offset` was deliberately left at 0.083: it only places the simplified hip collision cylinder,
not a joint, and 3.8 mm is well inside that cylinder's 46 mm radius.

Verified by expanding the corrected xacro and comparing physics bodies again:

```
worst disagreement across all 13 bodies: 4.4e-10
total mass  locosim 24.937000 kg   unitree 24.937000 kg   difference 0.0
joint origins: haa, hfe, kfe all match to 0.0
```

> **The install space holds a copy.** `aliengo_description` installs its files verbatim, and
> `$(find aliengo_description)` resolves to `ros_ws/install/share/`, so an edit to the source tree
> has no effect until the package is reinstalled. Both were synced here; after a `catkin_make
> install` the source values are reproduced.

### What the controller does about it

`training_model_check` in the config is `'warn'` (default) or `'off'`. It only ever **reports** —
at start-up, in simulation, one line per distinct discrepancy:

```
Deployed model matches the training model on all 13 bodies (within 2 %)
```

The check stays even though the description is now correct, because that is exactly what it guards:
a mismatch appearing here means the description has drifted — a stale install space, an edit, or a
policy retrained against a different asset — and the message says where to fix it.

There was briefly a `'match'` mode, and a `--match-training-model` flag, that pushed the trained
values into Gazebo at start-up. **Both are gone.** They were a workaround for the description being
wrong, and now that it is right at source the workaround is worse than nothing: a controller that
silently rewrites the physics engine's numbers leaves every other tool on the robot — rviz, the
other controllers, anyone else's Pinocchio model — looking at a different machine than the one being
simulated. The values belong in the robot's own xacro, which is where they now are.

`training_body_map` in the robot config carries the naming map (Gazebo `base_link` ↔ URDF `trunk`,
`lf/lh/rf/rh` + `hip/upperleg/lowerleg` ↔ `FL/RL/FR/RR` + `hip/thigh/calf`). An empty map disables
the check.

### Running this policy from the old controller

`quadruped_controller.py` drives `RlVelocityController` from
`components/rl_velocity_controller/`, and that class can now run either policy family:

```python
rl_policy = 'legacy'    # in quadruped_controller.py: 'legacy' or 'velocity'
```

* **`'legacy'`** (the default) is the original arrangement — a 48-float observation for the actor, a
  separate state-estimation network on a 3-frame history, and `_safe` variants of both, all under
  `components/rl_velocity_controller/policies/`.
* **`'velocity'`** runs the single-network policy from this directory, by delegating to
  `VelocityPolicy`. There is one implementation of the 228-float contract rather than two that can
  drift apart; the legacy class only adapts it to the `action(...)` signature that controller
  already calls.

Two consequences of the delegation, both reported rather than silently worked around:

* `use_nn_se=False` cannot be honoured. With it false the caller passes `base_lin_acc=None` and
  hands over a *measured* base velocity instead — and this policy consumes the accelerometer and
  regresses the velocity itself. There is no variant of it that takes a measured base velocity, so
  the flag is ignored with a message. Passing `base_lin_acc=None` raises rather than producing
  nonsense.
* The internal estimate is published on `/<robot>/se_nn_base_lin_vel`, the same topic the legacy
  estimator used, so anything already plotting it keeps working. `debug=True` still publishes
  ground truth on `/<robot>/gt_base_lin_vel` for comparison.

`policy_type="default"` maps to the `normal` variant and `policy_type="safe"` to `safe`, so the
existing call sites choose between the two networks unchanged.

**Retro-compatibility is checked, not asserted.** The pre-change class was loaded straight from git
alongside the new one and both were driven with identical inputs:

| case | steps | worst \|Δq_des\| |
|---|---|---|
| legacy, `use_nn_se=True, debug=True`, both policy types | 220 | **0.000e+00** |
| legacy, `use_nn_se=False` | 120 | **0.000e+00** |
| `'velocity'` through `action(...)` vs `VelocityPolicy` directly, across two variant switches | 200 | **0.000e+00** |

## Policy timing: does inference fit the tick?

The loop is **synchronous**, and that is the first half of the answer:

```
every tick (2 ms)   read state -> handler -> send_des_jstate -> publish telemetry
1 tick in 10        the handler additionally runs the network and the estimation head, and the
                    joint target it produces is sent on that same tick
```

There is no producer/consumer split, no queue and no worker thread, so **the action can never be
stale**: the tick that computes it is the tick that applies it. An inference tick that overruns
delays that one low-level command; the *policy* period is unaffected as long as the overrun stays
under 20 ms.

The second half is what it costs, and here two different numbers have to be kept apart:

| | measured | of the 2 ms tick |
|---|---|---|
| onnxruntime graph execution, back-to-back | 33 µs | 1.6 % |
| velocity-estimation head (numpy), back-to-back | 16 µs | 0.8 % |
| **whole inference path, back-to-back** | **85 µs** | **4.3 %** |
| whole inference path, measured in the running loop (Gazebo) | 2.7 ms mean, 36 ms worst | 135 % / 1800 % |

The last row is not policy cost. In the loop the call also contains however long the process spent
*off the CPU* — and in a container sharing cores with a physics engine that term dominates by more
than an order of magnitude. Reporting only the in-loop figure would blame the policy for Gazebo's
scheduling.

The back-to-back figure is the **minimum** of 400 runs, not the mean or median, and that detail
matters: running back-to-back does not by itself keep the process on the CPU. The benchmark fires
moments after the simulator launches, and while that settles the machine steals time from *most*
iterations — a median taken there came out at **2180 µs against a true cost near 50 µs**, which is
how this was caught. The fastest iteration is the one that was not interrupted. The median is
reported alongside it, and a large gap between the two is itself the signal that the machine was
busy while measuring.

So the controller reports both. At start-up it benchmarks the inference path before the loop
begins, and at shutdown it prints the benchmark, the in-loop figure and the gap:

```
Policy inference benchmark: 85 us of the 2000 us control tick (4.3 %), fastest of 400
  back-to-back runs; median 114 us
...
  Policy inference costs 85 us of the 2000 us control tick (4.3 %), fastest of 400 back-to-back runs.
  In the loop it took 2.691 ms on average and 35.503 ms at worst, over the tick on 284 of
  729 inference ticks (38.96 %) - the excess over the figure above is time the process spent
  off the CPU, not policy cost.
  In simulation that is expected: Gazebo shares these cores and the loop is descheduled inside
  the call. Judge this on the robot, where the two figures should agree.
```

**The gap between the two is the quantity to watch.** On the robot they should agree; if the
in-loop figure runs well above the benchmark there, something else on the control PC is stealing
the CPU — which is what `run_rl_controller.sh`, `SCHED_FIFO` and core isolation exist to prevent.
Live, the same information is on `policy/infer_ms` and `policy/infer_over_budget`.

### Immediate hand-over: no ramps, no blends

Asking for a policy hands it the robot on the next tick. There is nothing between the operator and
the network:

* the **command** is read live and fed as-is — no ramp, so whatever is asked for takes effect on
  the next inference tick;
* the **joint target** the network returns is applied unmodified — no blend from the posture being
  handed over from into the policy's own equilibrium;
* the same holds in both directions between the walking and safe-stop policies, and the observation
  history carries over untouched.

The one thing that *is* reset is the **stored command**, which is zeroed on **every** entry into a
policy — including the hand-back from the safe stop. That is not easing: no policy ever inherits a
velocity, so the operator always commands from a known state, and whatever they ask for still
arrives on the next tick with nothing in between. The hand-back is the case that matters most: it
happens because something went wrong, and it is the last moment a leftover command should launch the
robot. On a pad the stick is re-read every message, so a stick genuinely held takes effect
immediately anyway.

Measured — entering with `[0.35, -0.1, 0.2]` stored, commanding `0.25` while walking, then a safe
stop with `0.3` stored just before resuming:

```
--- entered rl          t+  0.0 ms  #1    fed=[0. 0. 0.]  stored=[0. 0. 0.]
--- entered safe_stop   t+ 18.0 ms  #153  fed=[0. 0. 0.]  stored=[0. 0. 0.]
--- entered rl          t+  0.0 ms  #203  fed=[0. 0. 0.]  stored=[0. 0. 0.]
```

**`auto_safe_stop_after` no longer hands back on its own.** It still hands *over* to the safe stop
after the command has been zero for the dwell, but leaving is deliberate — resume, stand down, or
collapse. A hand-back driven by the operator's command cannot work once every entry zeroes that
command: it would discard the very command that triggered it and drop back to the safe stop on the
next dwell.

Earlier versions eased both of these (`rl_command_ramp_duration`, `rl_blend_duration`). **Both are
gone, along with their config keys.** They were insurance against a step input on the handover tick,
and they bought it by making the robot not do what it had just been told for up to a second — which
is the wrong trade for a machine an operator is steering. The one command the state machine still
overrides is the safe stop's, held at zero because that is the only command that variant was trained
on, and it overrides it outright rather than fading it.

If the *command* needs smoothing, that belongs in the input device, where the operator can see it —
not in the state machine, where it silently delays what they asked for. The joystick already
low-passes mechanically through the stick itself and applies a dead zone; nothing digital is added.

The consequence worth knowing: **once commanded, the robot moves at once** — no ramp means the
first tick after a command is a step input. Every policy entry is safe from this, since the command
begins at zero there; the step is only ever one the operator just asked for. The safe stop and the
emergency damping are always one key away.

## Testing robustness: pushing the robot

In simulation the operator can shove the base and watch either controller deal with it. `p` applies
one push in a random horizontal direction, `P` a burst of `push_burst_count` of them at random
intervals, and `-` / `+` change the magnitude live — so a run can be walked up from a nudge to
something the policy has never seen, without restarting.

```
Disturbances armed on aliengo::base_link: 'p' one random push, 'P' a burst of 5, -/+ to change
  the 125 N magnitude
...
push force 200 N
PUSH #2: 200 N at    26 deg for 0.20 s [[180.4  86.4   0. ] N, state 'rl']
PUSH #3: 200 N at    47 deg for 0.20 s [[137.1 145.6   0. ] N, state 'rl']
```

**The default is a training-strength push, deliberately.** Training perturbs the robot with
`mdp.push_by_setting_velocity` over ±1.0 m/s in x and y; on a 24.94 kg robot that is an impulse of
about 24.9 N·s, which over the default `push_duration` of 0.2 s is **125 N**. So `p` at the default
asks the policy for something it has seen, and turning the force up is testing past its training —
which is the interesting direction, and now a keypress away.

> Two details of the training config are worth knowing when comparing. Its push interval is
> 10–15 s in the walking task and 2–3 s in the safe-stop task, so the safe policy saw them far more
> often. And its `velocity_range` is written `{"x": …, "y": …, "Z": …}` — `push_by_setting_velocity`
> only reads lowercase keys and *zeroes any axis it does not find*, so the vertical component was
> never actually applied. `push_vertical` here is off by default to match what was really trained.

Measured, walking at 0.2 m/s and then holding station, worst values reached:

| state | pushes | \|ω_base\| max | \|roll,pitch\| max | base height |
|---|---|---|---|---|
| `rl` (walking) | 4 | 2.13 rad/s | 0.072 rad | 0.301–0.398 m |
| `safe_stop` | 3 | 2.86 rad/s | 0.258 rad | 0.336–0.412 m |

Both recovered every time at 125–200 N; neither tripped the safety envelope.

Two implementation points. The wrench goes out through `/gazebo/apply_body_wrench` **on its own
thread** — that call is a round-trip to another process and the control loop has 2 ms, so the loop
only posts to a queue and never waits. And the force vector is published on the `push` topic as it
lands, latched, because a plot of a recovery is unreadable without knowing when the robot was hit.

The direction is sampled as a uniform *heading* rather than a uniform vector in the plane: the point
is to probe every direction equally, and a uniformly sampled xy vector clusters toward the diagonals.
`reference_frame` is `world` because that is the only frame this Gazebo service handles correctly —
it silently ignores any other.

On the real robot the keys report that pushes are simulation only and do nothing. Push the robot
yourself.

## Frame conventions, checked against ground truth

The observation the policy consumes is in the base frame, and a wrong rotation there is both easy to
introduce and invisible in a log. So it is verified against Gazebo's ground-truth odometry rather
than argued from the URDF. Over 10102 samples of walking and turning:

| quantity | vs ground truth **in the base frame** | vs ground truth **in the world frame** |
|---|---|---|
| `ang_vel_b` per-axis correlation | **0.997, 0.999, 1.000** | 0.588, 0.218, 1.000 |
| `ang_vel_b` per-axis RMS error | 0.005, 0.004, 0.001 rad/s | 0.069, 0.085, 0.002 rad/s |

against a signal RMS of 0.067 / 0.076 / 0.135 rad/s. The world-frame column is the control: if the
IMU were secretly reporting world-frame rates, that column would be the better fit. It is not, on
the two axes that a yaw rotation distinguishes. `projected_gravity` agrees with ground truth to an
RMS of 1.6e-4 per axis.

This is also what the URDFs say, and the two now agree: the IMU is rigidly at the base origin with
no rotation in both models (`imu_joint` has `rpy="0 0 0" xyz="0 0 0"` onto `trunk` in locosim, and
Isaac attaches `ImuCfg` to `/Robot/base` with the default identity offset), and the Gazebo plugin
publishes in the sensor frame with `rpyOffset 0 0 0`. **The angular velocity transform is correct.**

## Stopping: why the robot sometimes oscillates, and what actually fixes it

Commanding zero does not reliably bring the walking policy to rest. Over seven identical Gazebo
runs of *walk at 0.3 m/s for six seconds, then hold the command at zero*, measuring the residual
base angular rate 8–12 s after the stop:

| run | residual \|ω_base\| | lf_hfe peak-to-peak | dominant frequency | outcome |
|---|---|---|---|---|
| 1 | 0.0436 rad/s | 0.0230 rad | **3.45 Hz** | limit cycle |
| 2 | 0.0001 rad/s | 0.0033 rad | — | settled |
| 3 | 0.0003 rad/s | 0.0014 rad | — | settled |
| 4 | 0.0020 rad/s | 0.0878 rad | — | settled, after a 0.56 rad/s excursion at 4–8 s |
| 5 | 0.0001 rad/s | 0.0033 rad | — | settled |
| 6 | **0.1492 rad/s** | **0.0654 rad** | **3.82 Hz** | limit cycle, *growing* |
| 7 | 0.0001 rad/s | 0.0007 rad | — | settled |

Same controller, same model, same command sequence. Four settle to about 1e-4 rad/s — genuinely
still. Two lock into a non-decaying trot in place at the learned gait frequency; in run 6 it did
not merely persist but *grew*, from 0.088 rad/s at 4 s to 0.18 rad/s at 12 s. So the walking policy
has **two attractors at zero command**, a static stand and a stationary trot, and which one it lands
in depends on the gait state at the instant the command drops.

Three candidate explanations were tested and eliminated:

* **Not the angular-velocity frame.** Verified against ground truth to a correlation of 0.997 /
  0.999 / 1.000 — see [Frame conventions](#frame-conventions-checked-against-ground-truth).
* **Not loop timing.** Inference costs 89 µs of the 2 ms tick. The loop *does* get descheduled in
  simulation, but the jitter is the same in the runs that settle and the runs that do not (mean
  period 2.36–2.42 ms, p99 7.8–8.3 ms, over twice nominal on 2.5 %, 2.8 % and 4.2 % of ticks for
  runs 5, 6 and 7 — the failure is the middle one). Correlating loop-period p99 against residual
  motion in 0.5 s slices within each run gives **r = −0.28, −0.21, −0.04**: no positive
  relationship. See [Policy timing](#policy-timing-does-inference-fit-the-tick).
* **Not the model mismatch.** The [model discrepancy](#the-model-the-policy-was-trained-on) was
  real and one of its numbers was provably wrong, and it has since been **fixed at source** — the
  simulated robot now matches Unitree's to 4.4e-10 on every body. The oscillation survived it. Four
  runs of the same sequence on the corrected model gave residuals of 0.0064, **0.0915**, 0.0042 and
  0.0001 rad/s at 12–16 s: one clear limit cycle, one clean settle, two small standing residuals.
  Against four clean settles out of seven before the fix, that is not an improvement — and a first
  single-run A/B that had appeared to show an effect did not replicate. At a ~30 % failure rate,
  single-run comparisons of this behaviour prove nothing either way.

This is a property of the policy, and the training config says why it would be: `rel_standing_envs`
is 0.1 and the command resamples every 10 s, so a walk-then-stop transition is a small fraction of
what the walking policy ever saw. It was never pushed to reliably kill its own gait. That it does
settle reliably in Isaac Lab, at the same 10 %, points at what has *not* been equalised between the
two simulators — and after ruling out the frames, the timing and the rigid-body model, what is left
is the contact and solver behaviour: Gazebo/ODE at a 1 ms step against PhysX at 5 ms, with
different friction and contact stiffness, and a randomised 0–2 policy steps of actuator delay in
training against roughly zero here. Those are not things a deployment controller can match.

### What does work

The safe-stop policy settled in **every** run measured — 0.0002 to 0.003 rad/s of residual base
rate, lf_hfe under 0.012 rad, base height peak-to-peak under 2 mm. That is what it was trained for:
zero command was the *only* command it ever saw.

So the reliable way to stop this robot is to hand over to it. Either press it (`f` / RB), or set:

```python
"auto_safe_stop_after": 0.5,    # s of continuous zero command before handing over
```

which hands over automatically once the command has been zero for that long. Resuming is then
deliberate — `r` / X — since [every policy entry zeroes the stored
command](#immediate-hand-over-no-ramps-no-blends) and a command-driven hand-back would discard the
command that triggered it. Three runs of the same walk-then-stop sequence with it enabled, residual
base rate 8–16 s after the stop:

| run | 4–8 s | 8–12 s | 12–16 s |
|---|---|---|---|
| 1 | 0.0001 rad/s | 0.0001 | 0.0001 |
| 2 | 0.0008 rad/s | 0.0001 | 0.0001 |
| 3 | 0.0066 rad/s | 0.0026 | 0.0007 |

Three for three, against four out of seven without it. It is **off by default** because it is a
state change the operator did not ask for, it needs an explicit resume, and most of the time the
walking policy stops perfectly well — but if you need "stop" to mean stopped, this is the switch.
The dwell time matters: switching on the first zero tick would fire between two pushes of the
stick.

The proper fix is on the training side — more standing envs, or a reward that penalises motion at
zero command — at which point this switch stops being necessary.

## rviz and the world transform

rviz is launched with `world` as its fixed frame, and `robot_state_publisher` supplies only the
transforms *below* `base_link` — it turns joint states into link poses but cannot know where the
base is. Without a `world -> base_link` transform there is no path from the fixed frame to the
robot and rviz draws nothing, which is exactly what happened when rviz was first switched on here.

`BaseController` does have that broadcast, but it lives inside the full kinematics update that this
controller deliberately does not run, so it is published here instead — from `publishTelemetry`, at
the telemetry rate rather than the control rate, because rviz has no use for 500 Hz TF and the
traffic is not free. `publish_tf` in the config turns it off.

In simulation the translation comes from the ground-truth odometry. **On the robot it does not
exist**: nothing measures absolute position, the policy does not need it, and this controller runs
no odometry — so the translation stays at the origin and only the measured attitude is broadcast.
That is the honest thing to draw: the robot rendered in place, leaning the way it actually leans. It
is not a bug to fix by integrating the velocity estimate; that would drift and invite you to trust
a position nothing measured.

Verified end to end: `world -> base_link` resolves, and `world -> lf_foot` chains through
`robot_state_publisher`, so the whole robot is placed in the fixed frame.

## Real-robot notes

`--real` pings the robot before doing anything, folds in the `real` sub-dict of the config, switches
to the split hardware IMU topics, clips commands to the joint limits, and turns on the real-time
tuning. The loop rate is the same 500 Hz as in simulation. See
[Real-time behaviour and memory](#real-time-behaviour-and-memory) for what the loop does and does not
do, and [Real-time priority and core isolation](#real-time-priority-and-core-isolation) for the
launcher.

**Before the first hardware run:**

* keep the robot lying flat and still through calibration, and check the reported residual is near
  zero — if it is not, the policy is being fed a wrong acceleration and nothing downstream is
  trustworthy;
* have the emergency control in reach: `BACK` (or `LB+RB`) on the pad, `SPACE` on the keyboard. It
  works from any state, including mid-way through typing a velocity;
* prefer `./run_rl_controller.sh --real --input joy` over a bare `python3`, so the real-time priority
  covers start-up as well;
* start with the `real` command limits (0.3 m/s, 0.4 rad/s) and open them up once it walks well;
* be aware that `kd_rl = 0.5` is the training value and is deliberately soft — the policy expects it,
  but it will feel underdamped by hardware standards;
* watch the joint velocities. In Gazebo the swing legs peak near 17 rad/s at 0.3 m/s, which is close
  to the 16–20 rad/s the actuators were modelled with in training; real motors will saturate sooner
  than the simulation does, and the `max_joint_vel` safety limit is set at 25 rad/s.

## Configuration

`rl_controller_config.py` is the **only** configuration this controller reads. It does not import
`params.py` and nothing is merged with it: the controller hands the config module to
`BaseController` as its `external_conf`, which replaces `params.robot_params` for the lifetime of the
process. So a robot that runs this controller is described in one file, and tuning it cannot disturb
any other controller in the framework — nor can a change to `params.py` silently alter this one.

That is why the robot description fields appear there too, even though `params.py` has its own
copies — the two are deliberately independent:

| | comes from |
|---|---|
| `joint_names`, `ee_frames`, `q_fold`, `ip`, `spawn_*`, `control_dt` | `rl_controller_config.py` |
| gains, timing, safety, operator input, telemetry, real-time | `rl_controller_config.py` |
| the URDF / kinematics | the `<robot>_description` package, as always |
| `params.py` | **nothing** |

Resolution order, later winning:

1. `DEFAULTS`
2. `ROBOTS['<robot>']`
3. `ROBOTS['<robot>']['real']` — only with `--real`, and **normally empty**

The control parameters are deliberately **identical in simulation and on the robot**. The policy was
trained against a model of the machine, so a gain or a duration that has to differ between the two is
a modelling error to fix, not a difference to paper over — and a controller that behaves differently
in simulation is one you cannot test there. The `real` mechanism is kept for anything genuinely
hardware-only, but nothing uses it.

The only things that still resolve from `--real` are not modelling parameters: joint-limit clipping,
real-time scheduling, and the simulation-only spawn-height correction.

```python
ROBOTS = {
    "aliengo": {
        "joint_names": [...], "ee_frames": [...], "ip": "192.168.123.220",
        "q_fold": _legs(0.1, 1.6, -2.40),
        "real": {
            "stand_up_duration": 3.0,   # slower on hardware
            "max_lin_vel_cmd": 0.3,     # start conservative
        },
    },
}
```

Key groups, all documented inline in the file:

| Group | Keys |
|---|---|
| Robot description | `joint_names`, `ee_frames`, `ip`, `q_fold`, `spawn_x/y/z/R/P/Y` |
| Loop rate | `control_dt` (0.002 s = **500 Hz**, decimation 10) |
| Gains | `kp_fold`, `kd_fold`, `kp_stand`, `kd_stand`, `kp_rl`, `kd_rl`, `kd_damping` |
| Timing | `settle_duration`, `init_fold_duration`, `calibration_duration`, `stand_up_duration`, `stand_down_duration`, `damping_ramp_duration`, `damping_settle_duration` |
| Standstill | `auto_safe_stop_after`, `auto_safe_stop_deadband` |
| Disturbances (sim) | `push_force`, `push_duration`, `push_force_step`, `push_vertical`, `push_burst_count`, `push_burst_interval`, `push_body` |
| Postures | `q_stand`, `stand_up_via_fold` |
| Calibration | `calib_stable_window`, `calib_gyro_threshold`, `calib_joint_vel_threshold`, `calib_wait_timeout`, `calib_max_bias`, `calib_max_spread`, `calib_max_residual`, `skip_calibration` |
| Safety | `max_roll_pitch`, `safe_max_roll_pitch`, `max_joint_vel`, `clip_to_joint_limits` |
| Operator input | `max_lin_vel_cmd`, `max_ang_vel_cmd`, `key_lin_step`, `key_ang_step`, `key_speed_presets`, `joy_dead_zone` |
| Telemetry | `publish_telemetry`, `telemetry_rate`, `stats_rate`, `topic_namespace`, `publish_estimated_velocity`, `estimate_base_velocity`, `measure_inference_timing` |
| Real-time | `realtime`, `sched_fifo_priority`, `pin_cpu`, `pin_cpu_index`, `gc_managed`, `gc_gen0_threshold`, `gc_gen1_threshold`, `memory_growth_warn_mb`, `memory_check_rate` |
| Policy | `policy_name`, `safe_policy_variant` |
| Training model | `training_model_check`, `training_model_tol`, `training_body_map` |
| Simulation start-up | `fix_spawn_height`, `spawn_clearance`, `foot_radius`, `world_name`, `use_rviz`, `rviz_conf` |

`kp_rl` / `kd_rl` / `q_stand` default to `None`, meaning "take it from the policy's json". Overriding
`kd_rl` in particular is a deliberate departure from training — 0.5 is soft on purpose.

There are two low-level gain sets on purpose. `kp_fold` / `kd_fold` is used for the initial hold,
the `FOLD` move and the folded hold, where the feet are loaded and being dragged across the ground;
`kp_stand` / `kd_stand` is used to lift and hold the body. Stiff position control during the fold
move is a large part of what made the robot judder — see below. Both currently sit at 100 / 5, and
`kd_damping` at 10.

## Start-up: the spawn drop, and the vibration it caused

The robot used to ring for the first few seconds. The cause was in the launch, not the controller:

* `spawn_model` places the base at `params.py['spawn_z']` — 0.37 m for the Aliengo, a height sized
  for the `q_0` standing posture;
* but the robot is put into the SRDF `home` group state by the description's `go0` script, which is
  a sprawled pose (`haa = ±0.8`) whose feet are only **0.084 m** below the base;
* and `go0` **ignores the `go0_conf` argument the launch hands it** — it reads `argv[1]` as a sleep
  time and then always applies `home`, so `go0_conf:=standDown` never did anything.

Net effect: a **0.257 m free fall** onto the ground on every start, followed by the controller
stiffly holding a posture the robot cannot hold statically. That also made the accelerometer bias
unobservable, because calibration was averaging over a bouncing robot.

Three fixes, all in this controller:

1. `fix_spawn_height` reads the SRDF `home` posture, computes the matching base height with
   Pinocchio, and corrects `spawn_z` *before* the launch runs — no drop at all. Set it to `False` to
   use the stock spawn.
2. The joint target is claimed as soon as the first joint state arrives, still at the launch's own
   soft gains. Until something commands it, the low-level controller drives towards the `home`
   posture in its yaml — which has the hips 0.8 rad away from where the model was spawned — so it
   spent the whole start-up dragging the splayed legs inward under the robot's weight.
3. `INIT` holds whatever posture it inherits for `settle_duration` — the quintic in `FOLD` then
   starts from a joint at rest rather than a moving one — and the fold itself runs over
   `init_fold_duration` on the softer `kp_fold`/`kd_fold` gains. And `q_fold`
   itself moved: `params.py` puts the knees at −2.70 against a −2.775 limit, leaving 0.075 rad of
   margin, so the knees ended up parked on the stop carrying ~7.5 Nm to stay there. This controller
   uses −2.40, which leaves 0.375 rad and rests quietly.

Measured over the same start-up sequence in Gazebo:

| | original | now |
|---|---|---|
| spawn drop | 0.257 m free fall | none |
| peak base rate during the take-over hold | 4.07 rad/s | **0.83 rad/s** |
| mean base rate during `CALIBRATE` | 0.61 rad/s | **0.016 rad/s** |
| calibration residual | −2.15 m/s² | **−0.0000 m/s²** |
| knee margin to the joint limit while folded | 0.075 rad | **0.375 rad** |

A steady folded hold, once settled, sits at about 0.003 rad/s of base rate and 0.001–0.012 rad/s of
joint rate — i.e. the residual buzz is contact noise, not the controller.

On hardware none of the spawn work applies — the robot is already on the floor — but the take-over
hold, the eased fold, the softer fold gains and the stability-gated calibration all do, and they run
with exactly the same durations and gains as in simulation. See
[Configuration](#configuration) for why nothing here is tuned per-target.

## Calibration: first, and why it has to be still

The bias estimate is only meaningful while the base is not accelerating — any real motion inside the
window is indistinguishable from sensor bias, and a bias that absorbed part of a settling transient
is *worse* than no bias at all, because everything downstream then trusts it.

That is why calibration is the **first** thing the controller does, before any commanded motion at
all. The quietest the robot will ever be is the moment before something asks it to move, so the
procedure takes the operator at their word — the robot has been set down flat and left alone — and
commands nothing but "hold exactly where you are". That hold is not a no-op: the low-level
controller has been running since the launch with its own target, so the robot is usually still
drifting when this one takes over, and latching the measured position is what stops it.

The window is continuous, not opportunistic:

1. wait for `calib_stable_window` (1 s) of uninterrupted stillness — base rate under
   `calib_gyro_threshold` (0.05 rad/s) and joint rate under `calib_joint_vel_threshold`
   (0.1 rad/s), against the ~0.003 / ~0.012 rad/s a genuinely settled robot shows;
2. integrate for `calibration_duration` (3 s) more, still requiring stillness throughout;
3. if stillness is lost at any point, throw the window away and go back to step 1 — the controller
   says so when it happens;
4. give up after `calib_wait_timeout` (30 s), leaving the bias at zero and saying so plainly,
   rather than calibrating on a moving robot.

It runs **once**, and there is no automatic second attempt from another posture. The procedure
takes the operator at their word — the robot has been laid flat and left alone — and if it never
goes still then that assumption was wrong, which is worth being told rather than worked around.
Failure is not terminal: it continues to `FOLD` with a zero bias, and `c` (Y on the pad) re-runs it
from there as often as you like. That command is refused while the fold is still moving, since it
would only burn the timeout and report failure.

```
The robot never became still within 30 s: skipping calibration, the accelerometer bias stays
zero and the policy will see the accelerometer as it comes. Lay the robot flat on the ground
and re-trigger the calibration with 'c' (Y on the pad) from the folded hold - it can be
repeated as often as you like.
```

> **In Gazebo the first attempt will fail.** The simulated robot spawns in the splayed SRDF `home`
> posture, and a leg loaded against the ground there chatters at 0.05–0.10 rad/s — just over the
> threshold — so it never yields a continuous 1 s window however long you wait. Once folded it sits
> at 0.000 rad/s, so `c` calibrates cleanly. For simulation `--skip-calibration` is the honest
> option anyway: the Gazebo IMU is configured with `bias_mean 0` and `bias_stddev 0`, so there is no
> bias there to measure — the value it finds, `[-0.0002, -0.0, 0.004]`, is just noise.

`--skip-calibration` (or `skip_calibration` in the config) drops the procedure entirely: `INIT` goes
straight to `FOLD` and the bias stays at zero. The controller says so at start-up and warns again on
stand-up, because that offset then goes straight into the accelerometer history the policy acts on.

### The three acceptance checks

Three independent checks decide whether the result is usable, and all three are reported:

| check | meaning | on failure |
|---|---|---|
| `\|bias\| < calib_max_bias` (2 m/s²) | a plausible sensor offset | rejected, bias left at zero |
| per-axis spread `< calib_max_spread` (0.25 m/s²) | the robot really was still | window redone |
| `\| \|acc − bias\| − 9.806 \| < calib_max_residual` (0.5 m/s²) | gravity is all a static accelerometer sees | accepted with a warning |

A good calibration has spread in the fourth decimal and a residual that rounds to zero, as above.

An earlier version accumulated whichever samples happened to look quiet and skipped the rest. That
finishes faster and looks fine in the log, but it averages across a robot that is still settling.
If you see the spread or the residual come out large, the robot moved — put it down flat and
re-run the calibration with `c` rather than trusting the number.

## Telemetry: topics, rosbag, PlotJuggler

Nothing is accumulated in memory. Everything the controller knows is published, which is what makes
it recordable with `rosbag` and readable live in PlotJuggler — and what keeps the memory footprint
independent of how long the robot runs. (The previous in-memory arrays were about 70 MB of
preallocated buffers that stopped recording once full.)

Joint data is already on `/command` and `/<robot>/joint_states`, so it is not duplicated.

| Topic (under `/<robot>/rl/`) | Type | Rate | Content |
|---|---|---|---|
| `state` | `std_msgs/String` | on entry, latched | state machine state |
| `policy_variant` | `std_msgs/String` | on entry, latched | which network is driving: `normal` or `safe` |
| `velocity_command` | `geometry_msgs/TwistStamped` | 50 Hz | operator command fed to the policy |
| `action` | `std_msgs/Float64MultiArray` | 50 Hz | raw policy output, 12 joints in `joint_names` order |
| `estimated_base_lin_vel` | `geometry_msgs/Vector3Stamped` | 50 Hz | base linear velocity the actor regresses internally, base frame. Not advertised with `--no-estimator` |
| `imu_lin_acc` | `geometry_msgs/Vector3Stamped` | 100 Hz | bias-corrected specific force, as the policy sees it |
| `projected_gravity` | `geometry_msgs/Vector3Stamped` | 100 Hz | unit gravity in the base frame |
| `ang_vel_b` | `geometry_msgs/Vector3Stamped` | 100 Hz | body angular rate |
| `base_rpy` | `geometry_msgs/Vector3Stamped` | 100 Hz | attitude |
| `loop/dt_mean_ms`, `loop/dt_max_ms`, `loop/overruns` | `std_msgs/Float64` | 10 Hz | loop period statistics |
| `loop/compute_mean_ms`, `loop/compute_max_ms` | `std_msgs/Float64` | 10 Hz | computation time per tick |
| `memory/rss_mb` | `std_msgs/Float64` | 1 Hz | resident memory |
| `policy/inferences` | `std_msgs/Float64` | 10 Hz | cumulative inference count |
| `policy/infer_ms` | `std_msgs/Float64` | 50 Hz | cost of the last inference tick, against the 2 ms control period. Not advertised with `--no-inference-timing` |
| `policy/infer_over_budget` | `std_msgs/Float64` | 50 Hz | cumulative count of inference ticks that overran the control period. Not advertised with `--no-inference-timing` |
| `push` | `geometry_msgs/Vector3Stamped` | on each push, latched | the disturbance force applied, world frame (simulation only) |

Every message object is allocated once and refilled in place, so a publish costs a serialisation and
no allocation.

**Recording.** Either let the controller do it:

```bash
python3 rl_quadruped_controller.py --real --input joy --bag my_run
```

or copy the ready-made command the controller prints at start-up:

```bash
rosbag record /aliengo/joint_states /command /aliengo/rl/state /aliengo/rl/policy_variant \
  /aliengo/rl/velocity_command \
  /aliengo/rl/action /aliengo/rl/imu_lin_acc /aliengo/rl/projected_gravity /aliengo/rl/ang_vel_b \
  /aliengo/rl/base_rpy /aliengo/rl/loop/dt_mean_ms /aliengo/rl/loop/dt_max_ms \
  /aliengo/rl/loop/overruns /aliengo/rl/loop/compute_mean_ms /aliengo/rl/loop/compute_max_ms \
  /aliengo/rl/memory/rss_mb /aliengo/rl/policy/inferences
```

**PlotJuggler.** `rosrun plotjuggler plotjuggler`, then either stream live (ROS Topic Subscriber) or
open the bag. Useful overlays: `/command/position[i]` against `/aliengo/joint_states/position[i]`
for tracking, `/aliengo/rl/action/data[i]` for what the policy is asking for, and
`/aliengo/rl/loop/compute_max_ms` against the 2 ms budget for timing headroom.

## Real-time behaviour and memory

The loop runs at **500 Hz** (`control_dt = 0.002`), which divides the policy's 20 ms period exactly —
decimation 10. Measured over a full sequence in Gazebo:

| | |
|---|---|
| compute per tick, median | **0.14 ms** (7 % of the 2 ms budget) |
| compute per tick, p99 | **1.0 ms** (50 %) |
| policy inference alone | ~30 µs mean, ~43 µs p99 |
| resident memory | 188 MB, **+1.6 MB** over a 36 s run, flat throughout |

Those figures are with `rosbag` subscribed to every telemetry topic, which is the expensive case —
with no subscriber rospy skips serialisation entirely. The resident-memory curve is flat sample to
sample, including **+0.3 MB across the whole 15 s walking phase**, so the loop is not accumulating.

Note that in *simulation* the loop **period** follows Gazebo's real-time factor, because
`rate.sleep()` sleeps on sim time — so judge timing by the compute figures, which mean the same thing
in both places. The controller says so in its own shutdown report.

What the loop deliberately does not do: no `updateKinematics()` (no Pinocchio FK, mass matrix,
Jacobians, centroidal inertia or CoM), no state estimator, no leg odometry, no Pronto or mocap, no
inverse kinematics, no whole-body controller, no TF broadcast, and no in-memory logging.

What it does do:

* onnxruntime pinned to one thread with I/O binding, so a step is one graph execution and no buffer
  churn; BLAS/OMP thread counts forced to 1 before numpy loads;
* one preallocated `/command` message, preallocated telemetry messages, and precomputed joint-limit
  bounds, so a steady-state tick allocates nothing;
* `SCHED_FIFO` applied to **every thread of the process** — see below — or from outside by
  `run_rl_controller.sh`;
* optional core pinning, best paired with `isolcpus`;
* joint commands clipped to 90 % of the model limits on hardware.

**Garbage collection.** The startup heap is frozen out of every later collection (`gc.freeze`),
generations 0 and 1 keep collecting — both are bounded by their thresholds and cost microseconds —
and only the generation-2 full-heap scan is deferred, taken by hand when the machine enters a resting
state. The reasoning is that switching the collector off entirely, or deferring generation 1 as well,
leaves anything that survives a generation-0 pass accumulating for as long as the robot keeps
walking, since a long walk reaches no resting state at all. Keeping generation 1 in the loop bounds
that at negligible cost. At the end of a run `gc.get_count()` reads `(528, 2, 0)` against thresholds
of `(2000, 10, ...)` — collections are happening and nothing is piling up.

**Leak detection.** Resident memory is read from procfs once a second, published on
`memory/rss_mb`, and compared against the post-startup baseline. Growth past
`memory_growth_warn_mb` (64 MB) prints a warning once — a steady-state tick allocates nothing, so
real growth means something is leaking and that has to be visible rather than discovered when the
machine starts swapping. The shutdown report prints the final figure either way.

**Real-time priority is all-threads or nothing.** Raising only the control thread is not a
milder version of real-time scheduling, it is a hazard. rospy delivers joint states and the IMU on
background threads; a real-time main thread holding the GIL starves them, so the loop keeps running
at 500 Hz on an observation that has stopped updating. In simulation that reliably threw the robot
over about a second into walking, while reporting 200 ms "computation" times that were really the
loop waiting on threads it had itself starved. The controller now raises the control thread to
`sched_fifo_priority` and every other thread to ten below it, and if any thread cannot be raised it
puts them all back — a partially real-time process is the dangerous configuration, not a safer one.
Launching through `run_rl_controller.sh` sidesteps this, because threads inherit the scheduling
policy from the process at creation.

**Real-time priority belongs on hardware, not in simulation.** The default resolves `realtime` from
`--real` for a reason: in simulation Gazebo is essential work, so putting 26 FIFO threads above it
starves the simulator that feeds us. Measured over the same sequence, forcing `--realtime` in
Gazebo made timing *worse* — 8.1 % of ticks over 1.5x nominal against 2.46 % with it off. On the
real robot there is no simulator to starve, and the priority is what protects the loop from
everything else on the machine. The controller warns if you force it on in simulation.

State-entry ticks are the exception to all of the above: they pay for a `/set_pids` service
round-trip and a full collection, peaking around 8 ms. That is by design — it is why both are done at
a transition rather than in the loop — and the compute statistics report entry ticks separately so
they cannot hide the steady-state number. Steady-state ticks still show occasional multi-millisecond
outliers in this container, where Gazebo is competing for the same cores; that is what
`isolcpus` plus `run_rl_controller.sh` is for on the real robot.
