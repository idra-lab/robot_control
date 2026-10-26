"""Push-recovery sweep in Gazebo: the largest shove a policy survives, per command and direction.

    python3 push_sweep.py sweep --policy aliengo --out-dir ~/push_sweep/aliengo             # the grid
    python3 push_sweep.py plot ~/push_sweep/aliengo                                          # the figure

What it measures
----------------
For one velocity command and one push direction, the robot is stood up, handed to the policy and
pushed at 10, 20, 30 ... 400 N, one shove at a time, until it falls or has survived them all.  The
number reported is the first force it does **not** survive.

The grid that ``sweep`` runs:

* the walking (``normal``) variant at the zero command, and on three one-axis command sweeps -
  forward velocity, lateral velocity and yaw rate - from -0.5 to +0.5 in steps of 0.1;
* the ``safe`` variant at the zero command, the only command it was trained on;
* every one of those pushed forward, left, backward and right in the base frame.

Protocol
--------
The constants are in the ``protocol`` block below, and every one is written into each result file,
so a result says how it was made.

1. Start Gazebo and the controller, fold, stand up, start the policy.  For ``safe``, hand over to
   it the way the operator does, with the safe stop.
2. Hold the command for ``SETTLE_S`` of simulated time.  The mean base height over its last second is
   the reference height h0.  A fall here is a failure at 0 N.
3. For each force F: push the base with F newtons for the controller's ``push_duration`` (0.2 s), in
   the chosen direction of the base frame at the instant of the push, then watch for ``WATCH_S``,
   then wait for the robot to recover.  F survived = not fallen and recovered.
4. *Fallen*: the controller left the policy states (its own safety envelope tripped), or |roll| or
   |pitch| exceeded ``FALL_TILT``, or the base dropped below ``FALL_HEIGHT`` x h0.
   *Recovered*: |roll| and |pitch| below ``RECOVER_TILT``, |omega_xy| below ``RECOVER_RATE`` and the
   base above ``RECOVER_HEIGHT`` x h0, all continuously for ``RECOVER_HOLD_S``.  Not recovered within
   ``RECOVER_TIMEOUT_S`` after the watch window is a failure at that force.

A push is a force, not a torque: it goes through ``/gazebo/apply_body_wrench`` at the base origin, so
F newtons for 0.2 s is an impulse of 0.2 F N*s.  Training pushes are velocity kicks of up to 1 m/s,
i.e. ``body_mass`` x 1 m/s / 0.2 s = 125 N on the Aliengo and 80 N on the Go2 - the plot marks that
level.  The wrench is world-frame, the only frame the service honours, so the base-frame direction is
rotated by the base yaw when the push is sent.

Every wait runs on simulated time (/clock): a loaded machine makes a run slower, not different.

Why one Gazebo per (command, direction)
---------------------------------------
After a fall the controller collapses on joint damping and stops for good (DAMPING, then DONE); it
does not stand back up.  So every pair is its own Gazebo run, which also keeps the pairs independent
of each other.  ``sweep --docker IMAGE`` runs them in parallel containers, each on its own network so
their ROS masters and Gazebo servers cannot collide; without ``--docker`` it runs them one at a time
in the current ROS environment.

Repeatability
-------------
The protocol is deterministic - fixed forces, directions and durations - but Gazebo and the
controller are separate processes, not stepped in lockstep, and a push lands at whatever point of
the gait the clock says.  Two runs of the same pair can therefore differ by a force step or two.
``--repeats N`` runs every pair N times; the plot then shows the median and the min-max.

Every ``sweep`` call appends an entry to ``manifest.json`` next to the results: the protocol, the
md5 of the policy files, the robot_control commit (and whether its tree was dirty), the docker
image, the command line and which result files that call produced.
A sweep is resumable: pairs that already have a result file are skipped, so rerunning the same
command finishes an interrupted sweep, and deleting ``trials/`` starts over.

Output
------
::

    <out-dir>/manifest.json            how the sweep was run
    <out-dir>/trials/<pair>.json       one result per (variant, command, direction, repeat)
    <out-dir>/logs/<pair>.log          that run's controller and Gazebo output
    <out-dir>/push_sweep.png, .csv     written by ``plot``
"""

import argparse
import concurrent.futures
import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time as wall

# One thread for the numeric libraries, as run_rl_controller.sh sets: by default each library starts
# a thread per core and spins them, and several trials at once then starve their own control loops
# (a pair measured at a real-time factor of 0.89 fell to a 10 N push).  Set before numpy loads.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(HERE, "policies")

# =================================================================================================
# protocol
# =================================================================================================
FORCES = tuple(range(10, 401, 10))      # N, in the order they are applied
DIRECTIONS = {"forward": 0.0, "left": 90.0, "backward": 180.0, "right": 270.0}   # deg, base frame
COMMAND_STEPS = tuple(round(0.1 * i, 1) for i in range(-5, 6))   # m/s or rad/s, per axis
COMMAND_AXES = ("vx", "vy", "wz")
SETTLE_S = 4.0            # s of command before the first push
WATCH_S = 3.0             # s watched for a fall after each push
FALL_TILT = 1.0           # rad of |roll| or |pitch|
FALL_HEIGHT = 0.5         # x h0
RECOVER_TILT = 0.2        # rad
RECOVER_RATE = 1.0        # rad/s of |omega_xy|
RECOVER_HEIGHT = 0.8      # x h0
RECOVER_HOLD_S = 0.5      # s the recovered conditions must hold
RECOVER_TIMEOUT_S = 4.0   # s after the watch window
STARTUP_TIMEOUT_S = 240.0  # wall s to get from launch to the policy running

PROTOCOL = {k: v for k, v in globals().items()
            if k in ("FORCES", "DIRECTIONS", "COMMAND_STEPS", "SETTLE_S", "WATCH_S", "FALL_TILT",
                     "FALL_HEIGHT", "RECOVER_TILT", "RECOVER_RATE", "RECOVER_HEIGHT",
                     "RECOVER_HOLD_S", "RECOVER_TIMEOUT_S")}

CONTAINER_ENV = ("source /opt/ros/noetic/setup.bash && source /root/ros_ws/install/setup.bash && "
                 "export LOCOSIM_DIR=/root/ros_ws/src/locosim ROS_LOG_DIR=/tmp/roslog "
                 "ROS_HOSTNAME=localhost DISABLE_ROS1_EOL_WARNINGS=1 && "
                 "export PYTHONPATH=/opt/ros/noetic/lib/python3.8/site-packages:"
                 "$LOCOSIM_DIR/robot_control:$PYTHONPATH")
"""What the image's interactive bashrc sets up, for the non-interactive shell a trial runs in.  ROS
logs go to the container's /tmp so a sweep does not leave a log directory per pair in ~/.ros."""


def grid(repeats=1):
    """Every (variant, command, direction, repeat) of the sweep, interleaved by direction."""
    conditions = [("normal", (0.0, 0.0, 0.0))]
    for axis in range(3):
        for value in COMMAND_STEPS:
            if value != 0.0:
                command = [0.0, 0.0, 0.0]
                command[axis] = value
                conditions.append(("normal", tuple(command)))
    conditions.append(("safe", (0.0, 0.0, 0.0)))
    return [(variant, command, direction, repeat)
            for repeat in range(repeats)
            for variant, command in conditions
            for direction in DIRECTIONS]


def trial_name(variant, command, direction, repeat):
    vx, vy, wz = command
    return f"{variant}_vx{vx:+.1f}_vy{vy:+.1f}_wz{wz:+.1f}_{direction}_r{repeat}"


def policy_files(policy):
    contract = os.path.join(POLICY_DIR, f"{policy}_velocity.json")
    with open(contract) as f:
        variants = json.load(f).get("onnx_variants", {})
    return [contract] + [os.path.join(POLICY_DIR, name) for name in variants.values()]


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# =================================================================================================
# one trial, run inside the ROS environment
# =================================================================================================
def run_trial(args):
    import rospy
    from gazebo_msgs.srv import ApplyBodyWrench, ApplyBodyWrenchRequest

    from base_controllers.rl_quadruped.commandInterface import Event
    from base_controllers.rl_quadruped.rl_quadruped_controller import (POLICY_STATES,
                                                                       RlQuadrupedController, State)

    command = tuple(float(c) for c in args.cmd)
    direction = DIRECTIONS[args.direction]
    if args.variant == "safe" and any(command):
        sys.exit("the safe variant only runs at the zero command")
    forces = [f for f in FORCES if f <= args.max_force]

    controller = RlQuadrupedController(args.robot, input_device="none", policy_name=args.policy,
                                       skip_calibration=True, use_rviz=False, realtime=False,
                                       inference_timing=False, estimate_velocity=True)
    controller.startController(additional_args=["gui:=false"])

    result = {
        "robot": args.robot, "policy": args.policy, "variant": args.variant,
        "command": command, "direction": args.direction, "direction_deg": direction,
        "repeat": args.repeat,
        "push_duration": controller.cfg["push_duration"],
        "training_push_force": controller.cfg["body_mass"] * 1.0 / controller.cfg["push_duration"],
        "kp": controller.policy.cfg["kp"], "kd": controller.policy.cfg["kd"],
        "policy_md5": {os.path.basename(p): md5(p) for p in policy_files(args.policy)},
        "protocol": PROTOCOL, "forces_planned": forces,
        "h0": None, "pushes": [], "failure_force": None, "failure_reason": None,
        "max_survived": None, "error": None,
    }

    def sim_now():
        return rospy.Time.now().to_sec()

    def snapshot():
        pose = controller.basePoseW
        return (float(pose[2]), float(pose[3]), float(pose[4]), float(pose[5]),
                float(np.linalg.norm(controller.angVelB[:2])), controller.state)

    def fall_reason(h0):
        z, roll, pitch, _, _, state = snapshot()
        if state not in POLICY_STATES:
            return f"safety envelope ({state.value})"
        if max(abs(roll), abs(pitch)) > FALL_TILT:
            return f"tilt {np.degrees(max(abs(roll), abs(pitch))):.0f} deg"
        if h0 is not None and z < FALL_HEIGHT * h0:
            return f"height {z:.3f} m"
        return None

    def wait_for(states, timeout):
        end = wall.monotonic() + timeout
        while wall.monotonic() < end:
            if controller.state in states:
                return True
            wall.sleep(0.05)
        return False

    def recovered(stats):
        return stats["calm_since"] is not None and sim_now() - stats["calm_since"] >= RECOVER_HOLD_S

    def observe(seconds, h0, stats, until_recovered=False):
        """Watch for ``seconds`` of simulated time, or with ``until_recovered`` only until the robot
        is recovered.  Returns the fall reason, or None.  ``stats["calm_since"]`` is the start of
        the current stretch of recovered conditions, so recovery is judged on the robot's state
        at the end of the watch, not on a calm moment it has since lost."""
        start = sim_now()
        while sim_now() - start < seconds:
            reason = fall_reason(h0)
            z, roll, pitch, _, rate, _ = snapshot()
            stats["peak_tilt"] = max(stats["peak_tilt"], abs(roll), abs(pitch))
            stats["min_height"] = min(stats["min_height"], z)
            stats["peak_rate"] = max(stats["peak_rate"], rate)
            if reason:
                return reason
            calm = (max(abs(roll), abs(pitch)) < RECOVER_TILT and rate < RECOVER_RATE
                    and z > RECOVER_HEIGHT * h0)
            if not calm:
                stats["calm_since"] = None
            elif stats["calm_since"] is None:
                stats["calm_since"] = sim_now()
            if until_recovered and recovered(stats):
                return None
            wall.sleep(0.002)
        return None

    def push(apply_wrench, force):
        yaw = snapshot()[3]
        heading = yaw + np.radians(direction)
        request = ApplyBodyWrenchRequest()
        request.body_name = controller.cfg["push_body"] or f"{controller.robot_name}::base_link"
        request.reference_frame = "world"
        request.wrench.force.x = force * np.cos(heading)
        request.wrench.force.y = force * np.sin(heading)
        request.start_time = rospy.Time(0)
        request.duration = rospy.Duration(controller.cfg["push_duration"])
        response = apply_wrench(request)
        if not response.success:
            raise RuntimeError(f"push refused: {response.status_message}")

    def finish():
        if result["failure_force"] is None and result["error"] is None:
            result["max_survived"] = forces[-1] if forces else 0
        result["wall_s"] = wall.monotonic() - wall_start
        try:
            result["sim_s"] = sim_now() - sim_start if sim_start is not None else None
        except Exception:                               # rospy already shutting down
            result["sim_s"] = None
        # Below ~0.9 the machine could not keep Gazebo in real time; the waits are on sim time, but
        # the controller is a separate process, so check its loop was not starved too.
        if result["sim_s"]:
            result["real_time_factor"] = result["sim_s"] / (wall.monotonic() - wall_sim_start)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        outcome = (f"fails at {result['failure_force']} N ({result['failure_reason']})"
                   if result["failure_force"] is not None else
                   f"error: {result['error']}" if result["error"] else
                   f"survived every push up to {result['max_survived']} N")
        print(f"push_sweep: {os.path.basename(args.out)}: {outcome}", flush=True)
        controller.command.post(Event.QUIT)

    wall_start = wall.monotonic()
    sim_start = wall_sim_start = None

    def driver():
        nonlocal sim_start, wall_sim_start
        try:
            if not wait_for((State.FOLD,), STARTUP_TIMEOUT_S):
                raise RuntimeError("never reached FOLD")
            while controller._fold_moving:
                wall.sleep(0.2)
            wall.sleep(1.0)
            controller.command.post(Event.STAND_UP)
            if not wait_for((State.STANDING,), 60.0):
                raise RuntimeError("never reached STANDING")
            wall.sleep(1.5)
            controller.command.post(Event.START_RL)
            if not wait_for((State.RL,), 30.0):
                raise RuntimeError("never reached RL")
            if args.variant == "safe":
                controller.command.post(Event.SAFE_STOP)
                if not wait_for((State.SAFE_STOP,), 10.0):
                    raise RuntimeError("never reached SAFE_STOP")
            rospy.wait_for_service("/gazebo/apply_body_wrench", timeout=20.0)
            apply_wrench = rospy.ServiceProxy("/gazebo/apply_body_wrench", ApplyBodyWrench)

            sim_start, wall_sim_start = sim_now(), wall.monotonic()
            controller.command.velocity_cmd[:] = command
            heights = []
            start = sim_now()
            while sim_now() - start < SETTLE_S:
                reason = fall_reason(None)
                if reason:
                    result.update(failure_force=0, failure_reason=reason, max_survived=None)
                    return
                if sim_now() - start > SETTLE_S - 1.0:
                    heights.append(snapshot()[0])
                wall.sleep(0.002)
            h0 = result["h0"] = float(np.mean(heights))

            for force in forces:
                push_time = sim_now()
                stats = {"force": force, "peak_tilt": 0.0, "min_height": np.inf, "peak_rate": 0.0,
                         "calm_since": None}
                push(apply_wrench, force)
                reason = observe(controller.cfg["push_duration"] + WATCH_S, h0, stats)
                if reason is None and not recovered(stats):
                    reason = observe(RECOVER_TIMEOUT_S, h0, stats, until_recovered=True)
                    if reason is None and not recovered(stats):
                        reason = "no recovery"
                stats["outcome"] = "survived" if reason is None else reason
                # from the push to the start of the calm stretch that counted as recovered
                stats["recovery_time"] = (stats["calm_since"] - push_time
                                          if reason is None else None)
                del stats["calm_since"]
                result["pushes"].append(stats)
                if reason is not None:
                    result["failure_force"], result["failure_reason"] = force, reason
                    result["max_survived"] = max([p["force"] for p in result["pushes"]
                                                  if p["outcome"] == "survived"], default=0)
                    return
        except Exception as exc:                        # reported in the result, not lost
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            finish()

    threading.Thread(target=driver, daemon=True).start()
    try:
        controller.mainLoop()
    except Exception as exc:
        print("push_sweep: mainLoop raised", type(exc).__name__, exc, flush=True)
    # rospy and Gazebo threads would otherwise hold the process up after the result is written.
    os._exit(0 if os.path.exists(args.out) else 1)


# =================================================================================================
# the sweep
# =================================================================================================
def git_state(path):
    try:
        head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        branch = subprocess.run(["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        # The uncommitted changes themselves, so a dirty tree can still be reconstructed.
        diff = subprocess.run(["git", "-C", path, "diff", "HEAD"], capture_output=True, text=True,
                              check=True).stdout
        return {"commit": head, "branch": branch, "dirty": bool(dirty), "diff": diff}
    except (OSError, subprocess.CalledProcessError):
        return None


def to_container(path, lab_home):
    """Host path -> the same file inside the container, where lab_home is mounted at /root."""
    rel = os.path.relpath(os.path.realpath(path), os.path.realpath(lab_home))
    if rel.startswith(".."):
        sys.exit(f"{path} is not under {lab_home}, so the container cannot see it")
    return "/root/" + rel


def run_sweep(args):
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    lab_home = os.path.abspath(os.path.expanduser(args.lab_home))
    os.makedirs(os.path.join(out_dir, "trials"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    if not args.docker and args.jobs > 1:
        sys.exit("without --docker the trials share one ROS master and one Gazebo: use --jobs 1")

    image_id = None
    if args.docker:
        image_id = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", args.docker],
                                  capture_output=True, text=True).stdout.strip() or None
    todo = []
    for variant, command, direction, repeat in grid(args.repeats):
        name = trial_name(variant, command, direction, repeat)
        out = os.path.join(out_dir, "trials", name + ".json")
        if not os.path.exists(out):
            todo.append((name, variant, command, direction, repeat, out))

    # One entry per invocation, appended: a resumed or extended sweep may run on a later commit,
    # and each entry says which result files it produced.
    manifest_path = os.path.join(out_dir, "manifest.json")
    sweeps = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            previous = json.load(f)
        sweeps = previous.get("sweeps", [previous])     # a single-entry manifest of the old format
    sweeps.append({
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "command_line": " ".join(sys.argv),
        "robot": args.robot, "policy": args.policy, "repeats": args.repeats,
        "max_force": args.max_force, "protocol": PROTOCOL,
        "policy_md5": {os.path.basename(p): md5(p) for p in policy_files(args.policy)},
        "robot_control_git": git_state(os.path.join(HERE, "..", "..")),
        "docker_image": args.docker, "docker_image_id": image_id,
        "trials_run": [item[0] for item in todo],
    })
    with open(manifest_path, "w") as f:
        json.dump({"sweeps": sweeps}, f, indent=1)

    total = len(grid(args.repeats))
    print(f"push_sweep: {total - len(todo)} of {total} pairs already done, {len(todo)} to run "
          f"({args.jobs} at a time) -> {out_dir}", flush=True)

    def launch(item):
        name, variant, command, direction, repeat, out = item
        trial_args = ["trial", "--robot", args.robot, "--policy", args.policy, "--variant", variant,
                      "--cmd", *[f"{c:.1f}" for c in command], "--direction", direction,
                      "--repeat", str(repeat), "--max-force", str(args.max_force)]
        log_path = os.path.join(out_dir, "logs", name + ".log")
        container = None
        if args.docker:
            container = re.sub(r"[^a-zA-Z0-9_.-]", "", f"push_sweep_{name}")
            script = to_container(os.path.abspath(__file__), lab_home)
            inner = (f"{CONTAINER_ENV} && python3 -u {script} {' '.join(trial_args)} "
                     f"--out {to_container(out, lab_home)}; status=$?; "
                     f"chown {os.getuid()}:{os.getgid()} {to_container(out, lab_home)} 2>/dev/null; "
                     f"exit $status")
            # a private generated_urdf: the controller writes the URDF there at start-up
            cmd = ["docker", "run", "--rm", "--name", container, "-v", f"{lab_home}:/root",
                   "--tmpfs", "/root/ros_ws/src/locosim/robot_urdf/generated_urdf",
                   "-w", "/root", args.docker, "bash", "-c", inner]
        else:
            cmd = [sys.executable, "-u", os.path.abspath(__file__), *trial_args, "--out", out]
        with open(log_path, "w") as log:
            try:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.trial_timeout)
            except subprocess.TimeoutExpired:
                log.write(f"\npush_sweep: trial timed out after {args.trial_timeout} s\n")
                if container:
                    subprocess.run(["docker", "kill", container], capture_output=True)
        if os.path.exists(out):
            with open(out) as f:
                r = json.load(f)
            return name, (f"fails at {r['failure_force']} N ({r['failure_reason']})"
                          if r["failure_force"] is not None else
                          f"ERROR {r['error']}" if r["error"] else f"survived {r['max_survived']} N")
        return name, f"no result, see {log_path}"

    done = total - len(todo)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for name, outcome in pool.map(launch, todo):
            done += 1
            print(f"push_sweep: [{done}/{total}] {name}: {outcome}", flush=True)


# =================================================================================================
# the figure
# =================================================================================================
DIRECTION_STYLE = {   # fixed categorical order: colour and marker follow the direction everywhere
    "forward": ("#2a78d6", "o"), "left": ("#eb6834", "s"),
    "backward": ("#1baf7a", "^"), "right": ("#eda100", "D")}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_results(out_dir):
    results = []
    for path in sorted(glob.glob(os.path.join(out_dir, "trials", "*.json"))):
        with open(path) as f:
            results.append(json.load(f))
    return results


def outcome(r):
    """``(force, survived_every_push)`` for one result, or None if the run errored.  A run that
    survived every push is drawn at the largest force it took, with an open marker - which is not
    the same thing as failing at that force, so the flag travels with the number."""
    if r["error"]:
        return None
    if r["failure_force"] is not None:
        return float(r["failure_force"]), False
    return float(r["forces_planned"][-1]), True


def spread_labels(entries, min_gap):
    """Nudge end-of-line labels ``[(y, text)]`` apart so no two are closer than ``min_gap``."""
    placed = []
    for y, text in sorted(entries):
        if placed and y - placed[-1][0] < min_gap:
            y = placed[-1][0] + min_gap
        placed.append((y, text))
    return placed


def run_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    results = load_results(out_dir)
    if not results:
        sys.exit(f"no results in {out_dir}/trials")
    errors = [r for r in results if r["error"]]
    ceiling = max(max(r["forces_planned"]) for r in results)
    training = results[0]["training_push_force"]
    policy = results[0]["policy"]

    # (variant, command, direction) -> [(failure force, survived every push) per repeat]
    table = {}
    for r in results:
        value = outcome(r)
        if value is not None:
            table.setdefault((r["variant"], tuple(r["command"]), r["direction"]), []).append(value)

    def summary(key):
        """median, min, max of the plotted force, repeats, and whether every repeat survived."""
        values = table.get(key)
        if not values:
            return None
        forces = [force for force, _ in values]
        return (float(np.median(forces)), min(forces), max(forces), len(values),
                all(survived for _, survived in values))

    # the table view: every number the figure draws
    csv_path = os.path.join(out_dir, "push_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("variant,vx,vy,wz,direction,first_push_not_survived_median_N,min_N,max_N,repeats,"
                "repeats_that_survived_every_push\n")
        for key in sorted(table):
            variant, command, direction = key
            med, low, high, n, _ = summary(key)
            survived = sum(1 for _, s in table[key] if s)
            f.write(f"{variant},{command[0]:.1f},{command[1]:.1f},{command[2]:.1f},{direction},"
                    f"{med:.0f},{low:.0f},{high:.0f},{n},{survived}\n")

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), sharey=True,
                             gridspec_kw={"width_ratios": [3, 3, 3, 1.6]})
    fig.patch.set_facecolor(SURFACE)
    labels = {"vx": "forward velocity command (m/s)", "vy": "lateral velocity command (m/s)",
              "wz": "yaw-rate command (rad/s)"}
    top = ceiling * 1.08
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axhline(training, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.set_ylim(0, top)

    repeats = max(len(v) for v in table.values())
    for axis_index, (ax, axis) in enumerate(zip(axes[:3], COMMAND_AXES)):
        end_labels = []
        x_end = COMMAND_STEPS[-1]
        for direction, (colour, marker) in DIRECTION_STYLE.items():
            xs, med, low, high, open_x, open_y = [], [], [], [], [], []
            for value in COMMAND_STEPS:
                command = [0.0, 0.0, 0.0]
                command[axis_index] = value
                s = summary(("normal", tuple(command), direction))
                if s is None:
                    continue
                xs.append(value)
                med.append(s[0])
                low.append(s[0] - s[1])
                high.append(s[2] - s[0])
                if s[4]:
                    open_x.append(value)
                    open_y.append(s[0])
            if not xs:
                continue
            ax.errorbar(xs, med, yerr=[low, high] if repeats > 1 else None, color=colour,
                        linewidth=1.6, marker=marker, markersize=6, markerfacecolor=colour,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, capsize=2, elinewidth=1,
                        label=direction, zorder=3)
            ax.plot(open_x, open_y, linestyle="none", marker=marker, markersize=6,
                    markerfacecolor=SURFACE, markeredgecolor=colour, markeredgewidth=1.6, zorder=4)
            end_labels.append((med[-1], direction))
            x_end = xs[-1]
        for y, text in spread_labels(end_labels, min_gap=0.055 * top):
            ax.text(x_end + 0.05, y, text, va="center", fontsize=8.5, color=INK_2)
        ax.set_xlabel(labels[axis])
        ax.set_xticks(COMMAND_STEPS[::2])
        ax.set_xlim(COMMAND_STEPS[0] - 0.08, COMMAND_STEPS[-1] + 0.3)
        ax.set_title(f"walking policy, {axis} sweep", loc="left", fontsize=10.5, color=INK)
    axes[0].set_ylabel("first push not survived (N, held 0.2 s)")

    # zero command: walking vs safe, grouped by direction
    ax = axes[3]
    width = 0.36
    for i, (direction, (colour, marker)) in enumerate(DIRECTION_STYLE.items()):
        for j, variant in enumerate(("normal", "safe")):
            s = summary((variant, (0.0, 0.0, 0.0), direction))
            if s is None:
                continue
            x = j + (i - 1.5) * width / 2
            ax.vlines(x, 0, s[0], color=colour, linewidth=2, zorder=2)
            if repeats > 1:
                ax.vlines(x, s[1], s[2], color=colour, linewidth=1, zorder=2)
            ax.plot(x, s[0], marker=marker, markersize=6, linestyle="none", zorder=3,
                    markerfacecolor=SURFACE if s[4] else colour,
                    markeredgecolor=colour if s[4] else SURFACE, markeredgewidth=1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["walking", "safe"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_title("zero command", loc="left", fontsize=10.5, color=INK)
    axes[0].text(0.01, training - 0.01 * top, f"training push {training:.0f} N", ha="left",
                 va="top", fontsize=8.5, color=INK_2,
                 transform=blended_transform_factory(axes[0].transAxes, axes[0].transData))

    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, [f"pushed {n}" for n in names], loc="upper right", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.995, 1.0))
    fig.suptitle(f"{policy}: push recovery in Gazebo", x=0.01, ha="left", fontsize=12.5,
                 color=INK)
    spread = "median of %d runs, bar = min-max; " % repeats if repeats > 1 else "one run per point; "
    note = (f"Push at the base origin in the base-frame direction shown, increasing by "
            f"{FORCES[1] - FORCES[0]} N until the first failure ({spread}open marker = survived "
            f"every push up to {ceiling} N). Dashed line: training push strength.")
    if errors:
        note += f"  {len(errors)} run(s) errored and are not drawn."
    fig.text(0.01, 0.01, note, fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    png_path = os.path.join(out_dir, "push_sweep.png")
    fig.savefig(png_path, dpi=150, facecolor=SURFACE)
    print(f"push_sweep: wrote {png_path} and {csv_path}")


# =================================================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    trial = sub.add_parser("trial", help="one (variant, command, direction): runs Gazebo itself")
    trial.add_argument("--robot", default="aliengo")
    trial.add_argument("--policy", required=True, help="policy basename under policies/")
    trial.add_argument("--variant", choices=("normal", "safe"), default="normal")
    trial.add_argument("--cmd", nargs=3, type=float, default=(0.0, 0.0, 0.0),
                       metavar=("VX", "VY", "WZ"))
    trial.add_argument("--direction", choices=sorted(DIRECTIONS), required=True)
    trial.add_argument("--repeat", type=int, default=0, help="index, recorded in the result")
    trial.add_argument("--max-force", type=int, default=FORCES[-1])
    trial.add_argument("--out", required=True, help="result file (json)")

    sweep = sub.add_parser("sweep", help="the whole grid, resumable")
    sweep.add_argument("--robot", default="aliengo")
    sweep.add_argument("--policy", required=True, help="policy basename under policies/")
    sweep.add_argument("--out-dir", required=True)
    sweep.add_argument("--repeats", type=int, default=1)
    sweep.add_argument("--max-force", type=int, default=FORCES[-1])
    sweep.add_argument("--docker", default=None, metavar="IMAGE",
                       help="run each trial in its own container of IMAGE (from the host)")
    sweep.add_argument("--lab-home", default="~/trento_lab_home",
                       help="host directory the image mounts at /root (default: %(default)s)")
    sweep.add_argument("--jobs", type=int, default=1, help="trials at a time (needs --docker)")
    sweep.add_argument("--trial-timeout", type=float, default=1800.0, help="wall s per trial")

    plot = sub.add_parser("plot", help="figure and csv from a sweep's results")
    plot.add_argument("out_dir")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "trial":
        run_trial(args)
    elif args.mode == "sweep":
        run_sweep(args)
    else:
        run_plot(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
