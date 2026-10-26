"""Push recovery in Gazebo: how strong a push each policy survives, per velocity command and direction.

    python3 push_sweep.py sweep --robot go2 --out-dir ~/trento_lab_home/push_sweep/go2 \\
        --docker mfocchi/trento_lab_framework:locosim --jobs 8
    python3 push_sweep.py plot ~/trento_lab_home/push_sweep/go2

A push of strength k (m/s) is a force of body_mass x k / 0.2 s held for 0.2 s at the base origin,
in a base-frame direction: forward, left, backward or right.  Training kicked the base velocity by
up to ``TRAINING_KICK`` per horizontal axis: 0.5 m/s in the walking task, 1.5 m/s in the safe one.

Conditions: the walking policy (``normal``) at zero command and at vx, vy or wz = +-``--command-steps``
(one axis at a time), and the safe policy at zero command, each pushed in the four directions.

Every trial is one Gazebo run with exactly one push, so no trial depends on another:

1. Stand up, start the policy (safe: then the safe stop), hold the command ``SETTLE_S``; h0 is the
   mean base height over its last second.
2. Push at a moment of the next ``ONSET_JITTER_S`` drawn from a seed fixed by the trial's name.
3. Fallen: the controller left the policy states, |roll| or |pitch| > ``FALL_TILT``, or the base
   below ``FALL_HEIGHT`` x h0.  Survived: within ``WATCH_S`` + ``RECOVER_TIMEOUT_S`` the robot holds
   tilt < ``RECOVER_TILT``, |omega_xy| < ``RECOVER_RATE`` and height > ``RECOVER_HEIGHT`` x h0 for
   ``RECOVER_HOLD_S``.

Per condition the sweep is adaptive: ``--coarse-runs`` runs at every ``COARSE_KICKS`` level, more
levels in ``EXTEND_STEP`` up to ``MAX_KICK`` while the robot still survives the top one, then
``--fine-runs`` runs at every ``FINE_STEP`` inside the gap between the last level it always survived
and the first it never did.  ``plot`` fits P(survive) = 1 / (1 + exp((k - k50) / scale)) per
condition and reports k50, the push survived half the time, with a bootstrap 95 % interval (the gap
itself when the outcomes are separated), and k50 from two halves of the runs (every other run at
each kick level) as a repeatability check.

All waits are on simulated time; ``--docker`` runs every trial in its own container.  The sweep
resumes where it stopped and appends to ``manifest.json`` the design, the protocol, the policy md5s,
the robot_control commit and diff, and the docker image.

Output: ``manifest.json``, ``trials/<name>.json``, ``logs/<name>.log``; from ``plot``
``push_sweep.png`` (k50 per command and direction), ``push_survival.png`` (survival curves at zero
command) and ``push_sweep.csv``.
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
import zlib

# One thread for the numeric libraries (before numpy): parallel trials otherwise starve each other.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(HERE, "policies")

# =================================================================================================
# protocol
# =================================================================================================
# Largest kick per horizontal axis of push_by_setting_velocity in training (safe_rl
# velocity_env_cfg.py and safe_env_cfg.py), m/s.
TRAINING_KICK = {"normal": 0.5, "safe": 1.5}
DIRECTIONS = {"forward": 0.0, "left": 90.0, "backward": 180.0, "right": 270.0}   # deg, base frame
COMMAND_AXES = ("vx", "vy", "wz")
COARSE_KICKS = {"normal": (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                "safe": (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)}   # m/s
EXTEND_STEP = 0.5         # m/s, added above the coarse levels while the robot survives the top one
MAX_KICK = 6.0            # m/s
FINE_STEP = 0.1           # m/s, inside the survived-to-fallen gap
SETTLE_S = 4.0            # s of command before the push
ONSET_JITTER_S = 1.0      # s, the push lands at a seeded random moment of this window
WATCH_S = 3.0             # s watched for a fall after the push
FALL_TILT = 1.0           # rad of |roll| or |pitch|
FALL_HEIGHT = 0.5         # x h0
RECOVER_TILT = 0.2        # rad
RECOVER_RATE = 1.0        # rad/s of |omega_xy|
RECOVER_HEIGHT = 0.8      # x h0
RECOVER_HOLD_S = 0.5      # s the recovered conditions must hold
RECOVER_TIMEOUT_S = 4.0   # s after the watch window
STARTUP_TIMEOUT_S = 240.0  # wall s to get from launch to the policy running

PROTOCOL = {k: v for k, v in globals().items()
            if k in ("TRAINING_KICK", "DIRECTIONS", "COARSE_KICKS", "EXTEND_STEP", "MAX_KICK",
                     "FINE_STEP", "SETTLE_S", "ONSET_JITTER_S", "WATCH_S", "FALL_TILT", "FALL_HEIGHT",
                     "RECOVER_TILT", "RECOVER_RATE", "RECOVER_HEIGHT", "RECOVER_HOLD_S",
                     "RECOVER_TIMEOUT_S")}

CONTAINER_ENV = ("source /opt/ros/noetic/setup.bash && source /root/ros_ws/install/setup.bash && "
                 "export LOCOSIM_DIR=/root/ros_ws/src/locosim ROS_LOG_DIR=/tmp/roslog "
                 "ROS_HOSTNAME=localhost DISABLE_ROS1_EOL_WARNINGS=1 && "
                 "export PYTHONPATH=/opt/ros/noetic/lib/python3.8/site-packages:"
                 "$LOCOSIM_DIR/robot_control:$PYTHONPATH")
"""The image's interactive bashrc, for the non-interactive shell of a trial."""


def trial_name(variant, command, direction, kick, run):
    """Also the seed of the push moment, so a rerun pushes at the same instants."""
    name = f"{variant}_{direction}_k{kick:.2f}_r{run}"
    if any(command):
        name += "_vx{:+.2f}_vy{:+.2f}_wz{:+.2f}".format(*command)
    return name


def conditions(variants, command_steps):
    """(variant, command): the walking policy at zero command and on one-axis commands, the safe
    policy at zero command."""
    out = []
    if "normal" in variants:
        out.append(("normal", (0.0, 0.0, 0.0)))
        for axis in range(3):
            for step in command_steps:
                for sign in (-1.0, 1.0):
                    command = [0.0, 0.0, 0.0]
                    command[axis] = sign * step
                    out.append(("normal", tuple(command)))
    if "safe" in variants:
        out.append(("safe", (0.0, 0.0, 0.0)))
    return out


def robot_config(robot):
    sys.path.insert(0, os.path.join(HERE, "..", ".."))
    from base_controllers.rl_quadruped.rl_controller_config import get_config
    return get_config(robot)


def policy_files(robot):
    contract = os.path.join(POLICY_DIR, f"{robot}_velocity.json")
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

    controller = RlQuadrupedController(args.robot, input_device="none", skip_calibration=True,
                                       use_rviz=False, realtime=False)
    controller.startController(additional_args=["gui:=false"])

    result = {
        "robot": args.robot, "variant": args.variant,
        "command": command, "direction": args.direction, "direction_deg": direction,
        "repeat": args.repeat, "seed": args.seed, "onset_delay": None,
        "push_duration": controller.cfg["push_duration"], "body_mass": controller.cfg["body_mass"],
        "kp": controller.policy.cfg["kp"], "kd": controller.policy.cfg["kd"],
        "policy_md5": {os.path.basename(p): md5(p) for p in policy_files(args.robot)},
        "protocol": PROTOCOL, "forces_planned": [args.force],
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
        request.body_name = f"{controller.robot_name}::base_link"
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
            result["max_survived"] = args.force
        result["wall_s"] = wall.monotonic() - wall_start
        try:
            result["sim_s"] = sim_now() - sim_start if sim_start is not None else None
        except Exception:                               # rospy already shutting down
            result["sim_s"] = None
        # below ~0.9 Gazebo did not keep real time, and the controller may have been starved too
        if result["sim_s"]:
            result["real_time_factor"] = result["sim_s"] / (wall.monotonic() - wall_sim_start)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        outcome = (f"fails at {result['failure_force']} N ({result['failure_reason']})"
                   if result["failure_force"] is not None else
                   f"error: {result['error']}" if result["error"] else
                   f"survived {args.force} N")
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
                    result.update(failure_force=0, failure_reason=reason)
                    return
                if sim_now() - start > SETTLE_S - 1.0:
                    heights.append(snapshot()[0])
                wall.sleep(0.002)
            h0 = result["h0"] = float(np.mean(heights))

            stats = {"force": args.force, "peak_tilt": 0.0, "min_height": np.inf,
                     "peak_rate": 0.0, "calm_since": None}
            delay = float(np.random.default_rng(args.seed).uniform(0.0, ONSET_JITTER_S))
            result["onset_delay"] = delay
            reason = observe(delay, h0, dict(stats))
            if reason:
                result.update(failure_force=0, failure_reason=reason)
                return
            push_time = sim_now()
            push(apply_wrench, args.force)
            reason = observe(controller.cfg["push_duration"] + WATCH_S, h0, stats)
            if reason is None and not recovered(stats):
                reason = observe(RECOVER_TIMEOUT_S, h0, stats, until_recovered=True)
                if reason is None and not recovered(stats):
                    reason = "no recovery"
            stats["outcome"] = "survived" if reason is None else reason
            # from the push to the start of the calm stretch that counted as recovered
            stats["recovery_time"] = stats["calm_since"] - push_time if reason is None else None
            del stats["calm_since"]
            result["pushes"].append(stats)
            if reason is not None:
                result["failure_force"], result["failure_reason"] = args.force, reason
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


def load_results(out_dir):
    results = []
    for path in sorted(glob.glob(os.path.join(out_dir, "trials", "*.json"))):
        with open(path) as f:
            results.append(json.load(f))
    return results


def kick_of(r):
    return round(r["forces_planned"][0] * r["push_duration"] / r["body_mass"], 2)


def outcomes(out_dir):
    """(variant, command, direction) -> {kick: [survived per run]}, errored runs left out."""
    table = {}
    for r in load_results(out_dir):
        if r["error"]:
            continue
        key = (r["variant"], tuple(r["command"]), r["direction"])
        table.setdefault(key, {}).setdefault(kick_of(r), []).append(r["failure_force"] is None)
    return table


def gap(levels):
    """(highest kick survived by every run at it and below, lowest kick fallen by every run at it
    and above); None for the upper end while some run survives the top level."""
    kicks = sorted(levels)
    low = max([k for k in kicks if all(all(levels[j]) for j in kicks if j <= k)], default=0.0)
    high = min([k for k in kicks if not any(any(levels[j]) for j in kicks if j >= k)], default=None)
    return low, high


def run_sweep(args):
    cfg = robot_config(args.robot)
    mass, duration = cfg["body_mass"], cfg["push_duration"]
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    lab_home = os.path.abspath(os.path.expanduser(args.lab_home))
    os.makedirs(os.path.join(out_dir, "trials"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    if not args.docker and args.jobs > 1:
        sys.exit("without --docker the trials share one ROS master and one Gazebo: use --jobs 1")
    cases = [(variant, command, direction) for variant, command in
             conditions(args.variants, args.command_steps) for direction in DIRECTIONS]

    image_id = None
    if args.docker:
        image_id = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", args.docker],
                                  capture_output=True, text=True).stdout.strip() or None
    manifest_path = os.path.join(out_dir, "manifest.json")
    sweeps = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            sweeps = json.load(f).get("sweeps", [])
    sweeps.append({
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "command_line": " ".join(sys.argv), "robot": args.robot, "variants": args.variants,
        "command_steps": args.command_steps, "coarse_runs": args.coarse_runs,
        "fine_runs": args.fine_runs, "protocol": PROTOCOL,
        "policy_md5": {os.path.basename(p): md5(p) for p in policy_files(args.robot)},
        "robot_control_git": git_state(os.path.join(HERE, "..", "..")),
        "docker_image": args.docker, "docker_image_id": image_id,
    })
    with open(manifest_path, "w") as f:
        json.dump({"sweeps": sweeps}, f, indent=1)

    def trial(variant, command, direction, kick, run):
        name = trial_name(variant, command, direction, kick, run)
        force = round(mass * kick / duration, 1)
        seed = zlib.crc32(f"{args.robot}/{name}".encode())
        trial_args = ["trial", "--robot", args.robot, "--variant", variant,
                      "--cmd", *[f"{c:.2f}" for c in command], "--direction", direction,
                      "--force", str(force), "--seed", str(seed), "--repeat", str(run)]
        return name, trial_args, os.path.join(out_dir, "trials", name + ".json")

    def launch(item):
        name, trial_args, out = item
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
        if not os.path.exists(out):
            return name, f"no result, see {log_path}"
        with open(out) as f:
            r = json.load(f)
        return name, (f"ERROR {r['error']}" if r["error"] else
                      f"fell ({r['failure_reason']})" if r["failure_force"] is not None else
                      "survived")

    def run_stage(stage, wanted):
        """``wanted`` = [(variant, command, direction, kick, runs)]: top every level up to ``runs``."""
        items = []
        for variant, command, direction, kick, runs in wanted:
            for run in range(runs):
                item = trial(variant, command, direction, kick, run)
                if not os.path.exists(item[2]):
                    items.append(item)
        print(f"push_sweep: {stage}: {len(items)} trials to run ({args.jobs} at a time)", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for done, (name, result) in enumerate(pool.map(launch, items), 1):
                print(f"push_sweep: {stage} [{done}/{len(items)}] {name}: {result}", flush=True)

    run_stage("coarse", [(v, c, d, k, args.coarse_runs) for v, c, d in cases for k in COARSE_KICKS[v]])
    while True:
        table = outcomes(out_dir)
        extend = []
        for v, c, d in cases:
            levels = table.get((v, c, d), {})
            top = max(levels, default=0.0)
            if levels and any(levels[top]) and top + EXTEND_STEP <= MAX_KICK + 1e-9:
                extend.append((v, c, d, round(top + EXTEND_STEP, 2), args.coarse_runs))
        if not extend:
            break
        run_stage("extend", extend)
    fine = []
    for v, c, d in cases:
        low, high = gap(table.get((v, c, d), {}))
        if high is not None:
            steps = int(round((high - low) / FINE_STEP))
            fine += [(v, c, d, round(low + FINE_STEP * i, 2), args.fine_runs) for i in range(1, steps)]
    run_stage("fine", fine)


# =================================================================================================
# analysis and figures
# =================================================================================================
DIRECTION_STYLE = {   # fixed categorical order: colour and marker follow the direction everywhere
    "forward": ("#2a78d6", "o"), "left": ("#eb6834", "s"),
    "backward": ("#1baf7a", "^"), "right": ("#eda100", "D")}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def fit_survival(kicks, survived, rng=None, boot=1000):
    """Logistic P(survive) = 1 / (1 + exp((k - k50) / scale)) by maximum likelihood, with a
    parametric bootstrap 95 % interval on k50: outcomes simulated from the fit at the same kicks and
    runs, then refitted.  The scale is bounded below, so a sharp threshold gives k50 in the gap
    between the last survived and the first fallen kick, and that gap is then the interval."""
    from scipy.optimize import minimize
    kicks, survived = np.asarray(kicks, float), np.asarray(survived, float)

    def fit(k, y):
        # logit P(survive) = a + b k, convex in (a, b); scale = -1/b >= 0.02 m/s
        def nll(p):
            z = p[0] + p[1] * k
            return np.sum(np.logaddexp(0.0, z) - y * z)

        def grad(p):
            r = 1.0 / (1.0 + np.exp(-(p[0] + p[1] * k))) - y
            return np.array([r.sum(), (r * k).sum()])
        a, b = minimize(nll, [2.0, -1.0], jac=grad, method="L-BFGS-B",
                        bounds=[(None, None), (-50.0, -1e-3)]).x
        return -a / b, -1.0 / b

    k50, scale = fit(kicks, survived)
    if not boot:
        return float(k50), float(scale), float("nan"), float("nan")
    if survived.any() and not survived.all() and kicks[survived > 0.5].max() < kicks[survived < 0.5].min():
        # separated: k50 is anywhere in the gap, and the resolution is the kick spacing
        return float(k50), float(scale), float(kicks[survived > 0.5].max()), float(kicks[survived < 0.5].min())
    rng = rng or np.random.default_rng(0)
    p_survive = 1.0 / (1.0 + np.exp(np.clip((kicks - k50) / scale, -50, 50)))
    samples = [fit(kicks, (rng.random(len(kicks)) < p_survive).astype(float))[0] for _ in range(boot)]
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(k50), float(scale), float(low), float(high)


def analyse(results):
    """One row per (variant, command, direction) with the fit, or the bound when it never fell."""
    data = {}
    for r in results:
        key = (r["variant"], tuple(r["command"]), r["direction"])
        data.setdefault(key, []).append((kick_of(r), r["failure_force"] is None, r["repeat"]))
    rows = {}
    for key, points in sorted(data.items()):
        kicks, survived, runs = (np.array(x) for x in zip(*points))
        row = dict(pushes=len(points), points=points, censored=None)
        if survived.all():
            row.update(k50=kicks.max(), low=kicks.max(), high=np.inf, scale=np.nan, censored="above")
        elif not survived.any():
            row.update(k50=kicks.min(), low=0.0, high=kicks.min(), scale=np.nan, censored="below")
        else:
            row["k50"], row["scale"], row["low"], row["high"] = fit_survival(kicks, survived)
        # Repeatability: two halves taking every other run at each kick level; a level with a
        # single run (the coarse ones, far from the transition) is in both.
        half = np.zeros(len(points), dtype=int)
        for level in set(kicks):
            index = np.flatnonzero(kicks == level)
            half[index[np.argsort(runs[index])]] = [i % 2 if len(index) > 1 else -1
                                                   for i in range(len(index))]
        halves = []
        for h in (0, 1):
            mask = (half == h) | (half == -1)
            k50_half, scale_half = fit_survival(kicks[mask], survived[mask], boot=0)[:2]
            informative = 0 < survived[mask].mean() < 1 and scale_half < 2.0
            halves.append(k50_half if informative else np.nan)
        row["halves"] = halves
        rows[key] = row
    return rows


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def run_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    results = load_results(out_dir)
    errors = [r for r in results if r["error"]]
    results = [r for r in results if not r["error"]]
    if not results:
        sys.exit(f"no results in {out_dir}/trials")
    robot, mass, duration = results[0]["robot"], results[0]["body_mass"], results[0]["push_duration"]
    rtf = [r["real_time_factor"] for r in results if r.get("real_time_factor")]
    rows = analyse(results)

    csv_path = os.path.join(out_dir, "push_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("variant,vx,vy,wz,direction,pushes,kick50_mps,ci95_low_mps,ci95_high_mps,force50_N,"
                "scale_mps,censored,kick50_half_a_mps,kick50_half_b_mps\n")
        for (variant, command, direction), r in rows.items():
            f.write(f"{variant},{command[0]:.2f},{command[1]:.2f},{command[2]:.2f},{direction},"
                    f"{r['pushes']},{r['k50']:.3f},{r['low']:.3f},{r['high']:.3f},"
                    f"{r['k50'] * mass / duration:.0f},{r['scale']:.3f},{r['censored'] or ''},"
                    f"{r['halves'][0]:.3f},{r['halves'][1]:.3f}\n")

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    finite = [v for r in rows.values() for v in (r["k50"], r["low"], r["high"]) if np.isfinite(v)]
    top = 1.1 * max(finite + [TRAINING_KICK["safe"]])
    dashed = dict(color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    force_of = (lambda k: k * mass / duration, lambda f: f * duration / mass)

    def mark(ax, x, row, colour, marker, label=None):
        """k50 with its interval; an open marker is a bound (never fell, or always fell)."""
        err = [[row["k50"] - row["low"]], [min(row["high"], top) - row["k50"]]]
        ax.errorbar([x], [row["k50"]], yerr=err, color=colour, marker=marker, markersize=6,
                    markerfacecolor=SURFACE if row["censored"] else colour,
                    markeredgecolor=colour if row["censored"] else SURFACE, markeredgewidth=1.2,
                    capsize=2, elinewidth=1, linestyle="none", zorder=3, label=label)

    # --- k50 per command and direction ----------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), sharey=True,
                             gridspec_kw={"width_ratios": [3, 3, 3, 1.8]})
    fig.patch.set_facecolor(SURFACE)
    labels = {"vx": "forward velocity command (m/s)", "vy": "lateral velocity command (m/s)",
              "wz": "yaw-rate command (rad/s)"}
    for axis_index, (ax, axis) in enumerate(zip(axes[:3], COMMAND_AXES)):
        style(ax)
        ax.axhline(TRAINING_KICK["normal"], **dashed)
        values = sorted({c[axis_index] for (v, c, _) in rows if v == "normal"
                         and all(x == 0 for i, x in enumerate(c) if i != axis_index)})
        for direction, (colour, marker) in DIRECTION_STYLE.items():
            xs, ys = [], []
            for value in values:
                command = [0.0, 0.0, 0.0]
                command[axis_index] = value
                row = rows.get(("normal", tuple(command), direction))
                if row is None:
                    continue
                offset = (list(DIRECTION_STYLE).index(direction) - 1.5) * 0.012
                mark(ax, value + offset, row, colour, marker,
                     label=direction if not xs and axis_index == 0 else None)
                xs.append(value + offset)
                ys.append(row["k50"])
            ax.plot(xs, ys, color=colour, linewidth=1.3, zorder=2)
        span = (max(values) - min(values)) or 1.0
        ax.set_xticks(values)
        ax.set_xlim(min(values) - 0.1 * span, max(values) + 0.1 * span)
        ax.set_xlabel(labels[axis])
        ax.set_title(f"walking policy, {axis} commands", loc="left", fontsize=10.5, color=INK)
    axes[0].set_ylabel("push survived half the time (m/s kick)")
    axes[0].text(0.01, TRAINING_KICK["normal"] + 0.01 * top,
                 f"walking training kick {TRAINING_KICK['normal']} m/s", fontsize=8.5, color=INK_2,
                 transform=axes[0].get_yaxis_transform())

    ax = axes[3]
    style(ax)
    for j, variant in enumerate(("normal", "safe")):
        ax.hlines(TRAINING_KICK[variant], j - 0.45, j + 0.45, **dashed)
        for i, (direction, (colour, marker)) in enumerate(DIRECTION_STYLE.items()):
            row = rows.get((variant, (0.0, 0.0, 0.0), direction))
            if row is not None:
                mark(ax, j + (i - 1.5) * 0.18, row, colour, marker)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["walking", "safe"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, top)
    ax.set_title("zero command", loc="left", fontsize=10.5, color=INK)
    ax.secondary_yaxis("right", functions=force_of).set_ylabel(
        f"as a force held {duration:.1f} s (N)", color=INK_2)

    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, [f"pushed {n}" for n in names], loc="upper right", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.995, 1.0))
    fig.suptitle(f"{robot}: push recovery in Gazebo", x=0.01, ha="left", fontsize=12.5, color=INK)
    note = (f"Each point: the kick (force x {duration:.1f} s / {mass:.1f} kg) survived half the time, "
            f"from single pushes on a fresh start, with its 95 % interval; open marker = bound "
            f"(never fell / always fell). {len(results)} runs, real-time factor >= "
            f"{min(rtf):.2f}.\nDashed lines: largest training kick per axis, "
            f"{TRAINING_KICK['normal']} m/s walking and {TRAINING_KICK['safe']} m/s safe.")
    if errors:
        note += f"  {len(errors)} run(s) errored and are left out."
    fig.text(0.01, 0.01, note, fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    sweep_png = os.path.join(out_dir, "push_sweep.png")
    fig.savefig(sweep_png, dpi=150, facecolor=SURFACE)

    # --- survival curves at zero command --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    fig.patch.set_facecolor(SURFACE)
    zero = {k: r for k, r in rows.items() if not any(k[1])}
    for ax, variant, title in zip(axes, ("normal", "safe"), ("walking policy", "safe policy")):
        kmax = max([kick for k, r in zero.items() if k[0] == variant for kick, _, _ in r["points"]],
                   default=1.0)
        grid_k = np.linspace(0, kmax * 1.05, 400)
        style(ax)
        ax.axvline(TRAINING_KICK[variant], **dashed)
        ax.text(TRAINING_KICK[variant], 1.04, f" training kick {TRAINING_KICK[variant]} m/s",
                fontsize=8.5, color=INK_2, va="bottom")
        for direction, (colour, marker) in DIRECTION_STYLE.items():
            row = zero.get((variant, (0.0, 0.0, 0.0), direction))
            if row is None:
                continue
            levels = sorted({k for k, _, _ in row["points"]})
            rate = [np.mean([s for k, s, _ in row["points"] if k == level]) for level in levels]
            count = [sum(1 for k, _, _ in row["points"] if k == level) for level in levels]
            ax.scatter(levels, rate, s=[12 + 8 * c for c in count], color=colour, marker=marker,
                       edgecolor=SURFACE, linewidth=0.8, zorder=3)
            if not row["censored"]:
                curve = 1.0 / (1.0 + np.exp(np.clip((grid_k - row["k50"]) / row["scale"], -50, 50)))
                ax.plot(grid_k, curve, color=colour, linewidth=1.6, zorder=2,
                        label=f"pushed {direction}: {row['k50']:.2f} m/s "
                              f"[{row['low']:.2f}, {row['high']:.2f}]")
        ax.set_xlim(0, kmax * 1.05)
        ax.set_ylim(-0.03, 1.12)
        ax.set_xlabel("push as a velocity kick (m/s)")
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.07), frameon=False, fontsize=8.5,
                  title="50 % survival [95 % CI]", title_fontsize=8.5)
        ax.secondary_xaxis("top", functions=force_of).set_xlabel(
            f"force held {duration:.1f} s (N)", color=INK_2)
    axes[0].set_ylabel("survived (fraction of runs)")
    fig.suptitle(f"{robot}: survival of a single push at zero command, Gazebo", x=0.01, ha="left",
                 fontsize=12.5, color=INK)
    fig.text(0.01, 0.01, "Every point is one Gazebo run with one push; marker size = runs at that "
             "level; curves: logistic fits.", fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    survival_png = os.path.join(out_dir, "push_survival.png")
    fig.savefig(survival_png, dpi=150, facecolor=SURFACE)
    print(f"push_sweep: wrote {sweep_png}, {survival_png} and {csv_path}")


# =================================================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    trial = sub.add_parser("trial", help="one push in one Gazebo run")
    trial.add_argument("--robot", default="aliengo")
    trial.add_argument("--variant", choices=("normal", "safe"), default="normal")
    trial.add_argument("--cmd", nargs=3, type=float, default=(0.0, 0.0, 0.0),
                       metavar=("VX", "VY", "WZ"))
    trial.add_argument("--direction", choices=sorted(DIRECTIONS), required=True)
    trial.add_argument("--force", type=float, required=True, help="N, held for push_duration")
    trial.add_argument("--seed", type=int, default=0, help="seeds the moment of the push")
    trial.add_argument("--repeat", type=int, default=0, help="run index, recorded in the result")
    trial.add_argument("--out", required=True, help="result file (json)")

    sweep = sub.add_parser("sweep", help="the adaptive sweep, resumable")
    sweep.add_argument("--robot", default="aliengo")
    sweep.add_argument("--out-dir", required=True)
    sweep.add_argument("--variants", nargs="+", choices=("normal", "safe"), default=["normal", "safe"])
    sweep.add_argument("--command-steps", nargs="*", type=float, default=[0.25, 0.5],
                       help="|command| per axis for the walking policy, both signs (none: zero "
                            "command only)")
    sweep.add_argument("--coarse-runs", type=int, default=1)
    sweep.add_argument("--fine-runs", type=int, default=3)
    sweep.add_argument("--docker", default=None, metavar="IMAGE",
                       help="run each trial in its own container of IMAGE (from the host)")
    sweep.add_argument("--lab-home", default="~/trento_lab_home",
                       help="host directory the image mounts at /root (default: %(default)s)")
    sweep.add_argument("--jobs", type=int, default=1, help="trials at a time (needs --docker)")
    sweep.add_argument("--trial-timeout", type=float, default=900.0, help="wall s per trial")

    plot = sub.add_parser("plot", help="fits, figures and csv from a sweep's results")
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
