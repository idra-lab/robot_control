"""Deployment wrapper for the single-network velocity policy trained in Isaac Lab.

The policy comes from ``safe_rl.tasks.manager_based.velocity`` (gym id ``Rl-Velocity-Aliengo``).
It is a *single* network: the base linear velocity that older locosim policies received from an
external state estimator is regressed by a head inside the actor, so nothing outside this module
has to estimate the base state.  Everything the network needs is proprioceptive.

Observation contract (228 floats, exactly the declaration order of ``ObservationsCfg.PolicyCfg``)::

    [  0:  3)  velocity_commands       (vx, vy, wz)          current step only
    [  3: 18)  imu_lin_acc             3 x 5 history
    [ 18: 33)  imu_ang_vel             3 x 5 history
    [ 33: 48)  imu_projected_gravity   3 x 5 history
    [ 48:108)  joint_pos_rel          12 x 5 history
    [108:168)  joint_vel_rel          12 x 5 history
    [168:228)  actions                12 x 5 history

Every history block is stored oldest-frame-first / newest-frame-last, which is what
:class:`isaaclab.utils.buffers.CircularBuffer` returns (see its ``buffer`` property).  On the first
push that buffer replicates the sample across all slots, so :meth:`VelocityPolicy.reset` fills each
proprioceptive block with the current measurement and leaves the action block at zero.

The observation normalizer is baked into the ONNX graph, so raw (un-normalized) observations are fed.

The base linear velocity the actor regresses internally is not a graph output, so it cannot be read
back from a normal ``session.run``.  :class:`VelocityPolicy` therefore also evaluates the estimation
head itself, from weights read straight out of the same ONNX file - see :meth:`_loadEstimator`.  It
is the identical computation, not an approximation: reproducing the full actor this way agrees with
onnxruntime to 6e-05 on random observations.

The action is a joint position offset from the training default posture::

    q_des = q_default + action_scale * action

which is what ``JointPositionActionCfg(scale=0.2, use_default_offset=True)`` does in the task.  The
configured action clip of +/-20 rad applies to the *processed* target and is far outside anything the
network produces, so it is not replicated here; the raw action fed back into the observation history
is unclipped, matching ``mdp.last_action``.

Joint ordering needs no permutation: the task declares its joints in ``locosim_joint_names``, which is
element-for-element ``conf.robot_params['aliengo']['joint_names']``.
"""

import json
import os
from time import perf_counter

import numpy as np
import onnxruntime as ort


def _read_varint(buf, index):
    result = 0
    shift = 0
    while True:
        byte = buf[index]
        index += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return result, index


def _protobuf_fields(buf):
    """Decode a protobuf message into ``(field_number, wire_type, value)`` triples.

    Just enough of the wire format to walk an ONNX file for its initializers.  The alternative would
    be a dependency on the ``onnx`` package, which is not installed in the locosim container and is
    not worth adding to read a handful of weight tensors.
    """
    index = 0
    fields = []
    while index < len(buf):
        key, index = _read_varint(buf, index)
        field_number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, index = _read_varint(buf, index)
        elif wire_type == 2:
            length, index = _read_varint(buf, index)
            value = buf[index:index + length]
            index += length
        elif wire_type == 5:
            value = buf[index:index + 4]
            index += 4
        elif wire_type == 1:
            value = buf[index:index + 8]
            index += 8
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")
        fields.append((field_number, wire_type, value))
    return fields


def load_onnx_initializers(path):
    """Return every float32 initializer of an ONNX model as ``{name: ndarray}``."""
    with open(path, "rb") as handle:
        data = handle.read()
    graphs = [value for number, _, value in _protobuf_fields(data) if number == 7]
    if not graphs:
        raise ValueError(f"{path} has no graph")
    tensors = {}
    for number, _, value in _protobuf_fields(graphs[0]):
        if number != 5:                       # GraphProto.initializer
            continue
        dims, name, raw, dtype = [], None, None, None
        for field, _, item in _protobuf_fields(value):
            if field == 1:
                dims.append(item)             # TensorProto.dims
            elif field == 2:
                dtype = item                  # TensorProto.data_type
            elif field == 8:
                name = item.decode()          # TensorProto.name
            elif field == 9:
                raw = item                    # TensorProto.raw_data
        if name is None or raw is None:
            continue
        if dtype != 1:                        # 1 == FLOAT
            continue
        array = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        tensors[name] = array.reshape(dims) if dims else array
    return tensors


class _PolicyNetwork:
    """One exported actor: its onnxruntime session, its I/O binding and its estimation head.

    A :class:`VelocityPolicy` owns one of these per variant and hands every one of them the *same*
    observation and action buffers.  Selecting a variant is then just a pointer move: the history
    blocks, the action block and the decimation counter carry straight over, which is what you want
    when the switch happens precisely because the robot is already in trouble.
    """

    def __init__(self, name, model_path, obs, out, estimate_out, num_threads=1, verbose=True,
                 enable_estimator=True):
        self.name = name
        self.model_path = model_path
        self._obs = obs
        self._obs_flat = obs[0]
        self._out = out
        # Shared with the owning VelocityPolicy and with the other variants: whichever network is
        # active writes the current estimate here, so telemetry never has to know which one ran.
        self.estimated_base_lin_vel = estimate_out

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name
        self.obs_size = int(self.session.get_inputs()[0].shape[-1])
        self.action_size = int(self.session.get_outputs()[0].shape[-1])

        # I/O binding pins the input tensor to the shared observation buffer and the output to the
        # shared action buffer, so a step costs one graph execution and no buffer churn.  It is a
        # pure optimization: if the runtime will not bind, fall back to the ordinary run() path.
        self._binding = None
        try:
            binding = self.session.io_binding()
            self._obs_ort = ort.OrtValue.ortvalue_from_numpy(self._obs)
            self._out_ort = ort.OrtValue.ortvalue_from_numpy(self._out)
            binding.bind_ortvalue_input(self._input_name, self._obs_ort)
            binding.bind_ortvalue_output(self._output_name, self._out_ort)
            # Only adopt the fast path if the input OrtValue really aliases the buffer: a runtime
            # that copies instead of wrapping would silently feed a stale observation forever.
            self._obs_flat[0] = 1.234
            aliases = float(self._obs_ort.numpy()[0, 0]) == np.float32(1.234)
            self._obs_flat[0] = 0.0
            if aliases:
                self._binding = binding
        except Exception as exc:  # pragma: no cover - depends on the onnxruntime build
            if verbose:
                print(f"VelocityPolicy[{name}]: I/O binding unavailable ({exc}), using run()")

        if enable_estimator:
            self._loadEstimator(verbose=verbose)
        else:
            # Turned off deliberately: the head is never read out of the file and never evaluated,
            # so estimated_base_lin_vel stays zero and an inference tick costs one graph execution
            # and nothing else.  has_estimator is the flag every caller already tests first.
            self.has_estimator = False

    def infer(self):
        if self._binding is not None:
            self.session.run_with_iobinding(self._binding)
            return self._out[0]
        out = self.session.run([self._output_name], {self._input_name: self._obs})[0]
        self._out[:] = out
        return self._out[0]

    # ------------------------------------------------------------------------------------------
    # state-estimation head
    # ------------------------------------------------------------------------------------------
    def _loadEstimator(self, verbose=True):
        """Prepare an independent evaluation of the actor's state-estimation head.

        The head regresses the base linear velocity from the same observation the policy consumes,
        but its output is consumed internally by the policy backbone and never leaves the graph, so
        ``session.run`` cannot return it.  Rather than perform surgery on the policy file - the one
        artefact the robot depends on - the weights are read out of it and the head is evaluated
        here.  Verified against onnxruntime by reproducing the *whole* actor this way: agreement to
        6e-05 on random observations, so the layer conventions and the activation are right.

        The normalizer is folded into the first layer, since
        ``W((x - mean) / std) + b == (W / std) x + (b - (W / std) mean)``.  That removes a
        normalisation pass and costs nothing.  Each variant carries its own normalizer, which is
        the reason the head has to be prepared per network rather than once.
        """
        self.has_estimator = False
        try:
            tensors = load_onnx_initializers(self.model_path)
        except (OSError, ValueError) as exc:
            print(f"VelocityPolicy[{self.name}]: cannot read the estimation head ({exc}); "
                  f"estimated_base_lin_vel stays zero")
            return

        needed = ["normalizer._mean", "actor.estimator.0.weight", "actor.estimator.0.bias",
                  "actor.estimator.2.weight", "actor.estimator.2.bias",
                  "actor.estimator.4.weight", "actor.estimator.4.bias"]
        if any(name not in tensors for name in needed):
            if verbose:
                print(f"VelocityPolicy[{self.name}]: this policy has no state-estimation head; "
                      f"estimated_base_lin_vel stays zero")
            return

        # The divisor is an anonymous constant in the exported graph, so it is identified by shape
        # rather than by name.
        mean = tensors["normalizer._mean"].reshape(-1)
        std = None
        for name, tensor in tensors.items():
            if name == "normalizer._mean" or tensor.size != mean.size:
                continue
            if tensor.reshape(-1).shape == mean.shape and "estimator" not in name \
                    and "backbone" not in name:
                std = tensor.reshape(-1)
                break
        if std is None:
            print(f"VelocityPolicy[{self.name}]: could not identify the normalizer scale; "
                  f"estimated_base_lin_vel stays zero")
            return

        w0 = (tensors["actor.estimator.0.weight"] / std).astype(np.float32)
        self._est_w = [np.ascontiguousarray(w0),
                       np.ascontiguousarray(tensors["actor.estimator.2.weight"].astype(np.float32)),
                       np.ascontiguousarray(tensors["actor.estimator.4.weight"].astype(np.float32))]
        self._est_b = [np.ascontiguousarray(
                           (tensors["actor.estimator.0.bias"] - w0 @ mean).astype(np.float32)),
                       np.ascontiguousarray(tensors["actor.estimator.2.bias"].astype(np.float32)),
                       np.ascontiguousarray(tensors["actor.estimator.4.bias"].astype(np.float32))]
        # Preallocated layer buffers, so an evaluation allocates nothing.  The second set holds the
        # negative branch of the activation; see runEstimator.
        self._est_h = [np.zeros(w.shape[0], dtype=np.float32) for w in self._est_w]
        self._est_neg = [np.zeros(w.shape[0], dtype=np.float32) for w in self._est_w]
        self.has_estimator = True
        if verbose:
            print(f"VelocityPolicy[{self.name}]: state-estimation head available "
                  f"({' -> '.join(str(w.shape[1]) for w in self._est_w)} -> "
                  f"{self._est_w[-1].shape[0]}), base linear velocity exposed as "
                  f"estimated_base_lin_vel")

    def runEstimator(self):
        """Evaluate the estimation head on the current observation, allocating nothing.

        The ELU is written as ``max(x, 0) + expm1(min(x, 0))``, which is exact - the positive branch
        contributes ``expm1(0) = 0`` and the negative branch contributes ``max = 0`` - and runs
        entirely through preallocated buffers.  The obvious boolean-mask form
        (``out[out < 0] = np.expm1(out[out < 0])``) allocates three temporaries per layer on every
        call, which showed up as a millisecond-scale tail once the garbage collector was tuned for
        the control loop.
        """
        activation = self._obs_flat
        last = len(self._est_w) - 1
        for index in range(len(self._est_w)):
            out = self._est_h[index]
            np.dot(self._est_w[index], activation, out=out)
            out += self._est_b[index]
            if index < last:
                negative = self._est_neg[index]
                np.minimum(out, 0.0, out=negative)
                np.expm1(negative, out=negative)
                np.maximum(out, 0.0, out=out)
                out += negative
            activation = out
        self.estimated_base_lin_vel[:] = self._est_h[-1]


class VelocityPolicy:
    """Runs the velocity policy at a fixed rate on top of a faster control loop.

    Args:
        robot_name: used to locate ``policies/<robot_name>_velocity.{onnx,json}``.
        dt: period of the *control* loop that calls :meth:`step`, in seconds.
        policy_rate: rate the network was trained at, in Hz.  ``dt`` times the resulting decimation
            must equal ``1 / policy_rate``, otherwise the joint velocities and the action history are
            sampled on a different time base than in training.
        num_threads: onnxruntime thread count.  One is both the fastest and the most predictable
            choice for a network this small; extra threads only add scheduling jitter.
        policy_dir: overrides the directory the policy is loaded from.
        enable_estimator: evaluate the actor's state-estimation head on every inference tick and
            expose the result as :attr:`estimated_base_lin_vel`.  The head is *diagnostic* here:
            the policy backbone already consumes its output inside the ONNX graph, and this second
            evaluation exists only so the estimate can be seen from outside.  ``False`` skips
            reading it out of the policy file, leaves :attr:`has_estimator` ``False`` and the
            estimate at zero, and drops a three-layer forward pass from every tick.
        measure_timing: time each inference tick and maintain :attr:`last_infer_ms`,
            :attr:`infer_ms_max`, :attr:`infer_ms_mean` and :attr:`infer_over_budget`.  ``False``
            drops the two ``perf_counter`` calls per tick and leaves those counters at zero.
            :meth:`benchmark` is unaffected - it times itself and is called explicitly.
    """

    def __init__(self, robot_name="aliengo", dt=0.004, policy_rate=None, num_threads=1,
                 policy_dir=None, verbose=True, enable_estimator=True, measure_timing=True):
        if policy_dir is None:
            policy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies")
        cfg_path = os.path.join(policy_dir, f"{robot_name}_velocity.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"No velocity policy config for '{robot_name}' at {cfg_path}")
        with open(cfg_path, "r") as f:
            self.cfg = json.load(f)

        self.robot_name = robot_name
        self.policy_dir = policy_dir
        self.dt = float(dt)
        self.obs_dim = int(self.cfg["obs_dim"])
        self.num_actions = int(self.cfg["num_actions"])
        self.history_length = int(self.cfg["history_length"])
        self.action_scale = float(self.cfg["action_scale"])
        self.q_default = np.asarray(self.cfg["q_default"], dtype=np.float64)
        self.kp = float(self.cfg["kp"])
        self.kd = float(self.cfg["kd"])
        self.policy_rate = float(self.cfg["policy_rate"] if policy_rate is None else policy_rate)
        self.enable_estimator = bool(enable_estimator)
        self.measure_timing = bool(measure_timing)

        # --- decimation from the control loop down to the policy rate -------------------------
        # The policy was trained with a fixed 20 ms control step (Isaac decimation 4 at a 5 ms
        # physics step).  Everything the network sees is tied to that period - the joint velocities,
        # the finite-differenced IMU, the action history - so the deployed policy period has to match
        # it exactly.  The control loop is then whatever integer multiple of it we run at: 500 Hz
        # gives a decimation of 10.  A non-integer ratio is refused rather than rounded, because
        # rounding it silently samples the whole observation on the wrong time base.
        policy_dt = 1.0 / self.policy_rate
        self.decimation = int(round(policy_dt / self.dt))
        if self.decimation < 1:
            raise ValueError(
                f"Control loop ({1.0 / self.dt:.1f} Hz) is slower than the policy rate "
                f"({self.policy_rate:.1f} Hz); it must be an integer multiple of it")
        if abs(self.decimation * self.dt - policy_dt) > 1e-12:
            raise ValueError(
                f"Control period {self.dt * 1e3:.4f} ms does not divide the policy period "
                f"{policy_dt * 1e3:.4f} ms: the closest decimation {self.decimation} would give "
                f"{self.decimation * self.dt * 1e3:.4f} ms. Choose a control rate that is an "
                f"integer multiple of {self.policy_rate:.0f} Hz.")
        self.policy_dt = policy_dt

        # --- observation buffer and per-term slices -------------------------------------------
        # One flat, reused float32 buffer: the whole point is that steady-state stepping does no
        # allocation at all.  '_blocks' maps a term to (start, width) of its history block.
        self._obs = np.zeros((1, self.obs_dim), dtype=np.float32)
        self._obs_flat = self._obs[0]
        self._blocks = {}
        offset = 0
        for name, width, length in self.cfg["obs_layout"]:
            if name == "velocity_commands":
                self._cmd_slice = slice(offset, offset + width)
            else:
                self._blocks[name] = (offset, width)
            if length != 1 and length != self.history_length:
                raise ValueError(f"Term '{name}' has history {length}, expected {self.history_length}")
            offset += width * length
        if offset != self.obs_dim:
            raise ValueError(f"obs_layout sums to {offset}, expected obs_dim {self.obs_dim}")

        self.velocity_cmd = np.zeros(3)
        self.prev_action = np.zeros(self.num_actions)
        self.action = np.zeros(self.num_actions)
        self.q_des = self.q_default.copy()
        self._decimation_counter = 0
        self._ready = False
        self.inference_count = 0

        # --- inference-tick timing ---------------------------------------------------------------
        # The policy is synchronous: an inference tick computes the action and returns the joint
        # target on the same control tick, so there is no way to act on a stale action.  What can
        # go wrong instead is that the inference tick overruns the control period, which delays
        # that one low-level command.  It is measured continuously rather than benchmarked once,
        # because it depends on what else is running on the machine.
        self.budget_ms = self.dt * 1e3
        self.last_infer_ms = 0.0
        self.infer_ms_max = 0.0
        self._infer_ms_sum = 0.0
        self.infer_over_budget = 0
        self.benchmark_ms = None
        self.benchmark_median_ms = None

        # --- networks -------------------------------------------------------------------------
        # The contract may name several exported actors: the normal walking policy and any
        # alternates trained on the same observation and action spaces - here a zero-command safe
        # stop.  They all share one observation buffer and one action buffer, so switching between
        # them mid-stride leaves the history continuous; only the weights change.
        variants = self.cfg.get("onnx_variants") or {"normal": self.cfg["onnx"]}
        self.default_variant = "normal" if "normal" in variants else sorted(variants)[0]

        self._out = np.zeros((1, self.num_actions), dtype=np.float32)
        self.estimated_base_lin_vel = np.zeros(3)
        self.networks = {}
        for name in sorted(variants):
            path = os.path.join(policy_dir, variants[name])
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Policy variant '{name}' of '{robot_name}' is missing: {path}")
            net = _PolicyNetwork(name, path, self._obs, self._out, self.estimated_base_lin_vel,
                                 num_threads=num_threads, verbose=verbose,
                                 enable_estimator=self.enable_estimator)
            # Sharing the buffers only works if the variants really do agree on the interface, so
            # a mismatch is refused here rather than producing garbage on the first switch.
            if net.obs_size != self.obs_dim:
                raise ValueError(f"Variant '{name}' expects an observation of {net.obs_size}, "
                                 f"config says {self.obs_dim}")
            if net.action_size != self.num_actions:
                raise ValueError(f"Variant '{name}' outputs {net.action_size} actions, "
                                 f"config says {self.num_actions}")
            self.networks[name] = net

        self._net = self.networks[self.default_variant]
        self.active_variant = self._net.name
        self.model_path = self._net.model_path
        # Reported as available only when *every* variant has a head: otherwise the published
        # estimate would quietly freeze at the last value the moment the variant changed.
        self.has_estimator = all(net.has_estimator for net in self.networks.values())

        if verbose:
            loaded = ", ".join(
                f"{name}{'*' if name == self.active_variant else ''}="
                f"{os.path.basename(net.model_path)}"
                for name, net in sorted(self.networks.items()))
            print(f"VelocityPolicy: loaded {loaded} (* = active) "
                  f"(obs {self.obs_dim}, actions {self.num_actions}, history {self.history_length})")
            print(f"VelocityPolicy: policy at {self.policy_rate:.0f} Hz, control at "
                  f"{1.0 / self.dt:.0f} Hz, decimation {self.decimation}, "
                  f"io_binding {'on' if self._net._binding is not None else 'off'}, "
                  f"estimator {'on' if self.has_estimator else 'off'}, "
                  f"tick timing {'on' if self.measure_timing else 'off'}")

    # ------------------------------------------------------------------------------------------
    # history helpers
    # ------------------------------------------------------------------------------------------
    def _fill(self, name, value):
        """Replicate ``value`` across every frame of a history block."""
        start, width = self._blocks[name]
        block = self._obs_flat[start:start + width * self.history_length]
        block.reshape(self.history_length, width)[:] = value

    def _push(self, name, value):
        """Drop the oldest frame of a history block and append ``value`` as the newest."""
        start, width = self._blocks[name]
        end = start + width * self.history_length
        obs = self._obs_flat
        obs[start:end - width] = obs[start + width:end]
        obs[end - width:end] = value

    # ------------------------------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------------------------------
    def reset(self, q, qd, lin_acc_b, ang_vel_b, projected_gravity_b, velocity_cmd=None):
        """Seed the history exactly the way an Isaac Lab episode reset does.

        The proprioceptive blocks are filled with the current measurement (a fresh
        :class:`CircularBuffer` replicates its first sample) and the action block with zeros, because
        the action buffer is zeroed on reset and the first observation is computed before any action
        has been applied.
        """
        self._fill("imu_lin_acc", np.asarray(lin_acc_b, dtype=np.float32))
        self._fill("imu_ang_vel", np.asarray(ang_vel_b, dtype=np.float32))
        self._fill("imu_projected_gravity", np.asarray(projected_gravity_b, dtype=np.float32))
        self._fill("joint_pos_rel", (np.asarray(q, dtype=np.float64) - self.q_default).astype(np.float32))
        self._fill("joint_vel_rel", np.asarray(qd, dtype=np.float32))
        self._fill("actions", 0.0)

        if velocity_cmd is not None:
            self.velocity_cmd = np.asarray(velocity_cmd, dtype=np.float64).copy()
        self.prev_action[:] = 0.0
        self.action[:] = 0.0
        self.estimated_base_lin_vel[:] = 0.0
        self.q_des = self.q_default.copy()
        # Run the network on the very first step() after a reset rather than waiting out a
        # decimation window with a stale target.
        self._decimation_counter = 0
        self._ready = True
        self.inference_count = 0

    def step(self, q, qd, lin_acc_b, ang_vel_b, projected_gravity_b, velocity_cmd=None):
        """Advance one *control* step and return the joint position target (12,).

        Call this every control tick.  The network is evaluated once every :attr:`decimation` ticks
        and the target is held in between, mirroring the ``decimation = 4`` of the training env.
        """
        if not self._ready:
            raise RuntimeError("VelocityPolicy.reset() must be called before step()")

        if self._decimation_counter == 0:
            tick_start = perf_counter() if self.measure_timing else 0.0
            if velocity_cmd is not None:
                self.velocity_cmd = np.asarray(velocity_cmd, dtype=np.float64)

            # Measured terms for this step.  The newest frame of the action block is still the
            # action produced by the previous inference, which is what mdp.last_action reports.
            self._push("imu_lin_acc", np.asarray(lin_acc_b, dtype=np.float32))
            self._push("imu_ang_vel", np.asarray(ang_vel_b, dtype=np.float32))
            self._push("imu_projected_gravity", np.asarray(projected_gravity_b, dtype=np.float32))
            self._push("joint_pos_rel",
                       (np.asarray(q, dtype=np.float64) - self.q_default).astype(np.float32))
            self._push("joint_vel_rel", np.asarray(qd, dtype=np.float32))
            self._obs_flat[self._cmd_slice] = self.velocity_cmd

            action = self._infer()

            if self._net.has_estimator:
                # Same observation the policy just consumed, so this is exactly the velocity the
                # backbone was handed.
                self._net.runEstimator()

            self.prev_action[:] = self.action
            self.action[:] = action
            self._push("actions", self._out[0])
            self.q_des = self.q_default + self.action_scale * self.action
            self.inference_count += 1

            if self.measure_timing:
                self.last_infer_ms = (perf_counter() - tick_start) * 1e3
                self._infer_ms_sum += self.last_infer_ms
                if self.last_infer_ms > self.infer_ms_max:
                    self.infer_ms_max = self.last_infer_ms
                if self.last_infer_ms > self.budget_ms:
                    self.infer_over_budget += 1

        self._decimation_counter += 1
        if self._decimation_counter >= self.decimation:
            self._decimation_counter = 0
        return self.q_des

    def _infer(self):
        return self._net.infer()

    def select(self, variant):
        """Make ``variant`` the active network and report whether that changed anything.

        The observation is deliberately left alone - history blocks, action block and decimation
        counter all carry over.  The variants share the observation contract, so the incoming
        network picks the motion up exactly where the outgoing one left it.  Calling :meth:`reset`
        here instead would hand a recovery policy a history claiming the robot has been standing
        still, which is the opposite of useful at the moment you reach for it.
        """
        if variant not in self.networks:
            raise KeyError(f"No policy variant '{variant}'; have {sorted(self.networks)}")
        if variant == self.active_variant:
            return False
        self._net = self.networks[variant]
        self.active_variant = variant
        self.model_path = self._net.model_path
        return True

    @property
    def variants(self):
        return sorted(self.networks)

    @property
    def infer_ms_mean(self):
        return self._infer_ms_sum / self.inference_count if self.inference_count else 0.0

    def benchmark(self, iterations=400):
        """Time the inference path back-to-back and return the cost of the work, in milliseconds.

        This is worth having separately from the in-loop measurement because the two answer
        different questions.  The in-loop figure is compute *plus* however long the process spent
        off the CPU inside the call, and on a machine sharing cores with a physics engine that
        second term dominates by more than an order of magnitude; reporting only that would blame
        the policy for the simulator's scheduling, and reporting only this one would hide real
        interference on the robot.  The gap between them is the interesting quantity.

        The reported figure is the **minimum**, not the mean or the median.  Running back-to-back
        does not by itself keep the process on the CPU: this runs moments after the simulator
        launched, and while that is settling the machine can steal time from *most* of the
        iterations - a median taken here came out at 2.2 ms against a true cost near 0.05 ms.  The
        fastest iteration is the one that was not interrupted, which is the closest this can get to
        the work alone.  The median is kept alongside it, because a large gap between the two is
        itself the signal that the machine was busy while measuring.

        Safe to call before the first :meth:`step`: it runs the network and the estimation head on
        whatever is in the observation buffer and writes only to scratch that the first real step
        overwrites.  It does not touch the history, the decimation counter or the counters.
        """
        samples = np.empty(iterations)
        for index in range(iterations):
            start = perf_counter()
            self._net.infer()
            if self._net.has_estimator:
                self._net.runEstimator()
            samples[index] = perf_counter() - start
        self.benchmark_ms = float(samples.min()) * 1e3
        self.benchmark_median_ms = float(np.median(samples)) * 1e3
        return self.benchmark_ms

    def timingSummary(self, simulated=False):
        """One or two lines on whether inference fits the control tick it runs on."""
        if not self.inference_count:
            return "policy never ran"
        lines = []
        if self.benchmark_ms is not None:
            note = ""
            if self.benchmark_median_ms is not None and \
                    self.benchmark_median_ms > 4.0 * self.benchmark_ms:
                note = (f" (the median of the same samples was "
                        f"{self.benchmark_median_ms * 1e3:.0f} us, so the machine was busy even "
                        f"while benchmarking)")
            lines.append(
                f"Policy inference costs {self.benchmark_ms * 1e3:.0f} us of the "
                f"{self.budget_ms * 1e3:.0f} us control tick "
                f"({100.0 * self.benchmark_ms / self.budget_ms:.1f} %), fastest of "
                f"{'400'} back-to-back runs{note}.")
        if not self.measure_timing:
            lines.append("  Per-tick timing is off, so there is no in-loop figure to compare "
                         "against it.")
            return "\n".join(lines)
        late = self.infer_over_budget
        lines.append(
            f"  In the loop it took {self.infer_ms_mean:.3f} ms on average and "
            f"{self.infer_ms_max:.3f} ms at worst, over the tick on {late} of "
            f"{self.inference_count} inference ticks "
            f"({100.0 * late / self.inference_count:.2f} %)"
            + (" - the excess over the figure above is time the process spent off the CPU, not "
               "policy cost." if self.benchmark_ms is not None else "."))
        if simulated and late:
            lines.append("  In simulation that is expected: Gazebo shares these cores and the "
                         "loop is descheduled inside the call. Judge this on the robot, where "
                         "the two figures should agree.")
        lines.append("  The policy is synchronous - the action is computed and applied on the "
                     "same tick - so an overrun delays that one low-level command and can never "
                     "feed a stale action.")
        return "\n".join(lines)

    @property
    def observation(self):
        """The raw observation vector last handed to the network (read-only view)."""
        view = self._obs_flat.view()
        view.flags.writeable = False
        return view
