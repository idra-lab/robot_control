"""Runs the single-network velocity policy trained in safe_rl (``tasks/manager_based/velocity``).

The deployment contract is ``policies/<robot>_velocity.json``.  The observation (228 floats) is,
in order::

    velocity_commands       3       current step only
    imu_lin_acc             3 x 5   specific force averaged over the policy step
    imu_ang_vel             3 x 5
    imu_projected_gravity   3 x 5
    joint_pos_rel          12 x 5   q - q_default
    joint_vel_rel          12 x 5
    actions                12 x 5   raw network outputs

History blocks are oldest frame first, as Isaac Lab's ``CircularBuffer`` returns them.  The
observation normalizer is inside the ONNX graph, so raw observations are fed.  The action is a
joint position offset: ``q_des = q_default + action_scale * action``.

The actor regresses the base linear velocity in an internal head whose output is not a graph
output; :class:`_PolicyNetwork` evaluates that head separately from the weights in the same file.
"""

import json
import os

import numpy as np
import onnxruntime as ort

POLICY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies")


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
    """Decode a protobuf message into ``(field_number, wire_type, value)`` triples (no onnx package
    in the container)."""
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
        if name is None or raw is None or dtype != 1:   # 1 == FLOAT
            continue
        array = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        tensors[name] = array.reshape(dims) if dims else array
    return tensors


class _PolicyNetwork:
    """One exported actor: its onnxruntime session and its velocity-estimation head.

    Every variant is bound to the same observation and action buffers, so switching variant keeps
    the history.
    """

    def __init__(self, name, model_path, obs, out, estimate_out, enable_estimator=True):
        self.name = name
        self.model_path = model_path
        self._obs = obs
        self._obs_flat = obs[0]
        self._out = out
        self.estimated_base_lin_vel = estimate_out

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name
        self.obs_size = int(self.session.get_inputs()[0].shape[-1])
        self.action_size = int(self.session.get_outputs()[0].shape[-1])

        # Bind input and output to the shared buffers, but only if the input really aliases the
        # buffer: a runtime that copies would feed a stale observation forever.
        self._binding = None
        try:
            binding = self.session.io_binding()
            self._obs_ort = ort.OrtValue.ortvalue_from_numpy(self._obs)
            self._out_ort = ort.OrtValue.ortvalue_from_numpy(self._out)
            binding.bind_ortvalue_input(self._input_name, self._obs_ort)
            binding.bind_ortvalue_output(self._output_name, self._out_ort)
            self._obs_flat[0] = 1.234
            aliases = float(self._obs_ort.numpy()[0, 0]) == np.float32(1.234)
            self._obs_flat[0] = 0.0
            if aliases:
                self._binding = binding
        except Exception as exc:
            print(f"VelocityPolicy[{name}]: I/O binding unavailable ({exc}), using run()")

        self.has_estimator = False
        if enable_estimator:
            self._loadEstimator()

    def infer(self):
        if self._binding is not None:
            self.session.run_with_iobinding(self._binding)
        else:
            self._out[:] = self.session.run([self._output_name], {self._input_name: self._obs})[0]
        return self._out[0]

    def _loadEstimator(self):
        """Read the estimation head (three linear layers, ELU) and fold the normalizer into its
        first layer: ``W((x - mean) / std) + b == (W / std) x + (b - (W / std) mean)``."""
        try:
            tensors = load_onnx_initializers(self.model_path)
        except (OSError, ValueError) as exc:
            print(f"VelocityPolicy[{self.name}]: cannot read the estimation head ({exc})")
            return
        needed = ["normalizer._mean"] + [f"actor.estimator.{i}.{p}" for i in (0, 2, 4)
                                         for p in ("weight", "bias")]
        if any(name not in tensors for name in needed):
            print(f"VelocityPolicy[{self.name}]: no state-estimation head in this policy")
            return

        # The normalizer scale is an anonymous constant: find it by shape.
        mean = tensors["normalizer._mean"].reshape(-1)
        std = next((t.reshape(-1) for n, t in tensors.items()
                    if n != "normalizer._mean" and t.size == mean.size
                    and "estimator" not in n and "backbone" not in n), None)
        if std is None:
            print(f"VelocityPolicy[{self.name}]: cannot identify the normalizer scale")
            return

        w0 = (tensors["actor.estimator.0.weight"] / std).astype(np.float32)
        self._est_w = [np.ascontiguousarray(w0),
                       np.ascontiguousarray(tensors["actor.estimator.2.weight"], dtype=np.float32),
                       np.ascontiguousarray(tensors["actor.estimator.4.weight"], dtype=np.float32)]
        self._est_b = [np.ascontiguousarray(tensors["actor.estimator.0.bias"] - w0 @ mean,
                                            dtype=np.float32),
                       np.ascontiguousarray(tensors["actor.estimator.2.bias"], dtype=np.float32),
                       np.ascontiguousarray(tensors["actor.estimator.4.bias"], dtype=np.float32)]
        self._est_h = [np.zeros(w.shape[0], dtype=np.float32) for w in self._est_w]
        self._est_neg = [np.zeros(w.shape[0], dtype=np.float32) for w in self._est_w]
        self.has_estimator = True

    def runEstimator(self):
        """Evaluate the head on the current observation without allocating.  ELU is written as
        ``max(x, 0) + expm1(min(x, 0))``."""
        activation = self._obs_flat
        last = len(self._est_w) - 1
        for index, (w, b) in enumerate(zip(self._est_w, self._est_b)):
            out = self._est_h[index]
            np.dot(w, activation, out=out)
            out += b
            if index < last:
                negative = self._est_neg[index]
                np.minimum(out, 0.0, out=negative)
                np.expm1(negative, out=negative)
                np.maximum(out, 0.0, out=out)
                out += negative
            activation = out
        self.estimated_base_lin_vel[:] = self._est_h[-1]


class VelocityPolicy:
    """Runs the policy at its trained rate on top of a faster control loop.

    Args:
        robot_name: loads ``policies/<robot_name>_velocity.json``.
        dt: period of the control loop that calls :meth:`step`; it must divide the policy period.
        enable_estimator: also evaluate the velocity-estimation head (``estimated_base_lin_vel``).
    """

    def __init__(self, robot_name, dt, enable_estimator=True):
        with open(os.path.join(POLICY_DIR, f"{robot_name}_velocity.json")) as f:
            self.cfg = json.load(f)

        self.dt = float(dt)
        self.obs_dim = int(self.cfg["obs_dim"])
        self.num_actions = int(self.cfg["num_actions"])
        self.history_length = int(self.cfg["history_length"])
        self.action_scale = float(self.cfg["action_scale"])
        self.q_default = np.asarray(self.cfg["q_default"], dtype=np.float64)
        self.kp = float(self.cfg["kp"])
        self.kd = float(self.cfg["kd"])
        self.policy_rate = float(self.cfg["policy_rate"])

        policy_dt = 1.0 / self.policy_rate
        self.decimation = int(round(policy_dt / self.dt))
        if self.decimation < 1 or abs(self.decimation * self.dt - policy_dt) > 1e-12:
            raise ValueError(f"The control period {self.dt * 1e3:.3f} ms does not divide the policy "
                             f"period {policy_dt * 1e3:.3f} ms")

        self._obs = np.zeros((1, self.obs_dim), dtype=np.float32)
        self._obs_flat = self._obs[0]
        self._blocks = {}
        offset = 0
        for name, width, length in self.cfg["obs_layout"]:
            if name == "velocity_commands":
                self._cmd_slice = slice(offset, offset + width)
            else:
                self._blocks[name] = (offset, width)
            if length not in (1, self.history_length):
                raise ValueError(f"Term '{name}' has history {length}, expected {self.history_length}")
            offset += width * length
        if offset != self.obs_dim:
            raise ValueError(f"obs_layout sums to {offset}, expected obs_dim {self.obs_dim}")

        self.prev_action = np.zeros(self.num_actions)
        self.action = np.zeros(self.num_actions)
        self.q_des = self.q_default.copy()
        self.imu_lin_acc_obs = np.zeros(3)       # the imu_lin_acc term of the last inference
        self._acc_sum = np.zeros(3)
        self._acc_count = 0
        self._decimation_counter = 0
        self._ready = False
        self.inference_count = 0

        self._out = np.zeros((1, self.num_actions), dtype=np.float32)
        self.estimated_base_lin_vel = np.zeros(3)
        self.networks = {}
        for name, filename in sorted(self.cfg["onnx_variants"].items()):
            net = _PolicyNetwork(name, os.path.join(POLICY_DIR, filename), self._obs, self._out,
                                 self.estimated_base_lin_vel, enable_estimator=enable_estimator)
            if net.obs_size != self.obs_dim or net.action_size != self.num_actions:
                raise ValueError(f"Variant '{name}' is {net.obs_size} -> {net.action_size}, the "
                                 f"contract says {self.obs_dim} -> {self.num_actions}")
            self.networks[name] = net
        self.default_variant = "normal"
        self._net = self.networks[self.default_variant]
        self.active_variant = self.default_variant
        self.has_estimator = all(net.has_estimator for net in self.networks.values())

        print(f"VelocityPolicy: {robot_name}, variants {self.variants}, obs {self.obs_dim}, "
              f"policy {self.policy_rate:.0f} Hz, decimation {self.decimation}, "
              f"io_binding {'on' if self._net._binding is not None else 'off'}, "
              f"estimator {'on' if self.has_estimator else 'off'}")

    def _fill(self, name, value):
        """Replicate ``value`` across every frame of a history block."""
        start, width = self._blocks[name]
        self._obs_flat[start:start + width * self.history_length].reshape(
            self.history_length, width)[:] = value

    def _push(self, name, value):
        """Drop the oldest frame of a history block and append ``value``."""
        start, width = self._blocks[name]
        end = start + width * self.history_length
        self._obs_flat[start:end - width] = self._obs_flat[start + width:end]
        self._obs_flat[end - width:end] = value

    def reset(self, q, qd, lin_acc_b, ang_vel_b, projected_gravity_b, velocity_cmd=None):
        """Seed the history as an Isaac Lab episode reset does: every frame is the current
        measurement, the action block is zero.  ``lin_acc_b`` is the accelerometer reading
        (specific force in the base frame, ``(0, 0, 9.81)`` when level)."""
        self.imu_lin_acc_obs[:] = lin_acc_b
        self._fill("imu_lin_acc", self.imu_lin_acc_obs)
        self._fill("imu_ang_vel", ang_vel_b)
        self._fill("imu_projected_gravity", projected_gravity_b)
        self._fill("joint_pos_rel", np.asarray(q) - self.q_default)
        self._fill("joint_vel_rel", qd)
        self._fill("actions", 0.0)
        self._obs_flat[self._cmd_slice] = 0.0 if velocity_cmd is None else velocity_cmd
        self._acc_sum[:] = 0.0
        self._acc_count = 0
        self.prev_action[:] = 0.0
        self.action[:] = 0.0
        self.estimated_base_lin_vel[:] = 0.0
        self.q_des = self.q_default.copy()
        self._decimation_counter = 0          # infer on the first step()
        self._ready = True
        self.inference_count = 0

    def step(self, q, qd, lin_acc_b, ang_vel_b, projected_gravity_b, velocity_cmd):
        """Advance one control tick and return the joint position target.

        The network runs every :attr:`decimation` ticks and the target is held in between.  The
        accelerometer is read on every tick, because ``imu_lin_acc`` is its mean over the step.
        """
        if not self._ready:
            raise RuntimeError("VelocityPolicy.reset() must be called before step()")

        self._acc_sum += lin_acc_b
        self._acc_count += 1

        if self._decimation_counter == 0:
            np.divide(self._acc_sum, self._acc_count, out=self.imu_lin_acc_obs)
            self._acc_sum[:] = 0.0
            self._acc_count = 0
            self._push("imu_lin_acc", self.imu_lin_acc_obs)
            self._push("imu_ang_vel", ang_vel_b)
            self._push("imu_projected_gravity", projected_gravity_b)
            self._push("joint_pos_rel", np.asarray(q) - self.q_default)
            self._push("joint_vel_rel", qd)
            self._obs_flat[self._cmd_slice] = velocity_cmd

            action = self._net.infer()
            if self._net.has_estimator:
                self._net.runEstimator()
            self.prev_action[:] = self.action
            self.action[:] = action
            self._push("actions", action)
            self.q_des = self.q_default + self.action_scale * self.action
            self.inference_count += 1

        self._decimation_counter = (self._decimation_counter + 1) % self.decimation
        return self.q_des

    def select(self, variant):
        """Make ``variant`` the active network, keeping the history."""
        if variant not in self.networks:
            raise KeyError(f"No policy variant '{variant}'; have {self.variants}")
        self._net = self.networks[variant]
        self.active_variant = variant

    @property
    def variants(self):
        return sorted(self.networks)
