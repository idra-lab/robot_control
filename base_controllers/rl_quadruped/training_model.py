# -*- coding: utf-8 -*-
"""Rigid-body properties of the model a policy was trained on, for comparison against the one it
is deployed on.

A position-control policy is a feedback law tuned to a particular set of link inertias: the joint
target it emits is turned into torque by a fixed PD, so the closed-loop response of each leg is set
by ``kp`` and ``kd`` against the leg's inertia.  Deploy the same policy on a robot whose leg
inertia differs by a factor and the response it learned to expect is no longer the response it
gets.  That is not a subtle effect on a quadruped: standing still is held by feedback, and a
mis-tuned leg answers a stop command with a residual limit cycle rather than a standstill.

So the model the policy was trained on is part of its deployment contract, and this module makes
that contract checkable.  It reads the URDF the training asset was built from and reports the
properties of each body *as a physics engine would see it* - fixed-joint children fused into their
parent, because that is what both Gazebo and PhysX do before they integrate anything.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

_INERTIA_KEYS = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")


def _rpy_to_rot(rpy):
    """Roll-pitch-yaw (URDF convention: R = Rz(y) Ry(p) Rx(r)) to a rotation matrix."""
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def _inertia_matrix(values):
    ixx, ixy, ixz, iyy, iyz, izz = values
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


def _matrix_to_values(matrix):
    return np.array([matrix[0, 0], matrix[0, 1], matrix[0, 2],
                     matrix[1, 1], matrix[1, 2], matrix[2, 2]])


class Body(object):
    """One physics body: mass, centre of mass and the inertia tensor about that centre."""

    __slots__ = ("name", "mass", "com", "inertia", "fused")

    def __init__(self, name, mass, com, inertia, fused=()):
        self.name = name
        self.mass = float(mass)
        self.com = np.asarray(com, dtype=float)
        self.inertia = np.asarray(inertia, dtype=float)      # 3x3, about com
        self.fused = tuple(fused)

    @property
    def values(self):
        """The six independent inertia components, in URDF/Gazebo order."""
        return _matrix_to_values(self.inertia)

    def __repr__(self):
        v = self.values
        return (f"Body({self.name}, m={self.mass:.4f}, "
                f"ixx={v[0]:.6f}, iyy={v[3]:.6f}, izz={v[5]:.6f})")


def _read_links_and_joints(path):
    root = ET.parse(path).getroot()
    links = {}
    for element in root.findall("link"):
        inertial = element.find("inertial")
        if inertial is None:
            continue
        origin = inertial.find("origin")
        xyz = np.zeros(3)
        rpy = np.zeros(3)
        if origin is not None:
            xyz = np.array([float(x) for x in origin.get("xyz", "0 0 0").split()])
            rpy = np.array([float(x) for x in origin.get("rpy", "0 0 0").split()])
        tensor = inertial.find("inertia")
        values = np.array([float(tensor.get(k, 0.0)) for k in _INERTIA_KEYS])
        # A non-identity inertial origin rotation expresses the tensor in a rotated frame; bring it
        # back into the link frame so everything downstream is in one frame.
        matrix = _inertia_matrix(values)
        if np.any(rpy):
            rotation = _rpy_to_rot(rpy)
            matrix = rotation @ matrix @ rotation.T
        links[element.get("name")] = dict(
            mass=float(inertial.find("mass").get("value")), com=xyz, inertia=matrix)
    joints = []
    for element in root.findall("joint"):
        origin = element.find("origin")
        xyz = np.zeros(3)
        rpy = np.zeros(3)
        if origin is not None:
            xyz = np.array([float(x) for x in origin.get("xyz", "0 0 0").split()])
            rpy = np.array([float(x) for x in origin.get("rpy", "0 0 0").split()])
        joints.append(dict(name=element.get("name"), type=element.get("type"),
                           parent=element.find("parent").get("link"),
                           child=element.find("child").get("link"), xyz=xyz, rpy=rpy))
    return links, joints


def load_bodies(path):
    """Return ``{link_name: Body}`` with every fixed-joint group merged into one physics body.

    A physics engine has no degree of freedom for a fixed joint, so it merges the links either side
    of one before integrating anything: Gazebo reports the *merged* mass and inertia for the
    surviving link, and PhysX does the same.  Comparing raw URDF links across two models would
    therefore compare quantities neither simulator ever uses - on the Aliengo the foot is fixed to
    the calf, the IMU to the trunk, and in the Isaac asset each actuator rotor is fixed to its leg
    link, so the bodies that actually move are 'calf + foot', 'trunk + imu' and 'thigh + rotor'.

    Groups are found by connectivity over the fixed joints, which is what makes this robust to the
    massless adapter links URDFs like to put at the root: locosim hangs ``trunk`` off a
    ``floating_base`` link with no inertia at all, and treating that as the parent to merge *into*
    would lose the trunk entirely.  Every member name of a group maps to the same merged body, so a
    lookup works whichever name the caller happens to know.
    """
    links, joints = _read_links_and_joints(path)
    all_names = set(links)
    for joint in joints:
        all_names.add(joint["parent"])
        all_names.add(joint["child"])

    # --- union-find over the fixed joints ------------------------------------------------------
    parent_of = {name: name for name in all_names}

    def find(name):
        while parent_of[name] != name:
            parent_of[name] = parent_of[parent_of[name]]
            name = parent_of[name]
        return name

    fixed = [j for j in joints if j["type"] == "fixed"]
    for joint in fixed:
        a, b = find(joint["parent"]), find(joint["child"])
        if a != b:
            parent_of[b] = a

    groups = {}
    for name in all_names:
        groups.setdefault(find(name), []).append(name)

    # --- pose of every link relative to its group representative -------------------------------
    # Only fixed joints are walked, so the relative pose inside a group is constant by definition.
    adjacency = {}
    for joint in fixed:
        adjacency.setdefault(joint["parent"], []).append((joint["child"], joint, +1))
        adjacency.setdefault(joint["child"], []).append((joint["parent"], joint, -1))

    bodies = {}
    for root, members in groups.items():
        pose = {root: (np.zeros(3), np.eye(3))}
        stack = [root]
        while stack:
            here = stack.pop()
            offset, rotation = pose[here]
            for other, joint, direction in adjacency.get(here, ()):
                if other in pose:
                    continue
                joint_rotation = _rpy_to_rot(joint["rpy"])
                if direction > 0:                      # here is the parent of other
                    pose[other] = (offset + rotation @ joint["xyz"], rotation @ joint_rotation)
                else:                                  # here is the child of other
                    inverse = joint_rotation.T
                    pose[other] = (offset - rotation @ inverse @ joint["xyz"], rotation @ inverse)
                stack.append(other)

        parts = []
        mass = 0.0
        weighted = np.zeros(3)
        for name in members:
            if name not in links:
                continue                              # a massless adapter link contributes nothing
            offset, rotation = pose[name]
            link = links[name]
            com = offset + rotation @ link["com"]
            inertia = rotation @ link["inertia"] @ rotation.T
            parts.append((link["mass"], com, inertia))
            mass += link["mass"]
            weighted = weighted + link["mass"] * com
        if not parts:
            continue
        com = weighted / mass if mass > 0.0 else weighted
        total = np.zeros((3, 3))
        for part_mass, part_com, part_inertia in parts:
            d = part_com - com
            total += part_inertia + part_mass * (float(d @ d) * np.eye(3) - np.outer(d, d))
        # The representative is whichever member carries mass and sits highest in the group; the
        # name is cosmetic, since every member resolves to this same body.
        with_mass = [n for n in members if n in links]
        body = Body(sorted(with_mass)[0] if root not in with_mass else root,
                    mass, com, total,
                    fused=tuple(sorted(n for n in with_mass if n != root)))
        for name in members:
            bodies[name] = body
    return bodies


def unique_bodies(bodies):
    """The distinct physics bodies in a mapping returned by :func:`load_bodies`."""
    seen, out = set(), []
    for body in bodies.values():
        if id(body) not in seen:
            seen.add(id(body))
            out.append(body)
    return out


def default_model_path(policy_dir, robot_name):
    """Path of the vendored training URDF for ``robot_name``, or None when it is not there."""
    path = os.path.join(policy_dir, f"{robot_name}_velocity_model.urdf")
    return path if os.path.exists(path) else None


def compare(training, deployed, name_map, mass_tol=1e-3, inertia_tol=1e-3):
    """Compare two ``{name: Body}`` sets through ``name_map`` (deployed name -> training name).

    Returns a list of dicts, one per compared body, worst relative inertia difference first.
    """
    report = []
    for deployed_name, training_name in name_map.items():
        if deployed_name not in deployed or training_name not in training:
            continue
        a, b = deployed[deployed_name], training[training_name]
        va, vb = a.values, b.values
        scale = max(np.abs(vb).max(), np.abs(va).max(), 1e-12)
        rel = np.abs(va - vb) / scale
        report.append(dict(
            deployed=deployed_name, training=training_name,
            mass=(a.mass, b.mass), d_mass=a.mass - b.mass,
            values=(va, vb), max_rel=float(rel.max()),
            ratio=np.where(np.abs(vb) > 1e-12, va / np.where(np.abs(vb) > 1e-12, vb, 1.0), np.nan),
            differs=(abs(a.mass - b.mass) > mass_tol or rel.max() > inertia_tol),
            d_com=float(np.abs(a.com - b.com).max())))
    report.sort(key=lambda r: -r["max_rel"])
    return report
