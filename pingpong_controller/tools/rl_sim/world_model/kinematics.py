"""Differentiable hinge-chain FK extracted from the instantiated MuJoCo model.

The calibrated base and every unselected joint are fixed context. Positions,
rotations and twist Jacobians share one chain; there is no independent learned
racket pose. Callers vmap the single-configuration function for batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax.numpy as jnp
import mujoco
import numpy as np


@dataclass(frozen=True)
class Hinge:
    selected_index: int
    joint_type: int
    axis: Any
    anchor: Any
    reference_q_rad: float
    fixed_q_rad: float


@dataclass(frozen=True)
class ChainBody:
    translation: Any
    rotation: Any
    joints: tuple[Hinge, ...]


@dataclass(frozen=True)
class KinematicChain:
    bodies: tuple[ChainBody, ...]
    joint_ids: tuple[int, ...]
    site_translation: Any
    site_rotation: Any
    geom_translation: Any
    geom_rotation: Any


class Kinematics(NamedTuple):
    site_position: Any
    site_rotation: Any
    geom_position: Any
    geom_rotation: Any
    site_jacobian: Any
    geom_jacobian: Any
    angular_jacobian: Any


def _rotation(quaternion):
    value = np.empty(9, np.float64)
    mujoco.mju_quat2Mat(value, np.asarray(quaternion, np.float64))
    return value.reshape(3, 3).astype(np.float32)


def extract_chain(model, fixed_qpos, *, joint_ids, site_id: int, geom_id: int) -> KinematicChain:
    """Extract a fixed-base context; fail instead of omitting a moving base."""
    joint_ids = tuple(int(j) for j in joint_ids)
    if not joint_ids or len(set(joint_ids)) != len(joint_ids):
        raise ValueError('Joint order must be nonempty and unique')
    fixed_qpos = np.asarray(fixed_qpos)
    if fixed_qpos.shape != (model.nq,) or not np.isfinite(fixed_qpos).all():
        raise ValueError('A complete finite calibrated qpos is required')
    body = int(model.site_bodyid[site_id])
    if body != int(model.geom_bodyid[geom_id]):
        raise ValueError('Racket site and collision geometry must share the same rigid body')
    ancestors = []
    while body:
        ancestors.append(body)
        body = int(model.body_parentid[body])
    bodies, found = [], set()
    for body in reversed(ancestors):
        joints = []
        start, count = int(model.body_jntadr[body]), int(model.body_jntnum[body])
        for joint in range(start, start + count):
            selected = joint_ids.index(joint) if joint in joint_ids else -1
            joint_type = int(model.jnt_type[joint])
            if (joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
                    or (selected >= 0 and joint_type != mujoco.mjtJoint.mjJNT_HINGE)):
                raise ValueError(f'Unsupported moving joint {joint} in fixed-base arm chain')
            if selected >= 0:
                found.add(joint)
            address = int(model.jnt_qposadr[joint])
            joints.append(Hinge(selected, joint_type, model.jnt_axis[joint].astype(np.float32).copy(),
                model.jnt_pos[joint].astype(np.float32).copy(),
                float(model.qpos0[address]), float(fixed_qpos[address])))
        bodies.append(ChainBody(model.body_pos[body].astype(np.float32).copy(),
                                _rotation(model.body_quat[body]), tuple(joints)))
    if found != set(joint_ids):
        raise ValueError('Selected arm joints must all be ancestors of the racket')
    return KinematicChain(tuple(bodies), joint_ids,
        model.site_pos[site_id].astype(np.float32).copy(), _rotation(model.site_quat[site_id]),
        model.geom_pos[geom_id].astype(np.float32).copy(), _rotation(model.geom_quat[geom_id]))


def _axis_rotation(axis, angle):
    x, y, z = jnp.asarray(axis)
    zero = jnp.zeros_like(x)
    cross = jnp.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    return jnp.eye(3, dtype=angle.dtype) + jnp.sin(angle)*cross + (1-jnp.cos(angle))*(cross@cross)


def forward_kinematics(chain: KinematicChain, q_rad) -> Kinematics:
    """Return site/surface poses and world Jacobians in the declared joint order."""
    if q_rad.shape != (len(chain.joint_ids),):
        raise ValueError('FK expects one configuration; vmap for independent lanes')
    position, rotation = jnp.zeros(3, q_rad.dtype), jnp.eye(3, dtype=q_rad.dtype)
    anchors, axes = [None]*len(chain.joint_ids), [None]*len(chain.joint_ids)
    for body in chain.bodies:
        position = position + rotation @ body.translation
        rotation = rotation @ body.rotation
        for joint in body.joints:
            if joint.joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                # V324 keeps the six base coordinates in qpos and constrains
                # them with equality joints. Use the supplied calibrated pose,
                # including a nonzero slide reference, instead of dropping it.
                position = position + rotation @ joint.axis * (joint.fixed_q_rad-joint.reference_q_rad)
                continue
            anchor = position + rotation @ joint.anchor
            if joint.selected_index >= 0:
                anchors[joint.selected_index] = anchor
                axes[joint.selected_index] = rotation @ joint.axis
                angle = q_rad[joint.selected_index] - joint.reference_q_rad
            else:
                angle = jnp.asarray(joint.fixed_q_rad-joint.reference_q_rad, q_rad.dtype)
            rotation = rotation @ _axis_rotation(joint.axis, angle)
            position = anchor - rotation @ joint.anchor
    site_position = position + rotation @ chain.site_translation
    geom_position = position + rotation @ chain.geom_translation
    axes, anchors = jnp.stack(axes), jnp.stack(anchors)
    return Kinematics(site_position, rotation @ chain.site_rotation,
        geom_position, rotation @ chain.geom_rotation,
        jnp.cross(axes, site_position-anchors).T,
        jnp.cross(axes, geom_position-anchors).T, axes.T)
