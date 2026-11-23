"""Private native-quaternion ancestry FK for projection implementation audits.

No dynamics solve, geometry scan or global precision change is performed.
The original matrix FK used by fitted physical components remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import mujoco
from mujoco.mjx._src import math
import numpy as np

from .kinematics import KinematicChain, Kinematics, extract_chain


@dataclass(frozen=True)
class ControlChain:
    matrix_chain: KinematicChain
    body_quaternions: tuple
    site_quaternion: Any
    geom_quaternion: Any

    @property
    def joint_ids(self):
        return self.matrix_chain.joint_ids


def extract_control_chain(model, fixed_qpos, *, joint_ids, site_id, geom_id):
    """Retain compiled quaternions rather than reconstructing them from matrices."""
    chain = extract_chain(model, fixed_qpos, joint_ids=joint_ids, site_id=site_id, geom_id=geom_id)
    ancestors = []
    body = int(model.site_bodyid[site_id])
    while body:
        ancestors.append(body)
        body = int(model.body_parentid[body])
    quaternions = tuple(np.asarray(model.body_quat[i], np.float32).copy() for i in reversed(ancestors))
    return ControlChain(chain, quaternions, np.asarray(model.site_quat[site_id], np.float32).copy(),
                        np.asarray(model.geom_quat[geom_id], np.float32).copy())


def forward_control_kinematics(chain: ControlChain, q_rad) -> Kinematics:
    """Use native MJX hinge/slide arithmetic for the racket ancestor chain only."""
    if q_rad.shape != (len(chain.joint_ids),):
        raise ValueError('Expected one configuration in declared joint order')
    base = chain.matrix_chain
    position = jnp.zeros(3, q_rad.dtype)
    quaternion = jnp.array([1., 0., 0., 0.], q_rad.dtype)
    anchors, axes = [None]*len(chain.joint_ids), [None]*len(chain.joint_ids)
    for body, body_quaternion in zip(base.bodies, chain.body_quaternions, strict=True):
        position = position+math.rotate(jnp.asarray(body.translation), quaternion)
        quaternion = math.quat_mul(quaternion, jnp.asarray(body_quaternion))
        for joint in body.joints:
            axis = math.rotate(jnp.asarray(joint.axis), quaternion)
            anchor = position+math.rotate(jnp.asarray(joint.anchor), quaternion)
            value = q_rad[joint.selected_index] if joint.selected_index >= 0 else jnp.asarray(joint.fixed_q_rad, q_rad.dtype)
            angle = value-joint.reference_q_rad
            if joint.joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                position = position+axis*angle
            else:
                if joint.selected_index >= 0:
                    axes[joint.selected_index], anchors[joint.selected_index] = axis, anchor
                quaternion = math.quat_mul(quaternion, math.axis_angle_to_quat(jnp.asarray(joint.axis), angle))
                position = anchor-math.rotate(jnp.asarray(joint.anchor), quaternion)
    site_position = position+math.rotate(jnp.asarray(base.site_translation), quaternion)
    geom_position = position+math.rotate(jnp.asarray(base.geom_translation), quaternion)
    axes, anchors = jnp.stack(axes), jnp.stack(anchors)
    return Kinematics(site_position, math.quat_to_mat(math.quat_mul(quaternion, jnp.asarray(chain.site_quaternion))),
        geom_position, math.quat_to_mat(math.quat_mul(quaternion, jnp.asarray(chain.geom_quaternion))),
        jnp.cross(axes, site_position-anchors).T, jnp.cross(axes, geom_position-anchors).T, axes.T)
