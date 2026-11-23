"""V324 bounded reference projection with full-chain differentiable FK.

The native linear projector, bounds and nonlinear hold rule are preserved.
The chain carries the calibrated fixed base; six waist and seven arm states
are supplied every call. No MJX forward or dynamics solve is performed here.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from cartesian_pose_action_projection_jax import (
    matrix_to_euler_xyz_jax, project_cartesian_pose_qvel_jax,
    relative_xyz_euler_rate_jacobian_jax,
)

from .kinematics import forward_kinematics
from .control_execution import projection_v324


@dataclass(frozen=True)
class ProjectionContract:
    dt_s: float
    q_lower: Any
    q_upper: Any
    velocity_limit: Any
    reference_horizon_s: Any
    world_to_policy_rotation: Any
    policy_origin_world: Any
    reference_rotation: Any
    position_lower: Any
    position_upper: Any
    outer_position_lower: Any
    outer_position_upper: Any
    roll_pitch_limit: Any
    outer_roll_pitch_limit: Any
    position_gain_s: float
    orientation_gain_s: float
    sweeps: int
    tolerance: float
    execution_unroll: int = 1
    fk_matmul_precision: str | None = None

    @classmethod
    def from_native(cls, env):
        """Copy actual constructor values; never infer them from a run name."""
        mapping = dict(q_lower='arm_lo', q_upper='arm_hi', velocity_limit='arm_vel_limit_rad_s',
            reference_horizon_s='recovered_rmp_qvel_reference_error_horizon_s_per_joint',
            world_to_policy_rotation='cartesian_world_to_policy_rotation',
            policy_origin_world='cartesian_policy_origin_world',
            reference_rotation='fixed_contact_racket_reference_rotation',
            position_lower='cartesian_pose_position_lower', position_upper='cartesian_pose_position_upper',
            outer_position_lower='cartesian_pose_outer_position_lower', outer_position_upper='cartesian_pose_outer_position_upper',
            roll_pitch_limit='cartesian_pose_roll_pitch_limit', outer_roll_pitch_limit='cartesian_pose_outer_roll_pitch_limit')
        if abs(float(env.dt)-.005) > 1e-10:
            raise ValueError('WMJ projection requires the native 5 ms control period')
        return cls(dt_s=float(env.dt), **{name: jnp.asarray(getattr(env, source), jnp.float32)
            for name, source in mapping.items()},
            position_gain_s=float(env.cfg.cartesian_pose_position_cbf_gain_s),
            orientation_gain_s=float(env.cfg.cartesian_pose_orientation_cbf_gain_s),
            sweeps=int(env.cfg.cartesian_pose_projection_sweeps),
            tolerance=float(env.cfg.cartesian_pose_projection_tolerance), execution_unroll=24,
            fk_matmul_precision='highest')


class ProjectionResult(NamedTuple):
    reference_q: Any
    accepted_qvel: Any
    correction_norm: Any
    retained_fraction: Any
    intervention: Any
    nonlinear_backstop: Any
    fallback_to_hold: Any
    current_safe: Any
    position_violation: Any
    orientation_violation: Any
    linear_max_violation: Any
    final_yaw_rad: Any


def project_reference(chain, contract: ProjectionContract, measured_q, reference_q,
                      nominal_qvel, *, kinematics=forward_kinematics) -> ProjectionResult:
    """Project (N,7) velocities using (N,13) measured arm-then-waist state."""
    if (len(chain.joint_ids) != 13 or measured_q.ndim != 2 or measured_q.shape[1] != 13
            or reference_q.shape != measured_q[:, :7].shape or nominal_qvel.shape != reference_q.shape):
        raise ValueError('Expected thirteen measured joints, seven references and velocities per lane')
    c = contract
    if c.fk_matmul_precision not in (None, 'highest'):
        raise ValueError('FK precision must be legacy default or full FP32')

    def pose(q):
        full = jnp.concatenate((q, measured_q[:, 7:]), axis=-1)
        # Precision applies only to this private FK. Preserve native projection
        # arithmetic and all simulator/global precision settings.
        precision = (nullcontext() if c.fk_matmul_precision is None
                     else jax.default_matmul_precision(c.fk_matmul_precision))
        with precision:
            fk = jax.vmap(lambda value: kinematics(chain, value))(full)
        position = jnp.einsum('ij,nj->ni', c.world_to_policy_rotation,
                              fk.site_position-c.policy_origin_world[None])
        rotation = jnp.einsum('ji,njk->nik', c.reference_rotation, fk.site_rotation)
        return fk, position, matrix_to_euler_xyz_jax(rotation)

    fk, position, rpy = pose(reference_q)
    translation = jnp.einsum('ij,njk->nik', c.world_to_policy_rotation, fk.site_jacobian[..., :7])
    angular = relative_xyz_euler_rate_jacobian_jax(rpy, c.reference_rotation, fk.angular_jacobian[..., :7])
    maximum_error = c.velocity_limit*c.reference_horizon_s
    lower_q = jnp.maximum(c.q_lower[None], measured_q[:, :7]-maximum_error[None])
    upper_q = jnp.minimum(c.q_upper[None], measured_q[:, :7]+maximum_error[None])
    if c.execution_unroll not in (1, 24):
        raise ValueError('Only the original loop or pinned V324 EXEC projection is supported')
    project = projection_v324 if c.execution_unroll == 24 else project_cartesian_pose_qvel_jax
    projection = project(nominal_qvel=nominal_qvel,
        position_m=position, relative_rpy_rad=rpy, translation_jacobian=translation,
        relative_rpy_jacobian=angular, position_lower_m=c.position_lower,
        position_upper_m=c.position_upper, roll_pitch_limit_rad=c.roll_pitch_limit,
        velocity_lower=jnp.maximum(-c.velocity_limit[None], (lower_q-reference_q)/c.dt_s),
        velocity_upper=jnp.minimum(c.velocity_limit[None], (upper_q-reference_q)/c.dt_s),
        velocity_scale=c.velocity_limit, position_gain_s=c.position_gain_s,
        orientation_gain_s=c.orientation_gain_s, projection_sweeps=c.sweeps)
    candidate_q = reference_q+projection.qvel*c.dt_s
    _, candidate_position, candidate_rpy = pose(candidate_q)

    def violations(p, angles):
        position_error = jnp.max(jnp.maximum(jnp.maximum(c.outer_position_lower[None]-p,
            p-c.outer_position_upper[None]), 0.), axis=-1)
        angle_error = jnp.max(jnp.maximum(jnp.abs(angles[:, :2])-c.outer_roll_pitch_limit[None], 0.), axis=-1)
        return position_error, angle_error

    current_p, current_r = violations(position, rpy)
    candidate_p, candidate_r = violations(candidate_position, candidate_rpy)
    current_safe = jnp.maximum(current_p, current_r) <= c.tolerance
    candidate_safe = jnp.maximum(candidate_p, candidate_r) <= c.tolerance
    accepted = current_safe & candidate_safe & (projection.max_violation <= c.tolerance)
    selected = jnp.where(accepted[:, None], candidate_q, reference_q)
    velocity = (selected-reference_q)/c.dt_s
    correction = jnp.linalg.norm(velocity-nominal_qvel, axis=-1)
    retained = jnp.linalg.norm(velocity, axis=-1)/jnp.maximum(jnp.linalg.norm(nominal_qvel, axis=-1), 1e-6)
    return ProjectionResult(selected, velocity, correction, retained, correction > 1e-6,
        current_safe & ~candidate_safe, ~accepted, current_safe,
        jnp.where(accepted, candidate_p, current_p), jnp.where(accepted, candidate_r, current_r),
        projection.max_violation, jnp.where(accepted, candidate_rpy[:, 2], rpy[:, 2]))
