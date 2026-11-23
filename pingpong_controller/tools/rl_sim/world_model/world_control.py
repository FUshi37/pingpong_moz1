"""Known WMJ control memory: clip, reference projection, native RMP and delays.

The output is five delayed1 ms commands for the existing execution proxy.
No learned model, physics solver or mesh/IK scan is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from .control_projection import ProjectionContract, project_reference
from .control_rmp import apply_output_delay, promote_rmp_indices, rmp_step_lanes


RMP_CONTEXT_FIELDS = ('jnt_kp', 'jnt_kd', 'estimator_joint_process_var',
    'estimator_joint_measure_var', 'joint_velocity_feedforward',
    'joint_acceleration_feedforward', 'acceleration_weight', 'target_filter_length')


class ControlState(NamedTuple):
    reference_q_rad: Any
    previous_accepted_action: Any
    rmp: Any
    output_delay: dict


class ControlContext(NamedTuple):
    rmp: dict
    action_scale_mult: Any
    output_delay_steps: Any


@dataclass(frozen=True)
class ControlContract:
    projection: ProjectionContract
    rmp_params: Any
    acceleration_limit_rad_s2: Any
    jerk_limit_rad_s3: Any
    action_velocity_scale: float

    @classmethod
    def from_native(cls, env):
        """Reject unsupported control paths instead of silently omitting them."""
        expected = dict(action_acc_limit=1., action_filter_tau_ms=0., action_jerk_limit=0.,
            enable_anti_windup=False, arm_action_limiter=True, action_command_mode='velocity',
            policy_integration_feedback_source='measured', enable_delay_conditioning=False,
            dr_randomize_latency=False, dr_rmp_target_update_hold_probability=0.,
            dr_rmp_target_update_hold_tail_probability=0., dr_execution_command_hold_probability=0.,
            dr_execution_command_hold_tail_probability=0., recovered_rmp_use_safety_bounds=True,
            recovered_rmp_use_jerk_limit=False)
        if any(getattr(env.cfg, key) != value for key, value in expected.items()):
            raise ValueError('Unsupported native preprocessing or additional hold/filter path')
        if (tuple(env.cfg.dr_action_latency_steps_range) != (0, 0)
                or not env.recovered_rmp_motion_mode or not env.recovered_rmp_direct_qvel_target
                or not env.recovered_rmp_bounded_qvel_reference or not env.cartesian_pose_projection_enabled
                or env.recovered_rmp_servo_steps != 5 or env.recovered_rmp_target_update_delay_steps != 3
                or abs(env.recovered_rmp_period_s-.001) > 1e-10):
            raise ValueError('Expected native bounded-QVEL projection and five1 ms RMP ticks')
        return cls(ProjectionContract.from_native(env), env.recovered_rmp_params,
            env.arm_acc_limit_rad_s2, env.recovered_rmp_jerk_limit_rad_s3,
            float(env.cfg.action_velocity_scale))


def control_from_native(native) -> tuple[ControlState, ControlContext]:
    """Restore full native controller memory; this is not a controller reset.

    Call after launch/setup. Geometry/seed admission remains an offline step.
    Episode-specific RMP parameters are retained, not inferred from filenames.
    """
    if np.any(np.asarray(native.action_latency_steps) != 0):
        raise ValueError('Additional native action latency is outside this control contract')
    buffers = dict(q=native.rmp_output_q_delay_buffer, dq=native.rmp_output_qd_delay_buffer,
                   ddq=native.rmp_output_qdd_delay_buffer)
    delays = np.asarray(native.rmp_output_delay_steps)
    n = native.arm_cmd_q.shape[0]
    if (any(value.ndim != 3 or value.shape[0] != n or value.shape[-1] != 7
            or value.shape != buffers['q'].shape for value in buffers.values())
            or delays.shape != (n, 7) or np.any(delays < 0) or np.any(delays > buffers['q'].shape[1])):
        raise ValueError('Invalid native output-delay seed')
    context = ControlContext({key:getattr(native, 'rmp_'+key) for key in RMP_CONTEXT_FIELDS},
                              native.action_scale_mult, native.rmp_output_delay_steps)
    state = ControlState(native.arm_cmd_q, native.prev_action,
                          promote_rmp_indices(native.rmp_state), buffers)
    return state, context


def control_step(state: ControlState, raw_action, measured_q_rad, context: ControlContext,
                 *, chain, contract: ControlContract):
    """Advance one200 Hz policy action using own measured/predicted13-joint state."""
    if (raw_action.shape != state.reference_q_rad.shape or raw_action.ndim != 2
            or raw_action.shape[-1] != 7 or context.action_scale_mult.shape != raw_action.shape[:1]
            or context.output_delay_steps.shape != raw_action.shape
            or set(state.output_delay) != {'q', 'dq', 'ddq'}):
        raise ValueError('Expected seven actions/references and matching per-lane control context')
    c = contract.projection
    clipped = jnp.clip(raw_action, -1., 1.)
    denominator = c.velocity_limit[None]*contract.action_velocity_scale*context.action_scale_mult[:, None]
    requested = jnp.clip(clipped*denominator, -c.velocity_limit[None], c.velocity_limit[None])
    projection = project_reference(chain, c, measured_q_rad, state.reference_q_rad, requested)
    accepted = jnp.clip(projection.accepted_qvel/jnp.maximum(denominator, 1e-6), -1., 1.)
    rmp, commands = rmp_step_lanes(state.rmp, projection.reference_q, context.rmp,
        q_min=c.q_lower, q_max=c.q_upper, velocity_limit=c.velocity_limit,
        acceleration_limit=contract.acceleration_limit_rad_s2,
        jerk_limit=contract.jerk_limit_rad_s3, params=contract.rmp_params)
    delayed, buffers = {}, {}
    for key in ('q', 'dq', 'ddq'):
        delayed[key], buffers[key] = apply_output_delay(state.output_delay[key], commands[key],
                                                         context.output_delay_steps)
    after = ControlState(projection.reference_q, accepted, rmp, buffers)
    return after, dict(raw_action=raw_action, clipped_action=clipped, accepted_action=accepted,
        requested_qvel_rad_s=requested, projection=projection,
        rmp_commands=commands, delayed_commands=delayed)
