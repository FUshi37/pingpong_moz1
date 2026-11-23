"""Explicit new-trial setup from an actually warmed V324 control chain.

These functions are initialization operations, never rollout transitions or
goal switches. A new ball invalidates old camera history. The warmed robot,
q_ref, filters, delay queues and shared queue phases remain together.
"""

from __future__ import annotations

import jax.numpy as jnp

from .observation_engine import init_camera


CONTROL_FIELDS = frozenset((
    'prev_action', 'actor_previous_action_scale', 'prev_arm_qvel',
    'prev_racket_pos', 'prev_racket_vel', 'action_buffer', 'action_latency_steps',
    'command_buffer', 'actuator_command_buffer', 'tau_act_episode', 'tau_act',
    'delay_steps', 'delay_bin_id', 'anti_windup_scale', 'action_scale_mult',
    'actuator_cmd_tau', 'actuator_cmd_gain', 'second_order_frequency_scale',
    'second_order_damping_scale', 'second_order_gain_scale',
    'second_order_delay_offset_steps', 'rng', 'total_env_steps',
))
CONTROL_PREFIXES = ('arm_', 'rmp_', 'compensation_', 'execution_', 'dr_')


def carry_warmed_controller(fresh, warmed):
    """Keep the whole controller tree, including its scalar queue phases.

fresh must be the original reset for this same population/context. Its task
and sensor bookkeeping is reset; no control state is independently sampled.
The nested rmp_state is retained wholesale rather than sliced by lane.
"""
    if type(fresh) is not type(warmed):
        raise ValueError('Fresh and warmed native state layouts differ')
    return fresh._replace(**{name: getattr(warmed, name) for name in fresh._fields
        if name in CONTROL_FIELDS or name.startswith(CONTROL_PREFIXES)})


def place_ball(environment, state, position_m, velocity_m_s):
    """Place a new unspun ball, then recompute native contact/kinematics caches."""
    env = environment
    position_m = jnp.asarray(position_m, jnp.float32)
    velocity_m_s = jnp.asarray(velocity_m_s, jnp.float32)
    expected = (state.step_count.shape[0], 3)
    if position_m.shape != expected or velocity_m_s.shape != expected:
        raise ValueError('New ball position and velocity must have shape (N,3)')
    qpos = state.data.qpos.at[:, env.ball_qadr:env.ball_qadr+3].set(position_m)
    qpos = qpos.at[:, env.ball_qadr+3:env.ball_qadr+7].set(jnp.array([1., 0., 0., 0.]))
    qvel = state.data.qvel.at[:, env.ball_vadr:env.ball_vadr+3].set(velocity_m_s)
    qvel = qvel.at[:, env.ball_vadr+3:env.ball_vadr+6].set(0.)
    data = state.data.replace(qpos=qpos, qvel=qvel,
        qacc_warmstart=state.data.qacc_warmstart.at[:, env.ball_vadr:env.ball_vadr+6].set(0.),
        xfrc_applied=state.data.xfrc_applied.at[:, env.ball_body_id].set(0.))
    data = env.batched_forward(state.model, data)
    return state._replace(data=data, reset_ball_pos=position_m,
        reset_ball_vel=velocity_m_s, reset_ball_spin_rad_s=jnp.zeros_like(position_m),
        prev_ball_pos=position_m, racket_launch_hold_steps=jnp.zeros_like(state.step_count),
        reset_falling_contact_active=jnp.ones_like(state.step_count, bool))


def prepare_warmup(environment, fresh, admitted_q_rad):
    """Start a cached admitted pose, with the ball away from the robot.

Only a genuine reset can enter this setup. The caller then runs ordinary
env.step actions for warmup. No mesh/IK query or ideal actuator replacement
is used here. The temporary remote ball and warmup are not task samples.
"""
    state = environment.reset_arm_replay_state(fresh, admitted_q_rad)
    n = state.step_count.shape[0]
    return place_ball(environment, state, jnp.broadcast_to(jnp.array([0., -3., 3.]), (n, 3)),
                      jnp.zeros((n, 3), jnp.float32))


def camera_cache_fields(camera, velocity_m_s):
    """Native valid_pos/valid_vel store last valid vectors, not validity bits."""
    return dict(step_count=camera.step_count, last_ball_obs_step=camera.last_sample_step,
        cached_ball_obs_pos=camera.held_position_m, cached_ball_obs_vel=velocity_m_s,
        actor_ball_position_history=camera.history_position_m,
        actor_ball_age_history=camera.history_age_s,
        ball_position_noise_epsilon_m=camera.epsilon_m,
        fixed_contact_ball_observed=camera.observed,
        fixed_contact_ball_predicted=jnp.zeros_like(camera.observed),
        fixed_contact_ball_state_valid=camera.observed,
        ball_obs_valid_pos=camera.held_position_m, ball_obs_valid_vel=velocity_m_s)


def begin_ball_trial(environment, fresh, warmed, position_m, velocity_m_s,
                     *, ar1_enabled: bool):
    """Begin a new physical ball trial while preserving actual warmed controls.

Camera and task clocks restart for this declared episode boundary. The scalar
RMP filter indices keep their warmed phases. Exact forks inside a trial must
restore a full seed instead of calling this function.
"""
    state = carry_warmed_controller(fresh, warmed)._replace(model=warmed.model, data=warmed.data)
    state = place_ball(environment, state, position_m, velocity_m_s)
    camera = init_camera(position_m, warmed.rng, ar1_enabled=ar1_enabled)
    state = state._replace(
        data=state.data.replace(time=jnp.zeros_like(state.data.time)),
        **camera_cache_fields(camera, velocity_m_s),
        ball_obs_age_seconds=jnp.zeros_like(state.ball_obs_age_seconds),
        racket_anchor=state.data.site_xpos[:, environment.racket_site_id])
    # Only the current racket pose is real; legacy camera padding uses its
    # registered age=1 s, while the new paired history is initialized separately.
    from fixed_plane_v92 import rotation_matrix_to_6d
    pose = jnp.concatenate((state.data.site_xpos[:, environment.racket_site_id],
        rotation_matrix_to_6d(state.data.site_xmat[:, environment.racket_site_id].reshape(-1, 3, 3))), -1)
    return state._replace(actor_racket_pose_history=jnp.broadcast_to(
        pose[:, None, :], state.actor_racket_pose_history.shape))
