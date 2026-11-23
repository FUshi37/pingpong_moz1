"""Known, causal camera module reused by learned and teacher worlds.

No hidden physical ball velocity, future contact or model uncertainty enters
the actor. Position input is sensor truth before measurement noise; the
result is kept apart from physical state. Missing-vision augmentation is off.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from ball_position_noise_ar1 import advance_noise, initialize_noise
from fixed_plane_v149 import make_simple_position_observation

from .paired_history import (
    FeedbackHistory, PairedHistory, append_ball, append_feedback, encode_history,
    init_feedback, init_history, reset_selected,
)

MEAN_M = (.000048, .000072, -.000227)
STD_M = (.001451, .001418, .002970)
RHO_AT_90HZ = (.935, .907, .790)


class CameraState(NamedTuple):
    rng: Any
    step_count: Any
    last_sample_step: Any
    held_position_m: Any
    history_position_m: Any
    history_age_s: Any
    epsilon_m: Any
    observed: Any


class ObservationHistory(NamedTuple):
    feedback: FeedbackHistory
    paired: PairedHistory


class WorldObservationState(NamedTuple):
    """Complete observation memory, carried across imagined control steps."""

    camera: CameraState
    history: ObservationHistory


def init_camera(position_m, native_next_keys, *, ar1_enabled: bool) -> CameraState:
    """Match V324 initialization from the simulator's post-reset RNG keys."""
    n = position_m.shape[0]
    keys = jax.vmap(lambda k: jax.random.fold_in(k, 25401))(native_next_keys)
    epsilon = (initialize_noise(keys, std_m=STD_M) if ar1_enabled
               else jnp.zeros((n, 3), jnp.float32))
    observed = (position_m + jnp.asarray(MEAN_M) + epsilon
                if ar1_enabled else position_m)
    zero = jnp.zeros(n, jnp.int32)
    return CameraState(native_next_keys, zero, zero, observed,
        jnp.broadcast_to(observed[:, None, :], (n, 8, 3)),
        jnp.ones((n, 8), jnp.float32).at[:, 0].set(0.),
        epsilon, jnp.ones(n, bool))


def advance_camera(state: CameraState, true_position_m, *, ar1_enabled: bool):
    """Advance one 5 ms tick using V324's exact split-11 RNG and floor clock."""
    keys = jax.vmap(lambda key: jax.random.split(key, 11))(state.rng)
    step = state.step_count + 1
    ratio = 90. * .005
    fresh = (jnp.floor(step.astype(jnp.float32) * ratio)
             > jnp.floor(state.last_sample_step.astype(jnp.float32) * ratio))
    epsilon = state.epsilon_m
    if ar1_enabled:
        epsilon = advance_noise(epsilon, keys[:, 1],
            dt_s=(step - state.last_sample_step).astype(jnp.float32) * .005,
            refresh=fresh, std_m=STD_M, rho_at_90hz=RHO_AT_90HZ)
        sampled = true_position_m + (jnp.asarray(MEAN_M) + epsilon)
    else:
        sampled = true_position_m
    held = jnp.where(fresh[:, None], sampled, state.held_position_m)
    last = jnp.where(fresh, step, state.last_sample_step)
    age_s = (step - last).astype(jnp.float32) * .005
    return CameraState(keys[:, 0], step, last, held,
        jnp.concatenate((held[:, None, :], state.history_position_m[:, :-1]), axis=1),
        jnp.concatenate((age_s[:, None], state.history_age_s[:, :-1]), axis=1),
        epsilon, state.observed), fresh


def advance_observation_history(history: ObservationHistory, q_rad,
                                observed_ball_position_m, sample_time_us, now_us,
                                observed, *, pose_from_q: Callable,
                                physical_reset_mask=None):
    """Carry old samples forward and append only a new ball-source timestamp.

    q_rad is measured/predicted encoder state at now_us: RightArm-0..6, with
    LegWaist-0..5 appended in the full-chain thirteen-joint mode.
    pose_from_q maps that state to site XYZ and rotation6D (N,9). It must use
    the matching fixed-base calibration. Cached simulator site fields can
    precede integration, so they are not accepted as same-time encoder FK.
    The native teacher and imagined world share this exact buffer operation.
    """
    reset = (jnp.zeros_like(now_us, bool) if physical_reset_mask is None
             else physical_reset_mask)
    feedback = reset_selected(history.feedback, reset)
    paired = reset_selected(history.paired, reset)
    feedback, accepted = append_feedback(feedback, q_rad, now_us, now_us,
                                         racket_pose=pose_from_q(q_rad))
    paired, diagnostics = append_ball(paired, feedback, observed_ball_position_m,
        sample_time_us, now_us, sample_time_us > paired.last_source_time_us,
        observed, pose_from_q)
    diagnostics['feedback_accepted'] = accepted
    return ObservationHistory(feedback, paired), diagnostics


def init_world_observation(ball_position_m, q_rad, native_next_keys, *,
                           pose_from_q: Callable, ar1_enabled: bool):
    """Initialize at physical reset, with one real sample and masked padding.

    A mid-episode imagination branch must restore its complete camera/history
    seed instead. This function never fabricates an earlier 640 ms of data.
    """
    camera = init_camera(ball_position_m, native_next_keys, ar1_enabled=ar1_enabled)
    n = ball_position_m.shape[0]
    empty = ObservationHistory(init_feedback(n, n_joints=q_rad.shape[-1]), init_history(n))
    history, diagnostics = advance_observation_history(empty, q_rad,
        camera.held_position_m, camera.last_sample_step*5000, camera.step_count*5000,
        camera.observed, pose_from_q=pose_from_q)
    return WorldObservationState(camera, history), diagnostics


def advance_world_observation(state: WorldObservationState, next_q_rad,
                              next_ball_position_m, *, pose_from_q: Callable,
                              ar1_enabled: bool):
    """Advance one imagined 5 ms tick using predicted physical endpoints only.

    Physics predicts the next q/ball state, never the flattened 969-D history.
    The known camera creates a fresh 90 Hz sample or holds its existing value;
    only fresh samples append paired long-history rows. Old physical samples
    and their realized noise stay fixed. Age/masking are evaluated at query
    time. No future MJX observations or teacher-forcing corrections enter here.
    Gradients through model-generated samples survive shifts and holds.
    """
    camera, fresh = advance_camera(state.camera, next_ball_position_m,
                                    ar1_enabled=ar1_enabled)
    history, diagnostics = advance_observation_history(state.history, next_q_rad,
        camera.held_position_m, camera.last_sample_step*5000, camera.step_count*5000,
        camera.observed, pose_from_q=pose_from_q)
    diagnostics['camera_fresh'] = fresh
    return WorldObservationState(camera, history), diagnostics


def reset_world_observation(state: WorldObservationState,
                            reset_seed: WorldObservationState, reset_mask):
    """Restore complete seed lanes on physical reset, cutting reset gradients.

    This also supports warmed seeds with existing valid history; target changes
    must never call it. The caller restores the matching physics/control state.
    """
    if reset_mask.shape != state.camera.step_count.shape:
        raise ValueError('Physical reset mask must have one entry per environment')
    if jax.tree.structure(state) != jax.tree.structure(reset_seed):
        raise ValueError('Observation reset seed structure mismatch')

    def select(old, new):
        if old.shape != new.shape or old.dtype != new.dtype:
            raise ValueError('Observation reset seed leaf shape/dtype mismatch')
        return jnp.where(reset_mask.reshape(reset_mask.shape+(1,)*(old.ndim-1)),
                         jax.lax.stop_gradient(new), old)

    return jax.tree.map(select, state, reset_seed)


def actor_observation(q, dq, q_ref, previous_executed_action, racket_pose,
                      camera: CameraState, task_goal_xyz):
    """Build (N,73); seven-joint ordering is RightArm-0 through RightArm-6."""
    prefix = make_simple_position_observation(
        q=q, dq=dq, q_ref_error=q_ref-q, previous_action=previous_executed_action,
        ball_positions=camera.history_position_m, racket_pose_now=racket_pose,
        ball_ages=camera.history_age_s, target_xy_m=task_goal_xyz[:, :2],
        ball_position_valid=camera.observed)
    height = (task_goal_xyz[:, 2:3] - 1.425) / .10
    return jnp.concatenate((prefix, height), axis=-1).astype(jnp.float32)


def actor_observation_with_history(q, dq, q_ref, previous_executed_action,
                                   racket_pose, camera, task_goal_xyz, history, now_us):
    """Build the 969-D WMJ-V1 actor input from current values and paired samples."""
    prefix = actor_observation(q, dq, q_ref, previous_executed_action, racket_pose,
                               camera, task_goal_xyz)
    return jnp.concatenate((prefix, encode_history(history, now_us)), axis=-1)
