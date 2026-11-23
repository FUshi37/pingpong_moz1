"""640 ms of timestamp-paired ball/racket samples, with no dense actor history.

Actor rows are newest first: ball XYZ (3), matching racket site XYZ (3),
rotation columns (6), age/0.640 (1), valid (1). Sixty-four slots accommodate
the roughly 58 fresh 90 Hz samples; padding and expired/unpaired rows are zero.
The internal 200 Hz encoder cache is used only for causal interpolation/FK.
It is never flattened into actor input. Times are episode-relative integer
microseconds, in one shared clock domain, not receipt times or Unix epochs.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

WINDOW_US = 640000
FEEDBACK_SLOTS = 129
BALL_SLOTS = 64
HISTORY_ROW_DIM = 14
HISTORY_INPUT_DIM = BALL_SLOTS * HISTORY_ROW_DIM
ACTOR_INPUT_DIM = 73 + HISTORY_INPUT_DIM
MAX_FEEDBACK_GAP_US = 5000


class FeedbackHistory(NamedTuple):
    sample_time_us: Any
    q_rad: Any
    valid: Any
    racket_pose: Any
    pose_valid: Any


class PairedHistory(NamedTuple):
    sample_time_us: Any
    received_time_us: Any
    ball_position_m: Any
    racket_pose: Any
    ball_valid: Any
    aligned: Any
    present: Any
    last_source_time_us: Any


def init_feedback(n_envs: int, *, n_joints: int = 7) -> FeedbackHistory:
    """Internal arm7 or arm7+support6 cache; neither is flattened for the actor."""
    if n_joints not in (7, 13):
        raise ValueError('Encoder history requires the declared seven/thirteen-joint order')
    return FeedbackHistory(jnp.full((n_envs, FEEDBACK_SLOTS), -1, jnp.int32),
        jnp.zeros((n_envs, FEEDBACK_SLOTS, n_joints), jnp.float32),
        jnp.zeros((n_envs, FEEDBACK_SLOTS), bool),
        jnp.zeros((n_envs, FEEDBACK_SLOTS, 9), jnp.float32),
        jnp.zeros((n_envs, FEEDBACK_SLOTS), bool))


def init_history(n_envs: int) -> PairedHistory:
    off = jnp.zeros((n_envs, BALL_SLOTS), bool)
    times = jnp.full((n_envs, BALL_SLOTS), -1, jnp.int32)
    return PairedHistory(times, times, jnp.zeros((n_envs, BALL_SLOTS, 3), jnp.float32),
        jnp.zeros((n_envs, BALL_SLOTS, 9), jnp.float32), off, off, off,
        jnp.full((n_envs,), -1, jnp.int32))


def _append(array, value, selected):
    shifted = jnp.concatenate((value[:, None], array[:, :-1]), axis=1)
    return jnp.where(selected.reshape(selected.shape+(1,)*(array.ndim-1)), shifted, array)


def append_feedback(state: FeedbackHistory, q_rad, sample_time_us, now_us, *, racket_pose=None):
    """Accept only arrived, finite, strictly ordered measured encoder packets."""
    if q_rad.shape != (state.valid.shape[0], state.q_rad.shape[-1]):
        raise ValueError('Measured q must match the declared encoder history shape/order')
    accepted = ((sample_time_us >= 0) & (sample_time_us <= now_us)
        & (sample_time_us > state.sample_time_us[:, 0]) & jnp.all(jnp.isfinite(q_rad), axis=-1))
    pose = jnp.zeros((q_rad.shape[0], 9), jnp.float32) if racket_pose is None else racket_pose
    if pose.shape != (q_rad.shape[0], 9):
        raise ValueError('Expected measured FK racket pose, shape (N,9)')
    pose_valid = jnp.all(jnp.isfinite(pose), axis=-1) & (racket_pose is not None)
    return FeedbackHistory(_append(state.sample_time_us, sample_time_us, accepted),
        _append(state.q_rad, jnp.nan_to_num(q_rad), accepted),
        _append(state.valid, jnp.ones_like(accepted), accepted),
        _append(state.racket_pose, jnp.nan_to_num(pose), accepted),
        _append(state.pose_valid, pose_valid, accepted)), accepted


def _aligned_pose(feedback: FeedbackHistory, sample_time_us, now_us, pose_from_q: Callable):
    times = feedback.sample_time_us
    usable = feedback.valid & (times <= now_us[:, None])
    lower = usable & (times <= sample_time_us[:, None])
    upper = usable & (times >= sample_time_us[:, None])
    lower_index = jnp.argmax(jnp.where(lower, times, -2147483647), axis=-1)
    upper_index = jnp.argmin(jnp.where(upper, times, 2147483647), axis=-1)
    lanes = jnp.arange(times.shape[0])
    lo, hi = times[lanes, lower_index], times[lanes, upper_index]
    span = hi-lo
    weight = jnp.clip((sample_time_us-lo).astype(jnp.float32)/jnp.maximum(span, 1), 0., 1.)
    qlo, qhi = feedback.q_rad[lanes, lower_index], feedback.q_rad[lanes, upper_index]
    # Interpolate measured joint coordinates, then FK. Do not average 6D
    # rotations or match a stale ball sample to the newest racket pose.
    pose = pose_from_q(qlo + weight[:, None]*(qhi-qlo))
    exact_pose = (span == 0) & feedback.pose_valid[lanes, lower_index]
    pose = jnp.where(exact_pose[:, None], feedback.racket_pose[lanes, lower_index], pose)
    if pose.shape != (times.shape[0], 9):
        raise ValueError('pose_from_q must return site XYZ plus rotation columns, shape (N,9)')
    valid = (jnp.any(lower, axis=-1) & jnp.any(upper, axis=-1)
        & (span >= 0) & (span <= MAX_FEEDBACK_GAP_US)
        & (sample_time_us >= 0) & (sample_time_us <= now_us)
        & jnp.all(jnp.isfinite(pose), axis=-1))
    return jnp.where(valid[:, None], pose, 0.), valid


def realign_pending(history: PairedHistory, feedback: FeedbackHistory, now_us, pose_from_q: Callable):
    """Resolve pending samples only once bracketing feedback has actually arrived."""
    age = now_us[:, None]-history.sample_time_us
    pending = history.present & history.ball_valid & ~history.aligned & (age >= 0) & (age <= WINDOW_US)

    def resolve(current):
        poses, matched = jax.vmap(lambda source: _aligned_pose(
            feedback, source, now_us, pose_from_q), in_axes=1, out_axes=1)(current.sample_time_us)
        take = pending & matched
        return current._replace(racket_pose=jnp.where(take[..., None], poses, current.racket_pose),
                                aligned=current.aligned | take)

    return jax.lax.cond(jnp.any(pending), resolve, lambda current: current, history)


def append_ball(history: PairedHistory, feedback: FeedbackHistory, ball_position_m,
                sample_time_us, now_us, fresh, ball_valid, pose_from_q: Callable):
    """Append a fresh source sample once; holds/duplicates do not consume slots."""
    n = history.present.shape[0]
    if ball_position_m.shape != (n, 3):
        raise ValueError('Expected observed ball XYZ, shape (N,3)')
    history = realign_pending(history, feedback, now_us, pose_from_q)
    time_valid = (sample_time_us >= 0) & (sample_time_us <= now_us)
    age_valid = (now_us-sample_time_us) <= WINDOW_US
    new = fresh & time_valid & age_valid & (sample_time_us > history.last_source_time_us)
    capacity_exceeded = new & history.present[:, -1] & (
        now_us-history.sample_time_us[:, -1] <= WINDOW_US)
    new &= ~capacity_exceeded
    observed = ball_valid & jnp.all(jnp.isfinite(ball_position_m), axis=-1)
    pose, paired = _aligned_pose(feedback, sample_time_us, now_us, pose_from_q)
    values = (sample_time_us, now_us, jnp.where(observed[:, None], ball_position_m, 0.),
              pose, observed, paired, jnp.ones_like(new))
    updated = [_append(old, value, new) for old, value in zip(history[:-1], values)]
    updated.append(jnp.where(new, sample_time_us, history.last_source_time_us))
    report = dict(inserted=new, paired=new & paired & observed,
        capacity_exceeded=capacity_exceeded,
        pending=new & ~paired & observed,
        invalid_measurement=fresh & ~observed,
        invalid_timestamp=fresh & (~time_valid | ~age_valid),
        duplicate=fresh & (sample_time_us == history.last_source_time_us),
        out_of_order=fresh & (sample_time_us < history.last_source_time_us))
    return PairedHistory(*updated), report


def encode_history(history: PairedHistory, now_us):
    """Emit only valid same-time pairs, retaining actual source age within 640 ms."""
    age_us = now_us[:, None]-history.sample_time_us
    valid = (history.present & history.ball_valid & history.aligned
             & (age_us >= 0) & (age_us <= WINDOW_US)
             & (history.received_time_us <= now_us[:, None]))
    rows = jnp.concatenate((history.ball_position_m, history.racket_pose,
        (age_us.astype(jnp.float32)/WINDOW_US)[..., None], valid[..., None]), axis=-1)
    rows = jnp.where(valid[..., None], rows, 0.)
    return rows.reshape(rows.shape[0], HISTORY_INPUT_DIM).astype(jnp.float32)


def reset_selected(history: PairedHistory | FeedbackHistory, reset_mask):
    """Clear either buffer only for physical resets, never for target changes."""
    empty = (init_feedback(history.valid.shape[0], n_joints=history.q_rad.shape[-1]) if isinstance(history, FeedbackHistory)
             else init_history(history.present.shape[0]))
    return jax.tree.map(lambda old, new: jnp.where(reset_mask.reshape(
        reset_mask.shape+(1,)*(old.ndim-1)), new, old), history, empty)
