"""Batched goal accounting independent of robot, controller and camera state.

Physical release latches a goal ID. Only its actual subsequent flight apex
can score that contact. Initialization launches, duplicate release reports and
apexes without an attributed contact cannot earn success. All lengths are SI.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class TaskState(NamedTuple):
    goal_xyz: Any
    goal_id: Any
    hit_count: Any
    apex_count: Any
    last_impact_id: Any
    flight_goal_xyz: Any
    flight_goal_id: Any
    flight_hit_index: Any
    awaiting_apex: Any
    arrival_hit: Any
    stable_streak: Any
    max_stable_streak: Any
    stability_start_hit: Any
    stability_qualified: Any
    failed: Any
    physical_failure: Any
    finished: Any


class TaskEvent(NamedTuple):
    completed: Any
    success: Any
    valid_apex: Any
    xy_error_m: Any
    z_error_m: Any
    vxy_m_s: Any


def init_task(goal_xyz) -> TaskState:
    """Initialize task bookkeeping only; physical reset is a separate operation."""
    goal = jnp.asarray(goal_xyz, jnp.float32)
    if goal.ndim != 2 or goal.shape[-1] != 3:
        raise ValueError('Expected goal_xyz with shape (N, 3)')
    shape = goal.shape[:-1]
    zero = jnp.zeros(shape, jnp.int32)
    off = jnp.zeros(shape, bool)
    return TaskState(goal, zero, zero, zero, zero - 1, goal, zero, zero,
                     off, zero, zero, zero, zero, off, off, off, off)


def record_release(
    state: TaskState, *, impact_id, physical_release, controlled_hit,
    initialization_launch=None,
) -> tuple[TaskState, Any]:
    """Latch each uniquely identified controlled contact's current task target.

    ``controlled_hit`` must come from the physical contact/outgoing-flight
    validator, never from a delayed reward-confirmation counter. Contact IDs
    increase throughout an episode, including ignored initialization contacts.
    """
    launch = False if initialization_launch is None else initialization_launch
    unique = physical_release & (impact_id > state.last_impact_id)
    accepted = unique & controlled_hit & ~jnp.asarray(launch) & ~state.finished
    hit_count = state.hit_count + accepted.astype(jnp.int32)
    # A new valid impact before the previous flight produced an apex is not a
    # missing label to be silently dropped from the success denominator.
    interrupted_flight = unique & state.awaiting_apex
    after = state._replace(
        last_impact_id=jnp.where(unique, impact_id, state.last_impact_id),
        hit_count=hit_count,
        flight_goal_xyz=jnp.where(accepted[:, None], state.goal_xyz, state.flight_goal_xyz),
        flight_goal_id=jnp.where(accepted, state.goal_id, state.flight_goal_id),
        flight_hit_index=jnp.where(accepted, hit_count, state.flight_hit_index),
        awaiting_apex=jnp.where(unique, accepted, state.awaiting_apex),
        failed=state.failed | interrupted_flight,
    )
    return after, accepted


def record_apex(
    state: TaskState, apex_xyz, apex_vxy_m_s, physical_apex,
    *, hits_per_goal: int = 16, latest_start_hit: int = 3,
    stable_apexes: int = 10, xy_tolerance_m: float = .02,
    z_tolerance_m: float = .02, vxy_limit_m_s: float = .05,
) -> tuple[TaskState, TaskEvent]:
    """Score a real trajectory apex, never a model's predicted next apex label."""
    valid = physical_apex & state.awaiting_apex & ~state.finished
    same_goal = state.flight_goal_id == state.goal_id
    delta = apex_xyz - state.flight_goal_xyz
    xy_error = jnp.linalg.norm(delta[:, :2], axis=-1)
    z_error = jnp.abs(delta[:, 2])
    finite = jnp.all(jnp.isfinite(apex_xyz), axis=-1) & jnp.isfinite(apex_vxy_m_s)
    good = (finite & same_goal & (xy_error <= xy_tolerance_m)
            & (z_error <= z_tolerance_m) & (apex_vxy_m_s >= 0.)
            & (apex_vxy_m_s <= vxy_limit_m_s))
    streak = jnp.where(valid, jnp.where(good, state.stable_streak + 1, 0),
                       state.stable_streak)
    run_start = state.flight_hit_index - streak + 1
    qualified_now = valid & good & (streak >= stable_apexes) & (run_start <= latest_start_hit)
    qualified = state.stability_qualified | qualified_now
    failed = state.failed | (valid & (~finite | ~same_goal))
    count = state.apex_count + valid.astype(jnp.int32)
    completed = valid & (state.flight_hit_index >= hits_per_goal)
    after = state._replace(
        apex_count=count,
        awaiting_apex=state.awaiting_apex & ~valid,
        arrival_hit=jnp.where(valid & good & (state.arrival_hit == 0),
                              state.flight_hit_index, state.arrival_hit),
        stable_streak=streak,
        max_stable_streak=jnp.maximum(streak, state.max_stable_streak),
        stability_start_hit=jnp.where(qualified_now & ~state.stability_qualified,
                                     run_start, state.stability_start_hit),
        stability_qualified=qualified,
        failed=failed,
        finished=state.finished | completed,
    )
    event = TaskEvent(completed, completed & qualified & ~failed & (count == hits_per_goal),
                      valid, jnp.where(valid, xy_error, 0.), jnp.where(valid, z_error, 0.),
                      jnp.where(valid, apex_vxy_m_s, 0.))
    return after, event


def record_failure(state: TaskState, physical_failure) -> tuple[TaskState, TaskEvent]:
    """Include dropped/numerically failed episodes in the original task denominator."""
    new_failure = physical_failure & ~state.finished
    zeros = jnp.zeros_like(state.goal_id, jnp.float32)
    off = jnp.zeros_like(new_failure)
    after = state._replace(failed=state.failed | new_failure,
                           physical_failure=state.physical_failure | new_failure,
                           finished=state.finished | new_failure)
    return after, TaskEvent(new_failure, off, off, zeros, zeros, zeros)


def switch_completed_goals(state: TaskState, next_goal_xyz) -> tuple[TaskState, Any]:
    """Publish after the final apex, preserving every external physical state.

    Only this task object changes. A failed physical episode cannot publish a
    new goal instead of undergoing its separately accounted physical reset.
    """
    ready = state.finished & ~state.awaiting_apex & ~state.physical_failure
    fresh = init_task(next_goal_xyz)._replace(
        goal_id=state.goal_id + 1,
        flight_goal_id=state.goal_id + 1,
        last_impact_id=state.last_impact_id,
    )
    after = jax.tree.map(lambda old, new: jnp.where(
        ready.reshape(ready.shape + (1,) * (old.ndim - ready.ndim)), new, old),
        state, fresh)
    return after, ready
