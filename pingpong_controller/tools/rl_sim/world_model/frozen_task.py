"""Finite, paired evaluation accounting over real substep event labels."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .course_schedule import init_schedule, finish_substep
from .task import init_task, record_release, record_apex, switch_completed_goals


class EvaluationState(NamedTuple):
    task: Any
    schedule: Any
    completed_goals: Any
    successful_goals: Any
    hits: Any
    apexes: Any
    elapsed_ms: Any
    done: Any
    success: Any
    failure_code: Any
    truncated: Any


def _select(mask, yes, no):
    return jax.tree.map(lambda a,b:jnp.where(mask.reshape(mask.shape+(1,)*(a.ndim-1)),a,b),yes,no)


def init_evaluation(goals_xyz_m, key, lane_ids):
    """Goals have shape(N,K,3); only the first target is initially published."""
    goals=jnp.asarray(goals_xyz_m,jnp.float32)
    if goals.ndim!=3 or goals.shape[-1]!=3 or goals.shape[1] not in (1,2,4):
        raise ValueError('Expected one, two or four finite evaluation goals')
    zero=jnp.zeros(goals.shape[0],jnp.int32);off=jnp.zeros_like(zero,bool)
    return EvaluationState(init_task(goals[:,0]),init_schedule(key,lane_ids),
        zero,zero,zero,zero,zero,off,off,zero,off)


def fail_evaluation(state, code):
    """Latch the first physical failure; code zero means no new failure."""
    failed=(code!=0)&~state.done
    task=state.task._replace(failed=state.task.failed|failed,
        physical_failure=state.task.physical_failure|failed,finished=state.task.finished|failed)
    return state._replace(task=task,done=state.done|failed,
        failure_code=jnp.where(failed,code,state.failure_code))


def consume_events(state,labels,goals_xyz_m,*,perturb,mass_kg=.004):
    """One1 ms tick; switch at actual apex and stop after the finite sequence.

    Completed goals with poor accuracy remain failed goals in the case, while
    physical continuity is retained to measure the rest of the fixed sequence.
    No unseen goals or scheduled disturbance values enter actor observations.
    """
    active=~state.done
    task,release=record_release(state.task,impact_id=labels['impact_id'],
        physical_release=labels['contact_release']&active,
        controlled_hit=labels['controlled_release']&active)
    task,event=record_apex(task,labels['apex_position_m'],
        jnp.linalg.norm(labels['apex_velocity_m_s'][:,:2],axis=-1),labels['actual_apex']&active)
    completed=state.completed_goals+event.completed.astype(jnp.int32)
    successful=state.successful_goals+event.success.astype(jnp.int32)
    final=event.completed&(completed==goals_xyz_m.shape[1])
    arrival_hit=task.arrival_hit
    stability_start_hit=task.stability_start_hit
    index=jnp.minimum(completed,goals_xyz_m.shape[1]-1)
    next_goal=goals_xyz_m[jnp.arange(len(index)),index]
    switched_task,ready=switch_completed_goals(task,next_goal)
    switched=ready&event.completed&~final&active
    task=_select(switched,switched_task,task)
    schedule,pulse=finish_substep(state.schedule,switched,active,
        enabled=perturb,mass_kg=mass_kg)
    elapsed=state.elapsed_ms+active.astype(jnp.int32)
    timeout=active&~final&(elapsed>=60000)
    candidate=state._replace(task=task,schedule=schedule,completed_goals=completed,
        successful_goals=successful,hits=state.hits+release.astype(jnp.int32),
        apexes=state.apexes+event.valid_apex.astype(jnp.int32),elapsed_ms=elapsed,
        done=state.done|final|timeout,success=final&(successful==goals_xyz_m.shape[1]),
        truncated=state.truncated|timeout)
    after=_select(active,candidate,state)
    return after,dict(valid_release=release,valid_apex=event.valid_apex,
        goal_completed=event.completed,goal_success=event.success,goal_switched=switched,
        final=final,pulse_start=pulse,xy_error_m=event.xy_error_m,z_error_m=event.z_error_m,
        vxy_m_s=event.vxy_m_s,arrival_hit=jnp.where(event.completed,arrival_hit,0),
        stability_start_hit=jnp.where(event.completed,stability_start_hit,0))
