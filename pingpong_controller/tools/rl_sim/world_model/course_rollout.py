"""Causal course rollouts and bounded complete-state event capture."""

from typing import Any, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

from .actor_critic import actor_return, frozen_value
from .course_env import course_step, course_critic_features
from .course_goals import draw_next_goals
from .full_prior_world import world_actor_observation
from .imagined_env import select_lanes
from .imagined_rollout import PolicyWindow, window_batch
from .migration import actor_mean
from .sampling import SamplingShortage


class CapturedSeed(NamedTuple):
    before: Any
    after: Any
    exploration_rng: Any
    seen: Any
    event_count: Any


CourseWindow=NamedTuple('CourseWindow',[(name,Any) for name in PolicyWindow._fields]+[('captures',Any)])


def _empty_capture(state,keys):
    n=state.episode_steps.shape[0]
    stopped=jax.tree.map(jax.lax.stop_gradient,state)
    return CapturedSeed(stopped,stopped,keys,jnp.zeros(n,bool),jnp.zeros(n,jnp.int32))


def _capture(existing,before,after,keys,event):
    eligible=event & ~(after.terminated|after.truncated|after.model_truncated)
    first=eligible & ~existing.seen
    before,after,keys=jax.tree.map(jax.lax.stop_gradient,(before,after,keys))
    return CapturedSeed(select_lanes(first,before,existing.before),
        select_lanes(first,after,existing.after),jnp.where(first[:,None],keys,existing.exploration_rng),
        existing.seen|eligible,existing.event_count+eligible.astype(jnp.int32))


def course_policy_window(actor,robot,auxiliary,environment,exploration_rng,goal_table,*,
                         horizon=64,exploration_std=.03,rematerialize=True,**step_options):
    """Keep one complete causal seed per category/lane, not the full state history.

    Captures are gradient-detached auxiliary outputs. Actor/critic objectives
    still consume only model-generated transitions. The caller preflights the
    cached goal graph before entering this compiled function.
    """
    n=environment.episode_steps.shape[0]
    if horizon<1 or exploration_rng.shape!=(n,2) or exploration_std<0.:
        raise ValueError('Invalid imagined course window')
    stage_id=step_options['stage_id']
    empty=_empty_capture(environment,exploration_rng)

    def tick(carry,_):
        state,keys,goal_capture,recovery_capture=carry
        split=jax.vmap(lambda k:jax.random.split(k,2))(keys)
        noise=jax.vmap(lambda k:jax.random.normal(k,(7,)))(split[:,1])
        mean=actor_mean(actor,world_actor_observation(state.world,state.task.goal_xyz))
        raw=mean+exploration_std*noise
        inputs=course_critic_features(state)
        goal,next_goal_keys,available,_=draw_next_goals(goal_table,state.task.goal_xyz,
            state.schedule.goal_rng,stage_id=stage_id)
        after,transition=course_step(robot,auxiliary,state,raw,goal,**step_options)
        switched=transition.task_events['goal_switched']>0
        # A missing graph successor is a configuration error discovered by
        # host preflight. Also surface it here; it must never silently hold.
        shortage=transition.valid & ~available
        after=after._replace(model_truncated=after.model_truncated|shortage,
            schedule=after.schedule._replace(goal_rng=jnp.where(switched[:,None],
                next_goal_keys,state.schedule.goal_rng)))
        transition=transition._replace(reward=jnp.where(shortage,0.,transition.reward),
            discount=jnp.where(shortage,0.,transition.discount),
            trace_continue=transition.trace_continue & ~shortage,
            model_truncated=transition.model_truncated|shortage,
            task_events=dict(transition.task_events,goal_sampling_shortage=shortage))
        goal_capture=_capture(goal_capture,state,after,split[:,0],switched)
        recovery_capture=_capture(recovery_capture,state,after,split[:,0],
            transition.task_events['disturbance_applied']>0)
        return (after,split[:,0],goal_capture,recovery_capture),(
            transition,inputs,mean,raw,after.world.control.previous_accepted_action)

    tick=jax.checkpoint(tick) if rematerialize else tick
    final,outputs=jax.lax.scan(tick,(environment,exploration_rng,empty,empty),None,length=horizon)
    state,keys,goal_capture,recovery_capture=final
    transitions,inputs,mean,raw,accepted=outputs
    return CourseWindow(state,keys,transitions,inputs,mean,raw,accepted,
        dict(goal_switch=goal_capture,recoverable_disturbance=recovery_capture))


def course_actor_objective(actor,target,robot,auxiliary,environment,keys,goal_table,**options):
    window=course_policy_window(actor,robot,auxiliary,environment,keys,goal_table,**options)
    values=frozen_value(target,window.transitions.next_critic_features)
    return -jnp.mean(actor_return(window_batch(window),values)),window


def record_captured_seeds(bank,window,*,split,source_id):
    """Populate a host bank using the actual event state, not the window endpoint."""
    validate_course_window(window)
    added={}
    for category,capture in window.captures.items():
        marker='goal_switched' if category=='goal_switch' else 'disturbance_applied'
        added[category]=bank.observe(capture.before,capture.after,capture.exploration_rng,
            {marker:capture.seen},split=split,source_id=source_id+'/'+category)
    return added


def validate_course_window(window):
    """Reject graph shortages before learning or committing the persistent pool."""
    if np.any(np.asarray(window.transitions.task_events['goal_sampling_shortage'])):
        raise SamplingShortage('Course window has missing goal successors; no commit permitted')
