"""Opt-in course environment with explicit current pulse/sequence memory.

The legacy2596-feature diagnostic environment remains unchanged. This state
adds the approved six visible values and private task RNG, with the same969-D
actor, learned weights, V324 control chain and residual-free contact model.
"""

from typing import Any, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

from .action_effects import model_event_trace
from .course_schedule import init_schedule, finish_substep, schedule_features
from .critic_features import critic_features, critic_layout
from .external_force import forced_physics_step
from .full_prior_world import FullPriorWorldState, full_prior_physics_step, measured_racket_pose
from .imagined_env import (ImaginedState, ImaginedTransition, select_lanes,
    finite_lanes, finite_physics, physical_bounds, task_substep)
from .imagined_rewards import control_reward, approach_reward
from .model_numerics import precision_scope
from .observation_engine import advance_world_observation
from .physical_events import physical_event_substep
from .task import record_failure
from .course_control import control_step


CourseState=NamedTuple('CourseState',[(name,Any) for name in ImaginedState._fields]+[('schedule',Any)])


def init_course(base,key,lane_ids=None):
    """Attach fresh sequence memory only to a new task, never a mid-task fork."""
    if np.any(np.asarray(base.task.hit_count)) or np.any(np.asarray(base.completed_goals)):
        raise ValueError('A mid-task course fork must restore its schedule memory')
    n=base.episode_steps.shape[0]
    ids=jnp.arange(n) if lane_ids is None else lane_ids
    if ids.shape!=(n,):
        raise ValueError('Logical lane identities must match the complete seed')
    return CourseState(*base,init_schedule(key,ids,base.episode_id))


def course_critic_features(state):
    base=critic_features(state.world,state.task,state.events,state.context,state.episode_steps)
    return jnp.concatenate((base,schedule_features(state.schedule)),axis=-1)


def course_critic_layout(state):
    layout=critic_layout(state.world,state.task,state.events,state.context,state.episode_steps)
    offset=layout['dimension'];fields=list(layout['fields'])
    for name,width,scale,unit in (
            ('sequence.completed_goals',1,4.,'goals'),('sequence.pulse_used',1,1.,'bool'),
            ('pulse.remaining_steps',1,20.,'1ms_ticks'),('pulse.force_world_n',3,.02,'N')):
        fields.append(dict(name=name,start=offset,end=offset+width,
            shape=[] if width==1 else [width],divisor=scale,unit=unit))
        offset+=width
    return dict(layout,schema='wmj_v1_critic_course_current_force',dimension=offset,
        fields=fields,excluded_schedule_fields=['group_id','pulse_rng','goal_rng','direction_key','direction_sign'])


def course_step(robot,auxiliary,state,raw_action,next_goal_xyz,*,stage_id,
                control_contract,episode_contract,reward_contract,**physics_options):
    """Advance5 physical ticks; arm a pulse only after a real goal-completing apex."""
    if stage_id not in ('S01','S02','S03','S04','S05'):
        raise ValueError('Unknown WMJ course')
    disturbance=stage_id in ('S04','S05')
    if stage_id=='S01':
        next_goal_xyz=state.task.goal_xyz
    robot,auxiliary=jax.tree.map(jax.lax.stop_gradient,(robot,auxiliary))
    active=~(state.terminated|state.truncated|state.model_truncated)
    action_valid=jnp.all(jnp.isfinite(raw_action),axis=-1)
    control,control_trace=control_step(state.world.control,jnp.where(action_valid[:,None],raw_action,0.),
        state.world.physics.q_rad,state.context,chain=physics_options['chain'],contract=control_contract)
    control_valid=finite_lanes(control,active.shape[0])

    def tick(carry,command):
        physics,events,task,schedule,valid=carry
        if disturbance:
            predicted,trace=forced_physics_step(robot,auxiliary,physics,command,
                schedule.force_world_n,**physics_options)
        else:
            predicted,trace=full_prior_physics_step(robot,auxiliary,physics,command,**physics_options)
        usable=valid & finite_physics(predicted)
        with precision_scope(physics_options.get('matmul_precision','highest')):
            event_trace=model_event_trace(trace,surface=physics_options['surface'],
                ball_radius_m=physics_options['ball_radius_m'],contact_contract=physics_options['contact_contract'])
        next_events,labels=physical_event_substep(events,event_trace,valid=usable,
            initialization_active=jnp.zeros_like(usable))
        next_task,terms,task_events=task_substep(task,labels,next_goal_xyz,contract=reward_contract)
        next_schedule,started=finish_substep(schedule,task_events['goal_switched'],usable,
            enabled=disturbance,mass_kg=physics_options['contact_contract'].mass_kg)
        applied=usable & (schedule.remaining_steps>0) & disturbance
        extra=dict(task_events,disturbance_started=started,disturbance_applied=applied,
            pulse_onset_delay_s=jnp.where(started,(1-labels['apex_fraction'])*.001,0.))
        return (select_lanes(usable,predicted,physics),next_events,next_task,next_schedule,usable),(terms,extra)

    (physics,events,task,schedule,usable),(release_terms,task_events)=jax.lax.scan(tick,
        (state.world.physics,state.events,state.task,state.schedule,active & action_valid & control_valid),
        control_trace['delayed_commands'])
    pose=measured_racket_pose(physics_options['chain'],physics.q_rad)
    within,bounds=physical_bounds(physics,pose,control_contract,episode_contract)
    failure=active & usable & ~within
    model_failure=active & ~usable
    task,_=record_failure(task,failure)
    observation,_=advance_world_observation(state.world.observation,physics.q_rad,physics.ball.position_m,
        pose_from_q=lambda q:measured_racket_pose(physics_options['chain'],q),ar1_enabled=stage_id=='S05')
    world=FullPriorWorldState(physics,control,observation,pose)
    world=select_lanes(active & ~model_failure,world,state.world)
    steps=state.episode_steps+active.astype(jnp.int32)
    timeout=active & ~failure & ~model_failure & (steps>=episode_contract.max_episode_steps)
    terms={key:jnp.sum(value,axis=0) for key,value in release_terms.items() if key!='total'}
    dense=control_reward(control.previous_accepted_action,state.world.control.previous_accepted_action,
        control_trace['clipped_action'],active=active,physical_failure=failure,dt_s=.005,contract=reward_contract)
    terms.update({key:value for key,value in dense.items() if key!='total'})
    terms['approach']=approach_reward(state.world.physics.ball.position_m,state.world.physics.ball.velocity_m_s,
        state.world.cached_racket_pose[:,:3],active=active,dt_s=.005,contract=reward_contract)
    terms={key:jnp.where(active & ~model_failure,value,0.) for key,value in terms.items()}
    counts={key:value.astype(jnp.int32).sum(axis=0) for key,value in task_events.items()
            if key!='pulse_onset_delay_s'}
    # A rejected control interval is atomic: no partial event/task/RNG commit.
    trusted = active & ~model_failure
    events, task, schedule = select_lanes(trusted, (events, task, schedule),
                                          (state.events, state.task, state.schedule))
    counts = {key:jnp.where(trusted, value, 0) for key,value in counts.items()}
    after=state._replace(world=world,events=events,task=task,schedule=schedule,episode_steps=steps,
        terminated=state.terminated|failure,truncated=state.truncated|timeout,
        model_truncated=state.model_truncated|model_failure,
        completed_goals=state.completed_goals+counts['goal_completed'],
        successful_goals=state.successful_goals+counts['goal_success'])
    diagnostics=dict(counts,pulse_onset_delay_s=jnp.where(trusted,
                     jnp.max(task_events['pulse_onset_delay_s'],axis=0),0.),
                     **{'bound_'+k:v for k,v in bounds.items()})
    transition=ImaginedTransition(sum(terms.values()),
        jnp.where(active & ~failure & ~model_failure,episode_contract.gamma,0.),
        active & ~failure & ~timeout & ~model_failure,active,failure,timeout,model_failure,
        terms,diagnostics,course_critic_features(after))
    return after,transition
