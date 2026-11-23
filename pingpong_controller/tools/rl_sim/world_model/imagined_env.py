"""Pure imagined physical/control/history/task transitions with explicit endings.

Goal proposals are exogenous task inputs, never teacher next states. The
caller supplies preaudited seeds and goal proposals; no online mesh/IK audit
or native simulation is introduced into an imagined step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .action_effects import model_event_trace
from .critic_features import critic_features
from .full_prior_world import FullPriorWorldState, full_prior_physics_step, measured_racket_pose
from .imagined_rewards import release_reward, control_reward, approach_reward
from .model_numerics import precision_scope
from .observation_engine import advance_world_observation
from .physical_events import init_physical_events, physical_event_substep
from .task import init_task, record_release, record_apex, record_failure, switch_completed_goals
from .world_control import control_step


@dataclass(frozen=True)
class EpisodeContract:
    """Existing physical domain; finite horizon is a separate bootstrap boundary."""

    max_episode_steps: int = 6000
    gamma: float = .9995
    ball_lower: tuple = (-.38,-.75,.98)
    ball_upper: tuple = (.38,-.28,1.50)
    racket_lower: tuple = (-.38,-.75,1.)
    racket_upper: tuple = (.38,-.28,1.15)
    roll_pitch_limit_rad: float = .03490658503988659

    @classmethod
    def from_config(cls,cfg):
        d=cfg['domain'];i=cfg['imagination']
        return cls(int(round(i['episode_limit_s']/.005)),i['gamma_5ms'],
            tuple(d['view_xy_lower_m'])+(d['ball_z_physical_bounds_m'][0],),
            tuple(d['view_xy_upper_m'])+(d['ball_z_physical_bounds_m'][1],),
            tuple(d['racket_xyz_lower_m']),tuple(d['racket_xyz_upper_m']),
            d['roll_pitch_limit_deg']*3.141592653589793/180.)


class ImaginedState(NamedTuple):
    world: Any
    context: Any
    events: Any
    task: Any
    episode_steps: Any
    episode_id: Any
    terminated: Any
    truncated: Any
    model_truncated: Any
    completed_goals: Any
    successful_goals: Any


class ImaginedTransition(NamedTuple):
    reward: Any
    discount: Any
    trace_continue: Any
    valid: Any
    terminated: Any
    truncated: Any
    model_truncated: Any
    reward_terms: Any
    task_events: Any
    next_critic_features: Any


def init_imagined(world,context,goal_xyz,*,events=None,task=None):
    """Prepare a coherent seed on the host before entering a compiled rollout.

    A mid-task fork supplies both task and physical event memory. A fresh task
    branch may start before contact with new bookkeeping, without resetting
    any camera, controller, physics or source history.
    """
    n=world.physics.q_rad.shape[0];zero=jnp.zeros(n,jnp.int32);off=jnp.zeros(n,bool)
    if goal_xyz.shape!=(n,3):
        raise ValueError('Expected independent task goals (N,3)')
    if events is None and (task is not None or np.any(np.asarray(world.physics.ball.previous_contact))):
        raise ValueError('A contact/task fork requires its complete physical event memory')
    return ImaginedState(world,context,init_physical_events(n) if events is None else events,
        init_task(goal_xyz) if task is None else task,zero,zero,off,off,off,zero,zero)


def select_lanes(mask,yes,no):
    return jax.tree.map(lambda a,b:jnp.where(mask.reshape(mask.shape+(1,)*(a.ndim-mask.ndim)),a,b),yes,no)


def reset_lanes(current,seed,mask):
    """Replace complete lanes, stop reset gradients, preserve continuing lanes."""
    fresh=jax.tree.map(jax.lax.stop_gradient,seed)
    fresh=fresh._replace(episode_id=current.episode_id+1)
    return select_lanes(mask,fresh,current)


def task_substep(task,labels,next_goal,*,contract):
    """Consume one physical event once, then publish goals after actual apexes."""
    after,accepted=record_release(task,impact_id=labels['impact_id'],
        physical_release=labels['contact_release'],controlled_hit=labels['controlled_release'])
    terms=release_reward(labels['release_position_m'],labels['release_velocity_m_s'],
        after.flight_goal_xyz,accepted,contract=contract)
    after,event=record_apex(after,labels['apex_position_m'],
        jnp.linalg.norm(labels['apex_velocity_m_s'][...,:2],axis=-1),labels['actual_apex'])
    after,switched=switch_completed_goals(after,next_goal)
    return after,terms,dict(goal_completed=event.completed,goal_success=event.success,
        goal_switched=switched,valid_release=accepted,valid_apex=event.valid_apex)


def finite_physics(state):
    return finite_lanes(state,state.q_rad.shape[0])


def finite_lanes(state,n):
    """Check numerical state with an explicit leading environment axis."""
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x).reshape(n,-1),axis=-1)
        for x in jax.tree.leaves(state)]),axis=0)


def physical_bounds(physics,pose,control_contract,episode_contract):
    """Same200 Hz endpoint domain checks; no geometry/self-collision scan."""
    c=episode_contract;p=control_contract.projection
    column0=pose[:,3:6];column1=pose[:,6:9]
    rotation=jnp.stack((column0,column1,jnp.cross(column0,column1)),axis=-1)
    relative=jnp.einsum('ji,njk->nik',p.reference_rotation,rotation)
    roll=jnp.arctan2(-relative[:,1,2],relative[:,2,2])
    pitch=jnp.arcsin(jnp.clip(relative[:,0,2],-1.,1.))
    tests=dict(ball=jnp.all((physics.ball.position_m>=jnp.asarray(c.ball_lower))
        & (physics.ball.position_m<=jnp.asarray(c.ball_upper)),axis=-1),
        racket=jnp.all((pose[:,:3]>=jnp.asarray(c.racket_lower))
        & (pose[:,:3]<=jnp.asarray(c.racket_upper)),axis=-1),
        orientation=(jnp.abs(roll)<=c.roll_pitch_limit_rad)&(jnp.abs(pitch)<=c.roll_pitch_limit_rad),
        joints=jnp.all((physics.q_rad[:,:7]>=p.q_lower)&(physics.q_rad[:,:7]<=p.q_upper),axis=-1))
    return jnp.all(jnp.stack(list(tests.values())),axis=0),tests


def imagined_step(robot_parameters,auxiliary,state,raw_action,next_goal_xyz,*,
                  control_contract,episode_contract,reward_contract,ar1_enabled,**physics_options):
    """One200 Hz transition; fixed model weights retain state/action derivatives."""
    robot_parameters,auxiliary=jax.tree.map(jax.lax.stop_gradient,(robot_parameters,auxiliary))
    active=~(state.terminated|state.truncated|state.model_truncated)
    valid_action=jnp.all(jnp.isfinite(raw_action),axis=-1)
    safe_action=jnp.where(valid_action[:,None],raw_action,0.)
    control,control_trace=control_step(state.world.control,safe_action,state.world.physics.q_rad,
        state.context,chain=physics_options['chain'],contract=control_contract)
    valid_control=finite_lanes(control,active.shape[0])
    def tick(carry,command):
        physics,events,task,valid=carry
        predicted,trace=full_prior_physics_step(robot_parameters,auxiliary,physics,command,**physics_options)
        usable=valid & finite_physics(predicted)
        with precision_scope(physics_options.get('matmul_precision','highest')):
            event_trace=model_event_trace(trace,surface=physics_options['surface'],
                ball_radius_m=physics_options['ball_radius_m'],contact_contract=physics_options['contact_contract'])
        next_events,labels=physical_event_substep(events,event_trace,valid=usable,
            initialization_active=jnp.zeros_like(usable))
        next_task,terms,task_events=task_substep(task,labels,next_goal_xyz,contract=reward_contract)
        return (select_lanes(usable,predicted,physics),next_events,next_task,usable),(terms,task_events)
    (physics,events,task,usable),(release_terms,task_events)=jax.lax.scan(tick,
        (state.world.physics,state.events,state.task,active & valid_action & valid_control),
        control_trace['delayed_commands'])
    pose=measured_racket_pose(physics_options['chain'],physics.q_rad)
    within,bounds=physical_bounds(physics,pose,control_contract,episode_contract)
    failure=active & usable & ~within
    model_failure=active & ~usable
    task,_=record_failure(task,failure)
    observation,_=advance_world_observation(state.world.observation,physics.q_rad,physics.ball.position_m,
        pose_from_q=lambda q:measured_racket_pose(physics_options['chain'],q),ar1_enabled=ar1_enabled)
    world=FullPriorWorldState(physics,control,observation,pose)
    world=select_lanes(active & ~model_failure,world,state.world)
    steps=state.episode_steps+active.astype(jnp.int32)
    timeout=active & ~failure & ~model_failure & (steps>=episode_contract.max_episode_steps)
    terms={key:jnp.sum(value,axis=0) for key,value in release_terms.items() if key!='total'}
    dense=control_reward(control.previous_accepted_action,state.world.control.previous_accepted_action,
        control_trace['clipped_action'],active=active,physical_failure=failure,dt_s=.005,contract=reward_contract)
    terms.update({key:value for key,value in dense.items() if key!='total'})
    terms['approach']=approach_reward(state.world.physics.ball.position_m,
        state.world.physics.ball.velocity_m_s,state.world.cached_racket_pose[:,:3],
        active=active,dt_s=.005,contract=reward_contract)
    terms={key:jnp.where(active & ~model_failure,value,0.) for key,value in terms.items()}
    goal_counts={key:value.astype(jnp.int32).sum(axis=0) for key,value in task_events.items()}
    after=state._replace(world=world,events=events,task=task,episode_steps=steps,
        terminated=state.terminated|failure,truncated=state.truncated|timeout,
        model_truncated=state.model_truncated|model_failure,
        completed_goals=state.completed_goals+goal_counts['goal_completed'],
        successful_goals=state.successful_goals+goal_counts['goal_success'])
    discount=jnp.where(active & ~failure & ~model_failure,episode_contract.gamma,0.)
    transition=ImaginedTransition(sum(terms.values()),discount,
        active & ~failure & ~timeout & ~model_failure,active,failure,timeout,model_failure,
        terms,dict(goal_counts,**{'bound_'+k:v for k,v in bounds.items()}),
        critic_features(world,task,events,state.context,steps))
    return after,transition
