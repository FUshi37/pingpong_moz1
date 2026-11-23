"""Explicit privileged critic layout with complete native controller memory."""

from __future__ import annotations

import math
import numpy as np
import jax.numpy as jnp

from .full_prior_world import world_actor_observation


SCHEMA='wmj_v1_critic_full_control'
RMP_FIELDS=('q','qd','qdd','command_q','command_qd','command_qdd','accepted_target',
    'ticks_since_target_update','target_x','target_covariance','target_q_buffer',
    'target_qd_buffer','target_qdd_buffer','target_buffer_index','output_q_buffer',
    'output_qd_buffer','output_qdd_buffer','output_buffer_index')
CONTEXT_SCALES=dict(jnt_kp=1e5,jnt_kd=1e4,estimator_joint_process_var=50.,
    estimator_joint_measure_var=1e-4,joint_velocity_feedforward=1.,
    joint_acceleration_feedforward=1.,acceleration_weight=.001,target_filter_length=10.)


def feature_blocks(world,task,events,context,elapsed_steps):
    """Return named (array, fixed divisor, unit) blocks with lane-first axes."""
    physics=world.physics;control=world.control;camera=world.observation.camera
    blocks=[('actor_observation',world_actor_observation(world,task.goal_xyz),1.,'existing_actor_encoding'),
        ('joint_q',physics.q_rad,math.pi,'rad'),('joint_dq',physics.dq_rad_s,5.,'rad/s'),
        ('ball_position',physics.ball.position_m,1.,'m'),('ball_velocity',physics.ball.velocity_m_s,3.,'m/s'),
        ('ball_spin',physics.ball.omega_world_rad_s,30.,'rad/s'),
        ('ball_contact',physics.ball.previous_contact,1.,'bool'),
        ('ball_contact_age',physics.ball.duration_s,.02,'s'),
        ('reference_q',control.reference_q_rad,math.pi,'rad'),
        ('previous_accepted_action',control.previous_accepted_action,1.,'normalized_action')]
    if tuple(control.rmp._fields)!=RMP_FIELDS:
        raise ValueError('Unregistered RMP memory layout')
    vector=np.array([math.pi,5.,50.],np.float32)
    for name in RMP_FIELDS:
        value=getattr(control.rmp,name)
        if name=='target_x':
            scale=vector;unit='q_dq_ddq'
        elif name=='target_covariance':
            scale=vector[:,None]*vector[None,:];unit='q_dq_ddq_covariance'
        elif name in ('target_buffer_index','output_buffer_index'):
            scale=(control.rmp.target_q_buffer.shape[1] if name.startswith('target')
                   else control.rmp.output_q_buffer.shape[1])
            unit='ring_index'
        elif name=='ticks_since_target_update':
            scale=5.;unit='1ms_ticks'
        elif 'qdd' in name:
            scale=50.;unit='rad/s2'
        elif 'qd' in name:
            scale=5.;unit='rad/s'
        else:
            scale=math.pi;unit='rad'
        blocks.append(('rmp.'+name,value,scale,unit))
    for key,scale,unit in (('q',math.pi,'rad'),('dq',5.,'rad/s'),('ddq',50.,'rad/s2')):
        blocks.append(('output_delay.'+key,control.output_delay[key],scale,unit))
    if not set(context.rmp).issubset(CONTEXT_SCALES):
        raise ValueError('Unregistered episode control parameter')
    for key in sorted(context.rmp):
        blocks.append(('context.'+key,context.rmp[key],CONTEXT_SCALES[key],'native_parameter'))
    blocks.extend([('context.action_scale',context.action_scale_mult,1.,'multiplier'),
        ('context.output_delay_steps',context.output_delay_steps,20.,'1ms_ticks'),
        ('camera.phase',camera.step_count%20,20.,'200Hz_phase'),
        ('camera.sample_age',camera.step_count-camera.last_sample_step,3.,'200Hz_ticks'),
        ('camera.ar1_memory',camera.epsilon_m,.003,'m'),
        ('episode.elapsed',elapsed_steps,6000.,'200Hz_ticks')])
    for name,value in events._asdict().items():
        scale={'impact_id':64.,'normal_impulse_n_s':.02,'contact_duration_steps':20.}.get(name,1.)
        blocks.append(('events.'+name,value,scale,'event_state'))
    for name,value in task._asdict().items():
        scale=1. if 'xyz' in name or value.dtype==jnp.bool_ else (
            64. if name.endswith('_id') else 16.)
        blocks.append(('task.'+name,value,scale,'m' if 'xyz' in name else 'task_state'))
    return blocks


def critic_features(world,task,events,context,elapsed_steps):
    """No RNG, future observations/labels, fitting statistics or hidden encoder."""
    n=world.physics.q_rad.shape[0]
    blocks=feature_blocks(world,task,events,context,elapsed_steps)
    if any(value.ndim<1 or value.shape[0]!=n for _,value,_,_ in blocks):
        raise ValueError('Critic fields must all have an explicit environment axis')
    return jnp.concatenate([(jnp.asarray(value,jnp.float32)/jnp.asarray(scale,jnp.float32)).reshape(n,-1)
        for _,value,scale,_ in blocks],axis=-1)


def critic_layout(*args):
    """Publish concrete slices, shapes, units and divisors before compilation."""
    fields=[];offset=0
    for name,value,scale,unit in feature_blocks(*args):
        width=int(np.prod(value.shape[1:]))
        fields.append(dict(name=name,start=offset,end=offset+width,shape=list(value.shape[1:]),
            divisor=np.asarray(scale).tolist(),unit=unit))
        offset+=width
    return dict(schema=SCHEMA,dimension=offset,fields=fields,
                hidden_sizes=[256,256],output_dim=1,output_initialization='zero',
                running_normalization=False,actor_observation_changed=False)
