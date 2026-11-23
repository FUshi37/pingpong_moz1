"""Training-only activation penalty for the unchanged39-D tanh contact model."""

import math

import jax.numpy as jnp

from .contact_losses import masked_mean
from .contact_model import contact_features, FEATURE_SCALE
from .physics_fitting import coupled_contact_objective
from .position_contact import POSE_KEYS, position_features


def activation_penalty(parameters, inputs, mask):
    """Penalize final-hidden preactivation beyond two; exclude noncontact rows."""
    value = inputs
    for layer in parameters[:-1]:
        before = value@layer['w']+layer['b']
        value = jnp.tanh(before)
    excess = jnp.mean(jnp.maximum(jnp.abs(before)-2.,0.)**2,axis=-1)
    saturated = jnp.mean(jnp.abs(value)>.995,axis=-1)
    return masked_mean(excess,mask),masked_mean(saturated,mask)


def regularized_contact_objective(parameters,surface,batch,motion,*,activation_penalty_weight,**options):
    """Retain recursive/auxiliary losses, adding only an explicit fitting term."""
    total,metrics = coupled_contact_objective(parameters,surface,batch,motion,**options)
    if activation_penalty_weight == 0.:
        return total,metrics
    state,racket = batch['before'],batch['racket']
    features,geometry = contact_features(surface,ball_position_m=state.position_m,
        ball_velocity_m_s=state.velocity_m_s,ball_omega_world_rad_s=state.omega_world_rad_s,
        previous_contact=state.previous_contact,duration_s=state.duration_s,
        ball_radius_m=options['ball_radius_m'],**{key:racket[key] for key in POSE_KEYS})
    scale = options['position_scale']
    positions = position_features(racket['joint_q_rad'],racket['delayed_q_command_rad'])
    inputs = jnp.concatenate((features/jnp.asarray(FEATURE_SCALE,jnp.float32),(positions-scale.mean)/scale.scale),axis=-1)
    penalty,saturated = activation_penalty(parameters,inputs,batch['valid'] & (geometry['sphere_gap_m']<=0.))
    weighted = activation_penalty_weight*penalty
    return total+weighted,dict(metrics,loss=total+weighted,unregularized_loss=total,
        activation_excess_squared=penalty,activation_penalty_loss=weighted,
        final_hidden_contact_saturated_fraction=saturated)


def validate_activation_transition(previous,current,*,start_stage):
    """A separate explicit loss stage may turn on the registered0.01 penalty."""
    old,new = dict(previous),dict(current)
    before,after = old.pop('activation_penalty_weight',0.),new.pop('activation_penalty_weight',0.)
    if any(not math.isfinite(x) or x<0 for x in (before,after)):
        raise ValueError('Activation penalty must be finite and nonnegative')
    if before == after and not start_stage:
        return
    if not (old == new and start_stage and before == 0. and after == .01):
        raise ValueError('Activation stage may only turn on0.01 without changing another recipe field')
