"""Opt-in 39-D contact: legacy19 plus q13 and delayed position-error7.

No additional joint velocity, command velocity, acceleration or future solver
quantity is accepted. The original pose/twist features remain unchanged.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .hybrid_contact import hybrid_contact_step
from .contact_losses import balanced_edge_loss, ballistic_apex, masked_mean


POSE_KEYS = frozenset(('geom_position_m', 'geom_rotation_world',
                      'racket_linear_velocity_m_s', 'racket_angular_velocity_rad_s'))
POSITION_KEYS = frozenset(('joint_q_rad', 'delayed_q_command_rad'))


class PositionScale(NamedTuple):
    mean: jax.Array
    scale: jax.Array


def position_features(q_rad, delayed_q_command_rad):
    """Return (...,20): RightArm q7, LegWaist q6, delayed arm target minus q7."""
    if (q_rad.shape[-1] != 13
            or delayed_q_command_rad.shape != q_rad.shape[:-1]+(7,)):
        raise ValueError('Position conditioning requires thirteen q and seven delayed position targets')
    return jnp.concatenate((q_rad, delayed_q_command_rad-q_rad[..., :7]), axis=-1)


def position_scale_from_statistics(stats: dict) -> PositionScale:
    """Frozen training-only scales; a0.1 mrad floor matches robot input scaling."""
    if (stats.get('source') != 'train' or stats.get('count', 0) < 1
            or np.shape(stats['mean']) != (20,) or np.shape(stats['variance']) != (20,)
            or not np.isfinite(stats['mean']).all() or not np.isfinite(stats['variance']).all()
            or np.any(np.asarray(stats['variance']) < 0)):
        raise ValueError('Position contact scaling requires finite training-only20-D statistics')
    return PositionScale(jnp.asarray(stats['mean'], jnp.float32),
        jnp.sqrt(jnp.maximum(jnp.asarray(stats['variance'], jnp.float32), 1e-8)))


def position_contact_step(parameters, surface, state, racket, *, ball_radius_m,
                          position_scale, contract=None, dt_s=.001):
    """Condition the first affine layer, retaining exact legacy contact mechanics.

W19*x19 + W20*x20 + b is the39-D first layer. Folding its position term
into a per-sample bias permits reuse of both unchanged19-D physical paths.
In a composed world q and targets must come from its own recursive state.
"""
    if set(racket) != POSE_KEYS | POSITION_KEYS:
        raise ValueError('Expected position-only conditioning and the original pose/twist fields')
    if parameters[0]['w'].shape[0] != 39:
        raise ValueError('Position-only contact requires a39-input checkpoint')
    extra = position_features(racket['joint_q_rad'], racket['delayed_q_command_rad'])
    extra = (extra-position_scale.mean)/position_scale.scale
    first = dict(parameters[0], w=parameters[0]['w'][:19],
        b=parameters[0]['b']+extra@parameters[0]['w'][19:])
    folded = (first, *parameters[1:])
    motion = {name:racket[name] for name in POSE_KEYS}
    return hybrid_contact_step(folded, surface, state, motion, ball_radius_m=ball_radius_m,
                               contract=contract, dt_s=dt_s)


def position_contact_rollout(parameters, surface, initial, racket, *, ball_radius_m,
                             position_scale, contract=None, dt_s=.001):
    """Keep the predicted ball state continuous across all1 ms substeps."""
    def tick(state, motion):
        return position_contact_step(parameters, surface, state, motion, ball_radius_m=ball_radius_m,
            position_scale=position_scale, contract=contract, dt_s=dt_s)
    return jax.lax.scan(tick, initial, racket)


def position_contact_loss(parameters, surface, batch, *, ball_radius_m,
                          position_scale, autoregressive, contract=None):
    """Use the existing seven losses and physical scales for both39-D methods."""
    options = dict(ball_radius_m=ball_radius_m, position_scale=position_scale, contract=contract)
    if autoregressive:
        initial = jax.tree.map(lambda value:value[0], batch['before'])
        _, trace = position_contact_rollout(parameters, surface, initial, batch['racket'], **options)
    else:
        _, trace = position_contact_step(parameters, surface, batch['before'], batch['racket'], **options)
    predicted, target, valid = trace['after'], batch['after'], batch['valid']

    def vector_error(left, right, scale):
        return jnp.mean(((left-right)/scale)**2, axis=-1)

    metrics = dict(
        position_loss=masked_mean(vector_error(predicted.position_m, target.position_m, .003), valid),
        velocity_loss=masked_mean(vector_error(predicted.velocity_m_s, target.velocity_m_s, .08), valid),
        spin_loss=masked_mean(vector_error(predicted.omega_world_rad_s, target.omega_world_rad_s, 30.), valid),
        entry_loss=balanced_edge_loss(trace['entry_logit'], batch['entry'], valid),
        release_loss=balanced_edge_loss(trace['release_logit'], batch['release'], valid))
    before, actual = trace['before'], batch['before']
    outgoing = valid & batch['controlled_release']
    metrics['outgoing_velocity_loss'] = masked_mean(vector_error(before.velocity_m_s, actual.velocity_m_s, .08), outgoing)
    metrics['next_apex_loss'] = masked_mean(vector_error(ballistic_apex(before.position_m, before.velocity_m_s),
        ballistic_apex(actual.position_m, actual.velocity_m_s), .01), outgoing)
    loss = sum(metrics.values())
    metrics.update(loss=loss,
        entry_mismatch=masked_mean((trace['entry'] != batch['entry']).astype(jnp.float32), valid),
        release_mismatch=masked_mean((trace['release'] != batch['release']).astype(jnp.float32), valid))
    return loss, metrics
