"""Physical-scale contact state, edge and outgoing-outcome supervision."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .contact_dynamics import contact_rollout, contact_step


def masked_mean(value, mask):
    mask = mask.astype(value.dtype)
    return jnp.sum(jnp.where(mask > 0, value, 0.))/jnp.maximum(mask.sum(), 1.)


def balanced_edge_loss(logit, label, valid):
    """Balance actual positive/negative edge labels without inventing positives."""
    error = jax.nn.softplus(logit)-label.astype(logit.dtype)*logit
    positive, negative = valid & label, valid & ~label
    groups = (positive.any().astype(logit.dtype)+negative.any().astype(logit.dtype))
    return (masked_mean(error, positive)+masked_mean(error, negative))/jnp.maximum(groups, 1.)


def ballistic_apex(position, velocity):
    """Known no-drag next apex for an upward outgoing state."""
    duration = jnp.maximum(velocity[..., 2], 0.)/9.81
    return position+velocity*duration[..., None]+jnp.array([0., 0., -4.905])*duration[..., None]**2


def contact_module_loss(parameters, surface, batch: dict, *, ball_radius_m: float,
                        autoregressive: bool):
    """Use true state only at t=0 in recursive mode; recorded robot FK is explicit."""
    if autoregressive:
        initial = jax.tree.map(lambda value: value[0], batch['before'])
        _, trace = contact_rollout(parameters, surface, initial, batch['racket'],
                                   ball_radius_m=ball_radius_m)
    else:
        _, trace = contact_step(parameters, surface, batch['before'], batch['racket'],
                                ball_radius_m=ball_radius_m)
    predicted, target = trace['after'], batch['after']
    valid = batch['valid']

    def vector_error(left, right, scale):
        return jnp.mean(jnp.square((left-right)/scale), axis=-1)

    metrics = dict(
        position_loss=masked_mean(vector_error(predicted.position_m, target.position_m, .003), valid),
        velocity_loss=masked_mean(vector_error(predicted.velocity_m_s, target.velocity_m_s, .08), valid),
        spin_loss=masked_mean(vector_error(predicted.omega_world_rad_s, target.omega_world_rad_s, 30.), valid),
        entry_loss=balanced_edge_loss(trace['entry_logit'], batch['entry'], valid),
        release_loss=balanced_edge_loss(trace['release_logit'], batch['release'], valid),
    )
    # Native release is the beginning of the first non-overlap interval.
    # Compare the recursive before-state, without another gravity tick.
    before, actual = trace['before'], batch['before']
    outgoing = valid & batch['controlled_release']
    metrics['outgoing_velocity_loss'] = masked_mean(
        vector_error(before.velocity_m_s, actual.velocity_m_s, .08), outgoing)
    metrics['next_apex_loss'] = masked_mean(vector_error(
        ballistic_apex(before.position_m, before.velocity_m_s),
        ballistic_apex(actual.position_m, actual.velocity_m_s), .01), outgoing)
    loss = sum(metrics.values())
    metrics.update(loss=loss,
        entry_mismatch=masked_mean((trace['entry'] != batch['entry']).astype(jnp.float32), valid),
        release_mismatch=masked_mean((trace['release'] != batch['release']).astype(jnp.float32), valid))
    return loss, metrics
