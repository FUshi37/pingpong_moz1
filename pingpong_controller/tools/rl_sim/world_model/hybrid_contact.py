"""Optional soft-contact prior plus the unchanged learned 6-D correction."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .contact_dynamics import ContactState, contact_step
from .contact_geometry import surface_features
from .contact_model import contact_prediction
from .soft_contact import soft_contact_increment


def hybrid_contact_step(parameters, surface, state, racket, *, ball_radius_m,
                        contract=None, learned=True, dt_s=.001):
    """Advance own ball state; native/recorded racket motion is a caller choice."""
    if contract is None:
        if not learned:
            raise ValueError('A base-only evaluation needs a physical prior')
        return contact_step(parameters, surface, state, racket, ball_radius_m=ball_radius_m, dt_s=dt_s)
    if dt_s != contract.dt_s:
        raise ValueError('Prior and contact integrator periods differ')
    geometry = surface_features(surface, state.position_m, racket['geom_position_m'],
                                 racket['geom_rotation_world'], ball_radius_m)
    prior = soft_contact_increment(contract, geometry,
        ball_position_m=state.position_m, ball_velocity_m_s=state.velocity_m_s,
        ball_omega_world_rad_s=state.omega_world_rad_s,
        **{name: value for name, value in racket.items() if name != 'geom_rotation_world'})
    dv, domega = prior.delta_velocity_world_m_s, prior.delta_omega_world_rad_s
    entry_logit, release_logit = jnp.zeros_like(state.duration_s), jnp.zeros_like(state.duration_s)
    if learned:
        residual = contact_prediction(parameters, surface,
            ball_position_m=state.position_m, ball_velocity_m_s=state.velocity_m_s,
            ball_omega_world_rad_s=state.omega_world_rad_s, previous_contact=state.previous_contact,
            duration_s=state.duration_s, ball_radius_m=ball_radius_m, **racket)
        dv, domega = dv+residual.delta_velocity_world_m_s, domega+residual.delta_omega_world_rad_s
        entry_logit, release_logit = residual.entry_logit, residual.release_logit
    velocity = state.velocity_m_s+jnp.array([0., 0., -9.81], jnp.float32)*dt_s+dv
    overlap = geometry['sphere_gap_m'] <= 0.
    entry, release = overlap & ~state.previous_contact, ~overlap & state.previous_contact
    duration = jnp.where(overlap, jnp.where(entry, 0., state.duration_s)+dt_s, 0.)
    after = ContactState(state.position_m+velocity*dt_s, velocity,
        state.omega_world_rad_s+domega, overlap, duration)
    return after, dict(before=state, after=after, overlap=overlap, entry=entry,
        release=release, entry_logit=entry_logit, release_logit=release_logit)


def hybrid_contact_rollout(parameters, surface, initial, racket, *, ball_radius_m,
                           contract=None, learned=True, dt_s=.001):
    """Recurse from only the initial ball state over the same 1 ms substeps."""
    def step(state, motion):
        return hybrid_contact_step(parameters, surface, state, motion,
            ball_radius_m=ball_radius_m, contract=contract, learned=learned, dt_s=dt_s)
    return jax.lax.scan(step, initial, racket)


def hybrid_contact_loss(parameters, surface, batch, *, ball_radius_m, autoregressive,
                        contract=None):
    """Use the existing normalized loss recipe for a controlled structure comparison."""
    from .contact_losses import (balanced_edge_loss, ballistic_apex, contact_module_loss,
                                 masked_mean)
    if contract is None:
        return contact_module_loss(parameters, surface, batch, ball_radius_m=ball_radius_m,
                                   autoregressive=autoregressive)
    if autoregressive:
        initial = jax.tree.map(lambda value: value[0], batch['before'])
        _, trace = hybrid_contact_rollout(parameters, surface, initial, batch['racket'],
            ball_radius_m=ball_radius_m, contract=contract)
    else:
        _, trace = hybrid_contact_step(parameters, surface, batch['before'], batch['racket'],
            ball_radius_m=ball_radius_m, contract=contract)
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
