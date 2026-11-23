"""Opt-in normal and tangential soft-prior correction; contact residual inputs remain39-D.

Six solver-stage terms are a separate physical-prior input, never neural
contact features. The previous prior and all existing checkpoints are intact.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
from mujoco.mjx._src.math import make_frame

from .contact_geometry import surface_features
from .contact_losses import balanced_edge_loss, ballistic_apex, masked_mean
from .hybrid_contact import hybrid_contact_step
from .position_contact import POSE_KEYS, POSITION_KEYS, position_contact_step
from .soft_contact import SoftContactPrediction, soft_contact_increment, softness


def tangent_prior_increment(contract, geometry, solver_terms, *, ball_position_m,
        ball_velocity_m_s, ball_omega_world_rad_s, geom_position_m,
        racket_linear_velocity_m_s, racket_angular_velocity_rad_s,
        gravity_m_s2=(0., 0., -9.81)):
    """Correct all three desired-force rows; retain the existing friction cap."""
    if solver_terms.shape != ball_position_m.shape[:-1]+(6,):
        raise ValueError('Expected geom-origin world linear/angular solver terms')
    normal, point = geometry['contact_normal_world'], geometry['contact_position_world_m']
    frame = jax.vmap(make_frame)(normal.reshape((-1, 3))).reshape(normal.shape[:-1]+(3, 3))
    lever = point-ball_position_m
    relative = (ball_velocity_m_s+jnp.cross(ball_omega_world_rad_s, lever)
        -racket_linear_velocity_m_s-jnp.cross(racket_angular_velocity_rad_s, point-geom_position_m))
    velocity = jnp.einsum('...ij,...j->...i', frame, relative)
    gravity = jnp.einsum('...ij,j->...i', frame, jnp.asarray(gravity_m_s2, jnp.float32))
    distance = geometry['sphere_gap_m']
    k, b, impedance = softness(contract, distance)
    aref = (-b*velocity).at[..., 0].add(-k*impedance*distance)
    rotation_lever = jnp.cross(lever[..., None, :], frame)
    inverse_effective_mass = 1/contract.mass_kg+jnp.sum(rotation_lever**2, axis=-1)/contract.inertia_kg_m2
    mu = jnp.asarray(contract.friction, jnp.float32)
    ratios = jnp.stack((jnp.asarray(1.), jnp.asarray(1/contract.impratio),
                        mu[0]**2/(contract.impratio*mu[1]**2)))
    regularizer = contract.reference_inverse_mass_kg_inv*(1-impedance)/impedance
    regularizer = jnp.maximum(regularizer[..., None]*ratios, mujoco.mjMINVAL)
    point_term = solver_terms[..., :3]+jnp.cross(solver_terms[..., 3:], point-geom_position_m)
    numerator = aref-gravity+jnp.einsum('...ij,...j->...i', frame, point_term)
    desired = numerator/(inverse_effective_mass+regularizer)
    normal_force = jnp.maximum(desired[..., 0], 0.)
    tangent = desired[..., 1:]
    friction_norm = jnp.sqrt(jnp.maximum(jnp.sum((tangent/mu)**2, axis=-1), 1e-24))
    tangent = tangent*jnp.minimum(1., normal_force/friction_norm)[..., None]
    force = jnp.concatenate((normal_force[..., None], tangent), axis=-1)
    force = jnp.where((distance < 0.)[..., None], force, 0.)
    world_force = jnp.einsum('...ji,...j->...i', frame, force)
    return SoftContactPrediction(world_force*(contract.dt_s/contract.mass_kg),
        jnp.cross(lever, world_force)*(contract.dt_s/contract.inertia_kg_m2), world_force, force, impedance)


def tangent_contact_step(parameters, surface, state, racket, solver_terms, *,
                        ball_radius_m, position_scale, contract, learned=True, dt_s=.001):
    """Keep original residual computation; replace its base increment only."""
    if set(racket) != POSE_KEYS | POSITION_KEYS:
        raise ValueError('Contact39 retains only original pose/twist and position inputs')
    options = dict(ball_radius_m=ball_radius_m, contract=contract, dt_s=dt_s)
    if learned:
        after, trace = position_contact_step(parameters, surface, state, racket,
            position_scale=position_scale, **options)
    else:
        after, trace = hybrid_contact_step(parameters, surface, state,
            {k: racket[k] for k in POSE_KEYS}, learned=False, **options)
    if solver_terms is None:
        return after, trace
    if contract is None:
        raise ValueError('A full-prior correction requires the registered soft prior')
    geometry = surface_features(surface, state.position_m, racket['geom_position_m'],
                                 racket['geom_rotation_world'], ball_radius_m)
    physical = dict(ball_position_m=state.position_m, ball_velocity_m_s=state.velocity_m_s,
        ball_omega_world_rad_s=state.omega_world_rad_s,
        **{k: racket[k] for k in POSE_KEYS if k != 'geom_rotation_world'})
    original = soft_contact_increment(contract, geometry, **physical)
    corrected = tangent_prior_increment(contract, geometry, solver_terms, **physical)
    dv = corrected.delta_velocity_world_m_s-original.delta_velocity_world_m_s
    dw = corrected.delta_omega_world_rad_s-original.delta_omega_world_rad_s
    after = after._replace(position_m=after.position_m+dv*dt_s,
        velocity_m_s=after.velocity_m_s+dv, omega_world_rad_s=after.omega_world_rad_s+dw)
    return after, dict(trace, after=after)


def tangent_contact_rollout(parameters, surface, initial, racket, solver_terms, **options):
    def tick(state, inputs):
        motion, terms = inputs
        return tangent_contact_step(parameters, surface, state, motion, terms, **options)
    return jax.lax.scan(tick, initial, (racket, solver_terms))


def tangent_contact_loss(parameters, surface, batch, terms, *, autoregressive, **options):
    """Use the unchanged normalized contact losses and true event masks."""
    if autoregressive:
        initial = jax.tree.map(lambda value: value[0], batch['before'])
        _, trace = tangent_contact_rollout(parameters, surface, initial, batch['racket'], terms, **options)
    else:
        _, trace = tangent_contact_step(parameters, surface, batch['before'], batch['racket'], terms, **options)
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
