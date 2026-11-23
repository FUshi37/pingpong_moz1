"""One-step normal/full prior diagnostics for coherent native replay states.

True solver terms are an offline oracle only. Run the same-replay diagnostic
through run_wmj_v1_tangent_diagnostic.sh; independent traces are never joined.
The rejected joined-record probe remains in its immutable evidence snapshot.
"""

from __future__ import annotations

import jax.numpy as jnp

from .contact_geometry import surface_features
from .native_contact_data import prepare_native_contact_batch
from .normal_contact import normal_prior_increment
from .robot_support_model import support_features
from .soft_contact import soft_contact_increment
from .solver_auxiliary import predict
from .tangent_contact import tangent_prior_increment


def prior_errors(options: dict, auxiliary: dict, raw: dict, oracle_terms):
    """Compare one-step velocity increments at identical native before-states."""
    batch = prepare_native_contact_batch(None, raw)
    ball, motion = batch['before'], batch['racket']
    geometry = surface_features(options['surface'], ball.position_m,
        motion['geom_position_m'], motion['geom_rotation_world'], options['ball_radius_m'])
    commands = {k: raw['delayed_'+k+'_command'] for k in ('q', 'dq', 'ddq')}
    features = support_features(raw['q_before'], raw['dq_before'], commands)
    terms = predict(auxiliary['parameters'], auxiliary['scale'], features)
    physical = dict(ball_position_m=ball.position_m, ball_velocity_m_s=ball.velocity_m_s,
        ball_omega_world_rad_s=ball.omega_world_rad_s,
        **{k: v for k, v in motion.items() if k != 'geom_rotation_world'})
    prior = options['contact_contract']
    outputs = dict(zero=soft_contact_increment(prior, geometry, **physical),
        predicted_normal=normal_prior_increment(prior, geometry, terms, **physical),
        predicted_full=tangent_prior_increment(prior, geometry, terms, **physical),
        oracle_normal=normal_prior_increment(prior, geometry, oracle_terms, **physical),
        oracle_full=tangent_prior_increment(prior, geometry, oracle_terms, **physical))
    truth = batch['after'].velocity_m_s-ball.velocity_m_s-jnp.array([0., 0., -.00981])
    normal = geometry['contact_normal_world']
    metrics = {}
    for name, output in outputs.items():
        error = output.delta_velocity_world_m_s-truth
        en = jnp.sum(error*normal, axis=-1)
        metrics[name] = dict(velocity_3d_m_s=jnp.linalg.norm(error, axis=-1),
            normal_m_s=jnp.abs(en), tangent_m_s=jnp.linalg.norm(error-en[..., None]*normal, axis=-1))
    return dict(metrics=metrics, features=features,
        near_rim=geometry['edge_signed_distance_m'] > -.01)
