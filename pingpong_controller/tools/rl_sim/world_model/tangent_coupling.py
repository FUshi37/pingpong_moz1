"""Causal robot-conditioned normal and tangential prior, separate from contact39 features."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .tangent_contact import tangent_contact_loss, tangent_contact_rollout
from .normal_coupling import acceleration_terms, own_robot_motion
from .normal_coupling import normal_coupled_objective, make_predictor as normal_predictor
from .physics_metrics import ball_metrics, robot_metrics
from .position_contact_data import prepare_position_contact_batch


def tangent_coupled_objective(parameters, robot_parameters, auxiliary, raw, *,
                             options, tangent_enabled):
    batch = prepare_position_contact_batch(None, raw)
    motion, robot, commands = own_robot_motion(robot_parameters, raw,
        scale=options['robot_scale'], chain=options['chain'])
    shared = dict(ball_radius_m=options['ball_radius_m'],
                  position_scale=options['position_scale'], contract=options['contact_contract'])
    if not tangent_enabled:
        return normal_coupled_objective(parameters, robot_parameters, auxiliary, raw,
            options=options, normal_enabled=True)
    own_terms = acceleration_terms(auxiliary, robot['q_before_rad'], robot['dq_before_rad_s'], commands)
    recursive, metrics = tangent_contact_loss(parameters, options['surface'], dict(batch, racket=motion),
        own_terms, autoregressive=True, **shared)
    # This is the existing separate teacher-forced training anchor. Its values
    # never correct the recursively generated robot/ball trajectory above.
    native_terms = acceleration_terms(auxiliary, raw['q_before'], raw['dq_before'], commands)
    anchor, aux_metrics = tangent_contact_loss(parameters, options['surface'], batch,
        native_terms, autoregressive=False, **shared)
    total = recursive+256.*anchor
    return total, dict(metrics, loss=total, recursive_loss=recursive,
        auxiliary_single_step_loss=anchor, auxiliary_weighted_loss=256.*anchor,
        **{'single_step_'+key: value for key, value in aux_metrics.items()})


def make_predictor(parameters, options, auxiliary, *, tangent_enabled, learned):
    """Build the same evaluation contract, with two explicit robot conditions."""
    if not tangent_enabled:
        return normal_predictor(parameters, options, auxiliary, normal_enabled=True, learned=learned)

    def predict_batch(raw):
        batch = prepare_position_contact_batch(None, raw)
        ball = jax.tree.map(lambda value: value[0], batch['before'])
        motion, robot, commands = own_robot_motion(parameters['robot'], raw,
            scale=options['robot_scale'], chain=options['chain'])
        shared = dict(ball_radius_m=options['ball_radius_m'], position_scale=options['position_scale'],
                      contract=options['contact_contract'], learned=learned)
        native_terms = own_terms = None
        if tangent_enabled:
            native_terms = acceleration_terms(auxiliary, raw['q_before'], raw['dq_before'], commands)
            own_terms = acceleration_terms(auxiliary, robot['q_before_rad'], robot['dq_before_rad_s'], commands)
        _, native = tangent_contact_rollout(parameters['contact'], options['surface'], ball,
            batch['racket'], native_terms, **shared)
        _, own = tangent_contact_rollout(parameters['contact'], options['surface'], ball,
            motion, own_terms, **shared)
        output = {}
        for mode, contact in (('recorded_native_robot', native), ('predicted_robot', own)):
            metrics, edge_counts = ball_metrics(contact, batch)
            finite = jnp.ones(raw['q_before'].shape[:2], bool)
            for value in contact['after'][:3]:
                finite &= jnp.all(jnp.isfinite(value), axis=-1)
            if mode == 'predicted_robot':
                metrics.update(robot_metrics(robot, raw, batch))
                finite &= jnp.all(jnp.isfinite(robot['q_after_rad']), axis=-1)
                finite &= jnp.all(jnp.isfinite(robot['dq_after_rad_s']), axis=-1)
            output[mode] = dict(metrics=metrics, events=edge_counts,
                nonfinite_windows=jnp.sum(~jnp.all(finite, axis=0)))
        return output
    return predict_batch
