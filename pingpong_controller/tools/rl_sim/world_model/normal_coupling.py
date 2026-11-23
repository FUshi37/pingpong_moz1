"""Causal robot-conditioned normal prior, separate from contact39 features."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .normal_contact import normal_contact_loss, normal_contact_rollout
from .physics_fitting import coupled_contact_objective
from .physics_metrics import ball_metrics, robot_metrics
from .position_contact_data import prepare_position_contact_batch
from .robot_support_model import support_features, support_robot_rollout
from .solver_auxiliary import predict
from .support_losses import racket_kinematics


def acceleration_terms(auxiliary, q, dq, commands):
    return predict(auxiliary['parameters'], auxiliary['scale'], support_features(q, dq, commands))


def own_robot_motion(robot_parameters, raw, *, scale, chain):
    """Recurse from initial13-D q/dq only; never use later measured joints."""
    commands = {key: raw['delayed_'+key+'_command'] for key in ('q', 'dq', 'ddq')}
    _, (after_q, after_dq) = support_robot_rollout(robot_parameters, scale,
        raw['q_before'][0], raw['dq_before'][0], commands)
    q = jnp.concatenate((raw['q_before'][:1], after_q[:-1]))
    dq = jnp.concatenate((raw['dq_before'][:1], after_dq[:-1]))
    pose = racket_kinematics(chain, q, dq)
    motion = dict(geom_position_m=pose['geom_position'], geom_rotation_world=pose['rotation'],
        racket_linear_velocity_m_s=pose['velocity'], racket_angular_velocity_rad_s=pose['omega'],
        joint_q_rad=q, delayed_q_command_rad=commands['q'])
    trace = dict(q_before_rad=q, dq_before_rad_s=dq, q_after_rad=after_q,
                 dq_after_rad_s=after_dq, racket_before=pose)
    return motion, trace, commands


def normal_coupled_objective(parameters, robot_parameters, auxiliary, raw, *,
                             options, normal_enabled):
    batch = prepare_position_contact_batch(None, raw)
    motion, robot, commands = own_robot_motion(robot_parameters, raw,
        scale=options['robot_scale'], chain=options['chain'])
    shared = dict(ball_radius_m=options['ball_radius_m'],
                  position_scale=options['position_scale'], contract=options['contact_contract'])
    if not normal_enabled:
        return coupled_contact_objective(parameters, options['surface'], batch, motion,
            ball_radius_m=shared['ball_radius_m'], position_scale=shared['position_scale'],
            contact_contract=shared['contract'], auxiliary_weight=256.)
    own_terms = acceleration_terms(auxiliary, robot['q_before_rad'], robot['dq_before_rad_s'], commands)
    recursive, metrics = normal_contact_loss(parameters, options['surface'], dict(batch, racket=motion),
        own_terms, autoregressive=True, **shared)
    # This is the existing separate teacher-forced training anchor. Its values
    # never correct the recursively generated robot/ball trajectory above.
    native_terms = acceleration_terms(auxiliary, raw['q_before'], raw['dq_before'], commands)
    anchor, aux_metrics = normal_contact_loss(parameters, options['surface'], batch,
        native_terms, autoregressive=False, **shared)
    total = recursive+256.*anchor
    return total, dict(metrics, loss=total, recursive_loss=recursive,
        auxiliary_single_step_loss=anchor, auxiliary_weighted_loss=256.*anchor,
        **{'single_step_'+key: value for key, value in aux_metrics.items()})


def make_predictor(parameters, options, auxiliary, *, normal_enabled, learned):
    """Build the same evaluation contract, with two explicit robot conditions."""
    def predict_batch(raw):
        batch = prepare_position_contact_batch(None, raw)
        ball = jax.tree.map(lambda value: value[0], batch['before'])
        motion, robot, commands = own_robot_motion(parameters['robot'], raw,
            scale=options['robot_scale'], chain=options['chain'])
        shared = dict(ball_radius_m=options['ball_radius_m'], position_scale=options['position_scale'],
                      contract=options['contact_contract'], learned=learned)
        native_terms = own_terms = None
        if normal_enabled:
            native_terms = acceleration_terms(auxiliary, raw['q_before'], raw['dq_before'], commands)
            own_terms = acceleration_terms(auxiliary, robot['q_before_rad'], robot['dq_before_rad_s'], commands)
        _, native = normal_contact_rollout(parameters['contact'], options['surface'], ball,
            batch['racket'], native_terms, **shared)
        _, own = normal_contact_rollout(parameters['contact'], options['surface'], ball,
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
