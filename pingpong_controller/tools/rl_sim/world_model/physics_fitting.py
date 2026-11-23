"""Contact fitting with own-state full-chain robot predictions and fixed39 inputs."""

from __future__ import annotations

import jax.numpy as jnp

from .position_contact import position_contact_loss
from .robot_support_model import support_robot_rollout
from .support_losses import racket_kinematics


def predicted_racket_motion(robot_parameters, raw: dict, *, scale, chain) -> dict:
    """Predict robot motion from only its initial q/dq and recorded commands.

    The frozen execution proxy has no ball input, so evaluating it before the
    contact scan is equivalent to the interleaved physical recursion. Future
    native joint states are labels only and cannot correct this trajectory.
    """
    commands = {key:raw['delayed_'+key+'_command'] for key in ('q', 'dq', 'ddq')}
    _, (after_q, after_dq) = support_robot_rollout(robot_parameters, scale,
        raw['q_before'][0], raw['dq_before'][0], commands)
    q = jnp.concatenate((raw['q_before'][:1], after_q[:-1]))
    dq = jnp.concatenate((raw['dq_before'][:1], after_dq[:-1]))
    pose = racket_kinematics(chain, q, dq)
    return dict(geom_position_m=pose['geom_position'], geom_rotation_world=pose['rotation'],
        racket_linear_velocity_m_s=pose['velocity'], racket_angular_velocity_rad_s=pose['omega'],
        joint_q_rad=q, delayed_q_command_rad=commands['q'])


def coupled_contact_objective(parameters, surface, batch, motion, *, ball_radius_m,
                               position_scale, contact_contract, auxiliary_weight: float):
    """Own-ball/own-robot recursion plus a separate native single-step anchor."""
    options = dict(ball_radius_m=ball_radius_m, position_scale=position_scale,
                   contract=contact_contract)
    recursive, metrics = position_contact_loss(parameters, surface, dict(batch, racket=motion),
                                                autoregressive=True, **options)
    auxiliary, aux_metrics = position_contact_loss(parameters, surface, batch,
                                                    autoregressive=False, **options)
    total = recursive+auxiliary_weight*auxiliary
    return total, dict(metrics, loss=total, recursive_loss=recursive,
        auxiliary_single_step_loss=auxiliary, auxiliary_weighted_loss=auxiliary_weight*auxiliary,
        **{'single_step_'+key:value for key, value in aux_metrics.items()})
