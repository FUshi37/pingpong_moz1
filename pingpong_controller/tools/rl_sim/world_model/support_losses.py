"""Thirteen-joint execution losses against native pose/twist supervision."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .kinematics import forward_kinematics
from .robot_support_model import support_robot_step, support_robot_rollout


def racket_kinematics(chain, q, dq):
    """Pose and world twist share the complete joint chain and the same time."""
    width, shape = len(chain.joint_ids), q.shape[:-1]
    if q.shape[-1] != width or q.shape != dq.shape:
        raise ValueError('FK state shape disagrees with the declared joint chain')
    fk = jax.vmap(lambda value: forward_kinematics(chain, value))(q.reshape(-1, width))
    velocity = jnp.einsum('nij,nj->ni', fk.geom_jacobian, dq.reshape(-1, width))
    omega = jnp.einsum('nij,nj->ni', fk.angular_jacobian, dq.reshape(-1, width))
    return dict(position=fk.site_position.reshape(shape+(3,)),
        geom_position=fk.geom_position.reshape(shape+(3,)),
        rotation=fk.geom_rotation.reshape(shape+(3, 3)),
        velocity=velocity.reshape(shape+(3,)), omega=omega.reshape(shape+(3,)))


def support_execution_loss(parameters, scale, batch, *, autoregressive, chain=None):
    """Predict arm and waist continuously; teacher future states are labels only.

    Native kinematic labels describe the preintegration interval. Their loss
    uses predicted states before that same interval, sampled every five ticks.
    There is no independent learned racket-pose output or future-state repair.
    """
    if autoregressive:
        _, (q, dq) = support_robot_rollout(parameters, scale,
            batch['q_before'][0], batch['dq_before'][0], batch['commands'])
    else:
        q, dq = support_robot_step(parameters, scale,
            batch['q_before'], batch['dq_before'], batch['commands'])
    # Native pose is recorded before integration. The preceding prediction
    # reaches that exact time in both bootstrap and autoregressive fitting.
    # Comparing teacher inputs to native labels would have zero model gradient.
    before_q = jnp.concatenate((batch['q_before'][:1], q[:-1]))
    before_dq = jnp.concatenate((batch['dq_before'][:1], dq[:-1]))
    mask = batch['valid'].astype(jnp.float32)
    denominator = jnp.maximum(mask.sum(), 1.)

    def mean_error(value, target, unit):
        return jnp.sum(jnp.mean(jnp.square((value-target)/unit), axis=-1)*mask)/denominator

    metrics = dict(
        arm_q_loss=mean_error(q[..., :7], batch['q_after'][..., :7], .01),
        arm_dq_loss=mean_error(dq[..., :7], batch['dq_after'][..., :7], .10),
        support_q_loss=mean_error(q[..., 7:], batch['q_after'][..., 7:], .001),
        support_dq_loss=mean_error(dq[..., 7:], batch['dq_after'][..., 7:], .01))
    if chain is not None:
        predicted = racket_kinematics(chain, before_q[4::5], before_dq[4::5])
        selected_mask = mask[4::5]
        selected_denominator = jnp.maximum(selected_mask.sum(), 1.)

        def pose_loss(value):
            return jnp.sum(value*selected_mask)/selected_denominator

        target_rotation = batch['racket_geom_rotation_before'][4::5]
        # Smooth chordal loss avoids arccos gradients at an exactly correct pose.
        rotation_error = jnp.sum((predicted['rotation']-target_rotation)**2, axis=(-1, -2))
        metrics['racket_rotation_loss'] = pose_loss(rotation_error/(2*.01**2))
        for key, target_key, unit in (
            ('position', 'racket_site_position_before_m', .003),
            ('velocity', 'racket_linear_velocity_before_m_s', .05),
            ('omega', 'racket_angular_velocity_before_rad_s', .10)):
            error = jnp.mean(((predicted[key]-batch[target_key][4::5])/unit)**2, axis=-1)
            metrics[f'racket_{key}_loss'] = pose_loss(error)
    loss = sum(metrics.values())
    return loss, {**metrics, 'loss': loss}
