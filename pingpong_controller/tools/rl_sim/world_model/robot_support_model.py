"""47->128->128->26 robot execution model with measured support motion.

Joint states are RightArm-0..6 followed by LegWaist-0..5. Commands remain
seven dimensional; waist actuators retain the native constant hold targets.
The 35 input and 14 output prefixes preserve the arm-only diagnostic model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .neural import mlp
from .robot_model import RobotScale, robot_features


JOINT_NAMES = tuple(f'RightArm-{i}' for i in range(7)) + tuple(
    f'LegWaist-{i}' for i in range(6))


def expand_arm_parameters(parameters):
    """Copy a 35/128/128/14 model; initialize all additional weights to zero.

    Arm predictions are preserved when normalization of the existing prefixes
    is retained. Shape-changing migration starts a fresh optimizer explicitly.
    """
    if (len(parameters) != 3 or tuple(p['w'].shape for p in parameters)
            != ((35, 128), (128, 128), (128, 14))
            or tuple(p['b'].shape for p in parameters) != ((128,), (128,), (14,))):
        raise ValueError('Expected the preserved 35/128/128/14 arm model')
    first = dict(parameters[0], w=jnp.pad(parameters[0]['w'], ((0, 12), (0, 0))))
    last = dict(parameters[-1], w=jnp.pad(parameters[-1]['w'], ((0, 0), (0, 12))),
                b=jnp.pad(parameters[-1]['b'], ((0, 12),)))
    return first, dict(parameters[1]), last


def support_features(q_rad, dq_rad_s, commands: dict):
    """Return (...,47) from thirteen measured joints and seven arm commands."""
    if q_rad.shape[-1] != 13 or dq_rad_s.shape != q_rad.shape:
        raise ValueError('Support execution requires thirteen q/dq values')
    if any(commands[name].shape != q_rad.shape[:-1]+(7,)
           for name in ('q', 'dq', 'ddq')):
        raise ValueError('Commands must retain seven right-arm components')
    return jnp.concatenate((robot_features(q_rad[..., :7], dq_rad_s[..., :7], commands),
                            q_rad[..., 7:], dq_rad_s[..., 7:]), axis=-1)


def support_residual_target(q, dq, next_q, next_dq, *, dt_s=.001):
    """Return arm delta-q/dq then support delta-q/dq, in 7/7/6/6 blocks."""
    delta_q, delta_dq = next_q-q-dq*dt_s, next_dq-dq
    return jnp.concatenate((delta_q[..., :7], delta_dq[..., :7],
                            delta_q[..., 7:], delta_dq[..., 7:]), axis=-1)


def support_scale_from_statistics(inputs: dict, residuals: dict) -> RobotScale:
    """Construct normalization exclusively from the training partition."""
    for stats, size in ((inputs, 47), (residuals, 26)):
        if (stats.get('source') != 'train' or stats.get('count', 0) < 1
                or np.shape(stats['mean']) != (size,)
                or np.shape(stats['variance']) != (size,)
                or not np.isfinite(stats['mean']).all()
                or not np.isfinite(stats['variance']).all()
                or np.any(np.asarray(stats['variance']) < 0)):
            raise ValueError('Support scaling requires finite training-only 47/26-D statistics')
    floors = jnp.array([1e-14]*7+[1e-8]*7+[1e-14]*6+[1e-8]*6, jnp.float32)
    return RobotScale(jnp.asarray(inputs['mean'], jnp.float32),
        jnp.sqrt(jnp.maximum(jnp.asarray(inputs['variance'], jnp.float32), 1e-8)),
        jnp.asarray(residuals['mean'], jnp.float32),
        jnp.sqrt(jnp.maximum(jnp.asarray(residuals['variance'], jnp.float32), floors)))


def support_robot_step(parameters, scale, q_rad, dq_rad_s, commands, *, dt_s=.001):
    """Advance all thirteen measured joints without observed future feedback."""
    features = support_features(q_rad, dq_rad_s, commands)
    residual = mlp(parameters, (features-scale.input_mean)/scale.input_scale)
    residual = residual*scale.residual_scale+scale.residual_mean
    delta_q = jnp.concatenate((residual[..., :7], residual[..., 14:20]), axis=-1)
    delta_dq = jnp.concatenate((residual[..., 7:14], residual[..., 20:26]), axis=-1)
    return q_rad+dq_rad_s*dt_s+delta_q, dq_rad_s+delta_dq


def support_robot_rollout(parameters, scale, q_rad, dq_rad_s, commands, *, dt_s=.001):
    """Autoregress arm and waist state; recorded commands isolate this module."""
    def tick(state, command):
        after = support_robot_step(parameters, scale, *state, command, dt_s=dt_s)
        return after, after
    return jax.lax.scan(tick, (q_rad, dq_rad_s), commands)
