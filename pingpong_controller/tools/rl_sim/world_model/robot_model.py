"""35->128->128->14 execution residual, conditioned on actual delayed commands.

This module predicts measured q/dq; the known RMP/projection path still owns
commands, filters and delays. Goal coordinates are never network inputs.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .neural import mlp


class RobotScale(NamedTuple):
    input_mean: Any
    input_scale: Any
    residual_mean: Any
    residual_scale: Any


def scale_from_training_statistics(inputs: dict, residuals: dict) -> RobotScale:
    """Only train-partition statistics may define learned dynamics scaling."""
    for stats, size in ((inputs, 35), (residuals, 14)):
        if (stats.get('source') != 'train' or stats.get('count', 0) < 1
                or np.shape(stats['mean']) != (size,) or np.shape(stats['variance']) != (size,)):
            raise ValueError('Robot scaling requires training-only 35/14-D statistics')
        if not np.isfinite(stats['mean']).all() or not np.isfinite(stats['variance']).all() or np.any(np.asarray(stats['variance']) < 0):
            raise ValueError('Invalid training normalization statistics')
    return RobotScale(jnp.asarray(inputs['mean'], jnp.float32),
        jnp.sqrt(jnp.maximum(jnp.asarray(inputs['variance'], jnp.float32), 1e-8)),
        jnp.asarray(residuals['mean'], jnp.float32),
        jnp.sqrt(jnp.maximum(jnp.asarray(residuals['variance'], jnp.float32),
                            jnp.array([1e-14]*7+[1e-8]*7, jnp.float32))))


def robot_features(q_rad, dq_rad_s, commands: dict):
    """Return (...,35); seven-joint order is RightArm-0 through RightArm-6."""
    return jnp.concatenate((q_rad, dq_rad_s, commands['q']-q_rad,
                            commands['dq'], commands['ddq']), axis=-1)


def robot_residual_target(q, dq, next_q, next_dq, *, dt_s=.001):
    return jnp.concatenate((next_q-q-dq*dt_s, next_dq-dq), axis=-1)


def robot_step(parameters, scale: RobotScale, q_rad, dq_rad_s, commands, *, dt_s=.001):
    """Apply a learned residual to the explicit inertial baseline.

    No clipping hides a divergent learned execution model. Qualification and
    model-truncation handling belong to its caller. FK derives racket state
    from these joint predictions, rather than a second independent pose head.
    """
    features = robot_features(q_rad, dq_rad_s, commands)
    normalized = (features-scale.input_mean)/scale.input_scale
    residual = mlp(parameters, normalized)*scale.residual_scale+scale.residual_mean
    return q_rad+dq_rad_s*dt_s+residual[..., :7], dq_rad_s+residual[..., 7:]


def robot_rollout(parameters, scale, q_rad, dq_rad_s, commands, *, dt_s=.001):
    """Module-level autoregression with a recorded delayed-command sequence.

    Only the first q/dq is observed. This isolated module diagnostic is not a
    full world rollout: the composed world must generate new delayed commands
    from its own predicted measured-feedback controller state.
    """
    def tick(state, command):
        after = robot_step(parameters, scale, *state, command, dt_s=dt_s)
        return after, after
    return jax.lax.scan(tick, (q_rad, dq_rad_s), commands)
