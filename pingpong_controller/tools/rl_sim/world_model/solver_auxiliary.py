"""Independent 47-input diagnostic of native solver-stage racket J*qacc.

The six outputs are world linear/angular terms at the racket geom origin.
They exclude Jdot*dq and are not measured accelerations. No production world
or contact model imports this module. Inputs retain the execution proxy's
RightArm-0..6 / LegWaist-0..5 ordering and existing delayed arm commands.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .neural import mlp
from .robot_model import RobotScale
from .robot_support_model import support_features


WIDTHS = (47, 128, 128, 6)


def execution_features(trace: dict):
    """Read only current measured joints and currently applied commands."""
    q = jnp.concatenate((trace['q_before'], trace['support_q_before']), axis=-1)
    dq = jnp.concatenate((trace['dq_before'], trace['support_dq_before']), axis=-1)
    commands = {key: trace[f'delayed_{key}_command'] for key in ('q', 'dq', 'ddq')}
    return support_features(q, dq, commands)


def normal_projection_coefficients(ball_normal, contact_offset):
    """Map racket geom-origin J*qacc to the robot normal constraint term."""
    return jnp.concatenate((-ball_normal, -jnp.cross(contact_offset, ball_normal)), axis=-1)


def scale_from_statistics(inputs: dict, outputs: dict) -> RobotScale:
    """Require finite training-only normalization for the exact 47/6 layout."""
    for stats, size in ((inputs, 47), (outputs, 6)):
        if (stats.get('source') != 'train' or stats.get('count', 0) < 1
                or np.shape(stats['mean']) != (size,)
                or np.shape(stats['variance']) != (size,)
                or not np.isfinite(stats['mean']).all()
                or not np.isfinite(stats['variance']).all()
                or np.any(np.asarray(stats['variance']) < 0)):
            raise ValueError('Expected training-only finite 47/6 statistics')
    return RobotScale(jnp.asarray(inputs['mean'], jnp.float32),
        jnp.sqrt(jnp.maximum(jnp.asarray(inputs['variance'], jnp.float32), 1e-8)),
        jnp.asarray(outputs['mean'], jnp.float32),
        jnp.sqrt(jnp.maximum(jnp.asarray(outputs['variance'], jnp.float32), 1.)))


def predict(parameters, scale, features):
    """Predict six terms using only the approved 47 execution inputs."""
    if features.shape[-1] != 47:
        raise ValueError('Auxiliary predictor accepts exactly 47 execution features')
    with jax.default_matmul_precision('highest'):
        normalized = mlp(parameters, (features-scale.input_mean)/scale.input_scale)
    return normalized*scale.residual_scale+scale.residual_mean


def supervised_loss(parameters, scale, features, targets):
    """Six-component normalized squared loss; native values are labels only."""
    error = (predict(parameters, scale, features)-targets)/scale.residual_scale
    return jnp.mean(error**2)


def normal_velocity_error(prediction, rows, *, mass_kg=.004, dt_s=.001):
    """Offline scalar equation diagnostic, not a learned contact transition.

    Native geometry/aref/regularizer/force here are evaluation labels. This
    teacher-conditioned quantity cannot qualify imagined H8 ball prediction.
    """
    robot_term = np.sum(np.asarray(prediction)*rows['normal_coefficients'], axis=-1)
    force = np.maximum((rows['aref']-rows['gravity']-robot_term)
                       /(1/mass_kg+rows['regularizer']), 0.)
    return (force-rows['force_n'])*dt_s/mass_kg
