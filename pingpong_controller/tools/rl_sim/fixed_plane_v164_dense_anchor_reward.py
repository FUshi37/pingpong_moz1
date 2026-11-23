"""Continuous position-only rewards for stationary centered juggling."""
import math

import jax.numpy as jnp


def dense_anchor_xy_penalties(ball_position_m, racket_position_m, anchor_m, *,
                              control_dt_s, ball_weight, racket_weight,
                              scale_m=.003):
    """Return two (N,) costs from physical (N,3) positions in metres.

    The common episode anchor prevents the ball and racket from drifting
    together without a contact-center cost. There is no XY deadband; height
    and control commands are independent of this reward computation.
    """
    weights = (float(ball_weight), float(racket_weight))
    dt, scale = float(control_dt_s), float(scale_m)
    if any(not math.isfinite(w) or w < 0. for w in weights):
        raise ValueError('Dense anchor weights must be finite and nonnegative')
    if not math.isfinite(dt) or dt <= 0. or not math.isfinite(scale) or scale <= 0.:
        raise ValueError('Dense anchor time and distance scales must be positive')
    ball, racket, anchor = map(jnp.asarray, (ball_position_m, racket_position_m, anchor_m))
    if (ball.ndim != 2 or ball.shape[-1] != 3 or racket.shape != ball.shape
            or anchor.shape not in (ball.shape, (3,), (1, 3))):
        raise ValueError('Dense anchor positions require matching (N,3) shapes')
    ball_distance = jnp.linalg.norm(ball[:, :2] - anchor[..., :2], axis=-1)
    racket_distance = jnp.linalg.norm(racket[:, :2] - anchor[..., :2], axis=-1)
    return (-weights[0] * dt / scale * ball_distance,
            -weights[1] * dt / scale * racket_distance)
