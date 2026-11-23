"""Bounded one-sided apex-error expansion cost between physical rebounds."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def late_apex_error_expansion_penalty(
    *,
    event: jax.Array,
    previous_valid: jax.Array,
    previous_error_m: jax.Array,
    current_apex_z_m: jax.Array,
    target_apex_z_m: jax.Array,
    deadband_m: float,
    scale_m: float,
    weight: float,
) -> jax.Array:
    """Penalize worsening only, so recovery after a weak rebound stays useful."""

    if not all(math.isfinite(x) for x in (deadband_m, scale_m, weight)):
        raise ValueError("V58 apex expansion values must be finite")
    if deadband_m < 0.0 or scale_m <= 0.0 or weight < 0.0:
        raise ValueError("V58 requires nonnegative band/weight and positive scale")
    current_error = jnp.abs(
        jnp.asarray(current_apex_z_m) - jnp.asarray(target_apex_z_m)
    )
    previous_error = jnp.asarray(previous_error_m)
    if current_error.shape != previous_error.shape:
        raise ValueError("V58 previous and current apex-error shapes must match")
    expansion = jnp.maximum(
        0.0, current_error - previous_error - deadband_m
    )
    penalty = -weight * jnp.clip(expansion / scale_m, 0.0, 2.0)
    return jnp.where(
        jnp.asarray(event, dtype=bool) & jnp.asarray(previous_valid, dtype=bool),
        penalty,
        0.0,
    )
