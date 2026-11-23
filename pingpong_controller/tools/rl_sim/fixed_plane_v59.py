"""Bounded upper-apex event cost for view-safe, regular physical rebounds."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def late_high_apex_penalty(
    *,
    event: jax.Array,
    confirmed_hits_before_event: jax.Array,
    apex_z_m: jax.Array,
    start_z_m: float,
    scale_m: float,
    weight: float,
) -> jax.Array:
    """Charge upper overshoot on every fourth-plus resolved physical launch."""

    if not all(math.isfinite(x) for x in (start_z_m, scale_m, weight)):
        raise ValueError("V59 upper-apex values must be finite")
    if scale_m <= 0.0 or weight < 0.0:
        raise ValueError("V59 requires positive scale and nonnegative weight")
    apex = jnp.asarray(apex_z_m)
    hits = jnp.asarray(confirmed_hits_before_event)
    if apex.shape != hits.shape:
        raise ValueError("V59 apex and hit-count shapes must match")
    excess = jnp.maximum(0.0, apex - start_z_m)
    penalty = -weight * jnp.clip(excess / scale_m, 0.0, 1.5)
    active = jnp.asarray(event, dtype=bool) & (hits >= 3)
    return jnp.where(active, penalty, 0.0)
