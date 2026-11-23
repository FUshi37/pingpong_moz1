"""Positive, two-sided late-apex credit for the opt-in V88 probe."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def positive_steady_height_credit(
    *,
    event: jax.Array,
    confirmed_hits_before_event: jax.Array,
    apex_z_m: jax.Array,
    previous_signed_error_m: jax.Array,
    previous_valid: jax.Array,
    weight: float,
) -> jax.Array:
    """Credit target-band rebounds without making fewer rebounds profitable.

    The adjacent factor rewards repeatability only when the current rebound is
    near the target. Earlier launch events have exactly zero credit.
    """
    apex_error_m = jnp.asarray(apex_z_m) - 1.425
    previous_error_m = jnp.asarray(previous_signed_error_m)
    target_quality = jnp.exp(-0.5 * (apex_error_m / 0.020) ** 2)
    adjacent_quality = jnp.exp(
        -0.5 * ((apex_error_m - previous_error_m) / 0.030) ** 2
    )
    adjacent_bonus = jnp.where(
        jnp.asarray(previous_valid, dtype=bool), adjacent_quality, 0.0
    )
    active = jnp.asarray(event, dtype=bool) & (
        jnp.asarray(confirmed_hits_before_event) >= 3
    )
    return jnp.where(
        active, weight * target_quality * (8.0 + 2.0 * adjacent_bonus), 0.0
    )
