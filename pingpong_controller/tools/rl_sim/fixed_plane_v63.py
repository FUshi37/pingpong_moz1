"""Deployment-observable previous-apex error suffix for the opt-in V63 actor."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def previous_apex_error_observation(
    signed_error_m: jax.Array,
    valid: jax.Array,
) -> jax.Array:
    """Return [signed last-apex error / 0.10 m, valid] per lane."""

    error = jnp.asarray(signed_error_m, dtype=jnp.float32)
    mask = jnp.asarray(valid, dtype=bool)
    if error.shape != mask.shape or error.ndim != 1:
        raise ValueError("previous apex error and valid must be [n_envs]")
    normalized = jnp.where(mask, jnp.clip(error / 0.10, -2.0, 2.0), 0.0)
    return jnp.stack((normalized, mask.astype(jnp.float32)), axis=-1)
