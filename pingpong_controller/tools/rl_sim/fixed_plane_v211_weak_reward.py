"""Bounded event cost for existing late physical weak-rebound resolutions."""

import math

import jax
import jax.numpy as jnp


def late_weak_event_penalty(*, weak_resolved_event: jax.Array,
                          confirmed_hits_before_event: jax.Array,
                          weight: float) -> jax.Array:
    """Charge once per weak resolution after three confirmed hits; inputs are (N,)."""
    if not math.isfinite(float(weight)) or weight < 0.:
        raise ValueError("Late weak penalty weight must be finite and nonnegative")
    event = jnp.asarray(weak_resolved_event, dtype=bool)
    hits = jnp.asarray(confirmed_hits_before_event)
    if event.ndim != 1 or hits.shape != event.shape:
        raise ValueError("Late weak event and hit counter must have identical (N,) shapes")
    return jnp.where(event & (hits >= 3), -float(weight), 0.)
