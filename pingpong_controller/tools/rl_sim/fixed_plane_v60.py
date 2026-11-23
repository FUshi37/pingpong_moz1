"""State transition for recoverable then terminal late weak rebounds."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def late_weak_streak_transition(
    *,
    previous: jax.Array,
    weak_event: jax.Array,
    good_event: jax.Array,
    limit: int,
) -> tuple[jax.Array, jax.Array]:
    """Count late weak outcomes; a good physical rebound clears the streak."""

    if limit not in (2, 3, 4):
        raise ValueError("V60 late weak streak limit must be 2, 3 or 4")
    prev = jnp.asarray(previous, dtype=jnp.int32)
    weak = jnp.asarray(weak_event, dtype=bool)
    good = jnp.asarray(good_event, dtype=bool)
    if prev.shape != weak.shape or prev.shape != good.shape:
        raise ValueError("V60 streak and physical-event shapes must match")
    # JAX booleans may be traced, so overlap is checked only for concrete data.
    if not isinstance(weak, jax.core.Tracer) and bool(jnp.any(weak & good)):
        raise ValueError("V60 weak and good physical events cannot overlap")
    next_count = jnp.where(weak, jnp.minimum(prev + 1, limit),
                           jnp.where(good, 0, prev))
    failed = weak & (next_count >= limit)
    return next_count, failed
