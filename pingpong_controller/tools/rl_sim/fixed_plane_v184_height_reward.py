"""Default-off upper-tail shaping for resolved fourth-plus ball impacts."""

import math

import jax
import jax.numpy as jnp


def upper_height_penalty(*, eligible: jax.Array, apex_z_m: jax.Array,
                         event_weight: float, shoulder_weight: float) -> jax.Array:
    """Return (N,) costs without changing the lower-height reward.

    The existing physical-event mask excludes launch and unresolved contacts.
    A quadratic shoulder starts at 1.435 m and reaches shoulder_weight at
    1.450 m. A separate event cost applies strictly above 1.450 m. Zero
    weights preserve the previous reward; contact counting is never altered.
    """
    for weight in (event_weight, shoulder_weight):
        if not math.isfinite(float(weight)) or float(weight) < 0.:
            raise ValueError("Upper-height weights must be finite and nonnegative")
    active = jnp.asarray(eligible, dtype=bool)
    apex = jnp.asarray(apex_z_m, dtype=jnp.float32)
    if active.ndim != 1 or active.shape != apex.shape:
        raise ValueError("Upper-height inputs must have identical (N,) shapes")
    shoulder = jnp.maximum((apex - 1.435) / .015, 0.)
    cost = float(event_weight) * (apex > 1.450).astype(jnp.float32)
    cost += float(shoulder_weight) * shoulder * shoulder
    return jnp.where(active, -cost, 0.).astype(jnp.float32)
