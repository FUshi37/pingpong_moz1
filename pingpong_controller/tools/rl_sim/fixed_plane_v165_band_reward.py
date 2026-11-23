"""Explicit event cost for already-defined physical steady-height violations."""
import math

import jax.numpy as jnp


def steady_band_violation_penalty(*, eligible, in_band, weight: float):
    """Return (N,) event costs; the existing V92 mask excludes the first 3 hits."""
    if not math.isfinite(float(weight)) or weight < 0.:
        raise ValueError('Band violation weight must be finite and nonnegative')
    eligible = jnp.asarray(eligible, dtype=bool)
    in_band = jnp.asarray(in_band, dtype=bool)
    if eligible.ndim != 1 or eligible.shape != in_band.shape:
        raise ValueError('Band masks must have identical (N,) shapes')
    return jnp.where(eligible & ~in_band, -float(weight), 0.).astype(jnp.float32)
