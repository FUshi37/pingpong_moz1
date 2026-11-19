"""Independent per-environment XYZ Gaussian AR(1) camera-position errors."""

import jax
import jax.numpy as jnp


def initialize_noise(keys: jax.Array, *, std_m: tuple[float, ...]) -> jax.Array:
    """Draw stationary epsilon in metres, shape (N, 3), at episode reset."""
    normal = jax.vmap(lambda key: jax.random.normal(key, (3,), dtype=jnp.float32))(keys)
    return normal * jnp.asarray(std_m, dtype=jnp.float32)


def advance_noise(
    epsilon_m: jax.Array, keys: jax.Array, *, dt_s: jax.Array,
    refresh: jax.Array, std_m: tuple[float, ...], rho_at_90hz: tuple[float, ...],
) -> jax.Array:
    """Advance (N, 3) stationary errors only on fresh observations.

    ``dt_s`` is the actual interval since the preceding camera update, (N,).
    Axis innovations and environment keys are independent. The conversion
    also handles 90/120 Hz sampling on the 200 Hz policy tick grid.
    """
    rho = jnp.asarray(rho_at_90hz, dtype=jnp.float32) ** (
        90. * jnp.asarray(dt_s, dtype=jnp.float32)[:, None])
    innovation = initialize_noise(keys, std_m=std_m)
    candidate = rho * epsilon_m + jnp.sqrt(jnp.maximum(0., 1. - rho**2)) * innovation
    return jnp.where(jnp.asarray(refresh)[:, None], candidate, epsilon_m)
