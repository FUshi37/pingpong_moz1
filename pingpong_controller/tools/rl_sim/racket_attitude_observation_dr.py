"""Episode-coherent local attitude observation bias; no physical mutation."""

import math

import jax
import jax.numpy as jnp


def sample_episode_tilt_rad(keys: jax.Array, *, max_angle_deg: float,
                            nominal_fraction: float) -> jax.Array:
    """Sample (N,3) local rotation vectors in an XY disk, with nominal episodes."""
    if not math.isfinite(max_angle_deg) or not 0. <= max_angle_deg <= 1.:
        raise ValueError("Observation tilt must be finite and between0 and1 degree")
    if not math.isfinite(nominal_fraction) or not 0. <= nominal_fraction <= 1.:
        raise ValueError("Nominal episode fraction must be between0 and1")
    uniforms = jax.vmap(lambda k: jax.random.uniform(
        jax.random.fold_in(k, 22301), (3,), dtype=jnp.float32))(keys)
    radius = math.radians(max_angle_deg) * jnp.sqrt(uniforms[:, 0])
    azimuth = 2. * math.pi * uniforms[:, 1]
    bias = jnp.stack((radius * jnp.cos(azimuth), radius * jnp.sin(azimuth),
                      jnp.zeros_like(radius)), axis=-1)
    return jnp.where((uniforms[:, 2] >= nominal_fraction)[:, None], bias, 0.)


def apply_local_tilt(rotation: jax.Array, rotation_vector_rad: jax.Array) -> jax.Array:
    """Right-compose a legal SO(3) rotation; preserve exact zero-bias rows."""
    vector = jnp.asarray(rotation_vector_rad, dtype=jnp.float32)
    x, y, z = vector[:, 0], vector[:, 1], vector[:, 2]
    zero = jnp.zeros_like(x)
    cross = jnp.stack((zero, -z, y, z, zero, -x, -y, x, zero), axis=-1)
    cross = cross.reshape((-1, 3, 3))
    angle = jnp.linalg.norm(vector, axis=-1)
    sine_scale = jnp.sinc(angle / math.pi)
    cosine_scale = .5 * jnp.square(jnp.sinc(angle / (2. * math.pi)))
    delta = (jnp.eye(3, dtype=vector.dtype)[None, :, :]
             + sine_scale[:, None, None] * cross
             + cosine_scale[:, None, None] * (cross @ cross))
    changed = rotation @ delta
    return jnp.where((angle > 0.)[:, None, None], changed, rotation)
