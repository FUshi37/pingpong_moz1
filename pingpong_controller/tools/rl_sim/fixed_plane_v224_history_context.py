"""Causal, deployment-available temporal context for the V224 actor."""

import jax
import jax.numpy as jnp
import numpy as np

BASE_DIM = 72
LAGS_TICKS = (10, 20, 40)
HISTORY_TICKS = 41
CONTEXT_DIM = 45
ACTOR_DIM = BASE_DIM + CONTEXT_DIM
CRITIC_DIM = 393 + CONTEXT_DIM
SCHEMA = "v224_v149_prefix72_observed_local_history45"


def initialize_history(observation: jax.Array) -> jax.Array:
    """Initialize (N,41,72) from the current represented observation."""
    return jnp.broadcast_to(observation[:, None, :],
                            (observation.shape[0], HISTORY_TICKS, BASE_DIM))


def advance_history(history: jax.Array, observation: jax.Array) -> jax.Array:
    """Store newest-first actual actor prefixes, after observation DR."""
    return jnp.concatenate((observation[:, None, :], history[:, :-1]), axis=1)


def _rotation(value, xp):
    x = value[..., :3]
    x = x / xp.maximum(xp.linalg.norm(x, axis=-1, keepdims=True), 1.e-6)
    y = value[..., 3:] - x * xp.sum(x * value[..., 3:], axis=-1, keepdims=True)
    y = y / xp.maximum(xp.linalg.norm(y, axis=-1, keepdims=True), 1.e-6)
    return xp.stack((x, y, xp.cross(x, y)), axis=-1)


def _make_context(history, step_count, *, masked, xp):
    if history.ndim != 3 or history.shape[1:] != (HISTORY_TICKS, BASE_DIM):
        raise ValueError("V224 history requires (N,41,72), newest first")
    if masked:
        return xp.zeros((history.shape[0], CONTEXT_DIM), dtype=history.dtype)
    current = history[:, 0]
    rotation = _rotation(current[:, 55:61], xp)
    parts = []
    for lag in LAGS_TICKS:
        old = history[:, lag]
        old_rotation = _rotation(old[:, 55:61], xp)
        # Both transformations use observed rotations. No physical FK matrix,
        # sampled bias, ball velocity, impact oracle or RMP state is accepted.
        delta_ball = xp.einsum("nji,nj->ni", rotation,
                                current[:, 28:31] - old[:, 28:31])
        delta_normal = xp.einsum("nji,nj->ni", rotation,
                                  rotation[:, :, 2] - old_rotation[:, :, 2])
        valid = ((step_count >= lag) & (current[:, 71] > .5)
                 & (old[:, 71] > .5))
        value = xp.concatenate((delta_ball, delta_normal, old[:, 14:21],
                                old[:, 61:62], valid[:, None].astype(history.dtype)),
                               axis=-1)
        parts.append(xp.where(valid[:, None], value, 0.))
    return xp.concatenate(parts, axis=-1)


def make_context_jax(
    history: jax.Array, step_count: jax.Array, *, masked: bool = False,
) -> jax.Array:
    """Return (N,45); lag blocks contain 3+3+7+1+1 SI/age/mask values."""
    return _make_context(jnp.asarray(history), jnp.asarray(step_count),
                         masked=masked, xp=jnp)


def make_context_numpy(
    history: np.ndarray, step_count: np.ndarray, *, masked: bool = False,
) -> np.ndarray:
    """Reference runtime calculation; a new adapter must reproduce its state."""
    return _make_context(np.asarray(history, dtype=np.float32),
                         np.asarray(step_count), masked=masked, xp=np)


def bounded_flight_impulse(
    velocity_m_s: jax.Array, delta_xy_m_s: jax.Array, *, step: jax.Array,
    scheduled_step: jax.Array, already_applied: jax.Array, hit_count: jax.Array,
    in_contact: jax.Array, clearance_m: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply at most one late XY impulse at the first eligible flight tick."""
    event = ((step >= scheduled_step) & (~already_applied) & (hit_count >= 3)
             & (~in_contact) & (clearance_m >= .05))
    delta = jnp.concatenate((delta_xy_m_s,
                            jnp.zeros_like(delta_xy_m_s[:, :1])), axis=-1)
    return velocity_m_s + jnp.where(event[:, None], delta, 0.), event
