"""Observation-only precontact gate for opt-in V53 residual experiments."""

from __future__ import annotations

import jax
import jax.numpy as jnp


V53_ACTOR_OBS_DIM = 96
V87_ACTOR_OBS_DIM = 98


def v53_residual_precontact_gate(obs: jax.Array) -> jax.Array:
    """Return (N, 1) mask from V53/V87 actor-visible observations.

    Slots 33/40/41/95 are current observed ball vz, predicted contact time,
    prediction validity, and the launch-to-steady progress respectively.
    The V53 schedule reaches progress 1.0 only after three confirmed hits,
    so all earlier approaches are untouched. No simulator hit counter or
    future contact result is read.
    """

    values = jnp.asarray(obs, dtype=jnp.float32)
    if values.ndim != 2 or values.shape[-1] not in (
        V53_ACTOR_OBS_DIM, V87_ACTOR_OBS_DIM
    ):
        raise ValueError("V53/V87 residual gate requires (N, 96/98) observations")
    active = (
        (values[:, 95] >= 0.999)
        & (values[:, 41] > 0.5)
        & (values[:, 33] < -0.2)
        & (values[:, 40] > 0.0)
        & (values[:, 40] < 0.13)
    )
    return active[:, None].astype(values.dtype)
