"""Finite-duration contact plus known 1 ms flight, with persistent contact age."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .contact_model import contact_prediction


class ContactState(NamedTuple):
    position_m: Any
    velocity_m_s: Any
    omega_world_rad_s: Any
    previous_contact: Any
    duration_s: Any


def contact_step(parameters, surface, state: ContactState, racket: dict, *,
                 ball_radius_m: float, dt_s: float = .001,
                 gravity_m_s2=(0., 0., -9.81)):
    """Apply one semi-implicit physical substep; no task or camera inputs."""
    prediction = contact_prediction(parameters, surface,
        ball_position_m=state.position_m, ball_velocity_m_s=state.velocity_m_s,
        ball_omega_world_rad_s=state.omega_world_rad_s,
        previous_contact=state.previous_contact, duration_s=state.duration_s,
        ball_radius_m=ball_radius_m, gravity_m_s2=gravity_m_s2, **racket)
    velocity = (state.velocity_m_s+jnp.asarray(gravity_m_s2, jnp.float32)*dt_s
                +prediction.delta_velocity_world_m_s)
    overlap = prediction.geometric_overlap
    entry = overlap & ~state.previous_contact
    release = ~overlap & state.previous_contact
    duration = jnp.where(overlap, jnp.where(entry, 0., state.duration_s)+dt_s, 0.)
    after = ContactState(state.position_m+velocity*dt_s, velocity,
        state.omega_world_rad_s+prediction.delta_omega_world_rad_s, overlap, duration)
    return after, dict(before=state, after=after, overlap=overlap, entry=entry,
        release=release, entry_logit=prediction.entry_logit, release_logit=prediction.release_logit)


def contact_rollout(parameters, surface, initial: ContactState, racket: dict, *,
                    ball_radius_m: float, dt_s: float = .001):
    """Recurse with known racket motion for module fitting and diagnostics.

    The composed world supplies racket motion from its own predicted robot FK.
    This isolated contact-module path may instead use recorded robot FK.
    """
    def tick(state, motion):
        return contact_step(parameters, surface, state, motion,
                            ball_radius_m=ball_radius_m, dt_s=dt_s)
    return jax.lax.scan(tick, initial, racket)
