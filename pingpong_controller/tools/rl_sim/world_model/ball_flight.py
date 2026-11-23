"""Known ball flight, separated from learned contact and task proposals.

The reference 1 ms dynamics use MuJoCo's semi-implicit position update.
Continuous ballistic formulas propose experiments and provide shaping signals;
neither their proposed contact nor their predicted apex is a measured event.
"""

from __future__ import annotations

import jax.numpy as jnp


def free_flight_substep(position_m, velocity_m_s, gravity_m_s2=(0., 0., -9.81), *, dt_s=.001):
    """Advance (...,3) world-frame state for one no-drag, no-contact substep."""
    velocity = velocity_m_s+jnp.asarray(gravity_m_s2)*dt_s
    return position_m+velocity*dt_s, velocity


def launch_to_contact(contact_position_m, incoming_velocity_m_s, time_to_contact_s,
                      gravity_m_s2=(0., 0., -9.81)):
    """Propose a pre-contact flight; the teacher determines the actual outcome."""
    gravity = jnp.asarray(gravity_m_s2)
    duration = jnp.asarray(time_to_contact_s)[..., None]
    velocity = incoming_velocity_m_s-gravity*duration
    position = contact_position_m-incoming_velocity_m_s*duration+.5*gravity*duration**2
    return position, velocity


def ballistic_apex(position_m, velocity_m_s, *, gravity_z_m_s2=-9.81):
    """Return predicted XYZ apex, time and validity for an upward free flight."""
    valid = jnp.isfinite(position_m).all(axis=-1) & jnp.isfinite(velocity_m_s).all(axis=-1)
    valid = valid & (velocity_m_s[..., 2] > 0.) & (gravity_z_m_s2 < 0.)
    time = jnp.where(valid, velocity_m_s[..., 2]/(-gravity_z_m_s2), 0.)
    displacement = velocity_m_s*time[..., None]
    displacement = displacement.at[..., 2].add(.5*gravity_z_m_s2*time**2)
    predicted = position_m+displacement
    return jnp.where(valid[..., None], predicted, position_m), time, valid


def velocity_for_apex(hit_position_m, target_apex_m, *, gravity_z_m_s2=-9.81):
    """Return an outgoing-velocity proposal, never an applied ball impulse."""
    height = target_apex_m[..., 2]-hit_position_m[..., 2]
    valid = ((height > 0.) & (gravity_z_m_s2 < 0.)
             & jnp.isfinite(hit_position_m).all(axis=-1) & jnp.isfinite(target_apex_m).all(axis=-1))
    vertical = jnp.sqrt(jnp.maximum(2*(-gravity_z_m_s2)*height, 1e-12))
    time = vertical/jnp.maximum(-gravity_z_m_s2, 1e-12)
    horizontal = (target_apex_m[..., :2]-hit_position_m[..., :2])/jnp.maximum(time[..., None], 1e-12)
    velocity = jnp.concatenate((horizontal, vertical[..., None]), axis=-1)
    return jnp.where(valid[..., None], velocity, 0.), valid
