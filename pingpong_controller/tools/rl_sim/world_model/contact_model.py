"""Finite-duration 1 ms contact residual in a physically consistent local frame.

The auxiliary entry/release heads supervise events. They cannot create an
impulse outside the selected finite contact geometry. Robot FK supplies the
racket pose/twist; goals and noisy camera measurements are absent.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from .contact_geometry import surface_features
from .neural import mlp


# SI reference scales, not statistics estimated from a holdout partition.
FEATURE_SCALE = (.075, .075, .02, 3., 3., 3., 30., 30., 30., 1., 1., 1.,
                 9.81, 9.81, 9.81, .02, .075, 1., .01)
DELTA_SCALE = (.5, .5, .5, 10., 10., 10.)


def world_spin(quaternion_wxyz, angular_velocity_local_rad_s):
    """Convert MuJoCo ball-local angular qvel to the world-frame model state."""
    quaternion = quaternion_wxyz/jnp.maximum(jnp.linalg.norm(quaternion_wxyz, axis=-1, keepdims=True), 1e-12)
    vector, scalar = quaternion[..., 1:], quaternion[..., :1]
    twice_cross = 2*jnp.cross(vector, angular_velocity_local_rad_s)
    return angular_velocity_local_rad_s+scalar*twice_cross+jnp.cross(vector, twice_cross)


def contact_features(surface, *, ball_position_m, ball_velocity_m_s,
                     ball_omega_world_rad_s, geom_position_m, geom_rotation_world,
                     racket_linear_velocity_m_s, racket_angular_velocity_rad_s,
                     ball_radius_m, previous_contact, duration_s,
                     gravity_m_s2=(0., 0., -9.81)):
    """Return (...,19): p, relative v, spin, racket omega, g, gaps, mode, age.

    Racket linear velocity is at its geom center. Relative contact-point
    velocity includes both ball spin and racket angular motion. A world-frame
    angular ball state is sufficient for an isotropic sphere; its orientation
    is not an additional input to the collision network.
    """
    geometry = surface_features(surface, ball_position_m, geom_position_m,
                                geom_rotation_world, ball_radius_m)
    frame = geometry['frame_world']
    point = geometry['contact_position_world_m']
    lever_ball = point-ball_position_m
    relative_v = (ball_velocity_m_s+jnp.cross(ball_omega_world_rad_s, lever_ball)
                  -racket_linear_velocity_m_s-jnp.cross(racket_angular_velocity_rad_s, point-geom_position_m))

    def local(vector):
        return jnp.einsum('...ji,...j->...i', frame, vector)

    features = jnp.concatenate((geometry['position_surface_m'], local(relative_v),
        local(ball_omega_world_rad_s), local(racket_angular_velocity_rad_s),
        local(jnp.broadcast_to(jnp.asarray(gravity_m_s2, jnp.float32), ball_position_m.shape)),
        geometry['sphere_gap_m'][..., None], geometry['edge_signed_distance_m'][..., None],
        jnp.asarray(previous_contact, jnp.float32)[..., None], duration_s[..., None]), axis=-1)
    return features, geometry


class ContactPrediction(NamedTuple):
    delta_velocity_world_m_s: Any
    delta_omega_world_rad_s: Any
    entry_logit: Any
    release_logit: Any
    geometric_overlap: Any
    features: Any


def contact_prediction(parameters, surface, **physical_inputs) -> ContactPrediction:
    """Predict contact-only increments; known gravity is applied separately.

    No contact hysteresis extends learned forces into free flight. The previous
    contact mode/age are inputs for finite-duration responses; a caller carries
    them across all five substeps and across control-window boundaries.
    """
    features, geometry = contact_features(surface, **physical_inputs)
    raw = mlp(parameters, features/jnp.asarray(FEATURE_SCALE, jnp.float32))
    delta = raw[..., :6]*jnp.asarray(DELTA_SCALE, jnp.float32)
    overlap = geometry['sphere_gap_m'] <= 0.
    delta = jnp.where(overlap[..., None], delta, 0.)
    frame = geometry['frame_world']
    return ContactPrediction(jnp.einsum('...ij,...j->...i', frame, delta[..., :3]),
        jnp.einsum('...ij,...j->...i', frame, delta[..., 3:]),
        raw[..., 6], raw[..., 7], overlap, features)
