"""Finite compiled racket-prism features for experiments and learned contact.

The V324 racket is a convex extruded polygon. Extract it once from its actual
compiled vertices, rather than substituting an infinite plane or ideal disk.
These ball/racket calculations are unrelated to offline robot self-collision
admission; no robot mesh queries occur in a runtime step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.spatial import ConvexHull

from fixed_contact_plane_training import racket_surface_frame


@dataclass(frozen=True)
class RacketSurface:
    normal_axis: int
    half_thickness_m: float
    vertices_xy_m: Any
    edge_outward_xy: Any
    edge_offset_m: Any


def surface_from_vertices(vertices, *, normal_axis: int) -> RacketSurface:
    """Validate the compiled two-layer convex prism before constructing features."""
    vertices = np.asarray(vertices, np.float64)
    if normal_axis not in (0, 1, 2) or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError('Compiled 3-D vertices and the declared thickness axis are required')
    if not np.isfinite(vertices).all():
        raise ValueError('Nonfinite racket mesh')
    normal = vertices[:, normal_axis]
    half = float(np.max(np.abs(normal)))
    if half <= 0 or not np.allclose(np.abs(normal), half, rtol=0., atol=1e-7):
        raise ValueError('Expected a centered two-layer compiled racket prism')
    axes = [(normal_axis+1) % 3, (normal_axis+2) % 3]
    first = np.unique(vertices[normal < 0][:, axes], axis=0)
    second = np.unique(vertices[normal > 0][:, axes], axis=0)
    if first.shape != second.shape or not np.allclose(first, second, rtol=0., atol=1e-7):
        raise ValueError('Racket faces must have the same finite polygon')
    hull = ConvexHull(first)
    if len(hull.vertices) != len(first):
        raise ValueError('Only the actual convex-prism contact mesh is supported')
    polygon = first[hull.vertices]
    edges = np.roll(polygon, -1, axis=0)-polygon
    outward = np.column_stack((edges[:, 1], -edges[:, 0]))
    outward /= np.linalg.norm(outward, axis=1)[:, None]
    offsets = np.sum(outward*polygon, axis=-1)
    if np.any(offsets <= 0.):
        raise ValueError('The compiled racket origin must lie inside its polygon')
    return RacketSurface(normal_axis, half, polygon.astype(np.float32),
                         outward.astype(np.float32), offsets.astype(np.float32))


def surface_features(surface: RacketSurface, ball_position_m, geom_position_m,
                     geom_rotation_world, ball_radius_m) -> dict:
    """Return exact prism gap and local physical features with leading batch axes."""
    local = jnp.einsum('...ji,...j->...i', geom_rotation_world, ball_position_m-geom_position_m)
    axis = surface.normal_axis
    xy = local[..., jnp.asarray([(axis+1) % 3, (axis+2) % 3])]
    vertices = jnp.asarray(surface.vertices_xy_m)
    edge = jnp.roll(vertices, -1, axis=0)-vertices
    relative = xy[..., None, :]-vertices
    weight = jnp.clip(jnp.sum(relative*edge, axis=-1)/jnp.sum(edge*edge, axis=-1), 0., 1.)
    offset = relative-weight[..., None]*edge
    # The floor only regularizes the zero-distance derivative (10 nanometers).
    edge_distance = jnp.sqrt(jnp.maximum(jnp.min(jnp.sum(offset*offset, axis=-1), axis=-1), 1e-16))
    halfspace = jnp.einsum('...i,ei->...e', xy, jnp.asarray(surface.edge_outward_xy))-jnp.asarray(surface.edge_offset_m)
    inside = jnp.all(halfspace <= 0., axis=-1)
    lateral = jnp.where(inside, -edge_distance, edge_distance)
    slab = jnp.abs(local[..., axis])-surface.half_thickness_m
    outside_squared = jnp.maximum(lateral, 0.)**2+jnp.maximum(slab, 0.)**2
    sdf = jnp.sqrt(jnp.maximum(outside_squared, 1e-16))+jnp.minimum(jnp.maximum(lateral, slab), 0.)
    nearest = jnp.argmin(jnp.sum(offset*offset, axis=-1), axis=-1)
    nearest_delta = jnp.take_along_axis(offset, nearest[..., None, None], axis=-2)[..., 0, :]
    lateral_normal = nearest_delta/edge_distance[..., None]*jnp.where(inside, -1., 1.)[..., None]
    z_sign = jnp.where(local[..., axis] >= 0., 1., -1.)
    outside_distance = jnp.sqrt(jnp.maximum(outside_squared, 1e-16))
    normal_xy = jnp.maximum(lateral, 0.)[..., None]*lateral_normal/outside_distance[..., None]
    normal_z = jnp.maximum(slab, 0.)*z_sign/outside_distance
    inside_prism = (lateral <= 0.) & (slab <= 0.)
    side_nearest = lateral > slab
    normal_xy = jnp.where(inside_prism[..., None],
        jnp.where(side_nearest[..., None], lateral_normal, 0.), normal_xy)
    normal_z = jnp.where(inside_prism, jnp.where(side_nearest, 0., z_sign), normal_z)
    local_normal = jnp.zeros_like(local).at[..., jnp.asarray([(axis+1) % 3, (axis+2) % 3])].set(normal_xy)
    local_normal = local_normal.at[..., axis].set(normal_z)
    contact_normal = jnp.einsum('...ij,...j->...i', geom_rotation_world, local_normal)
    closest = ball_position_m-sdf[..., None]*contact_normal
    sphere_point = ball_position_m-jnp.asarray(ball_radius_m)[..., None]*contact_normal
    frame = racket_surface_frame(geom_rotation_world, normal_axis=axis)
    sign = jnp.where(geom_rotation_world[..., 2, axis] >= 0., 1., -1.)
    relative_surface = jnp.stack((xy[..., 0], xy[..., 1]*sign,
        local[..., axis]*sign-surface.half_thickness_m), axis=-1)
    return dict(position_surface_m=relative_surface,
        signed_face_height_m=relative_surface[..., 2], edge_signed_distance_m=lateral,
        sphere_gap_m=sdf-ball_radius_m, inside_face=inside,
        normal_world=frame[..., :, 2], frame_world=frame,
        contact_normal_world=contact_normal, closest_prism_point_world_m=closest,
        contact_position_world_m=(closest+sphere_point)*.5)


def radial_boundary_distance(surface: RacketSurface, angle_rad):
    """Intersect an in-plane ray with the actual polygon for center/edge proposals."""
    direction = jnp.stack((jnp.cos(angle_rad), jnp.sin(angle_rad)), axis=-1)
    dot = jnp.einsum('...i,ei->...e', direction, jnp.asarray(surface.edge_outward_xy))
    distance = jnp.asarray(surface.edge_offset_m)/jnp.maximum(dot, 1e-12)
    return jnp.min(jnp.where(dot > 0., distance, jnp.inf), axis=-1)


def proposed_ball_center(surface: RacketSurface, geom_position_m, geom_rotation_world,
                         offset_xy_m, ball_radius_m):
    """Propose an upper-face sphere center; actual MJX decides hit versus miss."""
    frame = racket_surface_frame(geom_rotation_world, normal_axis=surface.normal_axis)
    # Offsets use the compiled polygon's tangent axes, with a normal sign
    # correction when its thickness axis points down in world coordinates.
    sign = jnp.where(geom_rotation_world[..., 2, surface.normal_axis] >= 0., 1., -1.)
    height = jnp.broadcast_to(jnp.asarray(ball_radius_m)+surface.half_thickness_m,
                              offset_xy_m.shape[:-1])
    local = jnp.stack((offset_xy_m[..., 0], offset_xy_m[..., 1]*sign, height), axis=-1)
    return geom_position_m+jnp.einsum('...ij,...j->...i', frame, local)
