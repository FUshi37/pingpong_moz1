"""Select legacy prism or the teacher's actual MJX sphere/mesh geometry.

Mesh construction is offline. Runtime evaluates one small ball/racket pair;
it never runs robot mesh admission, a rigid-body solver, or a simulator step.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco.mjx._src import collision_convex, mesh
from mujoco.mjx._src.collision_types import ConvexInfo, GeomInfo

from .contact_surface import RacketSurface, surface_from_vertices
from .contact_surface import surface_features as prism_features
from .contracts import file_hash


@dataclass(frozen=True)
class MJXContactSurface:
    prism: RacketSurface
    convex: Any
    geom_size_m: Any
    identity: dict


def surface_from_compiled_model(model, geom_id: int, *, normal_axis: int) -> MJXContactSurface:
    """Extract the exact merged faces and their ordering used by the teacher."""
    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError('Native contact features require the compiled racket mesh')
    mesh_id = int(model.geom_dataid[geom_id])
    start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    prism = surface_from_vertices(model.mesh_vert[start:start+count], normal_axis=normal_axis)
    convex = mesh.convex(model, mesh_id)
    digest = hashlib.sha256()
    for name, value in sorted(vars(convex).items()):
        array = np.asarray(value)
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    identity = dict(kind='v324_mjx_sphere_convex', merged_mesh_sha256=digest.hexdigest(),
        kernel_sha256=file_hash(Path(collision_convex.__file__)),
        mesh_builder_sha256=file_hash(Path(mesh.__file__)))
    return MJXContactSurface(prism, convex, jnp.asarray(model.geom_size[geom_id], jnp.float32), identity)


def surface_features(surface, ball_position_m, geom_position_m,
                     geom_rotation_world, ball_radius_m) -> dict:
    """Preserve local inputs, using native distance/contact point when selected."""
    if isinstance(surface, RacketSurface):
        return prism_features(surface, ball_position_m, geom_position_m,
                              geom_rotation_world, ball_radius_m)
    if not isinstance(surface, MJXContactSurface):
        raise TypeError('Unknown contact geometry contract')
    result = prism_features(surface.prism, ball_position_m, geom_position_m,
                            geom_rotation_world, ball_radius_m)
    shape = ball_position_m.shape[:-1]
    ball = jnp.asarray(ball_position_m).reshape((-1, 3))
    position = jnp.broadcast_to(geom_position_m, shape+(3,)).reshape((-1, 3))
    rotation = jnp.broadcast_to(geom_rotation_world, shape+(3, 3)).reshape((-1, 3, 3))
    radius = jnp.broadcast_to(jnp.asarray(ball_radius_m), shape).reshape((-1,))

    def query(p, x, rot, r):
        sphere = GeomInfo(p, jnp.eye(3), jnp.stack((r, jnp.zeros_like(r), jnp.zeros_like(r))))
        convex = ConvexInfo(x, rot, surface.geom_size_m, **vars(surface.convex))
        return collision_convex._sphere_convex(sphere, convex)

    distance, point, normal = jax.vmap(query)(ball, position, rotation, radius)
    point, normal = point.reshape(shape+(3,)), normal.reshape(shape+(3,))
    # MJX's normal points sphere -> convex; existing features use the opposite.
    result.update(sphere_gap_m=distance.reshape(shape), contact_position_world_m=point,
        contact_normal_world=-normal,
        closest_prism_point_world_m=2*point-(ball_position_m+normal*jnp.asarray(ball_radius_m)[..., None]))
    return result
