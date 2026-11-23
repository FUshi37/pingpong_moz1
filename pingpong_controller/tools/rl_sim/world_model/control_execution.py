"""Private reuse of the authenticated V324 EXEC projection-loop transformation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import FunctionType

import cartesian_pose_action_projection_jax as native
import mjx_topology_execution as execution

from .contracts import file_hash


@lru_cache(maxsize=1)
def _projection():
    if (file_hash(Path(execution.__file__)) != '3d1bed370804d4ec5c2a07bffbdadbb7c250b91f67ebda1d0801e122b2d416f0'
            or file_hash(Path(native.__file__)) != execution.PROJECTION_SHA256):
        raise ValueError('V324 EXEC/projection implementation changed')
    original = native.project_cartesian_pose_qvel_jax
    halfspaces = execution._clone(native._project_halfspaces_dykstra_jax,
                                  execution._Rewrite(projection=True), 1)
    namespace = dict(original.__globals__, _project_halfspaces_dykstra_jax=halfspaces)
    private = FunctionType(original.__code__, namespace, original.__name__,
                            original.__defaults__, original.__closure__)
    private.__kwdefaults__ = original.__kwdefaults__
    return private


def projection_v324(**kwargs):
    """Reuse exact native arithmetic and unroll24 without patching shared modules."""
    return _projection()(**kwargs)
