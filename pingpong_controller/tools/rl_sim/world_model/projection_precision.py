"""Opt-in full-FP32 coordinate transforms for free-attitude projection only.

The native EXEC halfspace solver, FK, bounds and hold inequalities retain
exactly their existing implementations. No global JAX setting is modified.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp

from .free_imagination import make_projector as free_projector, make_runtime as free_runtime
from .velocity_reward import clone_globals


class _CoordinatePrecision:
    """Forward NumPy operations, specializing only this method's frame products."""

    def __getattr__(self, name):
        return getattr(jnp, name)

    @staticmethod
    def einsum(*args, **kwargs):
        return jnp.einsum(*args, **dict(kwargs, precision=jax.lax.Precision.HIGHEST))


coordinate_precision = _CoordinatePrecision()


def make_projector():
    """Keep the existing projection bytecode with private FP32 frame products."""
    return clone_globals(free_projector(), jnp=coordinate_precision)


def native_projector(free_native_method):
    """Apply the same private frame-product precision to the native candidate."""
    return clone_globals(free_native_method, jnp=coordinate_precision)


def make_runtime(robot, auxiliary, options, control):
    """Use one candidate projector throughout physical and episodic recursion."""
    base = free_runtime(robot, auxiliary, options, control)
    projector = make_projector()
    controller = clone_globals(base.controller, project_reference=projector)
    world = clone_globals(base.world_step, control_step=controller)
    course = clone_globals(base.course_step, control_step=controller)
    return replace(base, mode='fp32_free_orientation_coordinate_precision_candidate',
        projector=projector, controller=controller, world_step=world, course_step=course)
