"""Explicit free-attitude imagination candidate; all shared paths stay intact."""

import ast
from dataclasses import dataclass
import hashlib
import inspect
import textwrap
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from . import control_projection, course_control, course_env, full_prior_world
from .free_orientation import free_physical_bounds
from .mixed_precision import PrecisionRuntime, make_runtime as precision_runtime
from .module_gradient_tools import private_clone


def make_projector() -> Callable:
    """Drop four angular halfspaces and the nonlinear angle test, retain XYZ."""
    original = control_projection.project_reference
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    changes = []

    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name == 'angle_error':
                    changes.append(name)
                    node.value = ast.parse('jnp.zeros_like(position_error)', mode='eval').body
            return self.generic_visit(node)

    tree = Rewrite().visit(tree)
    if changes != ['angle_error']:
        raise ValueError('Expected one private projection angle-error definition')

    def without_angles(function):
        def project(**kwargs):
            return function(**dict(kwargs,
                relative_rpy_rad=jnp.zeros_like(kwargs['relative_rpy_rad']),
                relative_rpy_jacobian=jnp.zeros_like(kwargs['relative_rpy_jacobian']),
                roll_pitch_limit_rad=jnp.zeros_like(kwargs['roll_pitch_limit_rad'])))
        return project

    scope = dict(original.__globals__,
        projection_v324=without_angles(control_projection.projection_v324),
        project_cartesian_pose_qvel_jax=without_angles(control_projection.project_cartesian_pose_qvel_jax),
        relative_xyz_euler_rate_jacobian_jax=lambda angles, rotation, angular: jnp.zeros_like(angular))
    exec(compile(ast.fix_missing_locations(tree), '<WMJ-free-attitude-projection>', 'exec'), scope)
    return scope[original.__name__]


@dataclass(frozen=True)
class FreeOrientationRuntime(PrecisionRuntime):
    projector: Callable
    bounds: Callable


def make_runtime(robot, auxiliary, options: dict, control) -> FreeOrientationRuntime:
    """Connect the same candidate controller to physical and episodic recursion."""
    base = precision_runtime('fp32', robot, auxiliary, options, control)
    projector = make_projector()
    controller = private_clone(course_control.control_step, overrides={'project_reference': projector})
    world = private_clone(full_prior_world.full_prior_world_step, overrides={'control_step': controller})
    course = private_clone(course_env.course_step,
        overrides={'control_step': controller, 'physical_bounds': free_physical_bounds})
    values = dict(base.__dict__, mode='fp32_free_orientation', controller=controller,
                  world_step=world, course_step=course)
    return FreeOrientationRuntime(**values, projector=projector, bounds=free_physical_bounds)


def select_verified_state(template, arrays: dict, manifest: dict, *, prefix: str,
                          lanes, original_count: int, context_sha256: str):
    """Authenticate full source leaves before selecting complete independent lanes.

    Shared scalar queue clocks are retained. All other native leaves have the
    registered leading batch axis; unrecognized axes fail rather than broadcast.
    """
    if manifest['schema'] != 'wmj_numeric_seed_v1' or manifest['context_sha256'] != context_sha256:
        raise ValueError('Seed context mismatch')
    lanes = np.asarray(lanes, dtype=np.int32)
    if lanes.ndim != 1 or np.any(lanes < 0) or np.any(lanes >= original_count):
        raise ValueError('Invalid source lanes')
    lookup = {row['path']: (f'leaf_{i:06d}', row) for i, row in enumerate(manifest['leaves'])}
    leaves, structure = jax.tree_util.tree_flatten_with_path(template)
    selected = []
    for path, example in leaves:
        name = prefix + jax.tree_util.keystr(path)
        key, row = lookup[name]
        value = np.asarray(arrays[key])
        if (list(value.shape) != row['shape'] or value.dtype.str != row['dtype']
                or hashlib.sha256(value.tobytes()).hexdigest() != row['sha256']):
            raise ValueError('Source leaf integrity mismatch: ' + name)
        if value.ndim:
            if value.shape[0] != original_count:
                raise ValueError('Unregistered native batch axis: ' + name)
            value = value[lanes]
        if value.shape != np.shape(example) or value.dtype != np.asarray(example).dtype:
            raise ValueError('Selected native layout mismatch: ' + name)
        selected.append(jnp.asarray(value))
    return jax.tree_util.tree_unflatten(structure, selected)
