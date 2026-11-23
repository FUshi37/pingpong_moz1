"""Use the pinned V324 projection unroll privately, without shared mutation."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))

import cartesian_pose_action_projection_jax as native
from mjx_topology_execution import make_collect


def test_private_projection_matches_pinned_execution_scope_and_restores_native():
    from world_model.control_execution import projection_v324
    before = native._project_halfspaces_dykstra_jax
    values = dict(nominal_qvel=jnp.ones((2, 7))*.3, position_m=jnp.zeros((2, 3)),
        relative_rpy_rad=jnp.zeros((2, 3)),
        translation_jacobian=jnp.tile(jnp.eye(3, 7)[None], (2, 1, 1)),
        relative_rpy_jacobian=jnp.tile(jnp.eye(3, 7)[None], (2, 1, 1)),
        position_lower_m=jnp.ones(3)*-.01, position_upper_m=jnp.ones(3)*.01,
        roll_pitch_limit_rad=jnp.ones(2)*.03, velocity_lower=jnp.ones((2, 7))*-2.,
        velocity_upper=jnp.ones((2, 7))*2., velocity_scale=jnp.ones(7)*2.,
        position_gain_s=20., orientation_gain_s=20., projection_sweeps=24)
    scoped, metadata = make_collect(lambda: native.project_cartesian_pose_qvel_jax(**values), mode='project')
    expected = jax.jit(scoped)()
    actual = jax.jit(lambda: projection_v324(**values))()
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(left, right, atol=1e-7)
    assert metadata['projection_unroll'] == 24
    assert native._project_halfspaces_dykstra_jax is before
