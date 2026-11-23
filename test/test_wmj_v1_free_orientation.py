"""Removing only racket attitude constraints retains position and joint limits."""

from pathlib import Path
from types import SimpleNamespace
import sys

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))


def projection_inputs():
    return dict(nominal_qvel=jnp.array([[.8, -.8, .2, .9, -.9, .1, 0.]]),
        position_m=jnp.array([[.99, -.99, 0.]]), relative_rpy_rad=jnp.array([[2., 1.57, -3.]]),
        translation_jacobian=jnp.eye(3, 7)[None],
        relative_rpy_jacobian=jnp.eye(3, 7, k=3)[None],
        position_lower_m=-jnp.ones(3), position_upper_m=jnp.ones(3),
        roll_pitch_limit_rad=jnp.full((2,), .035), velocity_lower=jnp.full((1, 7), -.5),
        velocity_upper=jnp.full((1, 7), .5), velocity_scale=jnp.ones(7),
        position_gain_s=20., orientation_gain_s=20., projection_sweeps=24)


def test_position_only_projection_preserves_xyz_and_velocity_boxes_at_any_angle():
    from world_model.free_orientation import position_only_projection
    kwargs = projection_inputs()
    result = position_only_projection(**kwargs)
    np.testing.assert_allclose(result.qvel, [[.2, -.2, .2, .5, -.5, .1, 0.]], atol=1e-6)
    assert float(result.max_violation[0]) < 1e-6
    # Disabled angular input cannot poison retained constraints near a singularity.
    kwargs['relative_rpy_jacobian'] = jnp.full((1, 3, 7), jnp.nan)
    np.testing.assert_array_equal(position_only_projection(**kwargs).qvel, result.qvel)


def test_removal_has_effect_on_angular_commands_while_original_projector_is_preserved():
    from world_model.control_execution import projection_v324
    from world_model.free_orientation import position_only_projection
    kwargs = projection_inputs(); kwargs['relative_rpy_rad'] = jnp.zeros((1, 3))
    kwargs['roll_pitch_limit_rad'] = jnp.full((2,), .001)
    before = projection_v324(**kwargs)
    free = position_only_projection(**kwargs)
    after = projection_v324(**kwargs)
    np.testing.assert_array_equal(before.qvel, after.qvel)
    assert abs(float(before.qvel[0, 3])) < .03
    assert float(free.qvel[0, 3]) == .5


def test_nonlinear_source_rewrite_changes_exactly_two_attitude_backstops():
    import ast
    from world_model.free_orientation import remove_reference_attitude_backstops
    source = '''def function(x):
    current_position_violation = x
    current_orientation_violation = x + 2
    candidate_position_violation = x + 1
    candidate_orientation_violation = x + 3
    return current_position_violation, candidate_position_violation, current_orientation_violation, candidate_orientation_violation
'''
    transformed = remove_reference_attitude_backstops(source)
    scope = {'jnp': jnp}; exec(compile(ast.parse(transformed), '<test>', 'exec'), scope)
    np.testing.assert_allclose(scope['function'](jnp.array([.3])), [[.3], [1.3], [0.], [0.]])
    with pytest.raises(ValueError):
        remove_reference_attitude_backstops(source.replace('candidate_orientation_violation =', 'wrong ='))


def test_actual_bounds_remove_attitude_only_and_still_reject_nonfinite_pose():
    from world_model.free_orientation import free_physical_bounds
    from world_model.imagined_env import EpisodeContract, physical_bounds
    control = SimpleNamespace(projection=SimpleNamespace(reference_rotation=jnp.eye(3),
        q_lower=-jnp.ones(7), q_upper=jnp.ones(7)))
    physics = SimpleNamespace(q_rad=jnp.zeros((4, 13)),
        ball=SimpleNamespace(position_m=jnp.tile(jnp.array([0., -.5, 1.3]), (4, 1))))
    # 90-degree roll, four lanes for attitude, workspace, joint and nonfinite checks.
    pose = jnp.tile(jnp.array([0., -.5, 1.1, 1., 0., 0., 0., 0., 1.]), (4, 1))
    pose = pose.at[1, 0].set(1.).at[3, 6].set(jnp.nan)
    physics.q_rad = physics.q_rad.at[2, 0].set(2.)
    _, old = physical_bounds(physics, pose, control, EpisodeContract())
    accepted, new = free_physical_bounds(physics, pose, control, EpisodeContract())
    np.testing.assert_array_equal(accepted, [True, False, False, False])
    for name in ('ball', 'racket', 'joints'):
        np.testing.assert_array_equal(old[name], new[name])
