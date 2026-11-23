"""Fail-closed source instrumentation and noninterference of substep recording."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model.teacher_trace import augment_source, recorded_loop, point_twist


SOURCE = '''def _step_impl(self, state, action):
    low_level_rmp_q_history = action
    low_level_rmp_qvel_history = action
    low_level_rmp_qdd_history = action
    def one_substep(index, carry):
        return carry
    (data, *metadata) = jax.lax.fori_loop(0, 5, one_substep, (state,))
    metrics = {}
    return state, action, action, action, metrics
'''


def test_instrumentation_requires_exactly_one_physics_loop_and_return():
    transformed = augment_source(SOURCE)
    assert '_wmj_recorded_loop' in transformed
    compile(transformed, '<test>', 'exec')
    with pytest.raises(ValueError, match='exactly one'):
        augment_source(SOURCE.replace('one_substep, (state,)', 'other_loop, (state,)'))
    with pytest.raises(ValueError, match='return'):
        augment_source(SOURCE.replace('return state, action, action, action, metrics', 'return state'))


def test_recording_is_independent_of_physics_carry_and_preserves_all_ticks():
    def substep(index, carry):
        return (carry[0] * 1.125 + index, carry[1] + 1)

    seed = (jnp.array([[.1, .2]], jnp.float32), jnp.array(0))
    sample = lambda index, before, after: {'q_before': before[0], 'q_after': after[0]}
    plain = jax.jit(lambda value: jax.lax.fori_loop(0, 5, substep, value))(seed)
    observed, trace = jax.jit(lambda value: recorded_loop(substep, value, 5, sample))(seed)
    for a, b in zip(plain, observed):
        np.testing.assert_array_equal(a, b)
    assert trace['q_before'].shape == (5, 1, 2)
    np.testing.assert_array_equal(trace['q_before'][1:], trace['q_after'][:-1])
    np.testing.assert_array_equal(trace['q_after'][-1], observed[0])


def test_native_point_twist_uses_preintegration_velocity_and_moving_base():
    # A translating base and a hinge rotating about world Z both move the point.
    axes = jnp.array([[[1., 0., 0.], [0., 0., 1.]]])
    anchors = jnp.zeros((1, 2, 3))
    point = jnp.array([[0., 2., 0.]])
    linear, angular = point_twist(point, axes, anchors, jnp.array([[.3, .4]]),
                                  jnp.array([False, True]))
    np.testing.assert_allclose(linear, [[-.5, 0., 0.]], atol=1e-7)
    np.testing.assert_allclose(angular, [[0., 0., .4]], atol=1e-7)
