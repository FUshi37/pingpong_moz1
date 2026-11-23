"""Mixed-phase world seeds retain independent native RMP ring-buffer clocks."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from recovered_rmp_jax import JointRMPParams, init_joint_rmp_state, joint_rmp_step
from world_model.control_rmp import promote_rmp_indices, rmp_step_lanes


def test_mixed_phase_lanes_match_separate_native_calls():
    params = JointRMPParams()
    lo, hi, vel, acc, jerk = (jnp.full(7, value) for value in (-3., 3., 4., 30., 700.))
    kwargs = dict(servo_steps=5, target_update_delay_steps=3, return_history=True)
    states, targets, expected = [], [], []
    for lane, warm_steps in enumerate((0, 1, 3)):
        q = jnp.full((1, 7), .1*lane)
        state = init_joint_rmp_state(q)
        for _ in range(warm_steps):
            state, *_ = joint_rmp_step(state, q+.02, lo, hi, vel, acc, jerk, params, .001, **kwargs)
        target = q+.03
        expected.append(joint_rmp_step(state, target, lo, hi, vel, acc, jerk, params, .001, **kwargs))
        states.append(promote_rmp_indices(state))
        targets.append(target)
    combined = jax.tree.map(lambda *values: jnp.concatenate(values), *states)
    result, histories = jax.jit(lambda state, target: rmp_step_lanes(
        state, target, {}, q_min=lo, q_max=hi, velocity_limit=vel,
        acceleration_limit=acc, jerk_limit=jerk, params=params))(combined, jnp.concatenate(targets))
    expected_states = [promote_rmp_indices(item[0]) for item in expected]
    reference = jax.tree.map(lambda *values: jnp.concatenate(values), *expected_states)
    for actual, target in zip(jax.tree.leaves(result), jax.tree.leaves(reference)):
        np.testing.assert_allclose(actual, target, atol=2e-6, rtol=1e-6)
    for i, name in enumerate(('q', 'dq', 'ddq')):
        np.testing.assert_allclose(histories[name], np.concatenate([item[i+1] for item in expected], axis=1),
                                   atol=2e-6, rtol=1e-6)


def test_delayed_history_preserves_per_joint_ticks_and_gradient():
    from world_model.control_rmp import apply_output_delay
    old = jnp.arange(19*7, dtype=jnp.float32).reshape(1, 19, 7)
    command = 1000+jnp.arange(5*7, dtype=jnp.float32).reshape(5, 1, 7)
    delay = jnp.array([[0, 1, 2, 3, 18, 19, 4]])
    result, buffer = apply_output_delay(old, command, delay)
    combined = np.concatenate((old, np.swapaxes(command, 0, 1)), axis=1)
    reference = np.array([[combined[0, 19+i-int(delay[0, j]), j] for j in range(7)] for i in range(5)])
    np.testing.assert_array_equal(result[:, 0], reference)
    np.testing.assert_array_equal(buffer, combined[:, -19:])
    derivative = jax.grad(lambda commands: apply_output_delay(old, commands, delay)[0].sum())(command)
    np.testing.assert_array_equal(derivative[:, 0, 0], 1.)
    np.testing.assert_array_equal(derivative[:, 0, 5], 0.)
