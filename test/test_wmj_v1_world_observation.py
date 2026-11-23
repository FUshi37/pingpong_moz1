"""Imagined observations carry real history forward without predicting it again."""

from functools import partial
from pathlib import Path
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model import observation_engine as observation
from world_model.native_observation import NativeObservationAdapter, camera_from_native
from world_model.paired_history import encode_history
from world_model.state_adapter import pack_numeric_state, restore_numeric_state


def pose_from_q(q):
    """Nonlinear FK fixture distinguishes a matching pose from a stale one."""
    angle = q[:, 0]
    zero = jnp.zeros_like(angle)
    return jnp.stack((angle**2, zero-.5, zero+1.1,
        jnp.cos(angle), jnp.sin(angle), zero,
        -jnp.sin(angle), jnp.cos(angle), zero), axis=-1)


def initial(n=1, *, ar1_enabled=False):
    return observation.init_world_observation(
        jnp.tile(jnp.array([[.1, -.5, 1.4]]), (n, 1)),
        jnp.zeros((n, 7)), jax.random.split(jax.random.PRNGKey(19), n),
        pose_from_q=pose_from_q, ar1_enabled=ar1_enabled)[0]


def step(state, q, ball, *, ar1_enabled=False):
    return observation.advance_world_observation(state, q, ball,
        pose_from_q=pose_from_q, ar1_enabled=ar1_enabled)


@pytest.mark.parametrize('noisy', [False, True])
def test_imagination_shifts_history_only_when_the_camera_samples(noisy):
    state = initial(ar1_enabled=noisy)
    advance = jax.jit(partial(step, ar1_enabled=noisy))
    fresh_count = 0
    for tick in range(1, 141):
        previous = state
        q = jnp.full((1, 7), tick*.001)
        ball = jnp.array([[tick*.0002, -.5, 1.4]])
        state, report = advance(state, q, ball)
        fresh = bool(report['inserted'][0])
        fresh_count += fresh
        old, new = previous.history.paired, state.history.paired
        if fresh:
            for name in ('sample_time_us', 'received_time_us', 'ball_position_m',
                         'racket_pose', 'ball_valid', 'aligned', 'present'):
                np.testing.assert_array_equal(getattr(new, name)[:, 1:],
                                              getattr(old, name)[:, :-1])
            np.testing.assert_allclose(new.racket_pose[:, 0], pose_from_q(q), atol=1e-7)
            np.testing.assert_array_equal(new.ball_position_m[:, 0], state.camera.held_position_m)
        else:
            for before, after in zip(jax.tree.leaves(old), jax.tree.leaves(new)):
                np.testing.assert_array_equal(before, after)
        # The preserved legacy eight slots advance even on camera holds.
        np.testing.assert_array_equal(state.camera.history_position_m[:, 1:],
                                      previous.camera.history_position_m[:, :-1])
        np.testing.assert_array_equal(state.camera.history_age_s[:, 1:],
                                      previous.camera.history_age_s[:, :-1])
    assert fresh_count == 63
    rows = np.asarray(encode_history(state.history.paired, state.camera.step_count*5000))
    rows = rows.reshape(64, 14)
    assert 57 <= rows[:, 13].sum() <= 59
    assert np.all(rows[rows[:, 13] == 0] == 0)


def test_long_history_age_advances_on_a_hold_without_rewriting_source_samples():
    state = initial()
    after, _ = step(state, jnp.ones((1, 7)), jnp.zeros((1, 3)))
    before_rows = encode_history(state.history.paired, state.camera.step_count*5000).reshape(64, 14)
    after_rows = encode_history(after.history.paired, after.camera.step_count*5000).reshape(64, 14)
    np.testing.assert_array_equal(after_rows[0, :12], before_rows[0, :12])
    np.testing.assert_allclose(after_rows[0, 12], .005/.640, atol=1e-8)
    assert after_rows[0, 13] == 1


def test_predicted_history_retains_gradients_through_shifts_and_holds():
    state = initial()

    def objective(delta):
        def body(current, tick):
            q = jnp.ones((1, 7)) * delta * tick
            ball = jnp.array([[.1, -.5, 1.4]]) + delta*tick
            current, _ = step(current, q, ball)
            return current, None

        after, _ = jax.lax.scan(body, state, jnp.arange(1, 10))
        rows = encode_history(after.history.paired, after.camera.step_count*5000).reshape(64, 14)
        # Tick 3 is now the fourth newest packet (9,7,5,3); no re-prediction.
        return rows[3, 0] + rows[3, 3]

    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.float32(.01))
    np.testing.assert_allclose(value, .1+3*.01+(3*.01)**2, atol=1e-7)
    np.testing.assert_allclose(gradient, 3+18*.01, atol=1e-6)


def test_full_observation_seed_restores_noise_phase_and_history():
    state = initial(n=2, ar1_enabled=True)
    advance = jax.jit(partial(step, ar1_enabled=True))
    q, ball = jnp.full((2, 7), .02), jnp.full((2, 3), .3)
    for _ in range(7):
        state, _ = advance(state, q, ball)
    arrays, manifest = pack_numeric_state(state, context_sha256='a'*64)
    restored = restore_numeric_state(state, arrays, manifest, context_sha256='a'*64)
    left, right = advance(state, q, ball), advance(restored, q, ball)
    for expected, actual in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('full_chain', [False, True])
def test_native_adapter_aligns_long_history_from_measured_q_and_preserves_prefix(full_chain):
    adapter = NativeObservationAdapter.__new__(NativeObservationAdapter)
    adapter.environment = SimpleNamespace(arm_qadr=jnp.arange(7), arm_vadr=jnp.arange(7),
                                          racket_site_id=0)
    adapter.full_chain = full_chain
    adapter.feedback_qadr = jnp.arange(13 if full_chain else 7)
    adapter.pose_from_q = (lambda q: pose_from_q(q[:, :7]+q[:, 7:8])) if full_chain else pose_from_q
    adapter._update = jax.jit(adapter._update_samples)
    world = initial()
    q = jnp.full((1, 7), .3)
    stale_pose = jnp.array([[.9, -.4, 1.2, 1., 0., 0., 0., 1., 0.]])
    measured = jnp.concatenate((q, jnp.full((1, 6), .2)), axis=-1) if full_chain else q
    native = SimpleNamespace(data=SimpleNamespace(qpos=measured, qvel=jnp.zeros_like(measured),
        site_xpos=stale_pose[:, None, :3], site_xmat=jnp.eye(3).reshape(1, 1, 9)),
        step_count=world.camera.step_count, last_ball_obs_step=world.camera.last_sample_step,
        cached_ball_obs_pos=world.camera.held_position_m, rng=world.camera.rng,
        actor_ball_position_history=world.camera.history_position_m,
        actor_ball_age_history=world.camera.history_age_s,
        ball_position_noise_epsilon_m=world.camera.epsilon_m,
        fixed_contact_ball_observed=world.camera.observed,
        arm_cmd_q=q, prev_action=jnp.zeros((1, 7)))
    history, _ = adapter.initialize(native)
    current_pose = adapter.pose_from_q(measured)
    np.testing.assert_allclose(history.paired.racket_pose[:, 0], current_pose, atol=1e-7)
    assert history.feedback.q_rad.shape[-1] == (13 if full_chain else 7)
    goal = jnp.array([[.1, -.5, 1.4]])
    expected_prefix = observation.actor_observation(q, native.data.qvel[:, :7], q,
        native.prev_action, stale_pose, camera_from_native(native), goal)
    obs = adapter.observation(native, history, goal)
    assert obs.shape == (1, 969)
    np.testing.assert_array_equal(obs[:, :73], expected_prefix)


def test_goal_changes_only_goal_slots_and_physical_reset_clears_selected_lane():
    state = initial(n=2)
    q, ball = jnp.ones((2, 7))*.01, jnp.ones((2, 3))*.1
    for _ in range(6):
        state, _ = step(state, q, ball)
    before = jax.tree.map(np.asarray, state)
    goal = jnp.tile(jnp.array([[0., -.5, 1.4]]), (2, 1))

    def observe(g):
        return observation.actor_observation_with_history(q, q, q, q,
            pose_from_q(q), state.camera, g, state.history.paired, state.camera.step_count*5000)

    old, new = np.asarray(observe(goal)), np.asarray(observe(goal+.01))
    changed = np.flatnonzero(np.any(old != new, axis=0))
    np.testing.assert_array_equal(changed, [69, 70, 72])
    for expected, actual in zip(jax.tree.leaves(before), jax.tree.leaves(state)):
        np.testing.assert_array_equal(actual, expected)
    reset = observation.reset_world_observation(state, initial(n=2), jnp.array([True, False]))
    assert int(reset.camera.step_count[0]) == 0
    assert int(reset.history.paired.present[0].sum()) == 1
    for actual, expected in zip(jax.tree.leaves(reset), jax.tree.leaves(state)):
        np.testing.assert_array_equal(actual[1], expected[1])
