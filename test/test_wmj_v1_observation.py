"""The original 72 slots and 90/200 Hz sample/hold semantics remain exact."""

from functools import partial
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from fixed_plane_v149 import make_simple_position_observation
from world_model.observation_engine import init_camera, advance_camera, actor_observation


def test_fractional_clock_produces_90_samples_per_second_and_holds_between():
    camera = init_camera(jnp.array([[0., -.5, 1.4]]),
                         jax.random.split(jax.random.PRNGKey(7), 1), ar1_enabled=False)

    def body(state, step):
        truth = jnp.array([[0., -.5, 1.]]) + step * .001
        updated, refresh = advance_camera(state, truth, ar1_enabled=False)
        return updated, (refresh, updated.held_position_m)

    camera, (refresh, positions) = jax.jit(lambda s: jax.lax.scan(
        body, s, jnp.arange(1, 201)))(camera)
    refresh = np.asarray(refresh[:, 0])
    assert refresh.sum() == 90
    np.testing.assert_array_equal(np.flatnonzero(refresh)[:5] + 1, [3, 5, 7, 9, 12])
    for i in range(1, 200):
        if not refresh[i]:
            np.testing.assert_array_equal(positions[i], positions[i - 1])


def test_history_keeps_age_at_each_policy_tick_and_reset_sentinels():
    camera = init_camera(jnp.array([[0., -.5, 1.4]]),
                         jax.random.split(jax.random.PRNGKey(8), 1), ar1_enabled=False)
    np.testing.assert_array_equal(camera.history_age_s, [[0., 1., 1., 1., 1., 1., 1., 1.]])
    for _ in range(4):
        camera, _ = advance_camera(camera, jnp.array([[0., -.5, 1.4]]), ar1_enabled=False)
    np.testing.assert_allclose(camera.history_age_s,
                               [[.005, 0., .01, .005, 0., 1., 1., 1.]], atol=1e-8)


def test_ar1_latent_and_held_position_change_only_on_fresh_samples():
    camera = init_camera(jnp.array([[0., -.5, 1.4]]),
                         jax.random.split(jax.random.PRNGKey(9), 1), ar1_enabled=True)
    step = jax.jit(partial(advance_camera, ar1_enabled=True))
    for _ in range(20):
        previous = camera
        camera, fresh = step(camera, jnp.array([[0., -.5, 1.4]]))
        if not bool(fresh[0]):
            np.testing.assert_array_equal(camera.epsilon_m, previous.epsilon_m)
            np.testing.assert_array_equal(camera.held_position_m, previous.held_position_m)
        else:
            assert not np.array_equal(camera.epsilon_m, previous.epsilon_m)


def test_actor_prefix_is_the_original_builder_with_independent_goal_xy():
    camera = init_camera(jnp.array([[.01, -.5, 1.4]]),
                         jax.random.split(jax.random.PRNGKey(10), 1), ar1_enabled=False)
    q = jnp.arange(7, dtype=jnp.float32)[None] * .05
    dq = q * .1
    q_ref = q + .001
    executed = jnp.full((1, 7), .02)
    pose = jnp.array([[0., -.49, 1.1, 1., 0., 0., 0., 1., 0.]])
    goal = jnp.array([[.15, -.65, 1.35]])
    obs = actor_observation(q, dq, q_ref, executed, pose, camera, goal)
    original = make_simple_position_observation(q=q, dq=dq, q_ref_error=q_ref-q,
        previous_action=executed, ball_positions=camera.history_position_m,
        racket_pose_now=pose, ball_ages=camera.history_age_s,
        target_xy_m=goal[:, :2], ball_position_valid=camera.observed)
    np.testing.assert_array_equal(obs[:, :72], original)
    np.testing.assert_allclose(obs[:, 72], [-.75], atol=2e-6)
    assert obs.shape == (1, 73)
