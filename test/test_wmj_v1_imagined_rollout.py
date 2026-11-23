"""End-to-end imagined gradients, continuity and numerical recovery."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_wmj_v1_control_projection import fixture
from test_wmj_v1_critic_features import setup_value
from test_wmj_v1_full_prior_world import components
from test_wmj_v1_world_control import setup_control
from world_model.actor_critic import init_critic, ImaginedBatch, lambda_returns
from world_model.critic_features import critic_features
from world_model.imagined_env import init_imagined, EpisodeContract
from world_model.imagined_rewards import RewardContract
from world_model.imagined_rollout import policy_window, actor_window_objective
from world_model.imagination_learner import (
    init_learner, critic_gradient_batch, apply_training_transaction,
)
from world_model.neural import init_mlp
from world_model.persistent_pool import init_pool


def test_rollout_split_retains_rng_history_and_finite_actor_gradient(fixture):
    world, task, events, context, _ = setup_value(fixture)
    ball = world.physics.ball._replace(
        position_m=jnp.array([[0., -.5, 1.4]]),
        velocity_m_s=jnp.array([[.01, 0., -.1]]))
    world = world._replace(physics=world.physics._replace(ball=ball))
    state = init_imagined(world, context, task.goal_xyz, events=events)
    robot, auxiliary, _, options = components()
    chain, _, _, _, control = setup_control(fixture)
    options = dict(options, chain=chain, control_contract=control,
        episode_contract=EpisodeContract(), reward_contract=RewardContract(),
        ar1_enabled=False)
    actor = dict(zip(('l1', 'l2', 'out'),
        init_mlp(jax.random.PRNGKey(4), (969, 256, 256, 7))))
    critic = init_critic(jax.random.PRNGKey(5), critic_features(
        world, task, events, context, state.episode_steps).shape[-1])
    keys = jax.random.split(jax.random.PRNGKey(6), 1)
    goals = jnp.broadcast_to(task.goal_xyz, (4, 1, 3))
    run = jax.jit(lambda s, k, g: policy_window(
        actor, robot, auxiliary, s, k, g, **options))
    whole = run(state, keys, goals)
    first = run(state, keys, goals[:2])
    second = run(first.environment, first.exploration_rng, goals[2:])
    for a, b in zip(jax.tree.leaves(whole.environment),
                    jax.tree.leaves(second.environment)):
        np.testing.assert_allclose(a, b, atol=3e-7, rtol=2e-6)
    np.testing.assert_array_equal(whole.exploration_rng, second.exploration_rng)
    np.testing.assert_allclose(whole.transitions.reward,
        jnp.concatenate((first.transitions.reward, second.transitions.reward)), atol=1e-7)
    gradient = jax.jit(jax.grad(lambda p: actor_window_objective(
        p, critic, robot, auxiliary, state, keys, goals, **options)[0]))(actor)
    leaves = jax.tree.leaves(gradient)
    assert all(np.isfinite(x).all() for x in leaves)
    assert sum(float(jnp.sum(x*x)) for x in leaves) > 0.


def test_critic_microbatch_and_update_authorization():
    critic = init_critic(jax.random.PRNGKey(4), 11)
    inputs = np.asarray(jax.random.normal(jax.random.PRNGKey(8), (7, 11)))
    targets = np.arange(7, dtype=np.float32) / 10.
    full, loss = critic_gradient_batch(critic, inputs, targets, microbatch=7)
    split, split_loss = critic_gradient_batch(critic, inputs, targets, microbatch=3)
    np.testing.assert_allclose(loss, split_loss, atol=1e-7)
    for a, b in zip(jax.tree.leaves(full), jax.tree.leaves(split)):
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=2e-6)
    with pytest.raises(PermissionError):
        apply_training_transaction(None, None, None, {})
    batch = ImaginedBatch(jnp.zeros((2, 1)), jnp.ones((2, 1)),
        jnp.ones((2, 1), bool), jnp.ones((2, 1), bool), origin='teacher')
    with pytest.raises(ValueError, match='model-generated'):
        lambda_returns(batch, jnp.zeros((2, 1)))
