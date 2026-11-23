"""Actual bounded optimizer transactions preserve frozen targets and resume RNG."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))


def test_pilot_transaction_updates_exact_counts_and_is_reproducible():
    from world_model.imagination_learner import init_learner
    from world_model.neural import init_mlp
    from world_model.pilot_learning import optimize_pilot
    from world_model.state_adapter import numeric_tree_digest
    actor = init_mlp(jax.random.PRNGKey(1), (2, 4, 1))
    critic = init_mlp(jax.random.PRNGKey(2), (3, 4, 1), zero_output=True)
    state = init_learner(actor, critic, None, seed=4)
    before = numeric_tree_digest(state)
    gradient = jax.tree.map(lambda x: jnp.ones_like(x), actor)
    inputs = np.arange(30, dtype=np.float32).reshape(10, 3) / 30
    target = np.linspace(0, 1, 10, dtype=np.float32)
    recipe = dict(actor_lr=1e-5, critic_lr=1e-4, critic_updates_per_actor=4,
        critic_batch=8, critic_microbatch=3, target_critic_ema=.01)
    with pytest.raises(PermissionError):
        optimize_pilot(state, gradient, inputs, target, recipe)
    after, report = optimize_pilot(state, gradient, inputs, target, recipe, authorized=True)
    again, repeated = optimize_pilot(state, gradient, inputs, target, recipe, authorized=True)
    assert numeric_tree_digest(state) == before
    assert numeric_tree_digest(after) == numeric_tree_digest(again)
    assert report == repeated
    assert int(after.actor_optimizer.step) == int(after.actor_updates) == 1
    assert int(after.critic_optimizer.step) == int(after.critic_updates) == 4
    for old, new, online in zip(jax.tree.leaves(state.target_critic),
                               jax.tree.leaves(after.target_critic), jax.tree.leaves(after.critic)):
        np.testing.assert_allclose(new, .99 * np.asarray(old) + .01 * np.asarray(online), atol=1e-7)
    assert report['imagined_rows'] == 10 and len(report['critic']) == 4


def test_pilot_rejects_nonfinite_or_nonimagined_targets_before_update():
    from world_model.pilot_learning import validate_learning_rows
    with pytest.raises(ValueError):
        validate_learning_rows(np.zeros((3, 2)), np.zeros(3), origin='teacher')
    with pytest.raises(FloatingPointError):
        validate_learning_rows(np.zeros((3, 2)), np.array([1., np.nan, 0.]))
