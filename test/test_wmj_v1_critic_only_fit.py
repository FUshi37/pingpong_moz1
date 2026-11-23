"""Critic-only learning preserves Actor state and full resume semantics."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                      'pingpong_controller/tools/rl_sim'))


def test_confirmed_critic_fit_storage_extension_is_exact_and_scoped():
    import copy
    import json
    from world_model.contracts import repository_root, validate_config
    cfg = json.loads((repository_root() /
        'pingpong_controller/tools/rl_2real/WMJ_V1_CONFIG.json').read_text())
    cfg['resources']['max_new_bytes'] = 6643777536
    validate_config(cfg)
    for cap in (6643777535, 6643777537):
        bad = copy.deepcopy(cfg); bad['resources']['max_new_bytes'] = cap
        with pytest.raises(ValueError, match='Storage budget'):
            validate_config(bad)
    bad = copy.deepcopy(cfg); bad.pop('critic_only_fit')
    with pytest.raises(ValueError, match='Storage budget'):
        validate_config(bad)
    for cap in (6*2**30, 6576668672, 6593445888):
        old = copy.deepcopy(cfg); old['resources']['max_new_bytes'] = cap
        validate_config(old)


def test_critic_only_matches_reference_four_steps_without_actor_update():
    from world_model.actor_critic import apply_accumulated_update, target_ema
    from world_model.critic_fit import make_update, update_session
    from world_model.imagination_learner import init_learner, critic_gradient_batch
    from world_model.neural import init_mlp
    from world_model.state_adapter import numeric_tree_digest
    actor = init_mlp(jax.random.PRNGKey(1), (2, 4, 1))
    critic = init_mlp(jax.random.PRNGKey(2), (3, 4, 1))
    initial = init_learner(actor, critic, None, seed=4)
    x = np.arange(30, dtype=np.float32).reshape(10, 3) / 30
    y = np.linspace(-10, 1, 10, dtype=np.float32)
    recipe = dict(critic_batch=8, critic_microbatch=4, critic_lr=1e-4,
                  target_critic_ema=.01, critic_updates_per_actor=4)
    fn = make_update(recipe)
    with pytest.raises(PermissionError):
        update_session(initial, x, y, fn)
    with pytest.raises(ValueError):
        update_session(initial, x, y, fn, authorized=True, origin='teacher')
    after, _ = update_session(initial, x, y, fn, authorized=True)
    p, opt, key = initial.critic, initial.critic_optimizer, initial.sampling_rng
    for _ in range(4):
        key, draw = jax.random.split(key)
        ids = np.asarray(jax.random.randint(draw, (8,), 0, len(x), dtype=jnp.int32))
        grad, _ = critic_gradient_batch(p, x[ids], y[ids], microbatch=4)
        p, opt, _ = apply_accumulated_update(p, opt, [grad], [8],
                                            learning_rate=1e-4, authorized=True)
    for a, b in zip(jax.tree.leaves((after.critic, after.critic_optimizer)),
                    jax.tree.leaves((p, opt))):
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=1e-5)
    expected = target_ema(initial.target_critic, p, .01)
    for a, b in zip(jax.tree.leaves(after.target_critic), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=1e-7)
    assert numeric_tree_digest((after.actor, after.actor_optimizer, after.actor_updates)) == \
        numeric_tree_digest((initial.actor, initial.actor_optimizer, initial.actor_updates))
    assert int(after.critic_updates) == int(after.critic_optimizer.step) == 4
    assert np.array_equal(after.sampling_rng, key)


def test_critic_only_resume_preserves_sampling_and_update_cap(tmp_path):
    from world_model.critic_fit import make_update, update_session
    from world_model.imagination_learner import init_learner
    from world_model.imagination_checkpoint import save_checkpoint, restore_checkpoint
    from world_model.neural import init_mlp
    from world_model.resources import StorageBudget
    from world_model.state_adapter import numeric_tree_digest
    p = init_mlp(jax.random.PRNGKey(5), (3, 4, 1))
    s = init_learner(p, p, None, seed=6)
    fn = make_update(dict(critic_batch=8, critic_microbatch=4, critic_lr=1e-4,
                         target_critic_ema=.01, critic_updates_per_actor=4))
    x = np.ones((10, 3), np.float32); y = np.arange(10, dtype=np.float32)
    a, _ = update_session(s, x, y, fn, authorized=True)
    ref = save_checkpoint(tmp_path/'fit.json', dict(learner=a), signature='test',
                          budget=StorageBudget(tmp_path, 2**20, 0), metadata={})
    restored = restore_checkpoint(ref, dict(learner=a), signature='test')['learner']
    b, _ = update_session(restored, x, y, fn, authorized=True)
    expected, _ = update_session(a, x, y, fn, authorized=True)
    assert numeric_tree_digest(b) == numeric_tree_digest(expected)
    with pytest.raises(ValueError):
        update_session(a, x, y, fn, authorized=True, initial_updates=0, maximum_updates=4)
    with pytest.raises(FloatingPointError):
        update_session(s, x, np.full(10, np.nan), fn, authorized=True)


def test_partial_resets_preserve_live_controller_and_history():
    from typing import NamedTuple
    from world_model.critic_fit import prepare_fit_state
    from world_model.imagination_learner import init_learner
    from world_model.neural import init_mlp
    from world_model.persistent_pool import PersistentPool
    class State(NamedTuple):
        episode_steps: object
        episode_id: object
        terminated: object
        truncated: object
        model_truncated: object
        queue: object
        history: object
    z = jnp.zeros(4, jnp.int32); off = jnp.zeros(4, bool)
    seed = State(z, z, off, off, off, jnp.arange(24).reshape(4, 6),
                 jnp.arange(32).reshape(4, 8))
    live = seed._replace(episode_steps=z+64, terminated=jnp.array([0, 1, 0, 0], bool),
                         queue=seed.queue+100, history=seed.history+200)
    keys = jax.random.split(jax.random.PRNGKey(4), 4)
    pool = PersistentPool(live, keys, jnp.array(0), jnp.zeros(2, jnp.uint32), jnp.array(1))
    p = init_mlp(jax.random.PRNGKey(5), (2, 3, 1))
    learner = init_learner(p, p, pool, seed=6)
    state = dict(learner=learner, reset_bank=seed, current_source_indices=jnp.arange(4))
    after, report = prepare_fit_state(state, persistent_envs=3)
    assert report['reset'] == [False, True, False, True]
    for i in (0, 2):
        assert np.array_equal(after['learner'].pool.environment.queue[i], live.queue[i])
        assert np.array_equal(after['learner'].pool.environment.history[i], live.history[i])
        assert np.array_equal(after['learner'].pool.exploration_rng[i], keys[i])
    for i in (1, 3):
        source = report['source_indices'][i]
        assert np.array_equal(after['learner'].pool.environment.queue[i], seed.queue[source])
        assert np.array_equal(after['learner'].pool.environment.history[i], seed.history[source])
        assert int(after['learner'].pool.environment.episode_id[i]) == 1
