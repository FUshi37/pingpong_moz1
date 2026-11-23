"""Fixed labels remain independent of the evolving target Critic."""

from pathlib import Path
import copy
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                      'pingpong_controller/tools/rl_sim'))


def test_fixed_fit_storage_extension_is_exact_and_preserves_old_caps():
    from world_model.contracts import repository_root, validate_config
    cfg = json.loads((repository_root() /
        'pingpong_controller/tools/rl_2real/WMJ_V1_CONFIG.json').read_text())
    cfg['critic_fixed_data_fit'] = dict(plan={'path': 'registered'})
    cfg['resources']['max_new_bytes'] = 6694109184
    validate_config(cfg)
    for value in (6694109183, 6694109185):
        bad = copy.deepcopy(cfg); bad['resources']['max_new_bytes'] = value
        with pytest.raises(ValueError, match='Storage budget'):
            validate_config(bad)
    bad = copy.deepcopy(cfg); bad.pop('critic_fixed_data_fit')
    with pytest.raises(ValueError, match='Storage budget'):
        validate_config(bad)
    bad = copy.deepcopy(cfg); bad['resources']['min_free_bytes'] = 8*2**30-1
    with pytest.raises(ValueError, match='Storage budget'):
        validate_config(bad)
    for value in (6*2**30, 6576668672, 6593445888, 6643777536, 6677331968):
        old = copy.deepcopy(cfg); old['resources']['max_new_bytes'] = value
        validate_config(old)


def test_changing_target_network_cannot_change_fixed_label_fit():
    from world_model.critic_fit import make_update
    from world_model.fixed_critic_fit import update_fixed_state, invariant_digest
    from world_model.imagination_learner import init_learner
    from world_model.neural import init_mlp
    from world_model.state_adapter import numeric_tree_digest
    p = init_mlp(jax.random.PRNGKey(1), (3, 8, 1))
    learner = init_learner(p, p, None, seed=2)
    initial = dict(learner=learner, fixed_lambda_targets=jnp.linspace(-10., 0., 12),
                   fixed_fit_transactions=jnp.array(0), reset_bank=jnp.zeros((3, 2)))
    different = dict(initial, learner=learner._replace(
        target_critic=jax.tree.map(lambda x:x+5., learner.target_critic)))
    x = np.arange(36, dtype=np.float32).reshape(12, 3)/30
    fn = make_update(dict(critic_batch=16, critic_microbatch=4, critic_lr=1e-4,
        critic_updates_per_actor=4, target_critic_ema=.05))
    with pytest.raises(PermissionError):
        update_fixed_state(initial, x, fn, initial_updates=0)
    a, _ = update_fixed_state(initial, x, fn, initial_updates=0, authorized=True)
    b, _ = update_fixed_state(different, x, fn, initial_updates=0, authorized=True)
    for key in ('critic', 'critic_optimizer', 'sampling_rng'):
        assert numeric_tree_digest(getattr(a['learner'], key)) == numeric_tree_digest(getattr(b['learner'], key))
    assert invariant_digest(initial) == invariant_digest(a)
    assert np.array_equal(a['fixed_lambda_targets'], initial['fixed_lambda_targets'])
    assert int(a['fixed_fit_transactions']) == 1
    with pytest.raises(ValueError):
        update_fixed_state(a, x, fn, initial_updates=0, maximum_updates=4, authorized=True)


def test_fixed_dataset_checkpoint_restores_targets_and_next_update(tmp_path):
    from world_model.critic_fit import make_update
    from world_model.fixed_critic_fit import update_fixed_state
    from world_model.imagination_learner import init_learner
    from world_model.imagination_checkpoint import save_checkpoint, restore_checkpoint
    from world_model.neural import init_mlp
    from world_model.resources import StorageBudget
    from world_model.state_adapter import numeric_tree_digest
    p = init_mlp(jax.random.PRNGKey(3), (3, 4, 1))
    s = dict(learner=init_learner(p, p, None, seed=4),
             fixed_lambda_targets=jnp.arange(10, dtype=jnp.float32),
             fixed_fit_transactions=jnp.array(0))
    fn = make_update(dict(critic_batch=8, critic_microbatch=4, critic_lr=1e-4,
        critic_updates_per_actor=4, target_critic_ema=.05))
    x = np.ones((10, 3), np.float32)
    a, _ = update_fixed_state(s, x, fn, initial_updates=0, authorized=True)
    ref = save_checkpoint(tmp_path/'full.json', a, signature='fixed',
                          budget=StorageBudget(tmp_path, 2**20, 0), metadata={})
    restored = restore_checkpoint(ref, a, signature='fixed')
    expected, _ = update_fixed_state(a, x, fn, initial_updates=0, authorized=True)
    actual, _ = update_fixed_state(restored, x, fn, initial_updates=0, authorized=True)
    assert numeric_tree_digest(actual) == numeric_tree_digest(expected)


def test_fixed_replay_pairs_float_tolerance_but_exact_source_identity():
    from world_model.train_fixed_critic import pair_rows
    previous = dict(value=np.array([1., 2.], np.float32), source=np.array([4, 5]))
    current = dict(value=previous['value']+1e-6, source=previous['source'].copy())
    tolerance = dict(atol=5e-5, rtol=1e-5)
    assert pair_rows(current, previous, tolerance)['passed']
    current['source'][0] = 3
    with pytest.raises(ValueError, match='source'):
        pair_rows(current, previous, tolerance)


def test_fixed_fit_plan_authenticates_original_recipe_and_sources():
    from world_model.contracts import load_config, authenticate, repository_root
    from world_model.train_fixed_critic import check_plan
    cfg = load_config()
    cfg['critic_fixed_data_fit']['authorized'] = True
    with pytest.raises(ValueError, match='guard changed'):
        check_plan(cfg)
    plan = json.loads(authenticate(repository_root(), cfg['critic_fixed_data_fit']['plan']).read_text())
    source = json.loads(authenticate(repository_root(), plan['source_plan']).read_text())
    assert plan['parent_critic_updates'] == 1144
    assert plan['recipe']['target_critic_ema'] == .05
    assert not set(source['train_indices']) & set(source['heldout_indices'])
    cfg['imagination']['critic_lr'] *= 2
    with pytest.raises(ValueError, match='scope'):
        check_plan(cfg)
