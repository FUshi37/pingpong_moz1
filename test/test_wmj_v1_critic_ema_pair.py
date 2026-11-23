"""EMA is the only changed learning field in the paired Critic trial."""

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


def test_ema_pair_changes_only_target_after_common_first_four_steps():
    from world_model.critic_fit import make_update, update_session
    from world_model.imagination_learner import init_learner
    from world_model.neural import init_mlp
    from world_model.state_adapter import numeric_tree_digest
    p = init_mlp(jax.random.PRNGKey(1), (3, 8, 1))
    state = init_learner(p, p, None, seed=2)
    recipe = dict(critic_batch=16, critic_microbatch=4, critic_lr=1e-4,
                  critic_updates_per_actor=4, target_critic_ema=.01)
    x = np.arange(36, dtype=np.float32).reshape(12, 3)/30
    y = np.linspace(-10, 1, 12, dtype=np.float32)
    arms = [update_session(state, x, y, make_update(dict(recipe, target_critic_ema=rate)),
                           authorized=True)[0] for rate in (.01, .05)]
    for field in state._fields:
        if field != 'target_critic':
            assert numeric_tree_digest(getattr(arms[0], field)) == numeric_tree_digest(getattr(arms[1], field))
    assert numeric_tree_digest(arms[0].target_critic) != numeric_tree_digest(arms[1].target_critic)
    for rate, arm in zip((.01, .05), arms):
        for old, online, target in zip(jax.tree.leaves(state.target_critic),
                                       jax.tree.leaves(arm.critic), jax.tree.leaves(arm.target_critic)):
            np.testing.assert_allclose(target, (1-rate)*old+rate*online, atol=1e-7)


def test_ema_budget_extension_retains_previous_caps_and_free_floor():
    from world_model.contracts import repository_root, validate_config
    cfg = json.loads((repository_root() /
        'pingpong_controller/tools/rl_2real/WMJ_V1_CONFIG.json').read_text())
    cfg['critic_ema_pair'] = dict(plan={'path': 'registered'})
    cfg['resources']['max_new_bytes'] = 6677331968
    validate_config(cfg)
    for value in (6677331967, 6677331969):
        bad = copy.deepcopy(cfg); bad['resources']['max_new_bytes'] = value
        with pytest.raises(ValueError, match='Storage budget'):
            validate_config(bad)
    bad = copy.deepcopy(cfg); bad.pop('critic_ema_pair')
    with pytest.raises(ValueError, match='Storage budget'):
        validate_config(bad)
    bad = copy.deepcopy(cfg); bad['resources']['min_free_bytes'] = 8*2**30-1
    with pytest.raises(ValueError, match='Storage budget'):
        validate_config(bad)
    for value in (6*2**30, 6576668672, 6593445888, 6643777536):
        old = copy.deepcopy(cfg); old['resources']['max_new_bytes'] = value
        validate_config(old)


def test_failure_histogram_counts_only_actual_failed_rows():
    from types import SimpleNamespace
    from world_model.train_critic_ema import failure_counts
    tr = SimpleNamespace(terminated=jnp.array([[True, False], [False, True]]),
        task_events=dict(bound_ball=jnp.array([[False, False], [False, True]]),
                         bound_racket=jnp.array([[True, True], [False, False]]),
                         valid_release=jnp.ones((2, 2), jnp.int32)))
    assert failure_counts(tr) == dict(ball=1, racket=1)


def test_pair_identity_allows_value_updates_but_detects_sampling_changes():
    from world_model.train_critic_ema import shared_digest
    from world_model.imagination_learner import init_learner
    from world_model.neural import init_mlp
    p = init_mlp(jax.random.PRNGKey(3), (3, 4, 1))
    learner = init_learner(p, p, None, seed=4)
    state = dict(learner=learner, reset_bank=jnp.zeros((4, 3)),
        validation_bank=jnp.ones((2, 3)), current_source_indices=jnp.arange(4),
        fit_transactions=jnp.array(0))
    changed = jax.tree.map(lambda x:x+1, p)
    assert shared_digest(state) == shared_digest(dict(state,
        learner=learner._replace(critic=changed, target_critic=changed)))
    assert shared_digest(state) != shared_digest(dict(state,
        learner=learner._replace(sampling_rng=jax.random.PRNGKey(99))))
