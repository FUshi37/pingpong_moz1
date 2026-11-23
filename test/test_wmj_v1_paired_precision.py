"""Paired training must isolate precision from randomness and optimizer dtypes."""
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))


def stochastic_probe(key):
    return jax.random.normal(key,(8,)),jax.random.randint(key,(8,),0,1024)


def test_random_streams_remain_exactly_fp32_int32_under_x64():
    from world_model.paired_precision import random32_clone
    f=random32_clone(stochastic_probe)
    key=jax.random.PRNGKey(12)
    with jax.enable_x64(False):expected=stochastic_probe(key)
    with jax.enable_x64(True):actual=f(key)
    for a,b in zip(actual,expected):
        assert a.dtype==b.dtype
        np.testing.assert_array_equal(a,b)


def test_mixed_targets_keep_value_math_and_fp32_critic_interface():
    from world_model.paired_precision import mixed_lambda_targets
    from world_model.actor_critic import ImaginedBatch
    with jax.enable_x64(True):
        r=jnp.array([[1.],[2.]],jnp.float64);valid=jnp.ones_like(r,bool)
        b=ImaginedBatch(r,jnp.ones_like(r)*.5,valid,valid)
        value=jnp.array([[3.],[4.]],jnp.float32)
        actual=mixed_lambda_targets(b,value,lambda_value=1.)
    assert actual.dtype==jnp.float32
    np.testing.assert_allclose(actual,[[3.],[4.]])


def test_parent_learning_state_preserves_optimizer_moments_and_target():
    from world_model.paired_precision import learning_state
    from world_model.imagination_learner import LearnerState
    values=[jnp.array([i],jnp.float32) for i in range(5)]
    s=LearnerState(*values,pool=None,sampling_rng=jax.random.PRNGKey(0),
        goal_rng=jax.random.PRNGKey(1),actor_updates=jnp.int32(20),critic_updates=jnp.int32(80))
    d=learning_state(s)
    assert set(d)==set(LearnerState._fields)-{'pool'}
    for k in d:assert d[k] is getattr(s,k)


def test_authorized_storage_extension_preserves_legacy_and_free_floor():
    import json
    from world_model.contracts import repository_root, validate_config
    import pytest
    cfg = json.loads((repository_root() / 'pingpong_controller/tools/rl_2real/WMJ_V1_CONFIG.json').read_text())
    for cap in (6 * 2**30, 6576668672):
        cfg['resources']['max_new_bytes'] = cap
        validate_config(cfg)
    cfg['resources']['max_new_bytes'] = 6576668673
    with pytest.raises(ValueError, match='Storage budget'):
        validate_config(cfg)
    cfg['resources']['max_new_bytes'] = 6576668672
    cfg['resources']['min_free_bytes'] = 8 * 2**30 - 1
    with pytest.raises(ValueError, match='Storage budget'):
        validate_config(cfg)


def test_fp32_optimizer_keeps_mixed_physics_and_small_increments_intact():
    from world_model.imagination_learner import init_learner
    from world_model.neural import init_mlp
    from world_model.pilot_learning import optimize_pilot
    from world_model.train_paired import assert_learning_fp32
    with jax.enable_x64(True):
        pool = {'q': jnp.array([1. + 1e-9], jnp.float64)}
    with jax.enable_x64(False):
        actor = init_mlp(jax.random.PRNGKey(11), (2, 4, 1))
        critic = init_mlp(jax.random.PRNGKey(12), (3, 4, 1), zero_output=True)
        state = init_learner(actor, critic, pool, seed=14)
        gradient = jax.tree.map(jnp.ones_like, actor)
        inputs = np.ones((10, 3), np.float32)
        targets = np.linspace(0, 1, 10, dtype=np.float32)
        recipe = dict(actor_lr=1e-5, critic_lr=1e-4, critic_updates_per_actor=4,
            critic_batch=8, critic_microbatch=3, target_critic_ema=.01)
        after, _ = optimize_pilot(state, gradient, inputs, targets, recipe, authorized=True)
    assert_learning_fp32(after)
    assert after.pool is pool
    assert np.asarray(after.pool['q']).dtype == np.float64
    assert np.asarray(after.pool['q'])[0] > 1.
