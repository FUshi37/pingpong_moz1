"""Numerical parity against the authenticated released V321 actor."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model.contracts import load_config, repository_root
from world_model.migration import load_v321_actor, expand_actor_73, expand_actor_history, actor_mean
from world_model.paired_history import ACTOR_INPUT_DIM


def test_real_checkpoint_preserves_raw_actions_without_output_squashing():
    cfg = load_config()
    source = load_v321_actor(repository_root(), cfg['lineage']['actor_source'])
    migrated = expand_actor_73(source)
    rng = np.random.default_rng(20261105)
    obs72 = rng.normal(0., .3, (256, 72)).astype(np.float32)
    obs72[:, 69:71] = [0., -.49]
    obs73 = np.concatenate([obs72, np.zeros((256, 1), np.float32)], axis=1)
    # Independent reference formula from the archived tanh/tanh/linear contract.
    pi = source
    ref = np.tanh(obs72 @ pi['l1']['w'] + pi['l1']['b'])
    ref = np.tanh(ref @ pi['l2']['w'] + pi['l2']['b'])
    ref = ref @ pi['out']['w'] + pi['out']['b']
    actual = np.asarray(actor_mean(migrated, jnp.asarray(obs73)))
    np.testing.assert_allclose(actual, ref, atol=5e-6, rtol=5e-6)
    np.testing.assert_array_equal(migrated['l1']['w'][:72], source['l1']['w'])
    np.testing.assert_array_equal(migrated['l1']['w'][72], 0.)
    # Zero added row makes all target-Z values equivalent before learning.
    varied = obs73.copy()
    varied[:, 72] = rng.uniform(-1.25, .25, len(varied))
    np.testing.assert_array_equal(actor_mean(migrated, varied), actual)


def test_migration_refuses_wrong_hash_and_double_migration():
    cfg = load_config()
    contract = dict(cfg['lineage']['actor_source'], sha256='0' * 64)
    with pytest.raises(ValueError, match='SHA'):
        load_v321_actor(repository_root(), contract)
    source = load_v321_actor(repository_root(), cfg['lineage']['actor_source'])
    with pytest.raises(ValueError, match='72'):
        expand_actor_73(expand_actor_73(source))


def test_new_height_input_can_learn_and_world_parameters_are_not_needed():
    cfg = load_config()
    pi = expand_actor_73(load_v321_actor(repository_root(), cfg['lineage']['actor_source']))
    obs = jnp.ones((4, 73), jnp.float32) * .1
    gradient = jax.grad(lambda p: actor_mean(p, obs).sum())(pi)
    assert float(jnp.linalg.norm(gradient['l1']['w'][72])) > 0.


def test_real_checkpoint_long_history_migration_preserves_actions_and_can_learn():
    cfg = load_config()
    source = load_v321_actor(repository_root(), cfg['lineage']['actor_source'])
    expanded = expand_actor_history(source)
    rng = np.random.default_rng(2026110503)
    obs = rng.normal(0., .2, (64, ACTOR_INPUT_DIM)).astype(np.float32)
    old = jax.jit(actor_mean)(source, obs[:, :72])
    new = jax.jit(actor_mean)(expanded, obs)
    np.testing.assert_allclose(old, new, atol=5e-6, rtol=5e-6)
    np.testing.assert_array_equal(expanded['l1']['w'][72:], 0.)
    gradient = jax.grad(lambda p: actor_mean(p, obs).sum())(expanded)
    assert float(jnp.linalg.norm(gradient['l1']['w'][73:])) > 0.
    with pytest.raises(ValueError, match='72'):
        expand_actor_history(expanded)
