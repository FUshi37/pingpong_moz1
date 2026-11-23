"""Pool resets and exact numerical learner recovery retain the whole contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_wmj_v1_control_projection import fixture
from test_wmj_v1_critic_features import setup_value
from world_model.imagined_env import init_imagined
from world_model.persistent_pool import init_pool,prepare_window,commit_window,add_count,count_as_int
from world_model.imagination_checkpoint import save_checkpoint,restore_checkpoint
from world_model.resources import StorageBudget
from world_model.neural import init_adam


def pool_fixture(fixture):
    world,task,events,context,_=setup_value(fixture)
    state=init_imagined(world,context,task.goal_xyz,events=events)
    state=jax.tree.map(lambda x:jnp.repeat(x,4,axis=0),state)
    return init_pool(state,jax.random.split(jax.random.PRNGKey(6),4))


def test_only_ended_persistent_lanes_and_all_short_branches_reset(fixture):
    pool=pool_fixture(fixture);fresh=pool.environment
    state=fresh._replace(episode_steps=jnp.array([20,30,40,50]),
        terminated=jnp.array([False,True,False,False]),completed_goals=jnp.array([1,0,2,0]))
    pool=pool._replace(environment=state)
    after,mask=prepare_window(pool,fresh,jax.random.split(jax.random.PRNGKey(8),4),persistent_envs=3)
    np.testing.assert_array_equal(mask,[False,True,False,True])
    np.testing.assert_array_equal(after.environment.episode_steps,[20,0,40,0])
    np.testing.assert_array_equal(after.environment.completed_goals,[1,0,2,0])
    committed=commit_window(after,after.environment,after.exploration_rng,jnp.ones((8,4),bool))
    assert count_as_int(committed.imagined_transition_words)==32
    words=add_count(jnp.array([7,2**32-10],jnp.uint32),32)
    assert count_as_int(words)==8*2**32+22


def test_learner_snapshot_restores_full_pool_and_rejects_other_model(tmp_path,fixture):
    pool=pool_fixture(fixture)
    actor={'w':jnp.arange(14,dtype=jnp.float32).reshape(2,7)}
    state=dict(pool=pool,actor=actor,actor_optimizer=init_adam(actor),
        critic={'w':jnp.zeros((3,1))},target={'w':jnp.zeros((3,1))},
        sampling_rng=jax.random.PRNGKey(10))
    budget=StorageBudget(tmp_path,64*2**20,0)
    reference=save_checkpoint(tmp_path/'resume.json',state,signature='test-runtime',budget=budget,
        metadata=dict(actor_updates=0,critic_updates=0))
    restored=restore_checkpoint(reference,state,signature='test-runtime')
    for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(restored)):
        np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError):
        restore_checkpoint(reference,state,signature='other-model')
    with pytest.raises(FileExistsError):
        save_checkpoint(tmp_path/'resume.json',state,signature='test-runtime',budget=budget,metadata={})
