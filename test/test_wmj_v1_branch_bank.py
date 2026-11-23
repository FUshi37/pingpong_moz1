"""Full-state branch ownership, causal admission and1024-lane reset semantics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_wmj_v1_control_projection import fixture
from test_wmj_v1_critic_features import setup_value
from world_model.imagined_env import init_imagined
from world_model.persistent_pool import init_pool
from world_model.branch_bank import BranchSeedBank, prepare_training_window


def initial(fixture):
    world,task,events,context,_=setup_value(fixture)
    return init_imagined(world,context,task.goal_xyz,events=events)


def test_bank_full_state_eviction_and_lossless_snapshot(fixture):
    state=initial(fixture)
    bank=BranchSeedBank(model_identity='frozen-a',stage_id='S01',capacity_per_category=2)
    for index in range(3):
        state=state._replace(episode_id=jnp.array([index]))
        bank.add_teacher_precontact(state,jnp.array([[index,8]],jnp.uint32),
            split='train',source_id=f'root/{index}',time_to_contact_s=.12)
    assert len(bank.rows)==2 and bank.evicted['precontact']==1
    tree,metadata=bank.snapshot()
    restored=BranchSeedBank.restore(tree,metadata,model_identity='frozen-a',stage_id='S01')
    assert restored.rows==bank.rows
    for a,b in zip(jax.tree.leaves(tree),jax.tree.leaves(restored.snapshot()[0])):
        np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match='identity'):
        BranchSeedBank.restore(tree,metadata,model_identity='frozen-b',stage_id='S01')
    contaminated=state._replace(terminated=jnp.array([True]))
    with pytest.raises(ValueError,match='ended'):
        bank.add_teacher_precontact(contaminated,jnp.zeros((1,2),jnp.uint32),
            split='train',source_id='invalid',time_to_contact_s=.12)
    noisy_bank=BranchSeedBank(model_identity='frozen-a',stage_id='S05')
    with pytest.raises(ValueError,match='noise'):
        noisy_bank.add_teacher_precontact(state,jnp.zeros((1,2),jnp.uint32),
            split='train',source_id='zero-noise-source',time_to_contact_s=.12)


def test_goal_branch_requires_observed_switch_and_retains_existing_history(fixture):
    state=initial(fixture)
    bank=BranchSeedBank(model_identity='frozen-a',stage_id='S02')
    later=state._replace(task=state.task._replace(goal_id=jnp.array([1]),
        goal_xyz=jnp.array([[.03,-.5,1.4]])),completed_goals=jnp.array([1]),episode_steps=jnp.array([900]))
    key=jnp.array([[3,4]],jnp.uint32)
    assert bank.observe(state,later,key,dict(goal_switched=jnp.array([0])),
        split='train',source_id='run/window')==0
    assert bank.observe(state,later,key,dict(goal_switched=jnp.array([1])),
        split='train',source_id='run/window')==1
    selected,_=bank.gather([0])
    assert int(selected.episode_steps[0])==900
    assert int(selected.completed_goals[0])==1
    for a,b in zip(jax.tree.leaves(later.world),jax.tree.leaves(selected.world)):
        np.testing.assert_array_equal(a,b)


def test_full_pool_keeps768_persistent_lanes_and_resets256_branches(fixture):
    seed=initial(fixture)
    bank=BranchSeedBank(model_identity='frozen-a',stage_id='S01')
    bank.add_teacher_precontact(seed,jnp.array([[9,10]],jnp.uint32),
        split='train',source_id='root/1',time_to_contact_s=.12)
    batch=jax.tree.map(lambda x:jnp.repeat(x,1024,axis=0),seed)
    batch=batch._replace(episode_steps=jnp.full(1024,25,jnp.int32),
        terminated=jnp.zeros(1024,bool).at[2].set(True))
    keys=jax.random.split(jax.random.PRNGKey(5),1024)
    pool=init_pool(batch,keys)
    persistent=jax.tree.map(lambda x:jnp.repeat(x,768,axis=0),seed)
    fresh_keys=jax.random.split(jax.random.PRNGKey(6),1024)
    after,report=prepare_training_window(pool,persistent,bank,np.random.default_rng(3),
        fresh_keys,minimum_precontact_lead_s=.08)
    assert report['persistent_resets']==1 and report['branch_resets']==256
    assert np.count_nonzero(np.asarray(after.environment.episode_steps)==25)==767
    np.testing.assert_array_equal(after.environment.world.physics.q_rad[:2],batch.world.physics.q_rad[:2])
    np.testing.assert_array_equal(after.exploration_rng[:2],keys[:2])
    np.testing.assert_array_equal(after.exploration_rng[768:],fresh_keys[768:])
    np.testing.assert_array_equal(after.environment.world.observation.history.paired.racket_pose[768],
        seed.world.observation.history.paired.racket_pose[0])
    np.testing.assert_array_equal(after.environment.world.observation.camera.epsilon_m[768],
        seed.world.observation.camera.epsilon_m[0])
