"""Course RNG, force memory and causal captured states survive short windows."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from types import SimpleNamespace
from test_wmj_v1_control_projection import fixture
from test_wmj_v1_course_env import setup_course
from world_model.course_goals import build_goal_table
from world_model.course_rollout import course_policy_window,record_captured_seeds
from world_model.branch_bank import BranchSeedBank
from world_model.neural import init_mlp
from world_model.imagination_checkpoint import save_checkpoint,restore_checkpoint
from world_model.resources import StorageBudget
from world_model.sampling import SamplingShortage
from world_model.course_learner import commit_course_windows, compact_course_window, compact_critic_arrays
from world_model.actor_critic import init_critic
from world_model.imagination_learner import critic_training_arrays


def test_missing_goal_successor_rejects_pool_commit():
    window = SimpleNamespace(transitions=SimpleNamespace(
        task_events={'goal_sampling_shortage':np.array([[True]])}))
    with pytest.raises(SamplingShortage, match='no commit'):
        commit_course_windows(None, [window])


def test_causal_capture_and_pulse_checkpoint_continuation(tmp_path,fixture):
    _,state,robot,auxiliary,options=setup_course(fixture)
    state=state._replace(task=state.task._replace(hit_count=jnp.array([16]),
        apex_count=jnp.array([15]),flight_hit_index=jnp.array([16]),
        awaiting_apex=jnp.array([True]),stability_qualified=jnp.array([True])),
        schedule=state.schedule._replace(goals_completed=jnp.array([2])))
    actor=dict(zip(('l1','l2','out'),init_mlp(jax.random.PRNGKey(9),(969,256,256,7),zero_output=True)))
    table=build_goal_table([[0.,-.5,1.4],[.04,-.5,1.4]],static_admitted=[True,True])
    keys=jax.random.split(jax.random.PRNGKey(8),1)
    run=jax.jit(lambda s,k:course_policy_window(actor,robot,auxiliary,s,k,table,
        horizon=4,stage_id='S04',exploration_std=0.,**options))
    first=run(state,keys)
    critic=init_critic(jax.random.PRNGKey(42),first.critic_inputs.shape[-1])
    expected=critic_training_arrays([first],critic)
    compact=compact_course_window(first,critic)
    for left,right in zip(expected,compact_critic_arrays([compact])):
        np.testing.assert_array_equal(left,right)
    for left,right in zip(jax.tree.leaves(first.environment),jax.tree.leaves(compact.environment)):
        np.testing.assert_array_equal(left,right)
    assert compact.transitions.next_critic_features.shape[-1]==0
    whole=jax.jit(lambda s,k:course_policy_window(actor,robot,auxiliary,s,k,table,
        horizon=8,stage_id='S04',exploration_std=0.,**options))(state,keys)
    second=run(first.environment,first.exploration_rng)
    for a,b in zip(jax.tree.leaves(whole.environment),jax.tree.leaves(second.environment)):
        np.testing.assert_allclose(a,b,atol=3e-7,rtol=2e-6)
    np.testing.assert_array_equal(whole.exploration_rng,second.exploration_rng)
    assert int(first.environment.schedule.remaining_steps[0])==1
    bank=BranchSeedBank(model_identity='frozen-a',stage_id='S04')
    counts=record_captured_seeds(bank,first,split='train',source_id='test/window')
    assert counts==dict(goal_switch=1,recoverable_disturbance=1)
    captured,_=bank.gather([0,1])
    np.testing.assert_array_equal(captured.episode_steps,[1,1])
    assert int(first.environment.episode_steps[0])==4
    snapshot=dict(environment=first.environment,exploration_rng=first.exploration_rng)
    ref=save_checkpoint(tmp_path/'pulse.json',snapshot,signature='test-course',
        budget=StorageBudget(tmp_path,64*2**20,0),metadata={'updates':0})
    restored=restore_checkpoint(ref,snapshot,signature='test-course')
    replay=run(restored['environment'],restored['exploration_rng'])
    for a,b in zip(jax.tree.leaves(second),jax.tree.leaves(replay)):
        np.testing.assert_array_equal(a,b)
