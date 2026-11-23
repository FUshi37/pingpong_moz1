"""Imagined task boundaries preserve physics and never cross reset gradients."""

from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_wmj_v1_control_projection import fixture
from test_wmj_v1_critic_features import setup_value
from test_wmj_v1_full_prior_world import components
from test_wmj_v1_world_control import setup_control
from world_model.full_prior_world import full_prior_world_step
from world_model.imagined_env import (
    EpisodeContract, init_imagined, imagined_step, task_substep, reset_lanes,
)
from world_model.imagined_rewards import RewardContract
from world_model.task import init_task


def test_goal_completion_changes_only_task_at_actual_apex():
    state=init_task(jnp.array([[0.,-.5,1.4]]))._replace(
        hit_count=jnp.array([16]),apex_count=jnp.array([15]),
        flight_hit_index=jnp.array([16]),awaiting_apex=jnp.array([True]),
        stability_qualified=jnp.array([True]))
    labels=dict(impact_id=jnp.array([16]),contact_release=jnp.array([False]),
        controlled_release=jnp.array([False]),release_position_m=jnp.zeros((1,3)),
        release_velocity_m_s=jnp.zeros((1,3)),actual_apex=jnp.array([True]),
        apex_position_m=jnp.array([[0.,-.5,1.4]]),apex_velocity_m_s=jnp.zeros((1,3)))
    goal=jnp.array([[.1,-.5,1.35]])
    after,terms,events=task_substep(state,labels,goal,contract=RewardContract())
    assert bool(events['goal_completed'][0]) and bool(events['goal_success'][0])
    assert int(after.goal_id[0])==1 and int(after.hit_count[0])==0
    np.testing.assert_array_equal(after.goal_xyz,goal)
    assert float(terms['total'][0])==0.


def test_imagined_world_matches_frozen_integration_and_split_resume(fixture):
    world,task,events,context,_=setup_value(fixture)
    chain,q,_,_,control_contract=setup_control(fixture)
    robot,auxiliary,_,options=components();options=dict(options,chain=chain)
    physics=world.physics._replace(ball=world.physics.ball._replace(
        position_m=jnp.array([[0.,-.5,1.4]]),velocity_m_s=jnp.array([[0.,0.,.1]])))
    world=world._replace(physics=physics)
    state=init_imagined(world,context,task.goal_xyz,events=events)
    contract=EpisodeContract()
    kwargs=dict(control_contract=control_contract,episode_contract=contract,
                reward_contract=RewardContract(),ar1_enabled=False,**options)
    step=jax.jit(lambda s:imagined_step(robot,auxiliary,s,jnp.zeros((1,7)),
        task.goal_xyz,**kwargs))
    after,transition=step(state)
    expected,_=jax.jit(lambda w:full_prior_world_step(robot,auxiliary,w,jnp.zeros((1,7)),
        context=context,control_contract=control_contract,task_goal_xyz=task.goal_xyz,
        ar1_enabled=False,**options))(world)
    for a,b in zip(jax.tree.leaves(after.world),jax.tree.leaves(expected)):
        np.testing.assert_allclose(a,b,atol=3e-7,rtol=2e-6)
    assert bool(transition.valid[0]) and not bool(after.terminated[0])
    assert int(after.episode_steps[0])==1
    short=replace(contract,max_episode_steps=1)
    timed,transition=imagined_step(robot,auxiliary,state,jnp.zeros((1,7)),task.goal_xyz,
        **dict(kwargs,episode_contract=short))
    assert bool(timed.truncated[0]) and not bool(timed.terminated[0])
    assert float(transition.discount[0])>0 and not bool(transition.trace_continue[0])
    padded,next_transition=imagined_step(robot,auxiliary,timed,jnp.ones((1,7)),task.goal_xyz,
        **dict(kwargs,episode_contract=short))
    assert float(next_transition.reward[0])==0 and not bool(next_transition.valid[0])
    np.testing.assert_array_equal(padded.world.physics.q_rad,timed.world.physics.q_rad)


def test_reset_selected_lanes_stops_all_seed_gradients(fixture):
    world,task,events,context,_=setup_value(fixture)
    state=init_imagined(world,context,task.goal_xyz,events=events)
    def value(x):
        seeded=state._replace(world=world._replace(physics=world.physics._replace(q_rad=world.physics.q_rad+x)))
        return reset_lanes(state,seeded,jnp.array([True])).world.physics.q_rad.sum()
    assert float(jax.grad(value)(.1))==0.
    def retained(x):
        current=state._replace(world=world._replace(physics=world.physics._replace(q_rad=world.physics.q_rad+x)))
        return reset_lanes(current,state,jnp.array([False])).world.physics.q_rad.sum()
    assert float(jax.grad(retained)(.1))==13.


def test_nonfinite_action_is_model_truncation_without_reward_or_state_poison(fixture):
    world,task,events,context,_=setup_value(fixture)
    state=init_imagined(world,context,task.goal_xyz,events=events)
    chain,_,_,_,control=setup_control(fixture)
    robot,auxiliary,_,options=components()
    after,transition=imagined_step(robot,auxiliary,state,jnp.full((1,7),jnp.nan),
        task.goal_xyz,control_contract=control,episode_contract=EpisodeContract(),
        reward_contract=RewardContract(),ar1_enabled=False,**dict(options,chain=chain))
    assert bool(after.model_truncated[0]) and not bool(after.terminated[0])
    assert float(transition.reward[0])==0. and float(transition.discount[0])==0.
    assert np.isfinite(transition.next_critic_features).all()
    for before,value in zip(jax.tree.leaves(state.world),jax.tree.leaves(after.world)):
        np.testing.assert_array_equal(before,value)


def test_mid_contact_seed_requires_physical_event_memory(fixture):
    world,task,events,context,_=setup_value(fixture)
    world=world._replace(physics=world.physics._replace(ball=world.physics.ball._replace(
        previous_contact=jnp.array([True]))))
    with pytest.raises(ValueError,match='event memory'):
        init_imagined(world,context,task.goal_xyz)
    with pytest.raises(ValueError,match='event memory'):
        init_imagined(world,context,task.goal_xyz,task=task)
