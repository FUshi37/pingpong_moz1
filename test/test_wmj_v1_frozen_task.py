"""Finite native evaluation must stop without manufacturing another goal."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
import jax
import jax.numpy as jnp
import numpy as np

from world_model.frozen_task import init_evaluation, consume_events, fail_evaluation


def labels(goal, impact=0, release=False, apex=False):
    return dict(impact_id=jnp.array([impact]),contact_release=jnp.array([release]),
        controlled_release=jnp.array([release]),actual_apex=jnp.array([apex]),
        apex_position_m=jnp.asarray(goal).reshape(1,3),apex_velocity_m_s=jnp.zeros((1,3)))


def test_initial_apex_and_finite_goal_completion():
    goals=jnp.array([[[0.,-.5,1.35],[.04,-.5,1.35]]])
    state=init_evaluation(goals,jax.random.PRNGKey(42),jnp.array([0]))
    state,event=consume_events(state,labels(goals[0,0],apex=True),goals,perturb=False)
    assert state.task.apex_count[0]==0
    for goal_index in range(2):
        for hit in range(16):
            state,_=consume_events(state,labels(goals[0,goal_index],goal_index*16+hit+1,release=True),goals,perturb=False)
            state,event=consume_events(state,labels(goals[0,goal_index],apex=True),goals,perturb=False)
        assert bool(state.done[0]) == (goal_index==1)
        assert int(state.completed_goals[0])==goal_index+1
    assert bool(state.success[0]) and int(state.hits[0])==32
    assert int(state.task.goal_id[0])==1
    previous=state
    state,_=consume_events(state,labels(goals[0,0],999,True,True),goals,perturb=False)
    for a,b in zip(jax.tree.leaves(previous),jax.tree.leaves(state)):
        np.testing.assert_array_equal(a,b)


def test_first_physical_failure_stops_case_and_preserves_reason():
    goals=jnp.array([[[0.,-.5,1.35]]])
    state=init_evaluation(goals,jax.random.PRNGKey(42),jnp.array([0]))
    state=fail_evaluation(state,jnp.array([2]))
    state=fail_evaluation(state,jnp.array([1]))
    state,_=consume_events(state,labels(goals[0,0],1,True,True),goals,perturb=False)
    assert int(state.failure_code[0])==2
    assert bool(state.done[0]) and not bool(state.success[0]) and int(state.hits[0])==0


def test_endpoint_failure_marks_a_goal_finished_in_the_same_control_tick():
    goals=jnp.array([[[0.,-.5,1.35]]])
    state=init_evaluation(goals,jax.random.PRNGKey(42),jnp.array([0]))
    state=state._replace(task=state.task._replace(finished=jnp.array([True])))
    state=fail_evaluation(state,jnp.array([5]))
    assert bool(state.task.failed[0]) and bool(state.task.physical_failure[0])
    assert int(state.failure_code[0])==5 and bool(state.done[0])


def test_four_goals_deliver_one_twenty_tick_pulse_without_fifth_goal():
    goals=jnp.array([[[0.,-.5,1.35]]*4])
    state=init_evaluation(goals,jax.random.PRNGKey(42),jnp.array([0]))
    starts=0;force_ticks=0
    for index in range(4):
        for hit in range(16):
            for _ in range(25):
                force_ticks+=int(state.schedule.remaining_steps[0]>0)
                state,event=consume_events(state,labels(goals[0,0]),goals,perturb=True)
                starts+=int(event['pulse_start'][0])
            state,_=consume_events(state,labels(goals[0,0],index*16+hit+1,release=True),goals,perturb=True)
            state,event=consume_events(state,labels(goals[0,0],apex=True),goals,perturb=True)
            starts+=int(event['pulse_start'][0])
    assert starts==1 and force_ticks==20
    assert bool(state.success[0]) and int(state.task.goal_id[0])==3
