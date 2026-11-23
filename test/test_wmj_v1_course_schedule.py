"""Exactly one bounded pulse per surviving four-goal group, without resets."""

from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.course_schedule import init_schedule, finish_substep, schedule_features, pulse_proposal


def test_one_uniform_internal_switch_per_group_and_antithetic_proposals():
    state=init_schedule(jax.random.PRNGKey(7),jnp.arange(6000))
    proposals=pulse_proposal(state)
    np.testing.assert_allclose(proposals[0::2],-proposals[1::2],atol=0.)
    assert np.linalg.norm(proposals[:,:2],axis=-1).max()<=.05
    assert np.abs(proposals[:,2]).max()<=.1
    counts=[]
    for goal in range(4):
        state,started=finish_substep(state,jnp.ones(6000,bool),jnp.ones(6000,bool),
            enabled=True,mass_kg=.004)
        counts.append(np.asarray(started))
        for _ in range(20):
            state,_=finish_substep(state,jnp.zeros(6000,bool),jnp.ones(6000,bool),
                enabled=True,mass_kg=.004)
    np.testing.assert_array_equal(np.sum(counts,axis=0),1)
    assert not counts[3].any()
    assert all(abs(np.mean(x)-1/3)<.035 for x in counts[:3])
    assert np.all(np.asarray(state.group_id)==1)
    np.testing.assert_array_equal(schedule_features(state),0.)


def test_pulse_is20_substeps_and_padding_never_consumes_it():
    state=init_schedule(jax.random.PRNGKey(4),jnp.arange(2))
    state=state._replace(goals_completed=jnp.full(2,2,jnp.int32))
    state,started=finish_substep(state,jnp.ones(2,bool),jnp.ones(2,bool),enabled=True,mass_kg=.004)
    assert np.all(started) and np.all(np.asarray(state.remaining_steps)==20)
    force=np.asarray(state.force_world_n)
    impulse=np.zeros_like(force)
    for i in range(20):
        impulse+=np.asarray(state.force_world_n)*.001
        state,_=finish_substep(state,jnp.zeros(2,bool),jnp.ones(2,bool),enabled=True,mass_kg=.004)
        assert np.all(np.asarray(state.remaining_steps)==19-i)
    np.testing.assert_allclose(impulse,force*.02,rtol=1e-6)
    np.testing.assert_array_equal(state.force_world_n,0.)
    before=init_schedule(jax.random.PRNGKey(1),jnp.arange(2))._replace(
        remaining_steps=jnp.full(2,20,jnp.int32),force_world_n=jnp.ones((2,3))*.01)
    after,_=finish_substep(before,jnp.ones(2,bool),jnp.zeros(2,bool),enabled=True,mass_kg=.004)
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)


def test_critic_six_values_exclude_future_direction_and_rng():
    state=init_schedule(jax.random.PRNGKey(7),jnp.arange(2))
    altered=state._replace(direction_key=state.direction_key+1,pulse_rng=state.pulse_rng+2,
                          goal_rng=state.goal_rng+3,group_id=state.group_id+10)
    np.testing.assert_array_equal(schedule_features(state),schedule_features(altered))
    changed=state._replace(goals_completed=jnp.array([1,2]),pulse_used=jnp.array([True,False]),
        remaining_steps=jnp.array([10,0]),force_world_n=jnp.array([[.01,0.,-.02],[0.,0.,0.]]))
    np.testing.assert_allclose(schedule_features(changed),[[.25,1.,.5,.5,0.,-1.],[.5,0.,0.,0.,0.,0.]])
