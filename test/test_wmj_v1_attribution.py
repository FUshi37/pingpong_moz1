"""Frozen diagnostics cannot replace task labels or mutate the normal hook."""

from pathlib import Path
import sys

import pytest
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from world_model.native_evaluation import merge_observer_trace


def test_diagnostic_observer_defaults_to_identical_output():
    output = {'valid_release': 1}
    assert merge_observer_trace(output, None, None, None, None, None) is output
    enriched = merge_observer_trace(output, lambda *args: {'geom': 3}, None, None, None, None)
    assert enriched == {'valid_release': 1, 'geom': 3}
    assert output == {'valid_release': 1}


def test_diagnostic_observer_cannot_overwrite_task_labels():
    with pytest.raises(ValueError, match='overwrite'):
        merge_observer_trace({'valid_release': 1}, lambda *args: {'valid_release': 0},
                             None, None, None, None)


def test_collision_observer_reports_only_existing_other_ball_overlap():
    from world_model.audit_attribution_native import contact_observer
    env=SimpleNamespace(ball_geom_id=9,racket_contact_geom_id=8)
    contact=SimpleNamespace(geom=jnp.asarray([[[9,8],[9,3],[9,4]],[[9,8],[9,3],[9,4]]]),
        dist=jnp.asarray([[-.01,-.002,.01],[-.01,.01,.02]]))
    result=contact_observer(env,None,SimpleNamespace(contact=contact),
        {'release_position_m':jnp.zeros((2,3))})
    np.testing.assert_array_equal(result['other_geom_pair'],[[9,3],[-1,-1]])
    np.testing.assert_allclose(result['other_contact_dist_m'],[-.002,0.])


def test_return_component_sum_preserves_bootstrap_and_failure_masks():
    import jax
    from world_model.actor_critic import ImaginedBatch,actor_return
    a=jnp.asarray([[1.,2.],[3.,0.],[4.,0.]])
    b=jnp.asarray([[.1,.2],[.3,0.],[.4,0.]])
    valid=jnp.asarray([[True,True],[True,False],[True,False]])
    discount=jnp.asarray([[.9,0.],[.9,0.],[.9,0.]])
    batch=ImaginedBatch(a+b,discount,valid,valid)
    values=jnp.ones_like(a)*5
    zero=jnp.zeros_like(a)
    parts=(actor_return(batch._replace(reward=a),zero)+
           actor_return(batch._replace(reward=b),zero)+
           actor_return(batch._replace(reward=zero),values))
    np.testing.assert_allclose(parts,actor_return(batch,values),rtol=1e-6)


def test_apex_replay_pairs_physical_impact_ids_not_arrival_hit():
    from world_model.attribution_common import apex_replay_deltas
    events=[dict(case_id='a',impact_id=i,arrival_hit=0,valid_apex=True,
                 apex_position_m=[0.,0.,z]) for i,z in [(1,1.4),(2,1.2)]]
    assert apex_replay_deltas(events,events)==[0.,0.]


def test_initial_interface_uses_physical_tolerances_and_exact_memories():
    from world_model.attribution_checks import compare_initial_interfaces
    reference=dict(control_sha256='control',camera_history_sha256='camera',
        qpos=[[1.]],qvel=[[0.]],racket_position_m=[[1.]],racket_rotation=[[1.]])
    changed=dict(reference,qpos=[[1.000001]])
    assert compare_initial_interfaces(changed,reference)['passed']
    assert not compare_initial_interfaces(dict(reference,qpos=[[1.01]]),reference)['passed']
    assert not compare_initial_interfaces(dict(reference,control_sha256='changed'),reference)['passed']


def test_original_evaluator_factory_binds_every_global_and_closure(monkeypatch):
    import builtins
    import dis
    from world_model import audit_attribution_collision as collision
    monkeypatch.setattr(collision,'build_evaluation_step',lambda *args,**kw:(None,{}))
    monkeypatch.setattr(collision.EpisodeContract,'from_config',lambda cfg:None)
    p=dict(env=None,adapter=None,control=None,bundle=SimpleNamespace(stack=None),
           goals=jnp.zeros((128,1,3)))
    fn,_=collision.original_policy_step(p,None,{})
    missing=[op.argval for op in dis.get_instructions(fn) if op.opname=='LOAD_GLOBAL'
             and op.argval not in fn.__globals__ and not hasattr(builtins,op.argval)]
    assert missing==[]


def test_world_event_trace_retains_actual_substep_clock_and_release():
    import jax
    from world_model.attribution_checks import world_event_trace
    from world_model.frozen_task import init_evaluation,consume_events
    goal=jnp.array([[[0.,-.5,1.35]]])
    state=init_evaluation(goal,jax.random.PRNGKey(1),jnp.array([0]))
    for index in range(4):
        released=index==3
        labels=dict(impact_id=jnp.array([int(released)]),contact_release=jnp.array([released]),
            controlled_release=jnp.array([released]),actual_apex=jnp.array([False]),
            apex_position_m=jnp.zeros((1,3)),apex_velocity_m_s=jnp.zeros((1,3)),
            release_position_m=jnp.array([[0.,-.5,1.1]]),release_velocity_m_s=jnp.array([[0.,0.,2.]]))
        state,scoring=consume_events(state,labels,goal,perturb=False)
        trace=world_event_trace(state,scoring,labels)
        assert int(trace['elapsed_ms'][0])==index+1
    assert bool(trace['valid_release'][0])
    np.testing.assert_array_equal(trace['release_velocity_m_s'],[[0.,0.,2.]])


def test_native_reward_probe_factory_has_no_missing_bindings(monkeypatch):
    import builtins
    import dis
    from world_model import audit_attribution_native_gradients as probe
    monkeypatch.setattr(probe,'build_evaluation_step',lambda *args,**kwargs:(None,{}))
    monkeypatch.setattr(probe.EpisodeContract,'from_config',lambda cfg:None)
    p=dict(env=None,adapter=None,goals=None,bundle=SimpleNamespace(stack=None))
    cfg={'imagination_implementation':{'reward':{'coefficients':{}}}}
    fn,_=probe.native_reward_step(p,np.array([0]),cfg)
    missing=[op.argval for op in dis.get_instructions(fn) if op.opname=='LOAD_GLOBAL'
        and op.argval not in fn.__globals__ and not hasattr(builtins,op.argval)]
    assert not missing


def test_world_mode_factory_retains_scope_bindings():
    import builtins
    import dis
    from world_model.audit_attribution_modes import build_mode_step
    fn=build_mode_step(dict.fromkeys(('robot','auxiliary','context','control','goals',
        'physics','n','reward','episode')))
    missing=[op.argval for op in dis.get_instructions(fn) if op.opname=='LOAD_GLOBAL'
        and op.argval not in fn.__globals__ and not hasattr(builtins,op.argval)]
    assert not missing
