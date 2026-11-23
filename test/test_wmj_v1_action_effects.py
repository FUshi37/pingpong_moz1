"""Counterfactual outcomes retain missing events and compare paired changes."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))


def test_first_release_only_uses_its_own_subsequent_apex():
    from world_model.action_effects import summarize_outcomes
    labels={k:np.zeros((8,2),bool) for k in ('controlled_release','actual_apex','contact_entry','contact_release','valid')}
    labels['valid'][:]=True
    labels['controlled_release'][2,0]=True
    labels['contact_release'][2,0]=True
    labels['actual_apex'][1,0]=True  # Earlier launch cannot belong to this release.
    labels['actual_apex'][6,0]=True
    labels['impact_id']=np.ones((8,2),np.int32)
    labels['impact_id'][6,0]=2  # Later impact's apex cannot be attributed either.
    for key in ('release_position_m','release_velocity_m_s','apex_position_m'):
        labels[key]=np.zeros((8,2,3),np.float32)
    labels['release_velocity_m_s'][2,0,2]=2.
    result=summarize_outcomes(labels)
    assert result[0]['release_present'] and not result[0]['actual_apex_present']
    assert result[0]['analytic_apex_m'][2]>0.
    assert not result[1]['release_present']
    assert result[1]['outgoing_velocity_m_s'] is None


def test_delta_comparison_reports_direction_and_missing_counterfactuals():
    from world_model.action_effects import compare_effects
    def outcome(x,present=True):
        return dict(release_present=present,impact_id=1,entries=1,releases=1,
            outgoing_velocity_m_s=[x,0,2.] if present else None,
            analytic_apex_m=[x*.2,0,1.4] if present else None,
            actual_apex_present=False,actual_apex_m=None)
    native={'base':[outcome(0),outcome(0)],'plus':[outcome(.02),outcome(0,False)],
            'minus':[outcome(-.02),outcome(-.02)]}
    world={'base':[outcome(0),outcome(0)],'plus':[outcome(-.02),outcome(.02)],
           'minus':[outcome(.02),outcome(-.02)]}
    result=compare_effects(native,world)
    assert result['cases']==2 and result['paired_release_cases']==1
    assert result['outgoing_velocity_m_s']['direction_agreement_fraction']==0.
    assert result['outgoing_velocity_m_s']['cosine']['mean']==-1.
    assert result['missing_native_release_cases']==1


def test_counterfactual_clones_copy_every_batch_leaf_and_preserve_scalar_phase():
    import jax.numpy as jnp
    import pytest
    from world_model.audit_action_effects import clone_batch_lanes
    state=dict(q=jnp.arange(12).reshape(4,3),model=dict(mass=jnp.arange(4)),
               rng=jnp.arange(8).reshape(4,2),shared_phase=jnp.array(2))
    mapping=np.array([2,2,1,1])
    cloned=clone_batch_lanes(state,mapping)
    np.testing.assert_array_equal(cloned['q'][0],state['q'][2])
    np.testing.assert_array_equal(cloned['q'][0],cloned['q'][1])
    np.testing.assert_array_equal(cloned['rng'][2],cloned['rng'][3])
    np.testing.assert_array_equal(cloned['model']['mass'],[2,2,1,1])
    assert int(cloned['shared_phase'])==2
    with pytest.raises(ValueError):
        clone_batch_lanes(dict(unregistered_axis=jnp.zeros((3,7))),mapping)


def test_repeat_diagnostic_keeps_unresolved_and_clipped_effects_visible():
    from world_model.action_effects import compare_with_repeat
    def row(x):
        return dict(release_present=True,impact_id=1,actual_apex_present=True,
                    outgoing_velocity_m_s=[x,0,2.],actual_apex_m=[x,0,1.4])
    native=dict(base=[row(0)]*3,plus=[row(.01)]*3,minus=[row(-.01)]*3)
    world=dict(base=[row(0)]*3,plus=[row(-.01)]*3,minus=[row(.01)]*3)
    repeat=[row(.001),row(.01),row(0)]
    cases=[dict(effective_clipped_central_pulse_norm=v) for v in (.1,.1,0.)]
    result=compare_with_repeat(native,world,repeat,cases)
    velocity=result['outgoing_velocity_m_s']
    assert velocity['paired_cases']==3 and velocity['clipped_zero_cases']==1
    assert velocity['effect_exceeds_repeat_cases']==1
    assert velocity['resolved_direction_agreement_fraction']==0.
    assert velocity['unresolved_cases']==2
