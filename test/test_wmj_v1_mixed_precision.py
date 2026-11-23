"""Regression contracts for the opt-in numerical candidate, never its promotion."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))

from world_model.robot_model import RobotScale
from world_model.mixed_precision import normalized_prediction, robot_increment_step, control_boundary, make_runtime


def test_small_physical_increment_survives_without_changing_weights():
    params=({'w':jnp.zeros((47,26)), 'b':jnp.zeros(26)},)
    scale=RobotScale(jnp.zeros(47),jnp.ones(47),jnp.ones(26)*1e-8,jnp.ones(26))
    commands={k:jnp.zeros((1,7)) for k in ('q','dq','ddq')}
    with jax.enable_x64(True):
        q=jnp.ones((1,13),jnp.float64)*2;dq=jnp.zeros_like(q)
        after,_=robot_increment_step(params,scale,q,dq,commands)
        assert after.dtype==jnp.float64
        np.testing.assert_allclose(after-q,1e-8,rtol=1e-7)
    assert params[0]['w'].dtype==jnp.float32


def test_high_precision_normalization_retains_small_waist_signal():
    weights=jnp.zeros((47,6)).at[36,0].set(1.)
    params=({'w':weights,'b':jnp.zeros(6)},)
    scale=RobotScale(jnp.ones(47)*2,jnp.ones(47)*1e-4,jnp.zeros(6),jnp.ones(6))
    with jax.enable_x64(True):
        x=jnp.ones((1,47),jnp.float64)*2
        def f(a):
            return normalized_prediction(params,scale,x.at[0,36].add(a*1e-4))[0,0]
        v,t=jax.jvp(f,(jnp.array(0.,jnp.float64),),(jnp.array(1.,jnp.float64),))
        fd=(f(.001)-f(-.001))/.002
        np.testing.assert_allclose(fd,t,rtol=1e-5)
        assert v.dtype==jnp.float64


def test_control_boundary_keeps_fp32_and_restores_outer_precision():
    calls=[]
    def native(state,raw,q,context,*,chain,contract):
        calls.append((jax.config.x64_enabled,raw.dtype,q.dtype,chain.dtype,contract.dtype))
        return state,{'raw':raw}
    wrapped=control_boundary(native,jnp.ones(2),jnp.ones(2),reference=False)
    with jax.enable_x64(True):
        s=jnp.ones((1,7),jnp.float32)
        wrapped(s,jnp.ones((1,7),jnp.float64),jnp.ones((1,13),jnp.float64),s,chain=None,contract=None)
        assert jax.config.x64_enabled
    assert calls==[(False,jnp.float32,jnp.float32,jnp.float32,jnp.float32)]


def test_legacy_runtime_reuses_original_functions_and_state():
    from world_model.course_env import course_step
    from world_model.full_prior_world import full_prior_world_step
    runtime=make_runtime('fp32',None,None,{},None)
    assert runtime.course_step is course_step
    assert runtime.world_step is full_prior_world_step
    state=object()
    assert runtime.prepare_course(state) is state
    with pytest.raises(ValueError):make_runtime('unregistered',None,None,{},None)


def test_fp64_course_counter_reduction_keeps_scan_state_type():
    from typing import NamedTuple
    from world_model.mixed_precision import stable_course_counters
    class Counters(NamedTuple):
        completed_goals: object
        successful_goals: object
    with jax.enable_x64(True):
        initial=Counters(jnp.zeros(1,jnp.int32),jnp.zeros(1,jnp.int32))
        def tick(state,_):
            count=jnp.ones((5,1),jnp.int32).sum(axis=0)
            after=Counters(state.completed_goals+count,state.successful_goals+count)
            return stable_course_counters(after,state),None
        final,_=jax.lax.scan(tick,initial,None,length=3)
        assert final.completed_goals.dtype==jnp.int32
        np.testing.assert_array_equal(final.completed_goals,[15])


def test_h64_branch_comparison_includes_center_and_retains_changed_case():
    from world_model.audit_mixed_precision import difference_row
    zero=np.zeros((2,1,1),bool);different=np.ones_like(zero)
    plus=(np.array([[1.,0.]]),{'modes':different})
    minus=(np.array([[-1.,0.]]),{'modes':different})
    row=difference_row(plus,minus,{'modes':zero},np.zeros((1,2,7)),np.ones(7),.001)
    assert row['center_endpoint_changed_indices']==[0]
    assert row['informative_direct']==1
    assert row['common_mode_informative_direct']==0
    assert row['central_components'][0][0]==1000.


def test_action_outcome_archive_retains_missing_values_as_masks():
    from world_model.audit_mixed_actions import pack_rows
    rows=[{'event':True,'xyz':[1.,2.,3.]},{'event':False,'xyz':None}]
    arrays,_=pack_rows(rows,'candidate')
    np.testing.assert_array_equal(arrays['candidate__xyz__present'],[True,False])
    np.testing.assert_array_equal(arrays['candidate__xyz'],[[1.,2.,3.],[0.,0.,0.]])
    assert np.isfinite(arrays['candidate__xyz']).all()


def test_compact_archive_slabs_reconstruct_every_dtype_and_shape():
    from world_model.audit_mixed_actions import pack_numeric_slabs
    original={'a':np.array([[1.,2.]],np.float64),'b':np.array([False,True]),
              'c':np.arange(6,dtype=np.int32).reshape(2,3),'d':np.array([-3.],np.float64)}
    slabs,layout=pack_numeric_slabs(original)
    assert len(slabs)==3
    for key,value in original.items():
        row=layout[key]
        restored=slabs[row['slab']][row['offset']:row['offset']+row['count']].reshape(row['shape'])
        assert restored.dtype==value.dtype
        np.testing.assert_array_equal(restored,value)
