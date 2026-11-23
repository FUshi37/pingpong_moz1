"""Read-only attribution must not turn privileged labels into runtime inputs."""
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))


def test_capture_cut_preserves_native_collection_and_excludes_model_experiment():
    import inspect
    from world_model.audit_free_imagination import audit
    from world_model.error_attribution import capture_source
    source=capture_source(inspect.getsource(audit))
    assert 'range(128)' in source and 'assess_native_pair' in source
    assert 'def replay_controller' not in source
    assert 'world_bank=world_bank' in source
    with pytest.raises(ValueError): capture_source('def audit():\n    return 1\n')


def test_flat_solver_observer_preserves_original_trace_and_carry():
    from world_model.error_attribution import augmented_recorded_loop
    from world_model.teacher_trace import recorded_loop
    def substep(i, carry):return (carry[0]+1.,)
    def sample(i,before,after):return dict(value=before[0])
    labels=lambda env,current:dict(target=jnp.stack((current,current),axis=-1),
        closure_error=current*0,active_pair_count=jnp.ones_like(current),overlap=current>0)
    loop=augmented_recorded_loop(None,labels=labels)
    expected,tr=recorded_loop(substep,(jnp.array([0.]),),5,sample)
    got,trace=jax.jit(lambda:loop(substep,(jnp.array([0.]),),5,sample))()
    np.testing.assert_array_equal(got[0],expected[0])
    np.testing.assert_array_equal(trace['value'],tr['value'])
    assert trace['solver_target'].shape==(5,1,2)
    assert not any(isinstance(v,dict) for v in trace.values())


def test_prediction_windows_never_use_clamped_padding_as_evidence():
    from world_model.error_attribution import window_arrays
    values=np.arange(5*2).reshape(5,2)
    result,available=window_arrays(values,np.array([0,3]),np.array([0,1]),length=4)
    np.testing.assert_array_equal(result[:,0],[0,2,4,6])
    np.testing.assert_array_equal(available[:,1],[True,True,False,False])


def test_error_locations_exclude_post_failure_and_preserve_source_clock():
    from world_model.error_attribution import largest_prefix_errors
    errors=np.array([[.1,.2],[99.,.4]])
    prefix=np.array([[True,True],[False,True]])
    rows=largest_prefix_errors(errors,prefix,[117,843],limit=2)
    assert rows==[dict(source_index=843,lane=1,control=2,error=.4),
                  dict(source_index=843,lane=1,control=1,error=.2)]
    assert largest_prefix_errors(errors,np.zeros_like(prefix),[117,843])==[]
