"""Local precision probes must preserve production functions and report branches."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import enable_x64

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))


def test_private_precision_clone_preserves_original_and_fp32_values():
    from world_model.module_gradient_tools import private_clone
    from world_model.neural import mlp
    params=({'w':jnp.array([[2.]],jnp.float32),'b':jnp.array([1.],jnp.float32)},)
    old_globals=mlp.__globals__
    clone=private_clone(mlp,float64=True)
    with enable_x64(True):
        x=jnp.array([[.123456789012]],jnp.float64)
        assert mlp(params,x).dtype==jnp.float32
        assert clone(params,x).dtype==jnp.float64
        np.testing.assert_allclose(clone(params,x),2*x+1,rtol=1e-14)
    assert mlp.__globals__ is old_globals
    np.testing.assert_array_equal(private_clone(mlp)(params,jnp.ones((1,1))),mlp(params,jnp.ones((1,1))))


def test_precision_cast_keeps_queue_indices_and_booleans():
    from world_model.module_gradient_tools import cast_floats
    with enable_x64(True):
        values=cast_floats({'q':jnp.ones(2),'index':jnp.array(1,jnp.int32),'mask':jnp.array(True)},jnp.float64)
        assert values['q'].dtype==jnp.float64
        assert values['index'].dtype==jnp.int32
        assert values['mask'].dtype==jnp.bool_


def test_probe_distinguishes_smooth_derivative_and_crossed_branch():
    from world_model.module_gradient_tools import probe_direction
    with enable_x64(True):
        def smooth(a):
            return jnp.array([[jnp.sin(a),a*a+2*a]]),jnp.zeros((1,1),jnp.int32)
        result=probe_direction(smooth,[1e-4],dtype=jnp.float64)
        assert result['rows'][0]['relative_norm_error']['maximum']<1e-6
        assert result['rows'][0]['branch_changed_lanes']==0
        def switched(a):
            return jnp.array([[jnp.where(a>0,1.+a,a)]]),jnp.array([[a>0]])
        result=probe_direction(switched,[1e-4],dtype=jnp.float64)
        assert result['rows'][0]['branch_changed_lanes']==1


def test_cross_precision_excludes_changed_branches():
    from world_model.audit_module_gradient_details import compare_precisions
    value = np.zeros((2, 2))
    tangent = np.ones((2, 2))
    fp32 = (value, tangent, np.array([[0], [0]]))
    fp64 = (value, np.array([[1., 1.], [100., 100.]]), np.array([[0], [1]]))
    report = compare_precisions(fp32, fp64)
    assert report['changed_branch_indices'] == [1]
    assert report['common_branch_tangent_relative_error']['maximum'] == 0.


def test_input_rounding_detects_lost_small_displacement():
    from world_model.audit_module_gradient_details import input_rounding_rows
    feature = np.array([[2.]], np.float32)
    direction = np.array([[.2]], np.float32)
    scale = np.array([1e-4], np.float32)
    delta = direction*scale*.001
    result = input_rounding_rows(feature+delta, feature-delta, direction, scale, .001)
    assert result[0]['unchanged_fraction'] == 1.
    assert result[0]['normalized_direction_error_rms'] > .19
