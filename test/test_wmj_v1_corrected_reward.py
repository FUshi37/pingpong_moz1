"""Frozen reward probes retain masking, negative cases and corrected control."""
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))


def test_directional_probe_retains_unresolved_reversed_and_changed_modes():
    from world_model.corrected_reward import directional_summary
    center=np.zeros((4,5));tangent=np.ones((4,5));plus=np.full((4,5),.01)
    minus=-plus.copy();plus[1]*=-1;minus[1]*=-1
    plus[2]=minus[2]=0.
    modes=np.zeros((2,4,1),bool);changed=modes.copy();changed[:,3]=True
    criteria=dict(return_difference_floor=1e-5,repeat_multiplier=3.,relative_derivative_tolerance=.2,
                  minimum_resolved_sources=2,minimum_pass_fraction=.9)
    out=directional_summary(center,center,tangent,plus,minus,modes,modes,changed,.01,
                            [1,0,0,0,0],criteria)
    assert out['resolved_indices']==[0,1]
    assert out['mode_changed_indices']==[3]
    assert out['unresolved_indices']==[2,3]
    assert out['direction_agreement_fraction']==.5
    assert out['relative_derivative_pass_fraction']==.5
    assert not out['diagnostic_pass']


def test_component_return_keeps_failure_mask_and_replaces_exactly_two_costs():
    from world_model.corrected_reward import objective_returns
    from world_model.actor_critic import ImaginedBatch,actor_return
    from types import SimpleNamespace
    terms=dict(hit=jnp.ones((3,2)),apex_xy=-jnp.ones((3,2)),apex_z=-2*jnp.ones((3,2)),
               velocity_xy=-3*jnp.ones((3,2)),velocity_z=-4*jnp.ones((3,2)))
    tr=SimpleNamespace(reward_terms=terms,discount=jnp.array([[.9,.9],[0.,.9],[0.,.9]]),
        trace_continue=jnp.array([[True,True],[False,True],[False,True]]),
        valid=jnp.array([[True,True],[True,True],[False,True]]))
    values=jnp.full((3,2),5.)
    result=objective_returns(tr,values)
    replacement=terms['hit']+terms['velocity_xy']+terms['velocity_z']
    expected=actor_return(ImaginedBatch(replacement,tr.discount,tr.trace_continue,tr.valid),values)
    np.testing.assert_allclose(result@jnp.array([1.,-1.,-1.,1.,1.]),expected,rtol=1e-6)
    np.testing.assert_allclose(result[0,0], -2*(1+.9), rtol=1e-6)


def test_diagnostic_instrumentation_preserves_private_projection_and_global_path():
    from world_model.projection_precision import make_runtime
    from world_model.corrected_reward import observed_step
    from world_model.course_env import course_step
    original=course_step.__globals__['control_step']
    runtime=make_runtime(None,None,{},None)
    step=observed_step(runtime)
    assert step.__globals__['control_step'] is runtime.controller
    assert step.__globals__['physical_bounds'] is runtime.bounds
    assert course_step.__globals__['control_step'] is original
    assert runtime.projector.__globals__['jnp'].einsum is not jnp.einsum
