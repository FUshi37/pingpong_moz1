"""Stable WMJ derivatives must preserve native RMP forward commands."""

from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from recovered_rmp_jax import JointRMPParams, init_joint_rmp_state, joint_rmp_step
from world_model.course_control import joint_rmp_step as differentiable_step, braking_sqrt
from test_wmj_v1_control_projection import fixture
from test_wmj_v1_world_control import setup_control


def test_safe_braking_derivative_at_and_beyond_both_bounds():
    params = JointRMPParams()
    low, high = jnp.full(7,-1.), jnp.full(7,1.)
    edge = np.float32(1.)-np.asarray(params.angle_gap_rad,np.float32)
    q = jnp.array([[edge[0],edge[1]+.01,-edge[2],-edge[3]-.01,0.,.4,-.4]])
    state = init_joint_rmp_state(q)
    args = (q,low,high,jnp.ones(7)*2.,jnp.ones(7)*30.,jnp.ones(7)*700.,params,.001)
    options = dict(servo_steps=5,target_update_delay_steps=3,return_history=True)
    native = jax.jit(lambda s:joint_rmp_step(s,*args,**options))(state)
    adapted = jax.jit(lambda s:differentiable_step(s,*args,**options))(state)
    for left,right in zip(jax.tree.leaves(native),jax.tree.leaves(adapted)):
        np.testing.assert_array_equal(left,right)
    def objective(fn,position):
        result = fn(state._replace(q=position),*args,**options)
        return jnp.sum(result[0].q)+jnp.sum(result[0].qd)
    old_gradient = jax.jit(jax.grad(lambda x:objective(joint_rmp_step,x)))(q)
    gradient = jax.jit(jax.grad(lambda x:objective(differentiable_step,x)))(q)
    assert not np.isfinite(np.asarray(old_gradient)).all()
    assert np.isfinite(np.asarray(gradient)).all()
    for margin in (-.01,.01):
        def speed(delta):
            return braking_sqrt(2.*30.*jnp.maximum(delta,0.))
        numeric = (speed(margin+1e-5)-speed(margin-1e-5))/(2e-5)
        np.testing.assert_allclose(jax.grad(speed)(margin),numeric,rtol=.002,atol=1e-5)


def test_opt_in_control_adapter_preserves_complete_native_forward(fixture):
    from world_model.course_control import control_step
    from world_model.world_control import control_step as native_step
    chain,q,state,context,contract = setup_control(fixture)
    action = jnp.array([[.3,-.2,.1,2.,-.7,.02,-.02]])
    expected = native_step(state,action,q,context,chain=chain,contract=contract)
    actual = control_step(state,action,q,context,chain=chain,contract=contract)
    for left,right in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
        np.testing.assert_array_equal(left,right)
