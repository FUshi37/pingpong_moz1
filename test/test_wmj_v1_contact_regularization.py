"""Contact fitting can reduce saturated activations without changing inference."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.contact_regularization import activation_penalty, validate_activation_transition


def test_activation_penalty_uses_contact_mask_and_has_desaturating_gradient():
    parameters = (dict(w=jnp.zeros((39,4)), b=jnp.zeros(4)),
                  dict(w=jnp.zeros((4,3)), b=jnp.array([5.,-5.,1.])),
                  dict(w=jnp.zeros((3,8)), b=jnp.zeros(8)))
    inputs = jnp.ones((2,39))
    penalty, saturated = activation_penalty(parameters,inputs,jnp.array([True,False]))
    np.testing.assert_allclose(penalty,6.)
    np.testing.assert_allclose(saturated,2/3)
    gradient = jax.grad(lambda p: activation_penalty(p,inputs,jnp.array([True,False]))[0])(parameters)
    assert gradient[1]['b'][0]>0 and gradient[1]['b'][1]<0 and gradient[1]['b'][2]==0
    assert activation_penalty(parameters,inputs,jnp.array([False,False]))[0]==0


def test_activation_stage_does_not_change_other_recipe_fields():
    old = dict(controls=8,matmul_precision='highest')
    new = dict(old,activation_penalty_weight=.01)
    with pytest.raises(ValueError):validate_activation_transition(old,new,start_stage=False)
    validate_activation_transition(old,new,start_stage=True)
    validate_activation_transition(new,new,start_stage=False)
    for wrong in (dict(new,controls=16),dict(new,matmul_precision=None),dict(new,activation_penalty_weight=-1)):
        with pytest.raises(ValueError):validate_activation_transition(old,wrong,start_stage=True)


def test_zero_penalty_preserves_original_physical_loss_and_gradient():
    from test_wmj_v1_world_physics import setup_physics
    from world_model.physics_fitting import predicted_racket_motion, coupled_contact_objective
    from world_model.position_contact import position_contact_rollout
    from world_model.contact_regularization import regularized_contact_objective
    robot,contact,state,options = setup_physics()
    raw = dict(q_before=state.q_rad[None],dq_before=state.dq_rad_s[None],
        **{'delayed_'+k+'_command':jnp.zeros((1,1,7)) for k in ('q','dq','ddq')})
    motion = predicted_racket_motion(robot,raw,scale=options['robot_scale'],chain=options['chain'])
    kwargs = dict(ball_radius_m=.02,position_scale=options['position_scale'],contract=options['contact_contract'])
    _,trace = position_contact_rollout(contact,options['surface'],state.ball,motion,**kwargs)
    batch = dict(before=trace['before'],after=trace['after'],racket=motion,
        valid=jnp.ones((1,1),bool),entry=trace['entry'],release=trace['release'],controlled_release=jnp.zeros((1,1),bool))
    loss_options = dict(ball_radius_m=.02,position_scale=options['position_scale'],
        contact_contract=options['contact_contract'],auxiliary_weight=256.)
    expected = jax.value_and_grad(coupled_contact_objective,has_aux=True)(
        contact,options['surface'],batch,motion,**loss_options)
    actual = jax.value_and_grad(regularized_contact_objective,has_aux=True)(
        contact,options['surface'],batch,motion,activation_penalty_weight=0.,**loss_options)
    for left,right in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
        np.testing.assert_array_equal(left,right)
