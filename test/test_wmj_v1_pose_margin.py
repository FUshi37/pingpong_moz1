"""Independent pose-margin calibration preserves the default reward contract."""

from dataclasses import replace
import math
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))


def test_disabled_and_zero_weight_preserve_original_reward_and_gradient():
    from world_model.pose_margin import PoseMarginContract, augment_reward
    def value(x, contract):
        return augment_reward(x*x, jnp.full((2,), x), True, False, .005, contract)
    for contract in (PoseMarginContract(), replace(PoseMarginContract(), enabled=True)):
        np.testing.assert_array_equal(value(jnp.float32(.04), contract), jnp.float32(.04)**2)
        np.testing.assert_allclose(jax.grad(lambda x:value(x, contract))(.04), .08)


def test_hinge_is_zero_inside_c1_at_onset_and_unsaturated_at_failure():
    from world_model.pose_margin import PoseMarginContract, pose_cost
    c=replace(PoseMarginContract(), enabled=True, weight_per_s=1.)
    f=lambda x:pose_cost(jnp.array([x, 0.]), c)
    assert f(0.) == f(c.onset_rad) == 0.
    assert jax.grad(f)(c.onset_rad) == 0.
    np.testing.assert_allclose(f(c.limit_rad), 1., rtol=2e-6)
    assert f(c.limit_rad+.001)>1. and jax.grad(f)(c.limit_rad)>0.


def test_penalty_gradient_restores_each_axis_and_matches_local_difference():
    from world_model.pose_margin import PoseMarginContract, penalty
    c=replace(PoseMarginContract(), enabled=True, weight_per_s=2.)
    for value in ([1.7,-1.9],[-1.8,2.1]):
        x=jnp.deg2rad(jnp.array(value,dtype=jnp.float32))
        f=lambda a:penalty(a,True,False,.005,c)
        gradient=np.asarray(jax.grad(f)(x))
        assert np.all(gradient*np.asarray(x)<0.)
        eps=1e-5
        fd=np.array([(float(f(x+jnp.eye(2)[i]*eps))-float(f(x-jnp.eye(2)[i]*eps)))/(2*eps) for i in range(2)])
        np.testing.assert_allclose(gradient,fd,rtol=3e-4,atol=1e-5)


def test_terminal_interval_is_charged_but_padding_and_model_failure_are_not():
    from world_model.pose_margin import PoseMarginContract, penalty
    c=replace(PoseMarginContract(), enabled=True, weight_per_s=1.)
    angle=jnp.tile(jnp.array([c.limit_rad,0.]),(3,1))
    result=penalty(angle,jnp.array([True,False,True]),jnp.array([False,False,True]),.005,c)
    np.testing.assert_allclose(result,[-.005,0.,0.],rtol=2e-6)
    np.testing.assert_allclose(2*penalty(angle,True,False,.005,c),penalty(angle,True,False,.010,c))


def test_gradient_grid_rejects_height_loss_and_selects_smallest_feasible_weight():
    import json
    from world_model.pose_margin import coefficient_grid, select_weight
    # base,height,XY,margin,unit-penalty in a two-parameter diagnostic space.
    vectors=np.array([[1.,-.2],[1.,0.],[0.,1.],[0.,1.],[0.,1.]])
    criteria=dict(height_direction_retention_min=.8,pose_to_height_gradient_norm_max=.5,
                  mean_pose_to_abs_height_return_max=.25)
    rows=coefficient_grid(vectors@vectors.T,-1.,-.1,[0.,.125,.25,.5,1.],criteria)
    assert json.loads(json.dumps(rows,allow_nan=False)) == rows
    assert not rows[0]['eligible'] and not rows[1]['eligible']
    assert rows[2]['eligible'] and not rows[-1]['eligible']
    assert select_weight({'a':rows,'b':rows})==.25
    assert select_weight({'a':rows[:2]}) is None


def test_invalid_margin_configuration_fails_closed():
    from world_model.pose_margin import PoseMarginContract
    for kwargs in (dict(weight_per_s=-1.),dict(onset_rad=math.radians(2.)),dict(limit_rad=float('nan'))):
        with pytest.raises(ValueError):PoseMarginContract(**kwargs)


def test_pose_angles_preserve_existing_reference_frame_and_margin_convention():
    from world_model.pose_margin import roll_pitch_rad
    from world_model.paired_attribution import orientation_margin_deg
    roll, pitch, yaw = np.deg2rad([1.8, -1.6, 17.])
    rx=np.array([[1.,0.,0.],[0.,np.cos(roll),-np.sin(roll)],[0.,np.sin(roll),np.cos(roll)]])
    ry=np.array([[np.cos(pitch),0.,np.sin(pitch)],[0.,1.,0.],[-np.sin(pitch),0.,np.cos(pitch)]])
    rz=np.array([[np.cos(yaw),-np.sin(yaw),0.],[np.sin(yaw),np.cos(yaw),0.],[0.,0.,1.]])
    reference=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],np.float32)
    rotation=reference@rx@ry@rz
    pose=jnp.array(np.concatenate(([0.,-.5,1.1],rotation[:,0],rotation[:,1]))[None],jnp.float32)
    angles=roll_pitch_rad(pose,jnp.asarray(reference))
    np.testing.assert_allclose(angles[0],[roll,pitch],atol=1e-7)
    old=orientation_margin_deg(pose,jnp.asarray(reference),math.radians(2.))
    np.testing.assert_allclose(2.-jnp.max(jnp.abs(jnp.rad2deg(angles)),axis=-1),old,atol=2e-6)
