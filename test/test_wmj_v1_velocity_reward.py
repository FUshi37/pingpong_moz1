"""Outgoing-speed shaping is explicit, inverse-consistent and reward-only."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))


def test_inverse_target_and_position_scaled_speed_errors():
    from world_model.velocity_reward import velocity_costs
    from world_model.ball_flight import ballistic_apex, velocity_for_apex
    from world_model.imagined_rewards import RewardContract
    p=jnp.array([[.1,-.5,1.1]]); goal=jnp.array([[.13,-.48,1.4]])
    target,valid=velocity_for_apex(p,goal)
    apex,_,_=ballistic_apex(p,target)
    np.testing.assert_allclose(apex,goal,atol=2e-7)
    costs,info=velocity_costs(p,target,goal,jnp.array([True]),contract=RewardContract())
    assert bool(info['valid'][0]) and bool(valid[0])
    np.testing.assert_allclose(costs['velocity_xy'],0.,atol=1e-10)
    np.testing.assert_allclose(costs['velocity_z'],0.,atol=1e-10)
    np.testing.assert_allclose(info['scale_xyz_m_s'][0,:2]*info['time_s'][0],.1)
    np.testing.assert_allclose(info['scale_xyz_m_s'][0,2]*info['time_s'][0],.05)


def test_disabled_is_exact_and_enabled_replaces_only_two_costs():
    from world_model.velocity_reward import VelocityRewardContract, replacement_release_reward
    from world_model.imagined_rewards import RewardContract,release_reward
    args=(jnp.array([[0.,-.5,1.1]]),jnp.array([[.1,.2,1.8]]),jnp.array([[0.,-.5,1.4]]),jnp.array([True]))
    old=release_reward(*args,contract=RewardContract())
    off=replacement_release_reward(*args,contract=RewardContract(),candidate=VelocityRewardContract())
    assert old.keys()==off.keys()
    for k in old: np.testing.assert_array_equal(old[k],off[k])
    new=replacement_release_reward(*args,contract=RewardContract(),candidate=VelocityRewardContract(enabled=True))
    assert 'apex_xy' not in new and 'apex_z' not in new
    for k in old.keys()-{'apex_xy','apex_z','total'}: np.testing.assert_array_equal(new[k],old[k])
    np.testing.assert_allclose(new['total'],sum(v for k,v in new.items() if k!='total'))


def test_invalid_inverse_falls_back_without_hiding_old_penalty():
    from world_model.velocity_reward import VelocityRewardContract,replacement_release_reward,velocity_costs
    from world_model.imagined_rewards import RewardContract,release_reward
    p=jnp.array([[0.,-.5,1.4]]);g=jnp.array([[0.,-.5,1.3]]);v=jnp.array([[.1,0.,1.]])
    old=release_reward(p,v,g,jnp.array([True]),contract=RewardContract())
    new=replacement_release_reward(p,v,g,jnp.array([True]),contract=RewardContract(),candidate=VelocityRewardContract(enabled=True))
    np.testing.assert_array_equal(new['velocity_xy'],old['apex_xy'])
    np.testing.assert_array_equal(new['velocity_z'],old['apex_z'])
    f=lambda x:sum(velocity_costs(x,v,g,jnp.array([False]),contract=RewardContract())[0].values()).sum()
    np.testing.assert_array_equal(jax.grad(f)(p),jnp.zeros_like(p))


def test_low_upward_speed_keeps_direct_vertical_recovery_gradient():
    from world_model.velocity_reward import velocity_costs
    from world_model.imagined_rewards import RewardContract,release_reward
    p=jnp.array([[0.,-.5,1.1]]);g=jnp.array([[0.,-.5,1.4]])
    def old(z): return release_reward(p,jnp.array([[0.,0.,z]]),g,jnp.array([True]),contract=RewardContract())['apex_z'].sum()
    def new(z): return velocity_costs(p,jnp.array([[0.,0.,z]]),g,jnp.array([True]),contract=RewardContract())[0]['velocity_z'].sum()
    a,b=float(jax.grad(old)(1e-4)),float(jax.grad(new)(1e-4))
    assert np.isfinite(a+b) and b>0 and b>100*a


def test_private_reward_wiring_retains_free_controller_and_bounds():
    from world_model.free_imagination import make_runtime
    from world_model.velocity_reward import VelocityRewardContract,candidate_course_step
    runtime=make_runtime(None,None,{},None)
    assert candidate_course_step(runtime,VelocityRewardContract()) is runtime.course_step
    candidate=candidate_course_step(runtime,VelocityRewardContract(enabled=True))
    assert candidate.__globals__['control_step'] is runtime.controller
    assert candidate.__globals__['physical_bounds'] is runtime.bounds
    assert candidate.__globals__['task_substep'] is not runtime.course_step.__globals__['task_substep']


def test_coefficient_grid_recovers_identical_unit_costs_and_serializes():
    import json
    from world_model.calibrate_velocity_reward import coefficient_grid
    vectors=np.array([[1.,1.],[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
    criteria=dict(cost_ratio_min=.5,cost_ratio_max=2.,total_gradient_ratio_min=.5,
        total_gradient_ratio_max=2.,baseline_cosine_min=.5,height_direction_retention_min=.8)
    rows=coefficient_grid(vectors@vectors.T,[-4.,-2.,-2.,-2.,-2.],[.5,1.,2.],criteria)
    unit=next(r for r in rows if r['xy_weight']==r['z_weight']==1.)
    assert unit['eligible'] and abs(unit['baseline_gradient_cosine']-1.)<1e-12
    assert unit['mean_candidate_return']==-4.
    json.dumps(rows,allow_nan=False)
