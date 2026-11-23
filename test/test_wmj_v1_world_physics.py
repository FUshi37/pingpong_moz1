"""Joint and ball recursion share one causal full-chain physical state."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))

from world_model.contact_dynamics import ContactState
from world_model.contact_surface import surface_from_vertices
from world_model.kinematics import ChainBody, Hinge, KinematicChain
from world_model.neural import init_mlp
from world_model.position_contact import PositionScale, position_contact_rollout
from world_model.robot_model import RobotScale
from world_model.support_losses import racket_kinematics


def setup_physics():
    from world_model.world_physics import PhysicsState
    zero, eye, point = jnp.zeros(3), jnp.eye(3), jnp.array([.2, 0., 0.])
    joints = tuple(Hinge(i, 3, jnp.array([0., 0., 1.]), zero, 0., 0.) for i in range(13))
    chain = KinematicChain((ChainBody(zero, eye, joints),), tuple(range(13)),
        point, eye, point, eye)
    surface = surface_from_vertices(np.array([[x, y, z] for x in (-.075, .075)
        for y in (-.075, .075) for z in (-.001, .001)]), normal_axis=2)
    robot = init_mlp(jax.random.PRNGKey(1), (47, 16, 26), zero_output=True)
    contact = init_mlp(jax.random.PRNGKey(2), (39, 16, 8))
    state = PhysicsState(jnp.zeros((1, 13)), jnp.zeros((1, 13)),
        ContactState(jnp.array([[.2, 0., .019]]), jnp.array([[.1, 0., -.5]]),
            jnp.zeros((1, 3)), jnp.zeros(1, bool), jnp.zeros(1)))
    options = dict(chain=chain, surface=surface,
        robot_scale=RobotScale(jnp.zeros(47), jnp.ones(47), jnp.zeros(26), jnp.ones(26)),
        position_scale=PositionScale(jnp.zeros(20), jnp.ones(20)),
        ball_radius_m=.02, contact_contract=None)
    return robot, contact, state, options


def test_full_joint_feedback_and_ballistic_substeps():
    from world_model.world_physics import physics_control_step
    robot, contact, state, options = setup_physics()
    dq = state.dq_rad_s.at[:, 7:].set(.03)
    state = state._replace(dq_rad_s=dq,
        ball=state.ball._replace(position_m=jnp.array([[.2, 0., 1.]])))
    command = {name:jnp.zeros((5, 1, 7)) for name in ('q', 'dq', 'ddq')}
    after, trace = jax.jit(lambda s,c:physics_control_step(robot,contact,s,c,**options))(state,command)
    np.testing.assert_allclose(after.q_rad, state.q_rad+5*.001*dq, atol=1e-9)
    np.testing.assert_array_equal(after.dq_rad_s,dq)
    expected_v = state.ball.velocity_m_s+jnp.array([0.,0.,-9.81])*.005
    expected_p = state.ball.position_m+state.ball.velocity_m_s*.005+jnp.array([0.,0.,-9.81])*(15e-6)
    np.testing.assert_allclose(after.ball.velocity_m_s,expected_v,atol=2e-7)
    np.testing.assert_allclose(after.ball.position_m,expected_p,atol=2e-7)
    np.testing.assert_array_equal(trace['q_before_rad'][1:],trace['q_after_rad'][:-1])
    expected_pose = racket_kinematics(options['chain'],trace['q_before_rad'],trace['dq_before_rad_s'])
    np.testing.assert_allclose(trace['racket_before']['rotation'],expected_pose['rotation'],atol=1e-7)
    assert float(jnp.linalg.norm(expected_pose['rotation'][-1]-expected_pose['rotation'][0])) > 1e-4


def test_static_robot_matches_contact_component_and_split_rollout():
    from world_model.world_physics import physics_rollout, physics_control_step
    robot,contact,state,options = setup_physics()
    commands = {name:jnp.zeros((10,1,7)) for name in ('q','dq','ddq')}
    after,trace = physics_rollout(robot,contact,state,commands,**options)
    middle,_ = physics_control_step(robot,contact,state,jax.tree.map(lambda x:x[:5],commands),**options)
    split,_ = physics_control_step(robot,contact,middle,jax.tree.map(lambda x:x[5:],commands),**options)
    for a,b in zip(jax.tree.leaves(after),jax.tree.leaves(split)):
        np.testing.assert_array_equal(a,b)
    pose = racket_kinematics(options['chain'],state.q_rad,state.dq_rad_s)
    motion = dict(geom_position_m=pose['geom_position'],geom_rotation_world=pose['rotation'],
        racket_linear_velocity_m_s=pose['velocity'],racket_angular_velocity_rad_s=pose['omega'],
        joint_q_rad=state.q_rad,delayed_q_command_rad=commands['q'][0])
    motion = jax.tree.map(lambda x:jnp.broadcast_to(x,(10,)+x.shape),motion)
    expected,_ = position_contact_rollout(contact,options['surface'],state.ball,motion,
        ball_radius_m=.02,position_scale=options['position_scale'])
    for a,b in zip(jax.tree.leaves(after.ball),jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a,b)


def test_delayed_command_gradient_and_contract_rejection():
    from world_model.world_physics import physics_substep, physics_control_step
    robot,contact,state,options = setup_physics()
    command = {name:jnp.zeros((1,7)) for name in ('q','dq','ddq')}
    def value(q):
        after,_ = physics_substep(robot,contact,state,dict(command,q=q),**options)
        return after.ball.velocity_m_s[0,2]
    gradient = jax.jit(jax.grad(value))(command['q'])
    delta = jnp.zeros((1,7)).at[0,0].set(1e-3)
    numerical = (value(command['q']+delta)-value(command['q']-delta))/(2e-3)
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient)>1e-6
    np.testing.assert_allclose(gradient[0,0],numerical,rtol=.03,atol=2e-5)
    with pytest.raises(ValueError):
        physics_substep(robot,contact,state,dict(command,future_q=state.q_rad),**options)
    with pytest.raises(ValueError):
        physics_substep(robot,contact,state._replace(q_rad=state.q_rad[:,:7]),command,**options)
    with pytest.raises(ValueError):
        physics_control_step(robot,contact,state,
            {name:jnp.zeros((4,1,7)) for name in command},**options)
