"""Normal-only auxiliary integration preserves old39-D contact behavior."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from test_wmj_v1_soft_contact import contract, inputs
from test_wmj_v1_world_physics import setup_physics
from world_model.position_contact import position_contact_step
from world_model.soft_contact import soft_contact_increment, softness
from world_model.support_losses import racket_kinematics
from world_model.normal_contact import normal_prior_increment, normal_contact_step


def test_zero_acceleration_retains_prior_and_normal_sign_is_correct():
    prior = contract()
    geometry, values = inputs(.019, (0., 0., -1.))
    baseline = soft_contact_increment(prior, geometry, **values)
    zero = normal_prior_increment(prior, geometry, jnp.zeros(6), **values)
    for actual, expected in zip(zero, baseline):
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)
    predicted = normal_prior_increment(prior, geometry, jnp.array([0., 0., 100., 0., 0., 0.]), **values)
    imp = softness(prior, geometry['sphere_gap_m'])[2]
    expected = 100./(1/prior.mass_kg+prior.reference_inverse_mass_kg_inv*(1-imp)/imp)
    np.testing.assert_allclose(predicted.force_contact_n[0]-baseline.force_contact_n[0], expected, rtol=3e-6)


def test_angular_term_uses_contact_lever_and_friction_and_free_flight_hold():
    prior = contract()
    geometry, values = inputs(.017, (10., -5., -2.))
    geometry = dict(geometry, contact_position_world_m=jnp.array([.02, .01, -.0015]))
    omega_term = jnp.array([20., 30., 40.])
    linear = jnp.cross(omega_term, geometry['contact_position_world_m']-values['geom_position_m'])
    a = normal_prior_increment(prior, geometry, jnp.concatenate((jnp.zeros(3), omega_term)), **values)
    b = normal_prior_increment(prior, geometry, jnp.concatenate((linear, jnp.zeros(3))), **values)
    np.testing.assert_allclose(a.force_world_n, b.force_world_n, atol=1e-7)
    assert np.linalg.norm(np.asarray(a.force_contact_n[1:])/prior.friction) <= float(a.force_contact_n[0])*(1+1e-6)
    free = normal_prior_increment(prior, dict(geometry, sphere_gap_m=jnp.array(.01)), jnp.ones(6)*1e6, **values)
    np.testing.assert_array_equal(free.delta_velocity_world_m_s, jnp.zeros(3))


def test_disabled_contact_is_exact_legacy_and_preserves_gradients():
    _, parameters, state, options = setup_physics()
    pose = racket_kinematics(options['chain'], state.q_rad, state.dq_rad_s)
    racket = dict(geom_position_m=pose['geom_position'], geom_rotation_world=pose['rotation'],
        racket_linear_velocity_m_s=pose['velocity'], racket_angular_velocity_rad_s=pose['omega'],
        joint_q_rad=state.q_rad, delayed_q_command_rad=jnp.zeros((1, 7)))
    kw = dict(ball_radius_m=.02, position_scale=options['position_scale'], contract=contract())
    old = position_contact_step(parameters, options['surface'], state.ball, racket, **kw)
    new = normal_contact_step(parameters, options['surface'], state.ball, racket, None, **kw)
    for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new)):
        np.testing.assert_array_equal(a, b)
    old_grad = jax.grad(lambda p: position_contact_step(p, options['surface'], state.ball, racket, **kw)[0].velocity_m_s.sum())(parameters)
    new_grad = jax.grad(lambda p: normal_contact_step(p, options['surface'], state.ball, racket, None, **kw)[0].velocity_m_s.sum())(parameters)
    for a, b in zip(jax.tree.leaves(old_grad), jax.tree.leaves(new_grad)):
        np.testing.assert_array_equal(a, b)


def test_normal_term_derivative_matches_finite_difference():
    prior = contract()
    geometry, values = inputs(.018, (.1, .05, -1.))
    function = lambda a: normal_prior_increment(prior, geometry, a, **values).delta_velocity_world_m_s
    point = jnp.array([0., 0., 50., 0., 0., 0.])
    derivative = jax.jit(jax.jacfwd(function))(point)
    direction = jnp.eye(6)[2]*.1
    numeric = (function(point+direction)-function(point-direction))/.2
    np.testing.assert_allclose(derivative[:, 2], numeric, atol=2e-6, rtol=.002)


def test_recursive_auxiliary_ignores_future_native_joints_and_keeps_old_motion():
    from world_model.normal_coupling import own_robot_motion, acceleration_terms
    from world_model.physics_fitting import predicted_racket_motion
    from world_model.neural import init_mlp
    from world_model.robot_model import RobotScale
    robot, _, state, options = setup_physics()
    commands = {k: jnp.zeros((10, 1, 7)) for k in ('q', 'dq', 'ddq')}
    raw = dict(q_before=jnp.broadcast_to(state.q_rad, (10, 1, 13)),
        dq_before=jnp.broadcast_to(state.dq_rad_s, (10, 1, 13)),
        **{f'delayed_{k}_command': v for k, v in commands.items()})
    motion, trace, command = own_robot_motion(robot, raw, scale=options['robot_scale'], chain=options['chain'])
    expected = predicted_racket_motion(robot, raw, scale=options['robot_scale'], chain=options['chain'])
    for a, b in zip(jax.tree.leaves(motion), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a, b)
    corrupted = dict(raw, q_before=raw['q_before'].at[1:].set(999.),
                     dq_before=raw['dq_before'].at[1:].set(-999.))
    other_motion, other, _ = own_robot_motion(robot, corrupted, scale=options['robot_scale'], chain=options['chain'])
    auxiliary = dict(parameters=init_mlp(jax.random.PRNGKey(42), (47, 128, 128, 6)),
        scale=RobotScale(jnp.zeros(47), jnp.ones(47), jnp.zeros(6), jnp.ones(6)))
    a = acceleration_terms(auxiliary, trace['q_before_rad'], trace['dq_before_rad_s'], command)
    b = acceleration_terms(auxiliary, other['q_before_rad'], other['dq_before_rad_s'], command)
    np.testing.assert_array_equal(a, b)
    for a, b in zip(jax.tree.leaves(motion), jax.tree.leaves(other_motion)):
        np.testing.assert_array_equal(a, b)


def test_zero_residual_matches_corrected_prior_without_double_application():
    from world_model.contact_geometry import surface_features
    from world_model.neural import init_mlp
    _, _, state, options = setup_physics()
    parameters = init_mlp(jax.random.PRNGKey(45), (39, 16, 8), zero_output=True)
    pose = racket_kinematics(options['chain'], state.q_rad, state.dq_rad_s)
    racket = dict(geom_position_m=pose['geom_position'], geom_rotation_world=pose['rotation'],
        racket_linear_velocity_m_s=pose['velocity'], racket_angular_velocity_rad_s=pose['omega'],
        joint_q_rad=state.q_rad, delayed_q_command_rad=jnp.zeros((1, 7)))
    terms = jnp.array([[0., 0., -100., 0., 0., 0.]])
    after, _ = normal_contact_step(parameters, options['surface'], state.ball, racket, terms,
        ball_radius_m=.02, position_scale=options['position_scale'], contract=contract())
    geometry = surface_features(options['surface'], state.ball.position_m,
                                 pose['geom_position'], pose['rotation'], .02)
    prior = normal_prior_increment(contract(), geometry, terms,
        ball_position_m=state.ball.position_m, ball_velocity_m_s=state.ball.velocity_m_s,
        ball_omega_world_rad_s=state.ball.omega_world_rad_s,
        **{k:v for k,v in racket.items() if k not in ('geom_rotation_world','joint_q_rad','delayed_q_command_rad')})
    velocity = state.ball.velocity_m_s+jnp.array([0., 0., -.00981])+prior.delta_velocity_world_m_s
    np.testing.assert_allclose(after.velocity_m_s, velocity, atol=2e-7)
    np.testing.assert_allclose(after.position_m, state.ball.position_m+velocity*.001, atol=2e-7)
