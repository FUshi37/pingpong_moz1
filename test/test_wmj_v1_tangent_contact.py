"""Full-prior correction retains the normal baseline and contact39 contract."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from test_wmj_v1_soft_contact import contract, inputs
from test_wmj_v1_world_physics import setup_physics
from world_model.normal_contact import normal_prior_increment
from world_model.soft_contact import soft_contact_increment
from world_model.tangent_contact import tangent_prior_increment, tangent_contact_step


def test_full_prior_matches_independent_relative_acceleration_equation():
    prior = contract()
    geometry, values = inputs(.017, (.01, -.02, -1.))
    for terms in (jnp.zeros(6), jnp.array([20., -10., 30., 1., 2., 3.])):
        point_term = terms[:3]+jnp.cross(terms[3:],
            geometry['contact_position_world_m']-values['geom_position_m'])
        expected = soft_contact_increment(prior, geometry, **values,
            gravity_m_s2=jnp.array([0., 0., -9.81])-point_term)
        actual = tangent_prior_increment(prior, geometry, terms, **values)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, atol=2e-7, rtol=2e-6)
    only_normal = jnp.array([0., 0., 100., 0., 0., 0.])
    a = tangent_prior_increment(prior, geometry, only_normal, **values)
    b = normal_prior_increment(prior, geometry, only_normal, **values)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, atol=2e-7, rtol=2e-6)


def test_tangent_sign_normal_load_friction_cap_and_free_flight():
    prior = contract()
    geometry, values = inputs(.017, (0., 0., -2.))
    base = normal_prior_increment(prior, geometry, jnp.zeros(6), **values)
    terms = jnp.array([20., -10., 0., 0., 0., 0.])
    moved = tangent_prior_increment(prior, geometry, terms, **values)
    np.testing.assert_allclose(moved.force_contact_n[0], base.force_contact_n[0], atol=1e-7)
    assert moved.force_world_n[0] > 0. and moved.force_world_n[1] < 0.
    saturated = tangent_prior_increment(prior, geometry, terms*1e5, **values)
    assert np.linalg.norm(np.asarray(saturated.force_contact_n[1:])/prior.friction) <= float(saturated.force_contact_n[0])*(1+1e-6)
    free = tangent_prior_increment(prior, dict(geometry, sphere_gap_m=jnp.array(.01)), terms*1e5, **values)
    np.testing.assert_array_equal(free.delta_velocity_world_m_s, jnp.zeros(3))


def test_tangential_prediction_gradient_matches_finite_difference():
    geometry, values = inputs(.017, (.01, -.01, -2.))
    function = lambda terms: tangent_prior_increment(contract(), geometry, terms, **values).delta_velocity_world_m_s
    point = jnp.array([2., -3., 40., 1., -2., 3.])
    analytic = jax.jit(jax.jacfwd(function))(point)
    numeric = np.stack([np.asarray((function(point+jnp.eye(6)[i]*.1)
        -function(point-jnp.eye(6)[i]*.1))/.2) for i in range(6)], axis=-1)
    np.testing.assert_allclose(analytic, numeric, atol=2e-6, rtol=.003)


def test_contact39_zero_residual_applies_full_prior_once_and_keeps_gradients():
    from world_model.contact_geometry import surface_features
    from world_model.neural import init_mlp
    from world_model.support_losses import racket_kinematics
    _, _, state, options = setup_physics()
    parameters = init_mlp(jax.random.PRNGKey(45), (39, 16, 8), zero_output=True)
    pose = racket_kinematics(options['chain'], state.q_rad, state.dq_rad_s)
    racket = dict(geom_position_m=pose['geom_position'], geom_rotation_world=pose['rotation'],
        racket_linear_velocity_m_s=pose['velocity'], racket_angular_velocity_rad_s=pose['omega'],
        joint_q_rad=state.q_rad, delayed_q_command_rad=jnp.zeros((1, 7)))
    terms = jnp.array([[40., -20., -100., 2., -3., 1.]])
    kwargs = dict(ball_radius_m=.02, position_scale=options['position_scale'], contract=contract())
    after, _ = tangent_contact_step(parameters, options['surface'], state.ball, racket, terms, **kwargs)
    geometry = surface_features(options['surface'], state.ball.position_m, pose['geom_position'], pose['rotation'], .02)
    prior = tangent_prior_increment(contract(), geometry, terms,
        ball_position_m=state.ball.position_m, ball_velocity_m_s=state.ball.velocity_m_s,
        ball_omega_world_rad_s=state.ball.omega_world_rad_s,
        **{k:v for k,v in racket.items() if k not in ('geom_rotation_world','joint_q_rad','delayed_q_command_rad')})
    velocity = state.ball.velocity_m_s+jnp.array([0., 0., -.00981])+prior.delta_velocity_world_m_s
    np.testing.assert_allclose(after.velocity_m_s, velocity, atol=3e-7)
    np.testing.assert_allclose(after.position_m, state.ball.position_m+velocity*.001, atol=3e-7)
    gradient = jax.grad(lambda p: tangent_contact_step(p, options['surface'], state.ball, racket, terms, **kwargs)[0].velocity_m_s.sum())(parameters)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(gradient))
    assert any(np.any(np.asarray(x) != 0.) for x in jax.tree.leaves(gradient))
