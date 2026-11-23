"""Independent normal-contact dynamics and physical invariants for the prior."""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.soft_contact import SoftContactContract, soft_contact_increment, softness


def contract():
    return SoftContactContract(mass_kg=.004, inertia_kg_m2=6.4e-7, dt_s=.001,
        solref=(.006836, .228086), solimp=(.04, .7, .014298, .04, 8.021856),
        friction=(.13268182450948787, .11820649851379933),
        reference_inverse_mass_kg_inv=250., impratio=1., reference_safe=True)


def inputs(height, velocity, spin=(0., 0., 0.)):
    ball = jnp.array([0., 0., height], jnp.float32)
    point = jnp.array([0., 0., (height-.02)*.5], jnp.float32)
    geometry = dict(sphere_gap_m=jnp.asarray(height-.02, jnp.float32),
        contact_position_world_m=point, contact_normal_world=jnp.array([0., 0., 1.]))
    return geometry, dict(ball_position_m=ball, ball_velocity_m_s=jnp.asarray(velocity, jnp.float32),
        ball_omega_world_rad_s=jnp.asarray(spin, jnp.float32), geom_position_m=jnp.zeros(3),
        racket_linear_velocity_m_s=jnp.zeros(3), racket_angular_velocity_rad_s=jnp.zeros(3))


def test_normal_prior_matches_independent_mujoco_single_contact():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep=".001" integrator="implicitfast" cone="elliptic" iterations="100" tolerance="1e-12"/>
      <worldbody><geom name="floor" type="plane" size="1 1 .1"/>
        <body pos="0 0 .02"><freejoint/><geom name="ball" type="sphere" size=".02" mass=".004"/></body>
      </worldbody><contact><pair geom1="floor" geom2="ball" condim="3"
        solref=".006836 .228086" solimp=".04 .7 .014298 .04 8.021856"
        friction=".13268182450948787 .11820649851379933 .0001545 .00002014 .00002014"/>
      </contact></mujoco>''')
    data = mujoco.MjData(model)
    prior = replace(contract(), reference_inverse_mass_kg_inv=float(model.body_invweight0[1, 0]))
    for height, vz in ((.0199, -2.), (.019, -1.), (.017, .3), (.0199, 2.)):
        mujoco.mj_resetData(model, data)
        data.qpos[2], data.qvel[2] = height, vz
        mujoco.mj_step(model, data)
        geometry, values = inputs(height, (0., 0., vz))
        predicted = soft_contact_increment(prior, geometry, **values)
        velocity = np.array([0., 0., vz-.00981])+np.asarray(predicted.delta_velocity_world_m_s)
        np.testing.assert_allclose(velocity, data.qvel[:3], atol=2e-6, rtol=2e-6)


def test_free_flight_is_exact_zero_and_friction_stays_in_native_ellipse():
    prior = contract()
    geometry, values = inputs(.03, (10., -5., -2.))
    free = soft_contact_increment(prior, geometry, **values)
    np.testing.assert_array_equal(free.delta_velocity_world_m_s, np.zeros(3))
    np.testing.assert_array_equal(free.delta_omega_world_rad_s, np.zeros(3))
    geometry, values = inputs(.017, (10., -5., -2.))
    contact = soft_contact_increment(prior, geometry, **values)
    force = np.asarray(contact.force_contact_n)
    assert force[0] > 0.
    assert np.linalg.norm(force[1:]/np.asarray(prior.friction)) <= force[0]*(1+1e-6)
    assert float(jnp.dot(contact.force_world_n[:2], values['ball_velocity_m_s'][:2])) < 0.
    expected_torque = jnp.cross(geometry['contact_position_world_m']-values['ball_position_m'], contact.force_world_n)
    np.testing.assert_allclose(contact.delta_omega_world_rad_s, expected_torque*.001/prior.inertia_kg_m2, atol=1e-6)


def test_prior_velocity_gradient_and_shared_translation_velocity():
    prior = contract()
    geometry, values = inputs(.017, (.1, .05, -1.))
    def predict(v):
        return soft_contact_increment(prior, geometry, **(values | {'ball_velocity_m_s': v})).delta_velocity_world_m_s
    p = values['ball_velocity_m_s']
    derivative = jax.jit(jax.jacfwd(predict))(p)
    epsilon = 1e-3
    finite = np.stack([np.asarray((predict(p+jnp.eye(3)[i]*epsilon)-predict(p-jnp.eye(3)[i]*epsilon))/(2*epsilon))
                       for i in range(3)], axis=-1)
    np.testing.assert_allclose(derivative, finite, atol=5e-5, rtol=2e-3)
    shift = jnp.array([.7, -.3, .9])
    shifted = soft_contact_increment(prior, geometry, **(values | {'ball_velocity_m_s': p+shift,
                                                               'racket_linear_velocity_m_s': shift}))
    np.testing.assert_allclose(shifted.delta_velocity_world_m_s, predict(p), atol=2e-7)


def test_native_softness_values_and_saturated_derivatives():
    from mujoco.mjx._src.constraint import _kbi
    prior = contract()
    native = SimpleNamespace(opt=SimpleNamespace(timestep=.001, disableflags=0))
    points = jnp.array([-.1, -.014298, -.003, -.0002, 0., .1])
    expected = jax.vmap(lambda p: _kbi(native, jnp.array(prior.solref), jnp.array(prior.solimp), p))(points)
    k, b, imp = softness(prior, points)
    np.testing.assert_allclose(k, expected[0], rtol=2e-7)
    np.testing.assert_allclose(b, expected[1], rtol=2e-7)
    np.testing.assert_allclose(imp, expected[2], atol=1e-7)
    derivative = jax.vmap(jax.grad(lambda p: softness(prior, p)[2]))(points)
    assert np.isfinite(derivative).all()
    np.testing.assert_array_equal(np.asarray(derivative)[[0, -1]], [0., 0.])


def test_hybrid_zero_residual_and_legacy_dynamics_are_preserved():
    from world_model.contact_dynamics import ContactState, contact_step
    from world_model.contact_surface import surface_from_vertices
    from world_model.hybrid_contact import hybrid_contact_step, hybrid_contact_rollout
    from world_model.neural import init_mlp
    surface = surface_from_vertices(np.array([[x,y,z] for x in (-.075,.075)
        for y in (-.075,.075) for z in (-.001,.001)]), normal_axis=2)
    state = ContactState(jnp.array([[0., 0., .02]]), jnp.array([[.1, 0., -1.]]),
        jnp.zeros((1,3)), jnp.array([False]), jnp.zeros(1))
    racket = dict(geom_position_m=jnp.zeros((1,3)), geom_rotation_world=jnp.eye(3)[None],
        racket_linear_velocity_m_s=jnp.zeros((1,3)), racket_angular_velocity_rad_s=jnp.zeros((1,3)))
    params = init_mlp(jax.random.PRNGKey(2), (19,16,8), zero_output=True)
    prior = contract()
    for learned in (False,True):
        result = hybrid_contact_step(params, surface, state, racket, ball_radius_m=.02,
                                     contract=prior, learned=learned)
        if not learned:
            base = result
        else:
            for left,right in zip(jax.tree.leaves(result),jax.tree.leaves(base)):
                np.testing.assert_array_equal(left,right)
    legacy = contact_step(params,surface,state,racket,ball_radius_m=.02)
    actual = hybrid_contact_step(params,surface,state,racket,ball_radius_m=.02)
    for left,right in zip(jax.tree.leaves(actual),jax.tree.leaves(legacy)):
        np.testing.assert_array_equal(left,right)
    history = jax.tree.map(lambda x:jnp.broadcast_to(x,(5,)+x.shape),racket)
    def final_height(p):
        after,_ = hybrid_contact_rollout(p,surface,state,history,ball_radius_m=.02,contract=prior)
        return after.position_m[0,2]
    gradient = jax.jit(jax.grad(final_height))(params)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(gradient))
    assert any(np.any(np.asarray(x)!=0) for x in jax.tree.leaves(gradient))
