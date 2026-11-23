"""Contact module recursion keeps its own physical state after the seed."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from world_model.contact_dynamics import ContactState, contact_rollout
from world_model.contact_losses import contact_module_loss
from world_model.contact_surface import surface_from_vertices
from world_model.neural import init_mlp


def example():
    surface = surface_from_vertices(np.array([[x, y, z] for x in (-.075, .075)
        for y in (-.075, .075) for z in (-.001, .001)]), normal_axis=2)
    params = init_mlp(jax.random.PRNGKey(9), (19, 16, 8), zero_output=True)
    state = ContactState(jnp.array([[0., 0., .5]]), jnp.array([[0., 0., -1.]]),
        jnp.zeros((1, 3)), jnp.zeros(1, bool), jnp.zeros(1))
    racket = dict(geom_position_m=jnp.zeros((10, 1, 3)),
        geom_rotation_world=jnp.broadcast_to(jnp.eye(3), (10, 1, 3, 3)),
        racket_linear_velocity_m_s=jnp.zeros((10, 1, 3)),
        racket_angular_velocity_rad_s=jnp.zeros((10, 1, 3)))
    return surface, params, state, racket


def test_flight_recursion_is_semi_implicit_and_never_invents_contact():
    surface, params, state, racket = example()
    end, trajectory = contact_rollout(params, surface, state, racket, ball_radius_m=.02)
    np.testing.assert_allclose(end.velocity_m_s, [[0., 0., -1.0981]], atol=2e-7)
    np.testing.assert_allclose(end.position_m, [[0., 0., .5-.01-9.81e-6*55]], atol=2e-7)
    assert not np.asarray(trajectory['overlap']).any()
    np.testing.assert_array_equal(end.omega_world_rad_s, state.omega_world_rad_s)


def test_autoregressive_loss_does_not_read_future_ball_inputs():
    surface, params, state, racket = example()
    _, trace = contact_rollout(params, surface, state, racket, ball_radius_m=.02)
    batch = dict(racket=racket, before=trace['before'], after=trace['after'],
        valid=jnp.ones((10, 1), bool), entry=jnp.zeros((10, 1), bool),
        release=jnp.zeros((10, 1), bool), controlled_release=jnp.zeros((10, 1), bool))
    first, metrics = contact_module_loss(params, surface, batch, ball_radius_m=.02, autoregressive=True)
    bad = batch['before']._replace(
        position_m=batch['before'].position_m.at[1:].add(100.),
        velocity_m_s=batch['before'].velocity_m_s.at[1:].add(100.))
    second, _ = contact_module_loss(params, surface, {**batch, 'before': bad},
                                  ball_radius_m=.02, autoregressive=True)
    np.testing.assert_allclose(first, second, atol=1e-7)
    assert float(metrics['position_loss']) < 1e-12
    assert float(metrics['velocity_loss']) < 1e-12
    gradients = jax.grad(lambda p: contact_module_loss(p, surface, batch,
        ball_radius_m=.02, autoregressive=True)[0])(params)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(gradients))


def test_contact_age_and_edges_carry_across_windows():
    surface, params, state, racket = example()
    state = state._replace(position_m=jnp.array([[0., 0., .020]]),
                           velocity_m_s=jnp.zeros((1, 3)))
    end, trace = contact_rollout(params, surface, state, racket, ball_radius_m=.02)
    assert np.asarray(trace['entry']).sum() == 1
    np.testing.assert_allclose(end.duration_s, .010, atol=1e-8)
    _, next_trace = contact_rollout(params, surface, end, racket, ball_radius_m=.02)
    assert not np.asarray(next_trace['entry']).any()
