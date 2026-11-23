"""Contact proxy respects finite geometry and ball/racket velocity frames."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.contact_surface import surface_from_vertices
from world_model.contact_model import contact_features, contact_prediction, world_spin
from world_model.neural import init_mlp


def surface():
    return surface_from_vertices(np.array([[x, y, z] for x in (-.075, .075)
        for y in (-.075, .075) for z in (-.001, .001)]), normal_axis=2)


def inputs(ball):
    return dict(ball_position_m=jnp.asarray([ball]), ball_velocity_m_s=jnp.array([[.2, 0., -1.]]),
        ball_omega_world_rad_s=jnp.zeros((1, 3)), geom_position_m=jnp.zeros((1, 3)),
        geom_rotation_world=jnp.eye(3)[None], racket_linear_velocity_m_s=jnp.array([[.2, 0., -1.]]),
        racket_angular_velocity_rad_s=jnp.zeros((1, 3)), ball_radius_m=.02,
        previous_contact=jnp.array([False]), duration_s=jnp.zeros(1))


def test_contact_features_use_relative_velocity_and_physical_local_frames():
    features, geometry = contact_features(surface(), **inputs([0., 0., .020]))
    assert features.shape == (1, 19)
    np.testing.assert_allclose(features[0, 3:6], 0., atol=1e-8)
    np.testing.assert_allclose(features[0, 12:15], [0., 0., -9.81], atol=1e-6)
    assert geometry['sphere_gap_m'][0] < 0
    # Ball qvel angular components are local; a rotated ball needs conversion.
    quaternion = jnp.array([[np.sqrt(.5), 0., 0., np.sqrt(.5)]], jnp.float32)
    np.testing.assert_allclose(world_spin(quaternion, jnp.array([[1., 0., 0.]])),
                               [[0., 1., 0.]], atol=2e-7)


def test_far_field_and_lateral_miss_cannot_receive_learned_impulse():
    parameters = init_mlp(jax.random.PRNGKey(2), (19, 256, 256, 128, 8), zero_output=True)
    parameters = parameters[:-1]+({**parameters[-1], 'b': jnp.ones(8)},)
    for ball in ([0., 0., .5], [.2, 0., .020]):
        result = contact_prediction(parameters, surface(), **inputs(ball))
        np.testing.assert_array_equal(result.delta_velocity_world_m_s, np.zeros((1, 3)))
        np.testing.assert_array_equal(result.delta_omega_world_rad_s, np.zeros((1, 3)))
    overlapping = contact_prediction(parameters, surface(), **inputs([0., 0., .020]))
    assert float(jnp.linalg.norm(overlapping.delta_velocity_world_m_s)) > 0
    gradient = jax.jacfwd(lambda p: contact_prediction(parameters, surface(),
        **(inputs([0., 0., .020]) | {'ball_position_m': p})).delta_velocity_world_m_s)(jnp.array([[0., 0., .020]]))
    assert np.isfinite(gradient).all()


def test_rim_contact_uses_its_actual_normal_for_spin_velocity():
    state = inputs([.094, 0., 0.])
    state['ball_omega_world_rad_s'] = jnp.array([[0., 0., 1.]])
    features, geometry = contact_features(surface(), **state)
    np.testing.assert_allclose(geometry['contact_normal_world'], [[1., 0., 0.]], atol=1e-7)
    np.testing.assert_allclose(features[0, 3:6], [0., -.0195, 0.], atol=1e-7)
