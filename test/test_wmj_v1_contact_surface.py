"""Finite racket geometry uses the compiled polygon, including corners and edges."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model import contact_surface


def surface():
    # Thickness is compiled X, with a deliberately rectangular finite face.
    vertices = np.array([[x, y, z] for x in (-.001, .001)
                         for y in (-.06, .06) for z in (-.04, .04)])
    return contact_surface.surface_from_vertices(vertices, normal_axis=0)


def test_prism_distance_distinguishes_face_edge_corner_and_underside():
    model = surface()
    rotation = jnp.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    points = jnp.array([[0., 0., .031], [.09, 0., .001], [.09, .08, .001], [0., 0., -.031]])
    feature = contact_surface.surface_features(model, points, jnp.zeros_like(points),
                                               jnp.broadcast_to(rotation, (4, 3, 3)), .02)
    np.testing.assert_allclose(feature['sphere_gap_m'], [.01, .01, .03, .01], atol=1e-7)
    assert feature['signed_face_height_m'][3] < 0
    np.testing.assert_allclose(feature['normal_world'], [[0., 0., 1.]]*4, atol=1e-7)


def test_contact_proposal_uses_finite_polygon_boundary_and_sphere_radius():
    model = surface()
    radius = contact_surface.radial_boundary_distance(model, jnp.array([0., jnp.pi/2, jnp.pi/4]))
    np.testing.assert_allclose(radius, [.06, .04, .04*np.sqrt(2)], atol=1e-7)
    rotation = jnp.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    point = contact_surface.proposed_ball_center(model, jnp.zeros((3,)), rotation,
                                                 jnp.array([0., 0.]), .02)
    np.testing.assert_allclose(point, [0., 0., .021], atol=1e-7)


def test_surface_distance_gradient_and_rotation_are_consistent():
    model = surface()
    rotation = jnp.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])

    def distance(point):
        return contact_surface.surface_features(model, point, jnp.zeros(3), rotation, .02)['sphere_gap_m']

    derivative = jax.jit(jax.grad(distance))(jnp.array([.09, .08, .001]))
    np.testing.assert_allclose(derivative, [.6, .8, 0.], atol=1e-6)


def test_finite_contact_normal_and_point_follow_rim_and_underside():
    model = surface()
    rotation = jnp.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    points = jnp.array([[.09, .08, .001], [0., 0., -.031]])
    features = contact_surface.surface_features(model, points, jnp.zeros_like(points),
        jnp.broadcast_to(rotation, (2, 3, 3)), .02)
    np.testing.assert_allclose(features['contact_normal_world'], [[.6, .8, 0.], [0., 0., -1.]], atol=1e-6)
    np.testing.assert_allclose(features['closest_prism_point_world_m'], [[.06, .04, .001], [0., 0., -.001]], atol=1e-6)
