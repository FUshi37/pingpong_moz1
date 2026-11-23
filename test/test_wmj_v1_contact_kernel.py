"""Learned-contact geometry follows the actual teacher's finite MJX mesh."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))

from world_model.contact_geometry import surface_from_compiled_model, surface_features
from world_model.contact_model import contact_features, contact_prediction
from world_model.contact_surface import surface_features as prism_features
from world_model.contracts import authenticate, load_config, repository_root
from world_model.neural import init_mlp
from world_model.contact_checkpoint import legacy_geometry_dispatch_compatible
from world_model.contracts import file_hash


@pytest.fixture(scope='module')
def surface():
    cfg = load_config()
    model = mujoco.MjModel.from_binary_path(str(authenticate(
        repository_root(), cfg['collection']['compiled_model_cache'])))
    return surface_from_compiled_model(model, model.geom('racket_rubber_fore').id,
                                       normal_axis=0)


def physical_inputs(position):
    shape = position.shape[:-1]
    return dict(ball_position_m=position, ball_velocity_m_s=jnp.zeros_like(position),
        ball_omega_world_rad_s=jnp.zeros_like(position), geom_position_m=jnp.zeros_like(position),
        geom_rotation_world=jnp.broadcast_to(jnp.eye(3), shape+(3, 3)),
        racket_linear_velocity_m_s=jnp.zeros_like(position),
        racket_angular_velocity_rad_s=jnp.zeros_like(position), ball_radius_m=.02,
        previous_contact=jnp.zeros(shape, bool), duration_s=jnp.zeros(shape))


def test_recorded_rim_cases_do_not_turn_cpu_overlap_into_mjx_impulse(surface):
    # Held training-audit examples, in the compiled racket's local frame.
    points = jnp.array([[.016996338963508606, -.040918007493019104, .07116875052452087],
                       [.018530838191509247, .06697386503219604, .06187525391578674],
                       [.020, 0., 0.], [.100, 0., 0.]], jnp.float32)
    _, geometry = contact_features(surface, **physical_inputs(points))
    np.testing.assert_allclose(geometry['sphere_gap_m'], [.0070473887, .0227447711, -.001, 1.], atol=2e-7)
    legacy = prism_features(surface.prism, points, jnp.zeros_like(points),
                            jnp.broadcast_to(jnp.eye(3), (4, 3, 3)), .02)
    assert float(legacy['sphere_gap_m'][0]) < 0.  # The original model's false contact.
    params = init_mlp(jax.random.PRNGKey(3), (19, 16, 8), zero_output=True)
    params = params[:-1]+({**params[-1], 'b': jnp.ones(8)},)
    prediction = contact_prediction(params, surface, **physical_inputs(points))
    np.testing.assert_array_equal(prediction.geometric_overlap, [False, False, True, False])
    np.testing.assert_array_equal(np.asarray(prediction.delta_velocity_world_m_s)[[0, 1, 3]], np.zeros((3, 3)))
    assert float(jnp.linalg.norm(prediction.delta_velocity_world_m_s[2])) > 0.


def test_native_contact_point_controls_spin_velocity_and_batch_shapes(surface):
    point = jnp.array([.020, 0., 0.])
    values = physical_inputs(point)
    values['ball_omega_world_rad_s'] = jnp.array([0., 0., 1.])
    features, geometry = contact_features(surface, **values)
    np.testing.assert_allclose(geometry['contact_position_world_m'], [.0005, 0., 0.], atol=1e-7)
    np.testing.assert_allclose(geometry['contact_normal_world'], [1., 0., 0.], atol=1e-7)
    # Relative velocity, transformed using the same surface frame as all other inputs.
    expected = geometry['frame_world'].T@jnp.cross(values['ball_omega_world_rad_s'],
        geometry['contact_position_world_m']-point)
    np.testing.assert_allclose(features[3:6], expected, atol=1e-7)
    multi = jnp.broadcast_to(point, (2, 3, 3))
    batched, _ = jax.jit(lambda p: contact_features(surface, **physical_inputs(p)))(multi)
    assert batched.shape == (2, 3, 19)
    np.testing.assert_allclose(batched[0, 0, :3], features[:3], atol=1e-7)


def test_native_geometry_is_differentiable_in_face_and_rim_modes(surface):
    def gap(p):
        return surface_features(surface, p, jnp.zeros(3), jnp.eye(3), .02)['sphere_gap_m']
    for point in (jnp.array([.018, .01, .02]),
                  jnp.array([.005, .079, .006])):
        actual = jax.jit(jax.grad(gap))(point)
        epsilon = 1e-5
        expected = np.array([(gap(point+jnp.eye(3)[i]*epsilon)-gap(point-jnp.eye(3)[i]*epsilon))
                             /(2*epsilon) for i in range(3)])
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=2e-3)


def test_legacy_geometry_keeps_exact_outputs(surface):
    points = jnp.array([[.018, .02, .01], [.01699633896, -.04091800749, .07116875052]])
    args = (surface.prism, points, jnp.zeros_like(points), jnp.broadcast_to(jnp.eye(3), (2, 3, 3)), .02)
    expected, actual = prism_features(*args), surface_features(*args)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_legacy_checkpoint_exception_is_limited_to_exact_dispatch_sources():
    import json
    root = repository_root()
    metadata = json.loads((root/'pingpong_controller/outputs/rl_sim/wmj_v1/contact_native_fit_a02/contact_u004096.json').read_text())
    current = {name: file_hash(root/'pingpong_controller/tools/rl_sim/world_model'/name)
               for name in ('contact_model.py', 'contact_geometry.py')}
    assert legacy_geometry_dispatch_compatible(metadata['source_sha256'], current)
    assert not legacy_geometry_dispatch_compatible(metadata['source_sha256'], current | {'contact_geometry.py': 'changed'})
    assert not legacy_geometry_dispatch_compatible(metadata['source_sha256'] | {'contact_model.py': 'unknown'}, current)
