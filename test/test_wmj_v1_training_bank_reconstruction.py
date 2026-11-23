"""Independent training bank goals and restore identity retain physical contracts."""

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from world_model.build_training_seed_bank import bank_signature, first_goal
from world_model.contact_surface import surface_from_vertices
from world_model.contact_geometry import MJXContactSurface
from world_model.contracts import load_config
from test_wmj_v1_control_projection import fixture


def test_initial_goal_uses_finite_face_inside_the_registered_domain():
    vertices = np.array([[x, y, z] for x in (-.075, .075)
                         for y in (-.075, .075) for z in (-.001, .001)])
    options = dict(surface=surface_from_vertices(vertices, normal_axis=2), ball_radius_m=.02)
    cfg = dict(domain=dict(task_goal_xyz_lower_m=[-.38, -.75, 1.3],
                           task_goal_xyz_upper_m=[.38, -.28, 1.45]))
    raw = dict(racket_geom_position_before_m=np.array([[[.39, -.5, 1.075]]]),
               racket_geom_rotation_before=np.eye(3)[None, None])
    goal, audits = first_goal(cfg, raw, options, np.array([1.4]))
    np.testing.assert_allclose(goal, [[.38, -.5, 1.4]])
    assert audits[0]['inside_face'] and audits[0]['positive_drop']
    wrapped = dict(options, surface=MJXContactSurface(options['surface'], None, np.ones(3), {}))
    actual, wrapped_audits = first_goal(cfg, raw, wrapped, np.array([1.4]))
    np.testing.assert_array_equal(actual, goal)
    assert wrapped_audits == audits
    invalid = dict(raw, racket_geom_position_before_m=np.array([[[.5, -.5, 1.075]]]))
    with pytest.raises(ValueError, match='finite-face'):
        first_goal(cfg, invalid, options, np.array([1.4]))


def test_bank_restore_identity_excludes_status_notes_but_includes_task_semantics():
    cfg = load_config(verify_inputs=False)
    kwargs = dict(model_references={'robot': 'frozen'}, source_rows=[{'source': 'train'}],
                  dataset_sha256='training-only')
    original = bank_signature(cfg, **kwargs)
    changed = copy.deepcopy(cfg)
    changed['imagination_implementation']['status'] = 'different_offline_report_status'
    assert bank_signature(changed, **kwargs) == original
    changed['domain']['roll_pitch_limit_deg'] = 3.
    assert bank_signature(changed, **kwargs) != original


def test_restore_shape_template_needs_only_the_authentic_reset(fixture, monkeypatch):
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace
    from test_wmj_v1_world_control import setup_control
    from world_model import build_training_seed_bank as module
    from world_model.full_prior_world import measured_racket_pose
    from world_model.observation_engine import init_world_observation
    chain, q, control, context, _ = setup_control(fixture)
    ball = jnp.array([[0., -.5, 1.4]])
    observation, _ = init_world_observation(ball, q, jax.random.split(jax.random.PRNGKey(1), 1),
        pose_from_q=lambda value: measured_racket_pose(chain, value), ar1_enabled=False)
    monkeypatch.setattr(module, 'read_control_seed', lambda *a, **k: (control, context))
    monkeypatch.setattr(module, 'read_initial_camera', lambda *a, **k: observation.camera)
    raw = dict(q_before=np.asarray(q)[None], dq_before=np.zeros((1, 1, 13), np.float32),
        ball_qpos_before=np.array([[[0., -.5, 1.4, 1., 0., 0., 0.]]], np.float32),
        ball_qvel_before=np.zeros((1, 1, 6), np.float32))
    values = {'state__racket_position_m': np.zeros((1, 1, 3), np.float32),
              'state__racket_rotation_world': np.eye(3, dtype=np.float32)[None, None]}
    state, actual_context = module.reset_source_template(SimpleNamespace(report={'context_sha256': 'test'}),
        {'first_seed_ref': {}}, np.array([0]), raw, values, {'chain': chain})
    np.testing.assert_array_equal(state.physics.q_rad, q)
    np.testing.assert_array_equal(state.control.rmp.q, control.rmp.q)
    assert actual_context is context
    assert int(state.observation.history.paired.present.sum()) == 1
    assert not state.physics.ball.previous_contact.any()
