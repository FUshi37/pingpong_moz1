"""Directed experiments propose incoming states without fabricating outcomes."""

from pathlib import Path
import sys

import numpy as np

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model.contracts import load_config
from world_model import generators
from world_model.contact_surface import surface_from_vertices


def test_pose_catalog_uses_admitted_static_poses_and_retains_unknown_regions():
    catalog = generators.load_pose_catalog(load_config())
    assert len(catalog.region_ids) == 119
    assert len(catalog.unknown_region_ids) == 70
    assert catalog.q_rad.shape[1] == 7
    assert np.isfinite(catalog.q_rad).all()
    assert set(catalog.pose_region_ids).issubset(set(catalog.region_ids))
    assert not set(catalog.unknown_region_ids) & set(catalog.pose_region_ids)


def test_directed_trials_obey_confirmed_incoming_domain_and_zero_initial_spin():
    cfg = load_config()
    catalog = generators.load_pose_catalog(cfg)
    surface = surface_from_vertices(catalog.geometry['racket']['compiled_vertices_local_m'], normal_axis=0)
    n = 1024
    rotation = np.broadcast_to(np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]), (n, 3, 3))
    position = np.tile([0., -.5, 1.1], (n, 1))
    plan = generators.propose_directed_contacts(np.random.default_rng(31), cfg, surface,
                                                position, rotation, ball_radius_m=.02)
    assert np.all(plan.initial_angular_velocity_rad_s == 0.)
    assert np.min(plan.incoming_velocity_m_s[:, 2]) >= -3.5
    assert np.max(plan.incoming_velocity_m_s[:, 2]) <= -1.
    assert np.max(np.linalg.norm(plan.incoming_velocity_m_s[:, :2], axis=-1)) <= .5
    assert np.all((plan.time_to_contact_s >= .08) & (plan.time_to_contact_s <= .16))
    assert len(np.unique(plan.offset_class)) == 4
    t = plan.time_to_contact_s[:, None]
    g = np.array([0., 0., -9.81])
    np.testing.assert_allclose(plan.initial_position_m+plan.initial_velocity_m_s*t+.5*g*t*t,
                               plan.proposed_contact_position_m, atol=2e-7)
    np.testing.assert_allclose(plan.initial_velocity_m_s+g*t,
                               plan.incoming_velocity_m_s, atol=5e-7)
    assert np.all(plan.target_apex_m >= cfg['domain']['task_goal_xyz_lower_m'])
    assert np.all(plan.target_apex_m <= cfg['domain']['task_goal_xyz_upper_m'])
    assert np.any(np.linalg.norm(plan.desired_outgoing_velocity_m_s-plan.initial_velocity_m_s, axis=-1) > 1.)


def test_generator_counts_and_correlated_actions_are_reproducible():
    cfg = load_config()
    first = generators.generator_assignments(np.random.default_rng(5), 1024, cfg)
    second = generators.generator_assignments(np.random.default_rng(5), 1024, cfg)
    np.testing.assert_array_equal(first, second)
    names, counts = np.unique(first, return_counts=True)
    assert dict(zip(names, counts)) == dict(robot_identification=256, directed_contact=410,
                                          closed_loop=205, boundaries_failure=153)
    actions, modes = generators.correlated_action_sequences(np.random.default_rng(8), 64, 1024)
    assert actions.shape == (64, 1024, 7)
    assert np.isfinite(actions).all() and np.max(np.abs(actions)) <= .25+1e-6
    prbs = actions[:, modes == 'prbs']
    # A 20--60 ms hold cannot become independent white noise at every 5 ms tick.
    assert np.mean(np.all(np.diff(prbs, axis=0) == 0., axis=-1)) > .65
