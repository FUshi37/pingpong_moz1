"""Offline pose preference cannot weaken admission or mix controller phases."""

from pathlib import Path
import sys
from typing import NamedTuple
from types import SimpleNamespace

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from world_model.flat_initialization import (
    candidate_targets, merge_complete_native, pose_preference, solve_flat_pose,
)
from world_model import flat_initialization


def model_fixture():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint type="slide" axis="1 0 0" range="-1 1" limited="true"/>
      <joint type="slide" axis="0 1 0" range="-1 1" limited="true"/>
      <joint type="slide" axis="0 0 1" range="0 2" limited="true"/>
      <geom type="sphere" size=".01" mass="1"/>
      <body><joint axis="1 0 0" range="-60 60" limited="true"/>
       <joint axis="0 1 0" range="-60 60" limited="true"/>
       <joint axis="0 0 1" range="-60 60" limited="true"/>
       <geom type="sphere" size=".01" mass="1"/>
       <body><joint axis="0 0 1" range="-60 60" limited="true"/>
        <geom type="box" size=".075 .075 .001" mass=".1"/><site name="racket"/>
       </body></body></body></worldbody></mujoco>''')


def test_flat_pose_objective_recovers_position_and_horizontal_orientation():
    model = model_fixture()
    result = solve_flat_pose(
        model, model.qpos0, np.arange(7), model.site('racket').id,
        np.eye(3), np.array([.25, -.4, 1.075]),
        np.array([0., 0., 1., .2, -.1, .3, .2]),
    )
    assert result['status'] == 'IK_FEASIBLE'
    assert result['position_error_m'] < 1e-5
    assert max(map(abs, result['relative_roll_pitch_deg'])) < 1e-3
    q = np.asarray(result['q_rad'])
    assert np.all(q >= model.jnt_range[:, 0])
    assert np.all(q <= model.jnt_range[:, 1])
    unknown = solve_flat_pose(
        model, model.qpos0, np.arange(7), model.site('racket').id,
        np.eye(3), np.array([5., -.4, 1.075]), q,
    )
    assert unknown['status'] == 'UNKNOWN'


def test_flatness_preference_cannot_promote_an_unadmitted_pose():
    good = dict(status='ADMITTED', relative_roll_pitch_deg=[.03, .04],
                racket_site_m=[0., -.5, 1.075])
    tilted = dict(good, relative_roll_pitch_deg=[1.9, 0.])
    bad = dict(good, status='UNKNOWN')
    goal = np.array([0., -.5, 1.4])
    assert pose_preference(good, goal) < pose_preference(tilted, goal)
    assert pose_preference(bad, goal) > pose_preference(tilted, goal)


class State(NamedTuple):
    q: object
    queue: object
    phase: object


def test_native_merge_keeps_whole_lanes_and_rejects_shared_phase_changes():
    old = State(jnp.zeros((2, 7)), jnp.zeros((2, 3, 7)), jnp.array(3))
    new = State(jnp.ones((2, 7)), jnp.ones((2, 3, 7)), jnp.array(3))
    merged = merge_complete_native(old, new, np.array([True, False]))
    np.testing.assert_array_equal(merged.q[:, 0], [1., 0.])
    np.testing.assert_array_equal(merged.queue[:, 0, 0], [1., 0.])
    with pytest.raises(ValueError, match='shared'):
        merge_complete_native(old, new._replace(phase=jnp.array(4)), np.array([True, False]))


def test_candidate_positions_stay_in_workspace_without_changing_ball_goal():
    cfg = dict(
        domain=dict(racket_xyz_lower_m=[-.38, -.75, 1.],
                    racket_xyz_upper_m=[.38, -.28, 1.15]),
        offline_initialization_search=dict(
            inward_xy_offsets_m=[0., .03, .06], include_nearest_atlas_xy=True,
            site_height_candidates_m=[1.075, 1.025, 1.125]))
    catalog = SimpleNamespace(position_m=np.array([[.285, -.6, 1.075]]))
    goal = np.array([.38, -.74, 1.4])
    original = goal.copy()
    targets = np.array(candidate_targets(goal, catalog, cfg))
    np.testing.assert_array_equal(goal, original)
    assert np.all(targets >= np.array(cfg['domain']['racket_xyz_lower_m']))
    assert np.all(targets <= np.array(cfg['domain']['racket_xyz_upper_m']))
    assert len(targets) == 12
    np.testing.assert_array_equal(targets[0, :2], goal[:2])
    assert np.linalg.norm(targets[3, :2] - goal[:2]) == pytest.approx(.03)


def test_static_candidates_require_geometry_and_capacity_before_native_warmup(monkeypatch):
    cfg = dict(
        domain=dict(roll_pitch_limit_deg=2.),
        offline_initialization_search=dict(
            ik_starts_per_target=1, ik_max_function_evaluations=120,
            ik_position_scale_m=.001, ik_orientation_scale_rad=np.pi / 180,
            ik_joint_prior_weight=.01, orientation_preference_resolution_deg=.1,
            static_candidates_per_case=2))
    targets = [np.array([x, -.5, 1.075]) for x in (0., .02, .04, .06)]
    monkeypatch.setattr(flat_initialization, 'candidate_targets', lambda *args: targets)

    def solve(*args, **kwargs):
        target = args[5]
        return dict(status='IK_FEASIBLE', q_rad=[*target, 0., 0., 0., 0.],
                    position_m=target.tolist(), relative_roll_pitch_deg=[0., 0.])

    def audit(qpos, *args):
        return dict(status='UNKNOWN' if qpos[0] == 0. else 'ADMITTED',
                    racket_site_m=qpos[:3].tolist(), relative_roll_pitch_deg=[0., 0.])

    auditor = SimpleNamespace(
        model=SimpleNamespace(nv=7), data=None, baseline=np.zeros(7), qadr=np.arange(7),
        jids=np.arange(7), site=0, velocity_limits=np.ones(7), audit_warmed=audit,
        catalog=SimpleNamespace(q_rad=np.zeros((1, 7)), position_m=np.array([[0., -.5, 1.075]])))
    monkeypatch.setattr(flat_initialization, 'solve_flat_pose', solve)
    monkeypatch.setattr(flat_initialization, 'pure_vertical_speed_capacity',
                        lambda *args: dict(status='PASS'))
    row = flat_initialization.search_case(auditor, dict(
        case_id='test', group='nominal', goals_xyz_m=[[0., -.5, 1.4]]), np.eye(3), cfg)
    assert [c['candidate_id'] for c in row['candidates']] == [1, 2]
    assert row['attempts'][0]['static_admission']['status'] == 'UNKNOWN'
    assert 'not_evaluated' in row['attempts'][3]['excluded_reason']
    assert all(c['status'] == 'ADMITTED' for c in row['candidates'])
    monkeypatch.setattr(flat_initialization, 'pure_vertical_speed_capacity',
                        lambda *args: dict(status='UNKNOWN'))
    rejected = flat_initialization.search_case(auditor, dict(
        case_id='test', group='nominal', goals_xyz_m=[[0., -.5, 1.4]]), np.eye(3), cfg)
    assert rejected['candidates'] == []


def test_nonexact_search_reference_does_not_override_actual_drop_admission(monkeypatch):
    # The selected racket XYZ is a proposal. The fixed ball goal and physical
    # constraints, rather than submillimetre proposal tracking, define validity.
    q = [.03, -.5, 1.075, 0., 0., 0., 0.]
    source = dict(case_id='case', group='boundary', goal_xyz_m=[.06, -.5, 1.4],
                  attempts=[dict(candidate_id=4, target_site_m=[.06, -.5, 1.075],
                                 ik=dict(status='UNKNOWN', q_rad=q, position_m=q[:3],
                                         position_error_m=.03, relative_roll_pitch_deg=[0., 0.]))])
    auditor = SimpleNamespace(
        model=SimpleNamespace(nv=7), data=None, baseline=np.zeros(7), qadr=np.arange(7),
        jids=np.arange(7), site=0, velocity_limits=np.ones(7),
        audit_warmed=lambda *args: dict(status='ADMITTED', racket_site_m=q[:3],
                                      relative_roll_pitch_deg=[0., 0.]))
    cfg = dict(domain=dict(roll_pitch_limit_deg=2.), offline_initialization_search=dict(
        static_candidates_per_case=8, orientation_preference_resolution_deg=.1))
    monkeypatch.setattr(flat_initialization, 'pure_vertical_speed_capacity',
                        lambda *args: dict(status='PASS'))
    result = flat_initialization.recover_nonexact_case(auditor, source, np.eye(3), cfg)
    assert len(result['candidates']) == 1
    assert result['candidates'][0]['ik_reference_status'] == 'UNKNOWN'
    assert result['attempts'][0]['ik']['position_error_m'] == .03
    auditor.audit_warmed = lambda *args: dict(status='UNKNOWN')
    rejected = flat_initialization.recover_nonexact_case(auditor, source, np.eye(3), cfg)
    assert rejected['candidates'] == []
