"""Fail-closed contracts for the independent WMJ-V1 research path."""

import copy
from pathlib import Path
import sys

import pytest

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model.contracts import load_config, validate_config
from world_model.datasets import SplitRegistry, require_imagination
from world_model.resources import StorageBudget, StorageLimitError


def test_confirmed_course_and_component_identity():
    cfg = load_config()
    assert cfg['identity']['training_version'] == 'V1'
    assert cfg['components']['CONTACT']['id'] == 'ball_racket_contact_real11_static_1ms_v1'
    assert cfg['components']['EXEC']['id'] == 'mjx_topology_execution_v2'
    assert [s['ar1_enabled'] for s in cfg['curriculum']['stages']] == [False] * 4 + [True]
    assert cfg['domain']['task_goal_xyz_upper_m'][1] == -0.28
    assert cfg['domain']['roll_pitch_limit_deg'] == 2.0
    assert cfg['policy']['actor_obs_dim'] == 969
    assert cfg['lineage']['actor_migration'] == 'copy_72_rows_zero_all_897_added_input_rows'
    history = cfg['observation']['paired_history']
    assert history['window_s'] == .640
    assert history['racket_dense_actor_history'] is False
    assert history['joint_action_actor_history'] is False


def test_history_cannot_become_a_dense_racket_or_joint_stream():
    for field in ('racket_dense_actor_history', 'joint_action_actor_history'):
        cfg = load_config()
        cfg['observation']['paired_history'][field] = True
        with pytest.raises(ValueError, match='history'):
            validate_config(cfg)


@pytest.mark.parametrize('field,value', [('base_input_dim', 35), ('output_dim', 14),
    ('action_dimension', 13), ('actor_dimension', 981),
    ('deployment_execution_proxy_required', True)])
def test_approved_support_model_keeps_policy_and_deployment_boundaries(field, value):
    cfg = load_config()
    cfg['world_model']['robot'][field] = value
    with pytest.raises(ValueError, match='support execution'):
        validate_config(cfg)


def test_directed_collection_starts_without_spin_but_keeps_native_rotation_dynamics():
    cfg = load_config()
    incoming = cfg['collection']['directed_conditions']
    assert incoming['incoming_vertical_velocity_m_s'] == [-3.5, -1.0]
    assert incoming['incoming_horizontal_speed_max_m_s'] == .5
    assert incoming['time_to_contact_s'] == [.08, .16]
    assert incoming['initial_ball_angular_velocity_rad_s'] == [0., 0., 0.]
    assert incoming['sample_initial_spin'] is False
    assert incoming['preserve_collision_induced_spin'] is True
    for key, value in (('sample_initial_spin', True), ('preserve_collision_induced_spin', False),
                       ('initial_ball_angular_velocity_rad_s', [1., 0., 0.])):
        changed = copy.deepcopy(cfg)
        changed['collection']['directed_conditions'][key] = value
        with pytest.raises(ValueError, match='directed collection'):
            validate_config(changed)


def test_imagined_history_uses_known_updates_and_no_future_teacher_samples():
    cfg = load_config()
    assert cfg['policy']['representation'] == 'direct_history_mlp'
    assert len(cfg['policy']['representation_candidates']) == 2
    assert all(c['status'] == 'candidate_only'
               for c in cfg['policy']['representation_candidates'])
    update = cfg['observation']['world_transition']
    assert update['re_predict_history'] is False
    assert update['future_teacher_observations'] is False
    assert update['fresh_long_history_rate_hz'] == 90.
    for field in ('re_predict_history', 'future_teacher_observations'):
        changed = copy.deepcopy(cfg)
        changed['observation']['world_transition'][field] = True
        with pytest.raises(ValueError, match='history transition'):
            validate_config(changed)


def test_geometry_admission_is_offline_and_has_no_added_clearance():
    audit = load_config()['domain']['self_collision_audit']
    assert audit['scope'] == 'offline_initialization_only'
    assert audit['method'] == 'convex_broad_phase_original_triangle_and_containment_refinement'
    assert audit['runtime_step_enabled'] is False
    assert audit['runtime_reset_scan_enabled'] is False
    assert audit['additional_clearance_m'] == 0.
    assert audit['ambiguous_result'] == 'UNKNOWN'
    assert audit['reset_source'] == 'preaudited_seed_bank'


@pytest.mark.parametrize('field,value', [('runtime_step_enabled', True),
    ('runtime_reset_scan_enabled', True), ('additional_clearance_m', .005),
    ('ambiguous_result', 'PASS')])
def test_initialization_audit_cannot_add_training_work_or_change_admission(field, value):
    cfg = load_config()
    cfg['domain']['self_collision_audit'][field] = value
    with pytest.raises(ValueError, match='self-collision'):
        validate_config(cfg)


@pytest.mark.parametrize('path,value', [
    (('identity', 'training_version'), 'V2'),
    (('collection', 'n_envs'), 512),
    (('imagination', 'logical_envs'), 512),
    (('task', 'hits_per_goal'), 19),
    (('domain', 'roll_pitch_limit_deg'), 5.0),
    (('execution', 'id'), 'legacy'),
    (('policy', 'output_activation'), 'tanh'),
    (('policy', 'actor_obs_dim'), 72),
])
def test_unreviewed_contract_changes_are_rejected(path, value):
    cfg = copy.deepcopy(load_config())
    cfg[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_noise_cannot_silently_enter_an_earlier_stage():
    cfg = copy.deepcopy(load_config())
    cfg['curriculum']['stages'][1]['ar1_enabled'] = True
    with pytest.raises(ValueError, match='AR1'):
        validate_config(cfg)


def test_confirmed_distance_disturbance_and_frozen_validation_contract():
    cfg = load_config()
    assert cfg['curriculum']['transition_distance_metric'] == 'xyz_euclidean_m'
    pulse = cfg['curriculum']['recovery_disturbance']
    assert pulse['duration_s'] == .020
    assert pulse['delta_vxy_norm_max_m_s'] == .05
    assert pulse['delta_vz_abs_max_m_s'] == .10
    stages = cfg['curriculum']['stages']
    assert [s['disturbance_enabled'] for s in stages] == [False]*3+[True]*2
    assert [s['validation']['goals_per_case'] for s in stages] == [1, 2, 2, 4, 4]
    for stage in stages:
        assert stage['validation']['cases'] == 128
        assert stage['validation']['duration_s'] == 60.
        assert stage['validation']['gates']['each_group_each_mode_success_rate_min'] == .9


@pytest.mark.parametrize('field,value', [('duration_s', .010),
    ('delta_vxy_norm_max_m_s', .10), ('delta_vz_abs_max_m_s', .20)])
def test_disturbance_cannot_change_without_qa(field, value):
    cfg = load_config()
    cfg['curriculum']['recovery_disturbance'][field] = value
    with pytest.raises(ValueError, match='disturbance'):
        validate_config(cfg)


def test_validation_groups_cannot_be_replaced_by_overall_success():
    cfg = load_config()
    cfg['curriculum']['stages'][3]['validation']['gates'] = {'overall_success_rate_min': .9}
    with pytest.raises(ValueError, match='validation'):
        validate_config(cfg)


def test_storage_reserves_project_capacity_and_existing_evidence(tmp_path):
    artifact = tmp_path / 'protected.bin'
    artifact.write_bytes(b'x' * 40)
    allocated = max(40, artifact.stat().st_blocks * 512)
    budget = StorageBudget(tmp_path, max_bytes=allocated + 60, min_free_bytes=200)
    budget.check(additional_bytes=60, free_bytes=260)
    with pytest.raises(StorageLimitError, match='budget'):
        budget.check(additional_bytes=61, free_bytes=1000)
    with pytest.raises(StorageLimitError, match='free'):
        budget.check(additional_bytes=10, free_bytes=209)
    assert (tmp_path / 'protected.bin').read_bytes() == b'x' * 40


def test_same_component_name_does_not_authorize_different_contact_parameters():
    cfg = copy.deepcopy(load_config())
    cfg['physics']['contact']['ball_racket_pair_solref'][0] *= 2
    with pytest.raises(ValueError, match='contact'):
        validate_config(cfg)


def test_same_version_does_not_authorize_a_different_control_delay():
    cfg = copy.deepcopy(load_config())
    cfg['control']['native_parameters']['recovered_rmp_target_update_delay_steps'] = 4
    with pytest.raises(ValueError, match='control'):
        validate_config(cfg)


def test_split_grouping_survives_append_and_root_action_forks():
    registry = SplitRegistry('wmj_group_split_v1')
    split = registry.assign(root_seed_family='root-A', episode='e0', session='s0')
    assert registry.assign(root_seed_family='root-A', episode='branch+', session='s1') == split
    assert registry.assign(root_seed_family='root-B', episode='e1', session='s0') == split
    restored = SplitRegistry.from_dict(registry.to_dict())
    assert restored.assign(root_seed_family='root-A', episode='branch-', session='s2') == split
    assert restored.assign(root_seed_family='root-C', episode='e2', session='s1') == split


def test_later_group_connections_cannot_move_existing_data_between_splits():
    registry = SplitRegistry('wmj_group_split_v1')
    first = registry.assign(root_seed_family='a', episode='0', session='a')
    for i in range(100):
        other = str(i)
        if registry.assign(root_seed_family=other, episode='0', session=other) != first:
            break
    else:
        pytest.fail('Expected distinct deterministic partitions')
    original = registry.to_dict()
    with pytest.raises(ValueError, match='leakage'):
        registry.assign(root_seed_family='a', episode='cross', session=other)
    assert registry.to_dict() == original


@pytest.mark.parametrize('origin', ['mjx_teacher', 'real_robot', 'mjx_diagnostic', None])
def test_actor_critic_refuse_non_imagined_training_sources(origin):
    with pytest.raises(ValueError, match='imagination'):
        require_imagination({'origin': origin, 'model_revision': 1, 'model_sha256': 'a' * 64})


def test_imagination_origin_also_requires_pinned_dynamics():
    require_imagination({'origin': 'imagination', 'model_revision': 1,
                         'model_sha256': 'a' * 64})
    with pytest.raises(ValueError):
        require_imagination({'origin': 'imagination'})
