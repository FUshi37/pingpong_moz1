"""Configuration identities and authenticated inputs for fixed WMJ-V1."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

CONTACT_ID = 'ball_racket_contact_real11_static_1ms_v1'
EXECUTION_ID = 'mjx_topology_execution_v2'
V321_SHA256 = '889eb976b18567d607ab9f86f35457c89f901e2ecf221b9391128cb471f0d001'


def repository_root() -> Path:
    """Return the project root independently of the caller's working directory."""
    return Path(__file__).resolve().parents[4]


def canonical_json(value: Any) -> str:
    """Reject nonfinite JSON and produce the bytes used for contract hashes."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def config_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def authenticate(root: Path, reference: dict) -> Path:
    """Authenticate an existing input before parsing or unpickling it."""
    path = Path(reference['path'])
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f'Required input is missing: {path}')
    if file_hash(path) != reference['sha256']:
        raise ValueError(f'SHA-256 mismatch: {path}')
    return path.resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(cfg: dict) -> None:
    """Fail closed on unreviewed changes to the user's fixed experiment."""
    canonical_json(cfg)
    _require(cfg['schema_version'] == 'wmj_experiment_config_v1', 'Unknown WMJ schema')
    identity = cfg['identity']
    _require(identity['training_version'] == 'V1'
             and identity['project'] == 'WMJ'
             and identity['profile'] == 'wmj_v1', 'WMJ-V1 identity is fixed')
    _require(cfg['training_protocol']['enabled'] is True, 'WMJ protocol must be enabled')
    _require(cfg['components']['CONTACT']['id'] == CONTACT_ID, 'CONTACT must match V324')
    _require(cfg['components']['EXEC']['id'] == EXECUTION_ID
             and cfg['execution']['id'] == EXECUTION_ID, 'EXEC must match V324')
    _require(cfg['lineage']['actor_source']['sha256'] == V321_SHA256,
             'Only the authenticated V321 actor is a source')
    _require(cfg['lineage']['actor_migration'] == 'copy_72_rows_zero_all_897_added_input_rows',
             'History actor migration must preserve72 and zero all897 added rows')
    policy = cfg['policy']
    _require((policy['actor_obs_dim'], policy['action_dim']) == (969, 7), 'Actor must be 969/7')
    _require(policy['representation'] == 'direct_history_mlp'
             and all(c['status'] == 'candidate_only' and c['requires_user_confirmation']
                     for c in policy['representation_candidates']),
             'Latent representation alternatives are candidates only')
    _require(policy['observation_schema'] == 'wmj_v1_prefix73_paired_ball_racket_640ms',
             'The confirmed paired-history observation schema changed')
    _require(policy['hidden_sizes'] == [256, 256]
             and policy['hidden_activation'] == 'tanh'
             and policy['output_activation'] == 'linear'
             and policy['normalization'] == 'none', 'Preserve the released actor architecture')
    _require(policy['goal_z_encoding'] == {'center_m': 1.425, 'scale_m': 0.10},
             'Goal Z encoding changed')
    control = cfg['control']
    _require(config_hash(control['native_parameters'])
             == 'b8d0b7810af2f54b7ae6749329c0b1d5e8e1c9a240cf62c722c7cb6b4049d6d1',
             'Native V324 control contract changed')
    _require((control['policy_dt_s'], control['physics_dt_s'], control['substeps'])
             == (0.005, 0.001, 5), 'Control timing changed')
    _require(control['action_semantics'] == 'persistent_measured_feedback_QVEL',
             'Action semantics changed')
    domain = cfg['domain']
    _require(domain['task_goal_xyz_lower_m'] == [-0.38, -0.75, 1.30]
             and domain['task_goal_xyz_upper_m'] == [0.38, -0.28, 1.45],
             'Candidate goal domain changed')
    _require(domain['racket_xyz_lower_m'] == [-0.38, -0.75, 1.0]
             and domain['racket_xyz_upper_m'] == [0.38, -0.28, 1.15]
             and domain['roll_pitch_limit_deg'] == 2.0, 'Racket domain changed')
    _require(domain['view_xy_lower_m'] == [-0.38, -0.75]
             and domain['view_xy_upper_m'] == [0.38, -0.28]
             and domain['view_hardware_certified'] is False, 'Simulated view contract changed')
    audit = domain['self_collision_audit']
    _require(audit == dict(scope='offline_initialization_only',
        method='convex_broad_phase_original_triangle_and_containment_refinement',
        runtime_step_enabled=False, runtime_reset_scan_enabled=False,
        reset_source='preaudited_seed_bank', additional_clearance_m=0.,
        surface_tolerance_m=1e-9, ambiguous_result='UNKNOWN',
        repair_or_fill_mesh_holes=False, trajectory_safety_certified=False,
        preparation_wall_time_in_total=True),
        'Confirmed offline self-collision contract changed')
    task = cfg['task']
    _require((task['hits_per_goal'], task['latest_stability_start_hit'],
              task['required_stable_apexes']) == (16, 3, 10), 'Goal counting changed')
    _require((task['xy_tolerance_m'], task['z_tolerance_m'], task['vxy_limit_m_s'])
             == (.02, .02, .05), 'Task success thresholds changed')
    stages = cfg['curriculum']['stages']
    _require([s['id'] for s in stages] == ['S01', 'S02', 'S03', 'S04', 'S05'],
             'The five-stage course is fixed')
    _require([s['ar1_enabled'] for s in stages] == [False, False, False, False, True],
             'AR1 must enter only at S05, at full V324 strength')
    course = cfg['curriculum']
    _require(course['transition_distance_metric'] == 'xyz_euclidean_m',
             'Transition distances must include XYZ')
    disturbance = course['recovery_disturbance']
    _require((disturbance['duration_s'], disturbance['delta_vxy_norm_max_m_s'],
              disturbance['delta_vz_abs_max_m_s'], disturbance['pulses_per_sequence'])
             == (.020, .05, .10, 1)
             and disturbance['kind'] == 'external_ball_force_pulse'
             and disturbance['trigger'] == 'actual_apex_at_goal_transition'
             and disturbance['reset_control_or_observation'] is False
             and [s['disturbance_enabled'] for s in stages] == [False]*3+[True]*2,
             'Confirmed recovery disturbance changed')
    for stage, goals in zip(stages, (1, 2, 2, 4, 4)):
        v = stage['validation']
        _require(v['mode'] == 'frozen_paired' and v['cases'] == 128
                 and v['duration_s'] == 60. and v['goals_per_case'] == goals
                 and v['conditions'] == [dict(id=name, cases=32)
                    for name in ('nominal', 'near_side', 'mirror', 'boundary')]
                 and v['gates'].get('each_group_each_mode_success_rate_min') == .9
                 and v['reset_after_failure'] is False,
                 'Confirmed per-group frozen validation changed')
    camera = cfg['observation']
    transition = camera['world_transition']
    _require(transition['kind'] == 'known_camera_and_history_roll_forward'
             and transition['new_state_source'] == 'predicted_physics_then_observation_pipeline'
             and transition['re_predict_history'] is False
             and transition['future_teacher_observations'] is False
             and transition['legacy_short_history_rate_hz'] == 200.
             and transition['fresh_long_history_rate_hz'] == 90.
             and transition['hold_between_fresh_samples'] is True
             and transition['past_measurements_and_noise'] == 'preserve'
             and transition['racket_pose_source'] == 'FK_of_same_timestamp_measured_or_predicted_q',
             'Confirmed world history transition changed')
    history = camera['paired_history']
    _require(history['window_s'] == .640 and history['ball_sample_slots'] == 64
             and history['features_per_sample'] == 14
             and history['racket_dense_actor_history'] is False
             and history['joint_action_actor_history'] is False
             and history['append_only_fresh_ball_samples'] is True
             and history['pose_alignment'] == 'ball_source_timestamp_causal_encoder_interpolation_FK'
             and history['source_time_units'] == 'episode_relative_us'
             and history['latent_compression_enabled'] is False,
             'Confirmed 640ms paired ball/racket history changed')
    _require(camera['rate_hz'] == 90.0 and camera['history_policy_ticks'] == 8
             and camera['fractional_rate_scheduler'] is True, 'Observation clock changed')
    noise = camera['ar1']
    _require(noise['mean_m'] == [.000048, .000072, -.000227]
             and noise['std_m'] == [.001451, .001418, .002970]
             and noise['rho_at_90hz'] == [.935, .907, .790], 'V324 AR1 parameters changed')
    collection = cfg['collection']
    directed = collection['directed_conditions']
    _require(directed['incoming_vertical_velocity_m_s'] == [-3.5, -1.0]
             and directed['incoming_horizontal_speed_max_m_s'] == .5
             and directed['time_to_contact_s'] == [.08, .16]
             and directed['initial_ball_angular_velocity_rad_s'] == [0., 0., 0.]
             and directed['sample_initial_spin'] is False
             and directed['preserve_collision_induced_spin'] is True
             and directed['keep_actual_failed_outcomes'] is True,
             'Confirmed directed collection domain or zero-initial-spin contract changed')
    _require((collection['n_envs'], collection['steps_per_round']) == (1024, 64),
             'MJX collection must use 1024 environments by 64 control steps')
    _require(collection['budget_rounds'] == [20, 40, 80], 'Collection budget changed')
    _require(math.isclose(sum(collection['generator_fractions'].values()), 1.0),
             'Generator probabilities must sum to one')
    imagined = cfg['imagination']
    _require((imagined['logical_envs'], imagined['persistent_envs'], imagined['branch_envs'])
             == (1024, 768, 256), 'Imagination population changed')
    _require(imagined['horizon'] in (64, 128), 'Unregistered imagination horizon')
    _require(imagined['microbatch'] > 0
             and imagined['logical_envs'] % imagined['microbatch'] == 0,
             'Microbatch must divide the logical population')
    _require(imagined['actor_training_source'] == 'imagination'
             and imagined['critic_training_source'] == 'imagination',
             'Actor and critic must learn exclusively from imagination')
    robot = cfg['world_model']['robot']
    _require((robot['base_input_dim'], robot['hidden_sizes'], robot['output_dim'])
             == (47, [128, 128], 26)
             and robot['state_joint_order'] == [f'RightArm-{i}' for i in range(7)]
                 + [f'LegWaist-{i}' for i in range(6)]
             and robot['action_dimension'] == 7 and robot['actor_dimension'] == 969
             and robot['deployment_execution_proxy_required'] is False
             and robot['support_commands'] == 'unchanged_native_constant_hold_targets',
             'Approved support execution contract changed')
    _require(cfg['physics']['ball_mass_kg'] == .004
             and cfg['physics']['ball_mass_range_kg'] == [.004, .004]
             and cfg['physics']['collision_outcome_residual_enabled'] is False,
             'Preserve the V324 mass/contact baseline')
    _require(config_hash(cfg['physics']['contact'])
             == '7cf8d66353df521f8eb6f8cd6e97463e0c25bb8090e9b5ca5c5f0228ba6d6800',
             'Native V324 contact contract changed')
    _require((cfg['physics']['solver_iterations'], cfg['physics']['line_search_iterations'])
             == (100, 50), 'Solver precision must remain V324-exact')
    resources = cfg['resources']
    # Explicit user extensions for paired and pose-margin trials, 2026-11-10.
    # Lower historical caps remain valid; the 8 GiB free-space floor is unchanged.
    pose_extension = (resources['max_new_bytes'] == 6593445888
                      and bool(cfg.get('pose_margin_trial', {}).get('plan')))
    # User-confirmed 48 MiB extension for the isolated Critic fit, 2026-11-11.
    critic_fit_extension = (resources['max_new_bytes'] == 6643777536
                            and bool(cfg.get('critic_only_fit', {}).get('plan')))
    # Separate user-confirmed EMA pairing allowance; historical caps stay valid.
    critic_ema_extension = (resources['max_new_bytes'] == 6677331968
                            and bool(cfg.get('critic_ema_pair', {}).get('plan')))
    # User-confirmed fixed-data Critic allowance, 2026-11-12; exact extra16 MiB.
    critic_fixed_extension = (resources['max_new_bytes'] == 6694109184
                              and bool(cfg.get('critic_fixed_data_fit', {}).get('plan')))
    critic_fixed_resume = (resources['max_new_bytes'] == 6710886400
                           and bool(cfg.get('critic_fixed_data_fit', {}).get('resource_resume_plan')))
    _require((0 < resources['max_new_bytes'] <= 6576668672
              or pose_extension or critic_fit_extension or critic_ema_extension
              or critic_fixed_extension or critic_fixed_resume)
             and resources['min_free_bytes'] >= 8 * 2**30, 'Storage budget changed')
    _require(0 < resources['max_run_wall_s'] <= 12 * 3600, 'Run exceeds the 12-hour budget')


def load_config(path: Path | None = None, *, verify_inputs: bool = True) -> dict:
    path = path or repository_root() / 'pingpong_controller/tools/rl_2real/WMJ_V1_CONFIG.json'
    cfg = json.loads(path.read_text(encoding='utf-8'))
    validate_config(cfg)
    if verify_inputs:
        for reference in cfg['authenticated_inputs'].values():
            authenticate(repository_root(), reference)
        authenticate(repository_root(), cfg['training_protocol'])
        authenticate(repository_root(), cfg['lineage']['actor_source'])
    return cfg
