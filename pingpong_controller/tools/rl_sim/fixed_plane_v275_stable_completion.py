"""Gate-only V257 continuation with explicit aggregate stability tolerances."""
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v275_stable_completion'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v257'
START_STAGE = 'fplc_v275_s09_near_y_035mm_bridge'
RETENTION_STAGE = 'fplc_v275_s01_workspace_alignment'
GATE_CHANGES = {
    'target_mean_hits': 12.7,
    'target_mean_len_frac': .995,
    'target_episode_truncation_rate': .99,
    'target_hit12_rate': .98,
    'min_recent_mean_hit_apex_abs_z_m': 1.37,
    'max_recent_mean_hit_apex_abs_z_m': 1.41,
    'max_recent_mean_hit_vxy': .026,
    'max_recent_rms_hit_vxy': .030,
    'max_recent_rms_steady_hit_racket_vxy': .060,
    'max_recent_rapid_steady_racket_vxy_rms_m_s': .060,
    'max_recent_rapid_steady_racket_omega_rms_rad_s': .30,
    'max_recent_mean_hit_contact_center_dist_m': .007,
    'min_recent_late_physical_in_band_rate': .97,
    'max_recent_late_physical_below_floor_rate': None,
    'max_recent_late_physical_above_ceiling_rate': None,
    'max_recent_late_adjacent_jump_m': None,
    'min_recent_late_adjacent_pairs': None,
    'min_recent_rapid_steady_hit4_apex_z_m': None,
    'max_recent_rapid_steady_surface_z_abs_error_m': None,
    'min_recent_fixed_contact_quality_components': None,
    'min_recent_fixed_contact_return_apex_score': None,
    'advance_validate_contact_quality_components': False,
    'min_updates': 30,
    'convergence_hold_updates': 16,
}


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Keep every V257 physical lesson, reward and numerical PPO value exact."""
    if len(source) != 17 or source[8].name != 'fplc_v257_s09_near_y_035mm_bridge':
        raise ValueError('V275 requires the complete original V257 course')
    return [replace(stage, name=stage.name.replace('fplc_v257_', 'fplc_v275_', 1),
        **GATE_CHANGES,
        notes=('V275 gate-only continuation of recovered full V257 state. '
            'Same17 lessons, rewards, physics, control, observation/reset/DR '
            'and uncapped1024x128 numerical PPO; rolling32/hold16, minimum30 '
            'updates and independent288-episode block. Length>=.995,full>=.99,hit12>=.98, '
            'hits>=12.7, absolute apex1.37--1.41m, ballXY mean/RMS<=.026/.030. '
            'Steady racketXY RMS<=.060,omegaRMS<=.30,center<=7mm; '
            'late1.40--1.45m band>=.97. Small separate tails, adjacent jump '
            'and shaped quality components remain diagnostic. Native block '
            'and nominal retention use the same core thresholds. Fresh '
            'same-S9 gate statistics; no sample rejection or forced advancement.'))
        for stage in source]


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v257_outcome_balance import validate_training_contract as previous
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V275 requires nominal retention validation')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name': 'fplc_v257_s01_workspace_alignment'}))


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(key) for key in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V275 requires full unreset72/393/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        updates = int(payload.get('stage_update', 0)) - 4429
        if (resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_index') != 9
                or payload.get('stage_name') != 'fplc_v257_s09_near_y_035mm_bridge'
                or updates < 0 or payload.get('step') != 2794455040 + updates * 131072
                or int(payload['opt'].t) != 691767 + updates * 32):
            raise ValueError('V275 requires latest full V257 S9 and fresh gate history')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V275 requires exact own stage/history continuation')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V275 rejects another source profile or from-scratch training')
