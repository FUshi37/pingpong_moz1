"""V274: aggregate stability gates for timely completion of the same DR course."""
from dataclasses import replace
from typing import Any, Mapping, Sequence

import fixed_plane_v273_gate_calibration as parent

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v274_stable_dr'
PARENT_PROFILE = parent.PROFILE
STAGE_NAMES = tuple(s.replace('v273_', 'v274_', 1) for s in parent.STAGE_NAMES)
START_STAGE = STAGE_NAMES[6]
SOURCE = dict(update=133, global_update=133, step=1704460288, adam_t=424766,
    sha256='7c43f6b6650214b6aafe0e8805c6664fd9a8ab015f22d4c62555cf19756bccbe')
GATE_CHANGES = {
    'target_mean_hits': 12.7,
    'target_episode_truncation_rate': .99,
    'target_hit12_rate': .98,
    'min_recent_mean_hit_apex_abs_z_m': 1.37,
    'max_recent_mean_hit_apex_abs_z_m': 1.41,
    'max_recent_mean_hit_vxy': .045,
    'max_recent_rms_hit_vxy': .055,
    'max_recent_mean_hit_racket_vxy': .045,
    'max_recent_rms_hit_racket_vxy': .055,
    'max_recent_mean_recurrent_hit_racket_vxy': .045,
    'max_recent_rms_steady_hit_racket_vxy': .055,
    'max_recent_rapid_steady_racket_vxy_rms_m_s': .055,
    'max_recent_rapid_steady_racket_omega_rms_rad_s': .22,
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
    'convergence_hold_updates': 16,
}


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Use the same core stability contract for DR lessons and S1 retention."""
    if tuple(s.name for s in source) != parent.STAGE_NAMES:
        raise ValueError('V274 requires the complete preserved V273 course')
    return [replace(stage, name=name, **GATE_CHANGES,
        notes=('V274 user-authorized aggregate DR stability completion. '
            'Keep all11 environments/rewards and numerical PPO exact. '
            'Length>=.995, full>=.99, hit12>=.98, hits>=12.7; '
            'mean absolute apex1.37--1.41m; ballXY mean/RMS<=.045/.055; '
            'recurrent racketXY mean<=.045, '
            'steady RMS<=.055m/s, omegaRMS<=.22rad/s, center<=7mm. '
            'Late apex1.40--1.45m band>=.97; single tails and adjacent jump '
            'remain diagnostics. Rolling32/hold16, independent288x1200 block '
            'and S1 retention use the same core contract. Fresh S7 history; '
            'full actor/critic/Adam; no forced advancement or update cap.'))
        for stage, name in zip(source, STAGE_NAMES, strict=True)]


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    """Restore the tested full parent; reject stale gates and forced advancement."""
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 398, 7)):
        raise ValueError('V274 requires full72/398/7 actor/critic/Adam without reset')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (requested_start_stage != START_STAGE or resume_curriculum_state
                or payload.get('stage_index') != 7 or payload.get('stage_name') != parent.STAGE_NAMES[6]
                or payload.get('stage_update') != SOURCE['update']
                or payload.get('global_update') != SOURCE['global_update']
                or payload.get('step') != SOURCE['step'] or int(payload['opt'].t) != SOURCE['adam_t']):
            raise ValueError('V274 pins the tested V273 source and fresh same-S7 gate history')
    elif source == PROFILE:
        index = payload.get('stage_index', 0)
        if not 1 <= index <= 11 or payload.get('stage_name') != STAGE_NAMES[index - 1]:
            raise ValueError('V274 own stage identity differs')
        if resume_curriculum_state:
            if requested_start_stage != payload['stage_name']:
                raise ValueError('V274 continuation requires exact causal stage history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError('V274 rejects another source profile')


validate_training_contract = parent.validate_training_contract
