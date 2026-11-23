"""V228 aligns existing late motion losses with unchanged strict RMS gates."""
from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

from fixed_plane_v227_nominal_reference import (
    PARENT_PROFILE, PARENT_SHA256, REPLAY_SHA256, validate_anchor_contract,
)

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v228'
START_STAGE = 'fplc_v228_s01_workspace_alignment'


def normalized_motion_weights(ball_rms_limit_m_s: float, racket_xy_limit_m_s: float,
                              omega_limit_rad_s: float) -> tuple[float, float]:
    """Equalize per-event squared loss at each frozen ball/racket RMS cap."""
    limits = (ball_rms_limit_m_s, racket_xy_limit_m_s, omega_limit_rad_s)
    if not all(math.isfinite(float(x)) and x > 0. for x in limits):
        raise ValueError('V228 needs finite positive unchanged RMS targets')
    ball_cost_at_gate = 2. * (ball_rms_limit_m_s / .025) ** 2
    return (ball_cost_at_gate * (.053 / racket_xy_limit_m_s) ** 2,
            ball_cost_at_gate * (.25 / omega_limit_rad_s) ** 2)


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Change only two existing reward weights; preserve the complete course."""
    if len(source) != 17 or source[0].name != 'fplc_v227_s01_workspace_alignment':
        raise ValueError('V228 requires the complete unchanged V227 course')
    stages = []
    for stage in source:
        if stage.cfg.fixed_plane_v166_physical_ball_xy_weight != 2.:
            raise ValueError('V228 preserves the V226/V227 ball XY objective')
        xy_weight, omega_weight = normalized_motion_weights(
            stage.max_recent_rms_hit_vxy,
            stage.max_recent_rapid_steady_racket_vxy_rms_m_s,
            stage.max_recent_rapid_steady_racket_omega_rms_rad_s)
        cfg = replace(stage.cfg, fixed_plane_v158_late_squared_vxy_weight=xy_weight,
                      fixed_plane_v158_late_squared_omega_weight=omega_weight)
        stages.append(replace(stage, cfg=cfg,
            name=stage.name.replace('fplc_v227_', 'fplc_v228_', 1),
            notes=stage.notes + ' V228 changes only late squared XY/omega '
            'weights2.5/2 to2.5948680958444186/2.593570971629849, '
            'equalizing per-event cost at the unchanged RMS gates. '
            'All phase eligibility, ball costs, height, control, DR, observations, '
            'PPO and graduation requirements stay unchanged.'))
    return stages


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
                            resume_curriculum_state: bool, reset_optimizer: bool,
                            reset_critic: bool, allow_obs_dim_migration: bool,
                            allow_bounded_probe_start: bool = False) -> None:
    """Pin passed S5; initial V228 learning always begins at stability lesson S1."""
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        from fixed_plane_v227_nominal_reference import validate_resume_payload as validate_s5
        if requested_start_stage != START_STAGE or allow_bounded_probe_start:
            raise ValueError('V228 must learn its S1 stability prefix first')
        validate_s5(payload,
            requested_start_stage='fplc_v227_s01_workspace_alignment',
            resume_curriculum_state=resume_curriculum_state,
            reset_optimizer=reset_optimizer, reset_critic=reset_critic,
            allow_obs_dim_migration=allow_obs_dim_migration)
        return
    if (source != PROFILE or reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ['obs_dim', 'critic_obs_dim', 'act_dim']) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V228 rejects capped/incompatible/reset continuation')
    if resume_curriculum_state:
        if requested_start_stage != payload.get('stage_name'):
            raise ValueError('V228 requires exact stage and causal history')
    else:
        from fixed_plane_completed_resume import validate_completed_stage
        validate_completed_stage(payload, next_stage=requested_start_stage)
