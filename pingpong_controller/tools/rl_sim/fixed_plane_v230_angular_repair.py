"""V230 repairs the paired angular-speed regression with one reward weight."""
from dataclasses import replace
from typing import Any, Mapping, Sequence

from fixed_plane_v229_noninferiority import (
    PARENT_PROFILE, PARENT_SHA256, REPLAY_SHA256, validate_anchor_contract,
)

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v230'
START_STAGE = 'fplc_v230_s01_workspace_alignment'
ANGULAR_WEIGHT_MULTIPLIER = 1.25


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Keep all17 lessons and gates; raise only existing late angular loss."""
    if len(source) != 17 or source[0].name != 'fplc_v229_s01_workspace_alignment':
        raise ValueError('V230 requires the complete preserved V229 course')
    return [replace(stage,
        cfg=replace(stage.cfg, fixed_plane_v158_late_squared_omega_weight=(
            stage.cfg.fixed_plane_v158_late_squared_omega_weight * ANGULAR_WEIGHT_MULTIPLIER)),
        name=stage.name.replace('fplc_v229_', 'fplc_v230_', 1),
        notes=stage.notes + ' V230 raises only late physical full-angular squared '
        'event weight by25%, from2.593570971629849 to3.2419637145373116. '
        'Keep the same event population/scales and every other reward, '
        'gate, domain, control, observation, reference and PPO value. '
        'Failed V229 is diagnostic evidence, never a parent or promoted model.')
        for stage in source]


def validate_paired_stability_contract(settings: Any) -> None:
    """Retain mandatory blocking same-domain/S5 comparisons, with actual CLI names."""
    if (settings.advance_mode != 'converged'
            or settings.advance_validation_mode != 'block'
            or settings.advance_retention_stage_name != START_STAGE
            or settings.advance_retention_final_stage_name not in (None, '')
            or settings.advance_eval_deterministic
            or settings.advance_eval_n_envs != 288
            or settings.advance_eval_steps != 1200):
        raise ValueError('V230 requires blocking288x1200 matched nominal S5 retention')


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
                            resume_curriculum_state: bool, reset_optimizer: bool,
                            reset_critic: bool, allow_obs_dim_migration: bool,
                            allow_bounded_probe_start: bool = False) -> None:
    """Start from passed S5 and reject failed/capped predecessor policies."""
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        from fixed_plane_v229_noninferiority import validate_resume_payload as validate_s5
        if requested_start_stage != START_STAGE or allow_bounded_probe_start:
            raise ValueError('V230 initial continuation must start at S1')
        validate_s5(payload, requested_start_stage='fplc_v229_s01_workspace_alignment',
            resume_curriculum_state=resume_curriculum_state, reset_optimizer=reset_optimizer,
            reset_critic=reset_critic, allow_obs_dim_migration=allow_obs_dim_migration)
        return
    if (source != PROFILE or reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ['obs_dim', 'critic_obs_dim', 'act_dim']) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V230 rejects capped/incompatible/failed/reset continuation')
    if resume_curriculum_state:
        if requested_start_stage != payload.get('stage_name'):
            raise ValueError('V230 requires exact stage and causal history')
    else:
        from fixed_plane_completed_resume import validate_completed_stage
        validate_completed_stage(payload, next_stage=requested_start_stage)
