"""V248 strengthens the two existing physical risk reward weights."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from fixed_plane_v247_physical_risk import validate_config

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v248'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v247'
START_STAGE = 'fplc_v248_s08_near_y_020mm_bridge'
RETENTION_STAGE = 'fplc_v248_s01_workspace_alignment'


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Triple existing outcome costs; retain every domain and gate."""
    if len(source) != 17 or source[7].name != 'fplc_v247_s08_near_y_020mm_bridge':
        raise ValueError('V248 requires the complete V247 curriculum')
    result = []
    for stage in source:
        cfg = replace(
            stage.cfg,
            v247_startup_physical_weak_penalty_weight=600.,
            v247_late_physical_band_penalty_weight=240.)
        validate_config(cfg)
        result.append(replace(
            stage, cfg=cfg,
            name=stage.name.replace('fplc_v247_', 'fplc_v248_', 1),
            notes='V248 changes only two existing physical outcome costs: '
                  'startup weak release200->600 and original resolved late '
                  'height-band violation80->240. Retain Huber40000, startup '
                  '15/25mm reward headroom and late-XY extra20. Keep all other '
                  'rewards,17 domains/gates,conv_len1,72/393/7,control,physics, '
                  'DR,reset and every original uncapped1024x128 PPO setting. '
                  'Restore latest full V247 actor/critic/Adam and genuine '
                  'prior graduations; only current reward statistics are new.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v247_physical_risk import validate_training_contract as previous
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V248 retains nominal absolute validation')
    previous(SimpleNamespace(**{
        **vars(settings),
        'advance_retention_stage_name': 'fplc_v247_s01_workspace_alignment'}))


def validate_resume_payload(
    payload: Mapping[str, Any], *, requested_start_stage: str,
    resume_curriculum_state: bool, reset_optimizer: bool,
    reset_critic: bool, allow_obs_dim_migration: bool,
) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim'))
            != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V248 requires full uncapped72/393/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        expected = str(payload.get('stage_name', '')).replace('fplc_v247_', 'fplc_v248_', 1)
        if (resume_curriculum_state or requested_start_stage != expected
                or int(payload.get('stage_index', 0)) < 8):
            raise ValueError('V248 starts new reward statistics in the current lesson')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V248 own continuation must restore exact history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V248 rejects another source')
