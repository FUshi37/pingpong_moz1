"""V246 consolidates physical apex, late damping and actual weak rebounds."""

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from fixed_plane_v245_startup_apex_margin import (
    StartupApexMarginConfig, validate_config as validate_margin_config,
)

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v246'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v245'
START_STAGE = 'fplc_v246_s08_near_y_020mm_bridge'
RETENTION_STAGE = 'fplc_v246_s01_workspace_alignment'


@dataclass(frozen=True)
class RewardConsolidationConfig(StartupApexMarginConfig):
    """Expose the existing default-off V211 reward; no state/observation changes."""

    fixed_plane_v211_late_weak_penalty_weight: float = 0.


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Change three reward weights; keep the entire physical/gate contract."""
    if (len(source) != 17
            or source[7].name != 'fplc_v245_s08_near_y_020mm_bridge'):
        raise ValueError('V246 requires the complete V245 curriculum')
    result = []
    for stage in source:
        cfg = replace(RewardConsolidationConfig(**vars(stage.cfg)),
            v241_physical_apex_huber_weight=40000.,
            v241_late_ball_xy_weight=12.,
            fixed_plane_v211_late_weak_penalty_weight=40.)
        validate_margin_config(cfg)
        if not (cfg.fixed_contact_plane_training_enabled
                and cfg.fixed_plane_v92_signed_height_enabled
                and cfg.fixed_plane_v34_resolve_weak_launch):
            raise ValueError('V246 requires the already active weak-resolution path')
        result.append(replace(stage, cfg=cfg,
            name=stage.name.replace('fplc_v245_', 'fplc_v246_', 1),
            notes='V246 changes only reward weights: physical outcome Huber '
                  '30000->40000, late ball-XY extra8->12, existing V211 '
                  'late actual weak-resolution penalty0->40. Keep startup '
                  '15/25mm reward headroom and all other rewards,17 domains, '
                  'absolute gates,conv_len1,72/393/7 and every uncapped '
                  '1024x128 PPO setting. No state/observation/control/physics '
                  'or weak-hit resolution change; latest full actor/critic/Adam.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v245_startup_apex_margin import (
        validate_training_contract as previous,
    )
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V246 retains nominal absolute validation')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name': 'fplc_v245_s01_workspace_alignment'}))


def validate_resume_payload(
    payload: Mapping[str, Any], *, requested_start_stage: str,
    resume_curriculum_state: bool, reset_optimizer: bool,
    reset_critic: bool, allow_obs_dim_migration: bool,
) -> None:
    dimensions = tuple(payload.get(k) for k in (
        'obs_dim', 'critic_obs_dim', 'act_dim'))
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or dimensions != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V246 requires full uncapped72/393/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        expected = str(payload.get('stage_name', '')).replace(
            'fplc_v245_', 'fplc_v246_', 1)
        if (resume_curriculum_state or requested_start_stage != expected
                or int(payload.get('stage_index', 0)) < 8):
            raise ValueError('V246 starts new reward statistics in current lesson')
    elif source == PROFILE:
        if (not resume_curriculum_state
                or requested_start_stage != payload.get('stage_name')):
            raise ValueError('V246 own continuation must restore exact history')
    else:
        raise ValueError('V246 rejects another source')
