"""V243 aligns startup anchor deadband with existing ball-placement support."""

from dataclasses import replace
import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v243'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v242'
START_STAGE = 'fplc_v243_s08_near_y_020mm_bridge'
RETENTION_STAGE = 'fplc_v243_s01_workspace_alignment'


def placement_deadband_m(
    *, base_deadband_m: float, local_xy_jitter_m: float,
) -> float:
    """Cover the radial support of the existing square local-UV placement."""
    if any(not math.isfinite(value) or value < 0.
           for value in (base_deadband_m, local_xy_jitter_m)):
        raise ValueError('Placement support and base deadband must be finite >=0')
    return float(base_deadband_m) + math.sqrt(2.) * float(local_xy_jitter_m)


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Change one startup reward parameter; keep every late cost and gate."""
    if (len(source) != 17
            or source[7].name != 'fplc_v242_s08_near_y_020mm_bridge'):
        raise ValueError('V243 requires the complete V242 curriculum')
    result = []
    for stage in source:
        cfg = replace(stage.cfg, early_racket_xy_anchor_deadband_m=(
            placement_deadband_m(
                base_deadband_m=stage.cfg.early_racket_xy_anchor_deadband_m,
                local_xy_jitter_m=stage.cfg.racket_launch_xy_jitter)))
        result.append(replace(stage, cfg=cfg,
            name=stage.name.replace('fplc_v242_', 'fplc_v243_', 1),
            notes='V243 changes only the existing startup anchor reward '
                  'deadband: old2mm + sqrt(2)*declared localUV ball jitter. '
                  'Zero-jitter S1 stays exact; S8 uses13.314mm. It permits '
                  'moving under the initially offset ball before hit3. '
                  'V236 count3->6 return and every late reward, all17 domains '
                  'and absolute gates,72/393/7,physics/control/DR/reset and '
                  'all numeric uncapped1024x128 PPO remain V242-exact. '
                  'Continue latest full V242actor/critic/Adam; no rollback.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v242_startup_objectives import (
        validate_training_contract as previous,
    )
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V243 retains nominal absolute validation')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name': 'fplc_v242_s01_workspace_alignment'}))


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
        raise ValueError('V243 requires full uncapped72/393/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        expected = str(payload.get('stage_name', '')).replace(
            'fplc_v242_', 'fplc_v243_', 1)
        if (resume_curriculum_state or requested_start_stage != expected
                or int(payload.get('stage_index', 0)) < 8):
            raise ValueError('V243 starts new reward statistics in current lesson')
    elif source == PROFILE:
        if (not resume_curriculum_state
                or requested_start_stage != payload.get('stage_name')):
            raise ValueError('V243 own continuation must restore exact history')
    else:
        raise ValueError('V243 rejects another source')
