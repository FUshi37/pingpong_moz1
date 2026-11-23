"""V244 permits startup racket tracking and strengthens steady ball damping."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v244'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v243'
START_STAGE = 'fplc_v244_s08_near_y_020mm_bridge'
RETENTION_STAGE = 'fplc_v244_s01_workspace_alignment'


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Change only two existing, independently phase-gated reward weights."""
    if (len(source) != 17
            or source[7].name != 'fplc_v243_s08_near_y_020mm_bridge'):
        raise ValueError('V244 requires the complete V243 curriculum')
    return [replace(stage,
        name=stage.name.replace('fplc_v243_', 'fplc_v244_', 1),
        cfg=replace(stage.cfg,
            fixed_plane_v128_early_xy_motion_weight=120.,
            v241_late_ball_xy_weight=8.),
        notes='V244 changes only existing reward weights: early racket-XY '
              'motion cost840->120 before confirmed hit3; all-cohort late '
              'ball-XY extra4->8 after hit3. Keep early ball-XY damping, '
              'physical launch-apex loss, placement allowance and finite '
              'return ramp. All17 domains and absolute gates,72/393/7, '
              'physics/control/DR/reset and every numeric uncapped1024x128 '
              'PPO setting are V243-exact. Continue latest complete state; '
              'no rollback, rejected batches or historical actor veto.')
        for stage in source]


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v243_placement_allowance import (
        validate_training_contract as previous,
    )
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V244 retains nominal absolute validation')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name': 'fplc_v243_s01_workspace_alignment'}))


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
        raise ValueError('V244 requires full uncapped72/393/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        expected = str(payload.get('stage_name', '')).replace(
            'fplc_v243_', 'fplc_v244_', 1)
        if (resume_curriculum_state or requested_start_stage != expected
                or int(payload.get('stage_index', 0)) < 8):
            raise ValueError('V244 starts new reward statistics in current lesson')
    elif source == PROFILE:
        if (not resume_curriculum_state
                or requested_start_stage != payload.get('stage_name')):
            raise ValueError('V244 own continuation must restore exact history')
    else:
        raise ValueError('V244 rejects another source')
