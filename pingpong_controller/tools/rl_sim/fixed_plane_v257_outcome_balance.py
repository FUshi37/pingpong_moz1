"""V257 strengthens weak startup outcomes and bounded flight XY quality."""
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from fixed_plane_v255_corner import validate_config

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v257'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v255'
START_STAGE = 'fplc_v257_s09_near_y_035mm_bridge'
RETENTION_STAGE = 'fplc_v257_s01_workspace_alignment'
REWARD_CHANGES = {
    'v247_startup_physical_weak_penalty_weight': 160.0,
    'v253_flight_xy_reward_weight_per_s': 200.0,
}


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Preserve all17 domains and gates; change only two existing weights."""
    if len(source) != 17 or source[8].name != 'fplc_v255_s09_near_y_035mm_bridge':
        raise ValueError('V257 requires the complete17-lesson V255 curriculum')
    result = []
    for stage in source:
        cfg = replace(stage.cfg, **REWARD_CHANGES)
        validate_config(cfg)
        result.append(replace(
            stage, cfg=cfg,
            name=stage.name.replace('fplc_v255_', 'fplc_v257_', 1),
            notes=(
                'V257 reward weights only: second/third physical weak-release '
                'cost80->160; bounded post-third flight XY quality100->200/s. '
                'Retain V255 preparation gain, every physical target, first '
                'launch and late band cost, all17 domains/gates,conv_len/full1, '
                '72/393/7 and uncapped1024x128 PPO. Restore the latest full '
                'actor/critic/Adam in the current lesson. No sample rejection, '
                'rollback, observation/control/physics/reset/DR changes or '
                'historical-model advancement veto.'),
        ))
    return result


def validate_training_contract(settings: Any) -> None:
    """Use the unchanged V255 numerical training contract."""
    from fixed_plane_v255_corner import validate_training_contract as previous
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V257 requires nominal absolute validation')
    previous(SimpleNamespace(**{
        **vars(settings),
        'advance_retention_stage_name': 'fplc_v255_s01_workspace_alignment',
    }))


def validate_resume_payload(
    payload: Mapping[str, Any], *, requested_start_stage: str,
    resume_curriculum_state: bool, reset_optimizer: bool,
    reset_critic: bool, allow_obs_dim_migration: bool,
) -> None:
    """Restore full state without resetting optimizer or changing lesson."""
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(key) for key in
                     ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V257 requires full uncapped72/393/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        expected = str(payload.get('stage_name', '')).replace('fplc_v255_', 'fplc_v257_', 1)
        if (resume_curriculum_state or requested_start_stage != expected
                or int(payload.get('stage_index', 0)) < 9):
            raise ValueError('V257 preserves the lesson and starts fresh reward statistics')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V257 requires exact own convergence history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V257 rejects another parent or from-scratch launch')
