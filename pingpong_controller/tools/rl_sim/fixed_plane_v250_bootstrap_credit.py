"""V250 strengthens only the existing bounded first-apex event credit."""
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v250'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v249'
START_STAGE = 'fplc_v250_s09_near_y_035mm_bridge'
RETENTION_STAGE = 'fplc_v250_s01_workspace_alignment'


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Give the existing first-apex objective four times its event weight."""
    if len(source) != 17 or source[8].name != 'fplc_v249_s09_near_y_035mm_bridge':
        raise ValueError('V250 requires the complete V249 curriculum')
    from fixed_plane_v249_bootstrap_energy import validate_config
    result = []
    for stage in source:
        cfg = replace(stage.cfg, fixed_plane_v94_first_hit_apex_reward_weight=64.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg,
            name=stage.name.replace('fplc_v249_', 'fplc_v250_', 1),
            notes='V250 changes only existing confirmed-first-apex reward '
                  'weight16->64. Keep the1.16m reward target/15mm sigma, '
                  'physical schedule1.14/1.249/1.4/1.425m,lateXYextra28 and '
                  'every other reward/domain/gate,conv_len1,72/393/7, '
                  'control/physics/DR/reset and all original uncapped1024x128 '
                  'PPO settings. Latest full249 actor/critic/Adam and genuine '
                  'graduations retained. No new state,mask or confirmation.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v249_bootstrap_energy import validate_training_contract as previous
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V250 retains nominal absolute validation')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name': 'fplc_v249_s01_workspace_alignment'}))


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
        raise ValueError('V250 requires full uncapped72/393/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        expected = str(payload.get('stage_name', '')).replace('fplc_v249_', 'fplc_v250_', 1)
        if (resume_curriculum_state or requested_start_stage != expected
                or int(payload.get('stage_index', 0)) < 9):
            raise ValueError('V250 starts new reward statistics in current lesson')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V250 own continuation requires exact history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V250 rejects another source')
