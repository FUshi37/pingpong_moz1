"""Prioritize actual episode survival through one existing failure reward."""
from dataclasses import replace
from typing import Any, Mapping, Sequence

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v285_survival_objective'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v284_reference_prepare'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V285 requires the complete original17 physical lessons')
    result = []
    for i, stage in enumerate(source):
        if stage.cfg.early_termination_remaining_horizon_penalty != 4.:
            raise ValueError('V285 requires the preserved V284 failure reward')
        cfg = replace(stage.cfg, early_termination_remaining_horizon_penalty=400.) if i >= 11 else stage.cfg
        result.append(replace(stage, cfg=cfg, notes=(
            'V285 S12+ reward-only survival objective: actual early failure remaining-horizon '
            'penalty4->400; the existing cost is proportional to real unfinished episode '
            'time and is zero at the horizon/nonfailure. All other rewards, physical lessons, '
            'V281 IDs, gates, DR, observations72/393/actions7 and numerical1024x128 PPO '
            'are exact. Restore latest full state and actual physical-primary history; '
            'all minibatches, no update cap/reset/rewind/rejection or forced advance.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v284_reference_prepare import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str,Any],*,requested_start_stage: str,
        resume_curriculum_state: bool,reset_optimizer: bool,reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration or payload.get('opt') is None
            or not {'pi','v'}.issubset(payload.get('params',{}))
            or tuple(payload.get(k) for k in ('obs_dim','critic_obs_dim','act_dim'))!=(72,393,7)
            or payload.get('args',{}).get('max_stage_updates')!=-1):
        raise ValueError('V285 requires complete uncapped72/393/7 state')
    source=payload.get('args',{}).get('curriculum_profile')
    if source==PARENT_PROFILE:
        u=int(payload.get('stage_update',0));tail=payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage!=START_STAGE
                or payload.get('stage_name')!=START_STAGE or payload.get('stage_index')!=12
                or u!=120 or payload.get('step')!=3188588544 or int(payload['opt'].t)!=787934
                or not isinstance(tail,list) or not tail or int(tail[-1]['stage_update'])!=u
                or tail[-1]['stage_name']!=START_STAGE):
            raise ValueError('V285 pins latest V284 S12u120 full state and unchanged physical primary history')
    elif source==PROFILE:
        if resume_curriculum_state:
            if requested_start_stage!=payload.get('stage_name'):
                raise ValueError('V285 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload),next_stage=requested_start_stage)
    else:
        raise ValueError('V285 rejects another source or from-scratch launch')
