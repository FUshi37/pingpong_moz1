"""Opt-in final mean tolerance; identical V317 environment and learner."""
from dataclasses import replace
from typing import Any, Mapping, Sequence

PROFILE='fixed_plane_limit_cycle_qvel_rmp_v319_mean0265_completion'
PARENT_PROFILE='fixed_plane_limit_cycle_qvel_rmp_v318_rms031_completion'
START_STAGE='fplc_v281_s16_inward_momentum_080mms'
RETENTION_STAGE='fplc_v281_s01_workspace_alignment'


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Change exactly two stage constants, without changing any cfg field."""
    if len(source)!=17 or source[15].name!=START_STAGE:
        raise ValueError('V319 requires all17 original lessons')
    if any(source[i].max_recent_mean_hit_vxy!=.026 for i in (15,16)):
        raise ValueError('V319 requires the original .026 mean gates')
    return [replace(s,max_recent_mean_hit_vxy=.0265) if i>=15 else s
            for i,s in enumerate(source)]


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v317_late_ball_margin_gain import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str,Any], *,requested_start_stage: str,
                            resume_curriculum_state: bool,reset_optimizer: bool,
                            reset_critic: bool,allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
        or payload.get('opt') is None
        or not {'pi','v'}.issubset(payload.get('params',{}))
        or tuple(payload.get(k) for k in ('obs_dim','critic_obs_dim','act_dim'))!=(72,393,7)
        or payload.get('args',{}).get('max_stage_updates')!=-1):
        raise ValueError('V319 requires complete uncapped72/393/7 state')
    source=payload.get('args',{}).get('curriculum_profile')
    if source==PARENT_PROFILE:
        u=int(payload.get('stage_update',0));tail=payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage!=START_STAGE
            or payload.get('stage_name')!=START_STAGE or payload.get('stage_index')!=16
            or u<1191 or payload.get('global_update')!=u
            or payload.get('step')!=3958898688+u*131072
            or int(payload['opt'].t)!=975998+u*32
            or not isinstance(tail,list) or len(tail)!=96
            or [int(r['stage_update']) for r in tail]!=list(range(u-95,u+1))
            or any(r['stage_name']!=START_STAGE for r in tail)):
            raise ValueError('V319 requires latest full V318 S16 and exact raw96 history')
    elif source==PROFILE:
        if resume_curriculum_state:
            if requested_start_stage!=payload.get('stage_name'):
                raise ValueError('V319 requires exact own causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload),next_stage=requested_start_stage)
    else:
        raise ValueError('V319 rejects another source or from-scratch launch')
