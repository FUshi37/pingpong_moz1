"""Opt-in repricing of the two existing physical preimpact reward terms."""
from dataclasses import replace
from typing import Any, Mapping, Sequence

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v302_preimpact_price'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v301_launch_center'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'


def build_stages(source: Sequence[Any]) -> list[Any]:
    from fixed_plane_v301_launch_center import validate_config
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V302 requires the complete unchanged V301 course')
    result = []
    for index, stage in enumerate(source):
        cfg = stage.cfg
        if index >= 11:
            if (cfg.v294_critical_preparation_scale != .25
                    or cfg.v295_mid_near_preparation_scale != .25):
                raise ValueError('Unexpected parent preimpact reward weights')
            cfg = replace(cfg, v294_critical_preparation_scale=1.,
                          v295_mid_near_preparation_scale=1.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=(stage.notes if index < 11 else
            'V302 reward-only: matched existing physical preimpact preparation '
            'weights .25->1 for critical and mid-near support. Same target, '
            'eligibility, timing and returned-reward path. V301 centering and '
            'every other reward,17 IDs/domains,gates,72/393/7,control,physics,DR, '
            'reset and numerical1024x128 PPO/BC/KL unchanged. Latest full '
            'actor/critic/Adam/history;all32,no reset,rejection,rollback,rewind, '
            'update cap or forced advancement. Real rolling,independent and '
            'strict nominal graduation required.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v301_launch_center import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *,
        requested_start_stage: str, resume_curriculum_state: bool,
        reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(key) for key in
                     ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V302 requires full uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('global_update', 0))
        total = update-3575
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE
                or payload.get('stage_index') != 12
                or int(payload.get('stage_update', 0)) != update
                or update < 3831
                or payload.get('step') != 3641442304+total*131072
                or int(payload['opt'].t) != 898494+total*32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update
                or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V302 requires latest V301 S12 full causal state '
                             'after at least256 actual updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V302 requires exact own causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V302 rejects another source or from-scratch launch')
