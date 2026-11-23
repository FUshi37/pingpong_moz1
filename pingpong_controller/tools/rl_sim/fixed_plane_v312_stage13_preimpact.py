"""Default-off S13 continuation using the tested native preparation loss."""
from dataclasses import replace
from typing import Any, Mapping, Sequence
from fixed_plane_v311_clean_preimpact import CleanPreimpactConfig, validate_config

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v312_stage13_preimpact'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v310_qualified_acquisition'
START_STAGE = 'fplc_v281_s13_inward_momentum_010mms_bridge'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v311_clean_preimpact_enabled'})


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Keep S1--S12 exact; restore one existing physical cost at S13--S17."""
    if len(source) != 17 or source[12].name != START_STAGE:
        raise ValueError('V312 requires all 17 original V310 lessons')
    result = []
    for i, stage in enumerate(source):
        cfg = CleanPreimpactConfig(**vars(stage.cfg),
                                   v311_clean_preimpact_enabled=i >= 12)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=stage.notes if i < 12 else
            'V312 S13 reward-only native preimpact restoration. Reuse the '
            'tested V311 helper only: clean_support times original V304 '
            'physical loss at native previouscount1/2. Original TTC/Y '
            'envelope and SI weights; no action labels or reference loss. '
            'Preserve every other V310 reward, first12 completed lessons, '
            'all17 domains/gates,72/393/7,control,physics,DR,reset and '
            'numeric1024x128 PPO/BC/KL. Latest full S13 state/rawhistory; '
            'all32,no rejection,rollback,rewind,reset,cap or forced advance.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v310_qualified_acquisition import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *,
                            requested_start_stage: str,
                            resume_curriculum_state: bool,
                            reset_optimizer: bool, reset_critic: bool,
                            allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in (
                'obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V312 requires complete uncapped 72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('stage_update', 0))
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE
                or payload.get('stage_index') != 13 or update < 128
                or payload.get('global_update') != update
                or payload.get('step') != 3912499200+update*131072
                or int(payload['opt'].t) != 964670+update*32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update
                or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V312 requires latest complete V310 S13 after 128 real updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V312 requires exact own causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V312 rejects another source or from-scratch launch')
