"""V268: price temporal action changes while retaining learned height/center."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import fixed_plane_v267_motion_alignment as parent

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v268_temporal_balance'
PARENT_PROFILE = parent.PROFILE
STAGE_NAMES = tuple(s.replace('v267_', 'v268_', 1) for s in parent.STAGE_NAMES)
START_STAGE = STAGE_NAMES[6]
SOURCE = dict(update=128, global_update=128, step=1642856448, adam_t=410020,
    sha256='12c5d9a29cdb151e18bed5accb3c22a4c9f54637fa46e9284a622cccead16a8f')
REWARD_CHANGES = {'fixed_plane_v196_late_action_variation_weight': .02}
CHANGES = parent.CHANGES | REWARD_CHANGES


@dataclass(frozen=True)
class TemporalBalanceConfig(parent.MotionAlignmentConfig):
    """Reward-only opt-in marker; retain the entire parent state/observation."""
    v268_enabled: bool = True


def validate_config(cfg: TemporalBalanceConfig) -> None:
    """Require the complete inherited contract and the registered late dose."""
    if not cfg.v268_enabled:
        raise ValueError('V268 requires its explicit reward marker')
    weight = cfg.fixed_plane_v196_late_action_variation_weight
    if weight not in (0., .02):
        raise ValueError('V268 rejects unregistered temporal reward weights')
    fields = vars(cfg).copy()
    fields.pop('v268_enabled')
    inherited = parent.MotionAlignmentConfig(**fields)
    if weight == .02:
        inherited = replace(inherited, fixed_plane_v196_late_action_variation_weight=.005)
    parent.validate_config(inherited)


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Keep all eleven lessons and gates; change one S7+ reward coefficient."""
    if tuple(s.name for s in source) != parent.STAGE_NAMES:
        raise ValueError('V268 requires the complete preserved V267 course')
    result = []
    for index, (stage, name) in enumerate(zip(source, STAGE_NAMES, strict=True)):
        cfg = TemporalBalanceConfig(**vars(stage.cfg))
        if index >= 6:
            cfg = replace(cfg, **REWARD_CHANGES)
        validate_config(cfg)
        result.append(replace(stage, name=name, cfg=cfg, notes=(
            'V268 experimental reward-only continuation from full terminal '
            'V267S7u128. The source has not graduated: XY RMS remains high. '
            'Increase existing post-third-hit action-variation weight.005->.02 '
            'only; keep original normalized actions, sampling, execution, '
            'height/center/XY rewards, uniform0.6, all11 physical lessons, '
            'PPO and every rolling32/hold32/block/retention gate. '
            'Restore actor/critic/Adam with fresh same-S7 reward history.')))
    return result


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    """Pin one complete ungraduated parent; never bypass native graduation."""
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 398, 7)):
        raise ValueError('V268 requires full72/398/7 actor/critic/Adam without reset')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (requested_start_stage != START_STAGE or resume_curriculum_state
                or payload.get('stage_index') != 7 or payload.get('stage_name') != parent.STAGE_NAMES[6]
                or payload.get('stage_update') != SOURCE['update']
                or payload.get('global_update') != SOURCE['global_update']
                or payload.get('step') != SOURCE['step'] or int(payload['opt'].t) != SOURCE['adam_t']):
            raise ValueError('V268 pins terminal V267S7u128 and fresh same-S7 reward history')
    elif source == PROFILE:
        index = payload.get('stage_index', 0)
        if not 1 <= index <= 11 or payload.get('stage_name') != STAGE_NAMES[index - 1]:
            raise ValueError('V268 own stage identity differs')
        if resume_curriculum_state:
            if requested_start_stage != payload['stage_name']:
                raise ValueError('V268 continuation requires exact causal stage history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError('V268 rejects another source profile')


validate_training_contract = parent.validate_training_contract
