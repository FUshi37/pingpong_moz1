"""V267: retain learned height and increase the two blocking motion prices."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import fixed_plane_v266_value_scale as parent

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v267_motion_alignment'
PARENT_PROFILE = parent.PROFILE
STAGE_NAMES = tuple(s.replace('v266_', 'v267_', 1) for s in parent.STAGE_NAMES)
START_STAGE = STAGE_NAMES[6]
SOURCE = dict(update=100, global_update=100, step=1626079232, adam_t=405924,
    sha256='3a945ea27e5ccbca61ededac54809d25147b4a8ef6807574ee498134b27345b9')
REWARD_CHANGES = {
    'fixed_plane_v158_late_squared_vxy_weight': 6.,
    'fixed_plane_v159_mean_center_penalty_weight': 4.,
}
CHANGES = parent.CHANGES | REWARD_CHANGES


@dataclass(frozen=True)
class MotionAlignmentConfig(parent.ValueScaleConfig):
    """Reward-only version marker; no new state, observation or control input."""
    v267_enabled: bool = True


def validate_config(cfg: MotionAlignmentConfig) -> None:
    """Keep the complete parent contract and require both new late prices."""
    if not cfg.v267_enabled:
        raise ValueError('V267 requires its explicit reward marker')
    pair = (cfg.fixed_plane_v158_late_squared_vxy_weight,
            cfg.fixed_plane_v159_mean_center_penalty_weight)
    if pair not in ((3., 1.5), (6., 4.)):
        raise ValueError('V267 rejects mixed motion/center reward coefficients')
    fields = vars(cfg).copy()
    fields.pop('v267_enabled')
    inherited = parent.ValueScaleConfig(**fields)
    if pair == (6., 4.):
        inherited = replace(inherited, fixed_plane_v158_late_squared_vxy_weight=4.5,
            fixed_plane_v159_mean_center_penalty_weight=3.)
    parent.validate_config(inherited)


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Retain all eleven domains and gates; increase only S7+ XY/center costs."""
    if tuple(s.name for s in source) != parent.STAGE_NAMES:
        raise ValueError('V267 requires the complete preserved V266 course')
    result = []
    for index, (stage, name) in enumerate(zip(source, STAGE_NAMES, strict=True)):
        cfg = MotionAlignmentConfig(**vars(stage.cfg))
        if index >= 6:
            cfg = replace(cfg, **REWARD_CHANGES)
        validate_config(cfg)
        result.append(replace(stage, name=name, cfg=cfg, notes=(
            'V267 experimental reward-only continuation from full V266S7u100. '
            'The source learned the height gates but has not graduated. '
            'Increase counted late XY-square4.5->6 and mean-center3->4 only. '
            'Keep height, adjacent, band/miss, mild variation and global0.6 '
            'prices; retain all11 physical lessons, control/observations/PPO '
            'and every original native rolling/hold32/block/retention gate. '
            'Restore actor/critic/Adam, start new reward history at the same S7.')))
    return result


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    """Pin one qualified partial parent, never reinterpret it as graduated."""
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 398, 7)):
        raise ValueError('V267 requires full72/398/7 actor/critic/Adam without reset')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (requested_start_stage != START_STAGE or resume_curriculum_state
                or payload.get('stage_index') != 7 or payload.get('stage_name') != parent.STAGE_NAMES[6]
                or payload.get('stage_update') != SOURCE['update']
                or payload.get('global_update') != SOURCE['global_update']
                or payload.get('step') != SOURCE['step'] or int(payload['opt'].t) != SOURCE['adam_t']):
            raise ValueError('V267 pins partial V266S7u100 and fresh same-S7 reward history')
    elif source == PROFILE:
        index = payload.get('stage_index', 0)
        if not 1 <= index <= 11 or payload.get('stage_name') != STAGE_NAMES[index - 1]:
            raise ValueError('V267 own stage identity differs')
        if resume_curriculum_state:
            if requested_start_stage != payload['stage_name']:
                raise ValueError('V267 continuation requires exact causal stage history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError('V267 rejects another source profile')


validate_training_contract = parent.validate_training_contract
