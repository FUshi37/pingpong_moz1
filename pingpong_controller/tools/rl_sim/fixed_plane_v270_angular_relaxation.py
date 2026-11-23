"""V270: retain learned XY quietness while reducing one competing angular price."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import fixed_plane_v269_conditional_quality as parent

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v270_angular_relaxation'
PARENT_PROFILE = parent.PROFILE
STAGE_NAMES = tuple(s.replace('v269_', 'v270_', 1) for s in parent.STAGE_NAMES)
START_STAGE = STAGE_NAMES[6]
SOURCE = dict(update=71, global_update=71, step=1652162560, adam_t=412234,
    sha256='1c5279d9b034d141cab09b86b6451b549cc8e006857a09ef7f3ed42cecb620e4')
REWARD_CHANGES = {'fixed_plane_v168_quality_weights': (6., 24., 6., 0., 0.)}
CHANGES = parent.CHANGES | REWARD_CHANGES


@dataclass(frozen=True)
class AngularRelaxationConfig(parent.ConditionalQualityConfig):
    """One existing reward coefficient, no new physical state or observation."""
    v270_enabled: bool = True


def validate_config(cfg: AngularRelaxationConfig) -> None:
    """Require the exact early or late angular dose and all inherited checks."""
    if cfg.v270_enabled is not True:
        raise ValueError('V270 requires its explicit reward marker')
    weights = tuple(cfg.fixed_plane_v168_quality_weights)
    if weights not in ((0.,) * 5, REWARD_CHANGES['fixed_plane_v168_quality_weights']):
        raise ValueError('V270 rejects mixed conditional-quality coefficients')
    fields = vars(cfg).copy()
    fields.pop('v270_enabled')
    inherited = parent.ConditionalQualityConfig(**fields)
    if weights != (0.,) * 5:
        inherited = replace(inherited, **{
            'fixed_plane_v168_quality_weights': parent.REWARD_CHANGES['fixed_plane_v168_quality_weights']})
    parent.validate_config(inherited)


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Keep all eleven lessons; quarter only the added S7+ angular price."""
    if tuple(s.name for s in source) != parent.STAGE_NAMES:
        raise ValueError('V270 requires the complete preserved V269 course')
    result = []
    for index, (stage, name) in enumerate(zip(source, STAGE_NAMES, strict=True)):
        cfg = AngularRelaxationConfig(**vars(stage.cfg))
        if index >= 6:
            cfg = replace(cfg, **REWARD_CHANGES)
        validate_config(cfg)
        result.append(replace(stage, name=name, cfg=cfg, notes=(
            'V270 experimental same-S7 reward continuation from full V269u71. '
            'The partial source learned XY quietness but failed height gates; '
            'it has never graduated. Reduce only conditional angular24->6 '
            'to test the height/reorientation tradeoff. Keep conditional '
            'center6/XY24, baseline angular2/center4, fixed references and '
            'all other rewards,11 domains, physics/control/observations/PPO '
            'and original rolling32/hold32/block/S1-retention gates. Restore '
            'full actor/critic/Adam with fresh same-S7 reward history.')))
    return result


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    """Pin the complete partial source and prohibit forced stage advancement."""
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 398, 7)):
        raise ValueError('V270 requires full72/398/7 actor/critic/Adam without reset')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (requested_start_stage != START_STAGE or resume_curriculum_state
                or payload.get('stage_index') != 7 or payload.get('stage_name') != parent.STAGE_NAMES[6]
                or payload.get('stage_update') != SOURCE['update']
                or payload.get('global_update') != SOURCE['global_update']
                or payload.get('step') != SOURCE['step'] or int(payload['opt'].t) != SOURCE['adam_t']):
            raise ValueError('V270 pins partial V269S7u71 and fresh same-S7 reward history')
    elif source == PROFILE:
        index = payload.get('stage_index', 0)
        if not 1 <= index <= 11 or payload.get('stage_name') != STAGE_NAMES[index - 1]:
            raise ValueError('V270 own stage identity differs')
        if resume_curriculum_state:
            if requested_start_stage != payload['stage_name']:
                raise ValueError('V270 continuation requires exact causal stage history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError('V270 rejects another source profile')


validate_training_contract = parent.validate_training_contract
