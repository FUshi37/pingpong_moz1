"""V266: retain relative reward prices and restore the original reward units."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import fixed_plane_v265_mild_variation as parent

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v266_value_scale'
STAGE_NAMES = tuple(s.replace('v265_', 'v266_', 1) for s in parent.STAGE_NAMES)
START_STAGE = STAGE_NAMES[6]
SOURCE = parent.SOURCE
PARENT_PROFILE = parent.PARENT_PROFILE
GLOBAL_REWARD_MULTIPLIER = .6
CHANGES = parent.CHANGES | {'fixed_plane_v167_global_reward_multiplier': GLOBAL_REWARD_MULTIPLIER}


@dataclass(frozen=True)
class ValueScaleConfig(parent.MildVariationConfig):
    """Expose the existing final reward multiplier without adding state."""
    v266_enabled: bool = True
    fixed_plane_v167_global_reward_multiplier: float = 1.


def validate_config(cfg: ValueScaleConfig) -> None:
    """Keep all parent contracts and require the registered phase-specific scale."""
    if not cfg.v266_enabled:
        raise ValueError('V266 requires its explicit reward marker')
    fields = vars(cfg).copy()
    fields.pop('v266_enabled')
    scale = fields.pop('fixed_plane_v167_global_reward_multiplier')
    inherited = parent.MildVariationConfig(**fields)
    parent.validate_config(inherited)
    late = inherited.fixed_plane_v196_late_action_variation_weight == parent.VARIATION_WEIGHT
    if scale != (GLOBAL_REWARD_MULTIPLIER if late else 1.):
        raise ValueError('V266 requires its exact early or late uniform reward units')


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Copy all eleven lessons and scale only S7+ final rewards uniformly."""
    if tuple(s.name for s in source) != parent.STAGE_NAMES:
        raise ValueError('V266 requires the complete preserved V265 course')
    result = []
    for index, (stage, name) in enumerate(zip(source, STAGE_NAMES, strict=True)):
        cfg = ValueScaleConfig(**vars(stage.cfg))
        if index >= 6:
            cfg = replace(cfg, fixed_plane_v167_global_reward_multiplier=GLOBAL_REWARD_MULTIPLIER)
        validate_config(cfg)
        result.append(replace(stage, name=name, cfg=cfg, notes=(
            'V266 retains every V265 relative reward price and applies only '
            'the existing uniform final reward multiplier0.6 at S7 onward. '
            'The common800-step frozen prefix mean becomes0.54274 versus '
            'original0.54443; this is approximate value-scale calibration, '
            'not full-episode or convergence evidence. Keep the original '
            'critic/Adam, PPO, all11 physical lessons, controls, observations '
            'and every native graduation/retention gate unchanged.')))
    return result


def validate_resume_payload(payload: Mapping[str, Any], **kwargs: Any) -> None:
    """Pin the common original state or exact own validated continuation."""
    source = payload.get('args', {}).get('curriculum_profile')
    proxy = dict(payload)
    if source == PROFILE:
        proxy['args'] = dict(payload['args'], curriculum_profile=parent.PROFILE)
        proxy['stage_name'] = payload['stage_name'].replace('v266_', 'v265_', 1)
        if not kwargs['resume_curriculum_state']:
            parent.validate_resume_payload(proxy, **(kwargs | {
                'requested_start_stage': proxy['stage_name'], 'resume_curriculum_state': True}))
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=kwargs['requested_start_stage'])
            return
    elif source != PARENT_PROFILE:
        raise ValueError('V266 rejects earlier candidate actors as parent states')
    kwargs = dict(kwargs, requested_start_stage=kwargs['requested_start_stage'].replace('v266_', 'v265_', 1))
    parent.validate_resume_payload(proxy, **kwargs)


validate_training_contract = parent.validate_training_contract
