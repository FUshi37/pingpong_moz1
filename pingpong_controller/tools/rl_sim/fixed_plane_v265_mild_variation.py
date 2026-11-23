"""V265: a measured small dose of the existing late action-variation cost."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import fixed_plane_v264_variance_balance as parent

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v265_mild_variation'
STAGE_NAMES = tuple(s.replace('v264_', 'v265_', 1) for s in parent.STAGE_NAMES)
START_STAGE = STAGE_NAMES[6]
SOURCE = parent.SOURCE
PARENT_PROFILE = parent.PARENT_PROFILE
VARIATION_WEIGHT = .005
CHANGES = parent.CHANGES | {'fixed_plane_v196_late_action_variation_weight': VARIATION_WEIGHT}


@dataclass(frozen=True)
class MildVariationConfig(parent.VarianceBalanceConfig):
    """Extend the exact parent schema with one existing numerical reward field."""
    v265_enabled: bool = True
    fixed_plane_v196_late_action_variation_weight: float = 0.


def validate_config(cfg: MildVariationConfig) -> None:
    """Require the complete parent and its phase-specific mild reward dose."""
    if not cfg.v265_enabled:
        raise ValueError('V265 requires its explicit reward marker')
    fields = vars(cfg).copy()
    fields.pop('v265_enabled')
    weight = fields.pop('fixed_plane_v196_late_action_variation_weight')
    inherited = parent.VarianceBalanceConfig(**fields)
    parent.validate_config(inherited)
    late = {key: getattr(inherited, key) for key in parent.CHANGES} == parent.CHANGES
    if weight != (VARIATION_WEIGHT if late else 0.):
        raise ValueError('V265 requires its exact complete early or late reward dose')


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Preserve all eleven physical domains and every graduation gate."""
    if tuple(s.name for s in source) != parent.STAGE_NAMES:
        raise ValueError('V265 requires the complete preserved V264 course')
    result = []
    for index, (stage, name) in enumerate(zip(source, STAGE_NAMES, strict=True)):
        cfg = MildVariationConfig(**vars(stage.cfg))
        if index >= 6:
            cfg = replace(cfg, fixed_plane_v196_late_action_variation_weight=VARIATION_WEIGHT)
        validate_config(cfg)
        result.append(replace(stage, name=name, cfg=cfg, notes=(
            'V265 adds only the existing late action-variation reward weight0.005 '
            'to V264 S7 onward. Current-source measured executed action-delta '
            'energy4.5135 gives an illustrative22.57-point cost per1000 eligible '
            'ticks, not a new filter or control path. Preserve all11 domains, '
            'parent rewards, physics, observations, PPO and native hold32/block/'
            'retention gates. Common full V258S7u824 source; no prior candidate '
            'actor promotion and no convergence claim from this scale audit.')))
    return result


def validate_resume_payload(payload: Mapping[str, Any], **kwargs: Any) -> None:
    """Restore the common original source or exact own full-state continuity."""
    source = payload.get('args', {}).get('curriculum_profile')
    proxy = dict(payload)
    if source == PROFILE:
        proxy['args'] = dict(payload['args'], curriculum_profile=parent.PROFILE)
        proxy['stage_name'] = payload['stage_name'].replace('v265_', 'v264_', 1)
        if not kwargs['resume_curriculum_state']:
            parent.validate_resume_payload(proxy, **(kwargs | {
                'requested_start_stage': proxy['stage_name'], 'resume_curriculum_state': True}))
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=kwargs['requested_start_stage'])
            return
    elif source != PARENT_PROFILE:
        raise ValueError('V265 rejects earlier reward candidates as parent actors')
    kwargs = dict(kwargs, requested_start_stage=kwargs['requested_start_stage'].replace('v265_', 'v264_', 1))
    parent.validate_resume_payload(proxy, **kwargs)


validate_training_contract = parent.validate_training_contract
