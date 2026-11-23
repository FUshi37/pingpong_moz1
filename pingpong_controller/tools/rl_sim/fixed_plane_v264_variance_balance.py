"""V264: allow small height corrections while tightening tails and contact center."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from fixed_plane_v258_gaussian_force import GaussianBoundedForceConfig
import fixed_plane_v261_reward_balance as source_contract
import fixed_plane_v263_tail_cost as parent

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v264_variance_balance'
STAGE_NAMES = tuple(s.replace('v263_', 'v264_', 1) for s in parent.STAGE_NAMES)
START_STAGE = STAGE_NAMES[6]
SOURCE = source_contract.SOURCE
PARENT_PROFILE = source_contract.PARENT_PROFILE
CHANGES = parent.CHANGES | {
    'fixed_plane_v191_margin_weight': 24.,
    'fixed_plane_v159_mean_center_penalty_weight': 3.,
    'fixed_plane_v156_adjacent_height_penalty_multiplier': .75,
}
EARLY_VALUES = source_contract.PARENT_VALUES | {
    'fixed_plane_v156_adjacent_height_penalty_multiplier': 3.,
    'fixed_plane_v165_steady_band_violation_weight': 30.,
    'termination_miss_penalty_base': 30.,
}
PARENT_REPAIR = {key: parent.CHANGES[key] for key in (
    'fixed_plane_v191_margin_weight',
    'fixed_plane_v159_mean_center_penalty_weight',
    'fixed_plane_v156_adjacent_height_penalty_multiplier')}


@dataclass(frozen=True)
class VarianceBalanceConfig(GaussianBoundedForceConfig):
    """Explicit reward schema; no extra simulator state or policy inputs."""
    v264_enabled: bool = True
    fixed_plane_v191_absolute_height_multiplier: float = 1.


def validate_config(cfg: VarianceBalanceConfig) -> None:
    """Accept complete early or late coefficients and every inherited contract."""
    if not cfg.v264_enabled:
        raise ValueError('V264 requires its explicit reward marker')
    values = {key: getattr(cfg, key) for key in CHANGES}
    if values not in (EARLY_VALUES, CHANGES):
        raise ValueError('V264 rejects partial or mixed reward coefficients')
    fields = vars(cfg).copy()
    fields.pop('v264_enabled')
    inherited = parent.TailCostConfig(**fields)
    if values == CHANGES:
        inherited = replace(inherited, **PARENT_REPAIR)
    parent.validate_config(inherited)


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Retain the complete eleven lessons, including all physical gates."""
    if tuple(s.name for s in source) != parent.STAGE_NAMES:
        raise ValueError('V264 requires the complete preserved V263 course')
    result = []
    for index, (stage, name) in enumerate(zip(source, STAGE_NAMES, strict=True)):
        fields = vars(stage.cfg).copy()
        fields.pop('v263_enabled')
        cfg = VarianceBalanceConfig(**fields)
        if index >= 6:
            cfg = replace(cfg, **CHANGES)
        validate_config(cfg)
        result.append(replace(stage, name=name, cfg=cfg, notes=(
            'V264 reward-only candidate from the common full V258S7u824 source. '
            'Retain V263 coefficients except adjacent1.5->0.75, bilateral '
            'height-margin12->24 and mean-center2.25->3. A2.5mm immediate '
            'height correction gains0.96 versus0.9 adjacent cost. This identity '
            'is a tuning hypothesis, not convergence evidence. Preserve all11 '
            'physical domains, observation/action/control contracts, PPO and '
            'every rolling, hold32, block and retention graduation gate.')))
    return result


def validate_resume_payload(payload: Mapping[str, Any], **kwargs: Any) -> None:
    """Pin the common original source or exact own full-state continuation."""
    proxy = dict(payload)
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PROFILE:
        proxy['args'] = dict(payload['args'], curriculum_profile=source_contract.PROFILE)
        proxy['stage_name'] = payload['stage_name'].replace('v264_', 'v261_', 1)
        if not kwargs['resume_curriculum_state']:
            source_contract.validate_resume_payload(proxy, **(kwargs | {
                'requested_start_stage': proxy['stage_name'], 'resume_curriculum_state': True}))
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=kwargs['requested_start_stage'])
            return
    elif source != PARENT_PROFILE:
        raise ValueError('V264 rejects preceding reward candidates as parent actors')
    kwargs = dict(kwargs, requested_start_stage=kwargs['requested_start_stage'].replace('v264_', 'v261_', 1))
    source_contract.validate_resume_payload(proxy, **kwargs)


validate_training_contract = source_contract.validate_training_contract
