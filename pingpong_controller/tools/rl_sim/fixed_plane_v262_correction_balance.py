"""V262: lower adjacent-height cost to permit small center-height corrections."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from fixed_plane_v258_gaussian_force import GaussianBoundedForceConfig
import fixed_plane_v261_reward_balance as parent

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v262_correction_balance'
STAGE_NAMES = tuple(s.replace('v261_', 'v262_', 1) for s in parent.STAGE_NAMES)
START_STAGE = STAGE_NAMES[6]
SOURCE = parent.SOURCE
PARENT_PROFILE = parent.PARENT_PROFILE
CHANGES = parent.CHANGES | {'fixed_plane_v156_adjacent_height_penalty_multiplier':1.5}


@dataclass(frozen=True)
class CorrectionBalanceConfig(GaussianBoundedForceConfig):
    """Reward-only metadata; no actor, critic, physical or history extension."""
    v262_enabled: bool = True
    fixed_plane_v191_absolute_height_multiplier: float = 1.


def validate_config(cfg: CorrectionBalanceConfig) -> None:
    """Accept only native early lessons or the complete correction candidate."""
    if not cfg.v262_enabled:
        raise ValueError('V262 requires its explicit reward schema')
    values = {key:getattr(cfg,key) for key in parent.CHANGES}
    adjacent = cfg.fixed_plane_v156_adjacent_height_penalty_multiplier
    if (values == parent.PARENT_VALUES and adjacent != 3.
            or values == parent.CHANGES and adjacent != 1.5):
        raise ValueError('V262 rejects mixed adjacent-height reward coefficients')
    source = vars(cfg).copy()
    source.pop('v262_enabled')
    parent.validate_config(replace(parent.RewardBalanceConfig(**source),
        fixed_plane_v156_adjacent_height_penalty_multiplier=3.))


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Preserve11 V261 lessons; halve only S7+ adjacent-height weight."""
    if tuple(s.name for s in source) != parent.STAGE_NAMES:
        raise ValueError('V262 requires the complete preserved V261 course')
    result = []
    for index,(stage,name) in enumerate(zip(source,STAGE_NAMES,strict=True)):
        fields = vars(stage.cfg).copy()
        fields.pop('v261_enabled')
        cfg = CorrectionBalanceConfig(**fields)
        if index >= 6:
            cfg = replace(cfg,fixed_plane_v156_adjacent_height_penalty_multiplier=1.5)
        validate_config(cfg)
        result.append(replace(stage,name=name,cfg=cfg,notes=(
            'V262 reward-only ablation from the same stopped V258S7u824. '
            'Keep V261 absolute-height2/margin12/XY4.5/center2.25; reduce '
            'late adjacent-height multiplier3->1.5. Five-mm immediate '
            'height correction gains3.84 versus3.6 adjacent cost (was7.2). '
            'Physical height/jump telemetry and every graduation gate remain '
            'unchanged; no promotion or convergence claim from this identity.')))
    return result


def validate_resume_payload(payload: Mapping[str, Any], **kwargs: Any) -> None:
    """Keep the identical stopped source; never reuse the negative V261 actor."""
    source = payload.get('args',{}).get('curriculum_profile')
    if source == parent.PROFILE:
        raise ValueError('V262 starts from the common V258 source, not the V261 actor')
    proxy = dict(payload)
    if source == PROFILE:
        proxy['args'] = dict(payload['args'],curriculum_profile=parent.PROFILE)
        proxy['stage_name'] = payload['stage_name'].replace('v262_','v261_',1)
        if not kwargs['resume_curriculum_state']:
            parent.validate_resume_payload(proxy,**(kwargs | {
                'requested_start_stage':proxy['stage_name'],
                'resume_curriculum_state':True}))
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload,next_stage=kwargs['requested_start_stage'])
            return
    kwargs = dict(kwargs,requested_start_stage=kwargs['requested_start_stage'].replace('v262_','v261_',1))
    parent.validate_resume_payload(proxy,**kwargs)


validate_training_contract = parent.validate_training_contract
