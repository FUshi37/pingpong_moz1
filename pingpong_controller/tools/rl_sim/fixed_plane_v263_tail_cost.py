"""V263: strengthen physical band costs while keeping misses more costly."""
from dataclasses import dataclass,replace
from typing import Any,Mapping,Sequence

from fixed_plane_v258_gaussian_force import GaussianBoundedForceConfig
import fixed_plane_v262_correction_balance as parent

PROFILE='fixed_plane_limit_cycle_qvel_rmp_v263_tail_cost'
STAGE_NAMES=tuple(s.replace('v262_','v263_',1) for s in parent.STAGE_NAMES)
START_STAGE=STAGE_NAMES[6]
SOURCE=parent.SOURCE
PARENT_PROFILE=parent.PARENT_PROFILE
CHANGES=parent.CHANGES|{'fixed_plane_v165_steady_band_violation_weight':120.,
                      'termination_miss_penalty_base':200.}


@dataclass(frozen=True)
class TailCostConfig(GaussianBoundedForceConfig):
    """Existing numerical reward fields only; no simulator-state extension."""
    v263_enabled:bool=True
    fixed_plane_v191_absolute_height_multiplier:float=1.


def validate_config(cfg:TailCostConfig)->None:
    """Require the full early or late pair and every preserved parent check."""
    if not cfg.v263_enabled:
        raise ValueError('V263 requires its explicit reward schema')
    late={key:getattr(cfg,key) for key in parent.CHANGES}==parent.CHANGES
    pair=(cfg.fixed_plane_v165_steady_band_violation_weight,cfg.termination_miss_penalty_base)
    if pair!=((120.,200.) if late else (30.,30.)):
        raise ValueError('V263 rejects a partial band/miss reward pair')
    fields=vars(cfg).copy();fields.pop('v263_enabled')
    parent.validate_config(replace(parent.CorrectionBalanceConfig(**fields),
        fixed_plane_v165_steady_band_violation_weight=30.,termination_miss_penalty_base=30.))


def build_stages(source:Sequence[Any])->list[Any]:
    """Retain11 lessons; S7+ uses120 band cost and200 miss base."""
    if tuple(s.name for s in source)!=parent.STAGE_NAMES:
        raise ValueError('V263 requires the complete preserved V262 course')
    result=[]
    for index,(stage,name) in enumerate(zip(source,STAGE_NAMES,strict=True)):
        fields=vars(stage.cfg).copy();fields.pop('v262_enabled')
        cfg=TailCostConfig(**fields)
        if index>=6:
            cfg=replace(cfg,fixed_plane_v165_steady_band_violation_weight=120.,
                        termination_miss_penalty_base=200.)
        validate_config(cfg)
        result.append(replace(stage,name=name,cfg=cfg,notes=(
            'V263 reward-only physical band cost30->120 and ball-miss base30->200 '
            'after S6. Keep V262 height2/margin12/XY4.5/center2.25/adjacent1.5. '
            'Miss cost discourages dropping to avoid a weak-contact band loss. '
            'All11 domains,physics,control,observations,PPO and every gate '
            'remain unchanged. Resume the common full V258S7u824 source; '
            'no promotion from return or a partial window.')))
    return result


def validate_resume_payload(payload:Mapping[str,Any],**kwargs:Any)->None:
    """Use the common full parent or an exact own continuation only."""
    source=payload.get('args',{}).get('curriculum_profile')
    if source in ('fixed_plane_limit_cycle_qvel_rmp_v261_reward_balance',parent.PROFILE):
        raise ValueError('V263 rejects preceding negative candidate actors')
    proxy=dict(payload)
    if source==PROFILE:
        proxy['args']=dict(payload['args'],curriculum_profile=parent.PROFILE)
        proxy['stage_name']=payload['stage_name'].replace('v263_','v262_',1)
        if not kwargs['resume_curriculum_state']:
            parent.validate_resume_payload(proxy,**(kwargs|{
                'requested_start_stage':proxy['stage_name'],'resume_curriculum_state':True}))
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload,next_stage=kwargs['requested_start_stage'])
            return
    kwargs=dict(kwargs,requested_start_stage=kwargs['requested_start_stage'].replace('v263_','v262_',1))
    parent.validate_resume_payload(proxy,**kwargs)


validate_training_contract=parent.validate_training_contract
