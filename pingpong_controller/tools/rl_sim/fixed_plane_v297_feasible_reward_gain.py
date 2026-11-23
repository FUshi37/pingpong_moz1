"""Default-off matched gain on V296's two existing reward terms."""
from dataclasses import dataclass, replace
import math
from typing import Any,Mapping,Sequence
import jax.numpy as jnp
from fixed_plane_v296_feasible_startup import FeasibleStartupConfig
PROFILE='fixed_plane_limit_cycle_qvel_rmp_v297_feasible_reward_gain'
PARENT_PROFILE='fixed_plane_limit_cycle_qvel_rmp_v296_feasible_startup'
START_STAGE='fplc_v281_s12_near_y_100mm'
RETENTION_STAGE='fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS=frozenset({'v297_feasible_reward_multiplier'})
@dataclass(frozen=True)
class FeasibleGainConfig(FeasibleStartupConfig):
    v297_feasible_reward_multiplier:float=1.
def extra_reward(action_cost,xy_cost,*,multiplier:float):
    if not math.isfinite(multiplier) or not 1.<=multiplier<=32.:
        raise ValueError('V297 reward multiplier must be finite in[1,32]')
    a,b=jnp.asarray(action_cost),jnp.asarray(xy_cost)
    if a.shape!=b.shape or a.ndim!=1:raise ValueError('V297 costs require matching(N,)')
    return (multiplier-1.)*a,(multiplier-1.)*b
def validate_config(cfg:FeasibleGainConfig)->None:
    from fixed_plane_v296_feasible_startup import validate_config as previous
    previous(cfg)
    gain=cfg.v297_feasible_reward_multiplier
    if not math.isfinite(gain) or not 1.<=gain<=32.:
        raise ValueError('Invalid V297 reward multiplier')
    if gain>1. and (cfg.v296_action_prepare_weight_per_s!=512. or cfg.v296_early_xy_extra_weight!=cfg.v241_late_ball_xy_weight or cfg.v296_early_xy_extra_weight<=0.):
        raise ValueError('V297 only scales the original coupled V296 rewards')
def build_stages(source:Sequence[Any])->list[Any]:
    if len(source)!=17 or source[11].name!=START_STAGE:raise ValueError('V297 requires all17 V296 lessons')
    result=[]
    for i,stage in enumerate(source):
        cfg=FeasibleGainConfig(**vars(stage.cfg),v297_feasible_reward_multiplier=16. if i>=11 else 1.)
        validate_config(cfg)
        result.append(replace(stage,cfg=cfg,notes='V297 reward-only matched16x gain on existing V296 startup-label and early physical-XY costs. Effective8192/s and128 event-extra from original512/s and8; preserve their ratio, exact eligibility/targets and every other reward. First11 neutral. Original all17 domains/gates,72/393/7,control/physics/DR and all numeric1024x128 PPO/BC/KL unchanged. Restore latest full actor/critic/Adam/primary history; no rewind/reset/rejection/rollback/update cap or forced advancement. Gain rationale is weak observed label response, not proven learned graduation.'))
    return result
def validate_training_contract(settings:Any)->None:
    from fixed_plane_v296_feasible_startup import validate_training_contract as previous
    previous(settings)
def validate_resume_payload(payload:Mapping[str,Any],*,requested_start_stage:str,resume_curriculum_state:bool,reset_optimizer:bool,reset_critic:bool,allow_obs_dim_migration:bool)->None:
    if reset_optimizer or reset_critic or allow_obs_dim_migration or payload.get('opt') is None or not {'pi','v'}.issubset(payload.get('params',{})) or tuple(payload.get(k) for k in ('obs_dim','critic_obs_dim','act_dim'))!=(72,393,7) or payload.get('args',{}).get('max_stage_updates')!=-1:
        raise ValueError('V297 requires full uncapped72/393/7 state')
    source=payload.get('args',{}).get('curriculum_profile')
    if source==PARENT_PROFILE:
        total=int(payload.get('global_update',0))-2919;tail=payload.get('stage_history_tail')
        if not resume_curriculum_state or requested_start_stage!=START_STAGE or payload.get('stage_name')!=START_STAGE or payload.get('stage_index')!=12 or total<1 or payload.get('step')!=3555459072+total*131072 or int(payload['opt'].t)!=877502+total*32 or not isinstance(tail,list) or not tail or int(tail[-1]['stage_update'])!=int(payload['stage_update']) or tail[-1]['stage_name']!=START_STAGE:
            raise ValueError('V297 requires latest full V296 S12 state and exact history/Adam')
    elif source==PROFILE:
        if resume_curriculum_state:
            if requested_start_stage!=payload.get('stage_name'):raise ValueError('V297 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload),next_stage=requested_start_stage)
    else:raise ValueError('V297 rejects another source or from-scratch launch')
