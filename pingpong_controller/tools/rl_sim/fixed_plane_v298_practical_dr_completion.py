"""Opt-in robust early-XY reward tail and disclosed modest DR tolerance."""
from dataclasses import dataclass,replace
import math
from typing import Any,Mapping,Sequence
import jax.numpy as jnp
from fixed_plane_v297_feasible_reward_gain import FeasibleGainConfig
PROFILE='fixed_plane_limit_cycle_qvel_rmp_v298_practical_dr_completion'
PARENT_PROFILE='fixed_plane_limit_cycle_qvel_rmp_v297_feasible_reward_gain'
START_STAGE='fplc_v281_s12_near_y_100mm'
RETENTION_STAGE='fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS=frozenset({'v298_early_xy_huber_transition_m_s'})
GATE_CHANGES=dict(target_mean_len_frac=.94,target_episode_truncation_rate=.91,target_mean_hits=11.5,target_hit12_rate=.86)
@dataclass(frozen=True)
class PracticalDRConfig(FeasibleGainConfig):
    v298_early_xy_huber_transition_m_s:float=0.
def robust_xy_extra(added_quadratic_cost,base_quadratic_cost,*,base_weight:float,normalizer_m_s:float,transition_m_s:float):
    """Preserve the quadratic core, use continuous monotone linear speed tails.

    Input arrays are(N,). Base cost is the exact V166 event cost before
    smooth near-side support. No sample, contact, termination or counter is
    changed. All older base physical penalties retain their original form.
    """
    if not all(math.isfinite(x) for x in (base_weight,normalizer_m_s,transition_m_s)) or base_weight<=0. or normalizer_m_s<=0. or transition_m_s<normalizer_m_s:
        raise ValueError('Invalid physical XY Huber units')
    added,base=jnp.asarray(added_quadratic_cost),jnp.asarray(base_quadratic_cost)
    if added.shape!=base.shape or added.ndim!=1:raise ValueError('XY Huber inputs require matching(N,)')
    q=jnp.maximum(-base/base_weight,0.);k=transition_m_s/normalizer_m_s
    ratio=jnp.where(q>k*k,(2.*k*jnp.sqrt(q)-k*k)/jnp.maximum(q,1e-12),1.)
    return added*ratio
def validate_config(cfg:PracticalDRConfig)->None:
    from fixed_plane_v297_feasible_reward_gain import validate_config as previous
    previous(cfg)
    t=cfg.v298_early_xy_huber_transition_m_s
    if not math.isfinite(t) or (t!=0. and not .025<=t<=.1):raise ValueError('Invalid V298 XY tail transition')
    if t>0. and (cfg.v297_feasible_reward_multiplier!=16. or cfg.v296_action_prepare_weight_per_s!=512. or cfg.fixed_plane_v166_physical_ball_xy_weight<=0.):raise ValueError('V298 requires the exact V297 reward base')
def build_stages(source:Sequence[Any])->list[Any]:
    if len(source)!=17 or source[11].name!=START_STAGE:raise ValueError('V298 requires all17 V297 lessons')
    result=[]
    for i,s in enumerate(source):
        cfg=PracticalDRConfig(**vars(s.cfg),v298_early_xy_huber_transition_m_s=.05 if i>=11 else 0.)
        validate_config(cfg)
        result.append(replace(s,cfg=cfg,**(GATE_CHANGES if i>=11 else {}),notes='V298 preserves V297 normal-speed rewards exactly through0.05m/s; only added early-XY penalties use monotone linear speed tails beyond that threshold to avoid rare quadratic return explosions. Every sample still trains; old base costs,events,counts,physics/DR/control and72/393/7 unchanged. RemainingS12--S17 absolute DR gates:len.94/full.91/hits11.5/P12.86,with unchanged apex1.37--1.41 and XY.026/.030,all safety/quality/view/coverage and strict nominal S1. Native rolling32/hold16/independent288x1200/nominal validation required. Restore latest finite full safety-stop state including every applied update/Adam/history;no reset,rewind,rejection,rollback,update cap or forced advancement. Numeric1024x128 PPO/BC/KL and loss guards unchanged.'))
    return result
def validate_training_contract(settings:Any)->None:
    from fixed_plane_v297_feasible_reward_gain import validate_training_contract as previous
    previous(settings)
def validate_resume_payload(payload:Mapping[str,Any],*,requested_start_stage:str,resume_curriculum_state:bool,reset_optimizer:bool,reset_critic:bool,allow_obs_dim_migration:bool)->None:
    if reset_optimizer or reset_critic or allow_obs_dim_migration or payload.get('opt') is None or not {'pi','v'}.issubset(payload.get('params',{})) or tuple(payload.get(k) for k in ('obs_dim','critic_obs_dim','act_dim'))!=(72,393,7) or payload.get('args',{}).get('max_stage_updates')!=-1:raise ValueError('V298 requires full uncapped72/393/7 state')
    source=payload.get('args',{}).get('curriculum_profile')
    if source==PARENT_PROFILE:
        total=int(payload.get('global_update',0))-3018;tail=payload.get('stage_history_tail')
        if not resume_curriculum_state or requested_start_stage!=START_STAGE or payload.get('stage_name')!=START_STAGE or payload.get('stage_index')!=12 or total<1 or payload.get('step')!=3568435200+total*131072 or int(payload['opt'].t)!=880670+total*32 or not isinstance(tail,list) or not tail or int(tail[-1]['stage_update'])!=int(payload['stage_update']) or tail[-1]['stage_name']!=START_STAGE:raise ValueError('V298 requires latest full V297 S12 state and exact history/Adam')
    elif source==PROFILE:
        if resume_curriculum_state:
            if requested_start_stage!=payload.get('stage_name'):raise ValueError('V298 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload),next_stage=requested_start_stage)
    else:raise ValueError('V298 rejects another source or from-scratch launch')
