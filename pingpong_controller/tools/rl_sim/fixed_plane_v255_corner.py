"""V255 focuses existing preparation credit on the reproduced near-side tail."""
from dataclasses import dataclass,replace
import math
from types import SimpleNamespace
import jax.numpy as jnp

from fixed_plane_v253_preimpact import PreparationRewardConfig

PROFILE='fixed_plane_limit_cycle_qvel_rmp_v255'
START_STAGE='fplc_v255_s09_near_y_035mm_bridge'
RETENTION_STAGE='fplc_v255_s01_workspace_alignment'
REWARD_CHANGES={'v255_near_preparation_gain':7.,
                'v253_flight_xy_reward_weight_per_s':100.}


@dataclass(frozen=True)
class CornerRewardConfig(PreparationRewardConfig):
    """Default-zero reward focus; no policy input or state extension."""
    v255_near_preparation_gain: float=0.


def near_preparation_multiplier(*,initial_y_m,group,gain: float):
    """Smoothly ramp1->1+gain between the original20/35mm reset bridges.

    Only the existing near-side group is affected. Nominal, mirrored and
    <=20mm episodes retain the exact old preparation reward; >=35mm saturates.
    Every PPO sample is still learned with its original sampling/loss weight.
    """
    if not math.isfinite(gain) or not 0.<=gain<=7.:
        raise ValueError('V255 near-side reward gain must be finite in[0,7]')
    fraction=jnp.clip((jnp.asarray(initial_y_m)-.020)/.015,0.,1.)
    smooth=fraction*fraction*(3.-2.*fraction)
    return jnp.where(jnp.asarray(group)==1,1.+gain*smooth,1.)


def validate_config(cfg):
    from fixed_plane_v253_preimpact import validate_config as previous
    previous(cfg)
    gain=cfg.v255_near_preparation_gain
    if not math.isfinite(gain) or not 0.<=gain<=7.:
        raise ValueError('V255 near-side reward gain must be finite in[0,7]')
    if gain>0. and (not cfg.v225_enabled or cfg.v253_preimpact_energy_weight_per_s<=0.):
        raise ValueError('V255 gain requires the existing preparation reward')


def build_stages(source):
    if len(source)!=17 or source[8].name!='fplc_v253_s09_near_y_035mm_bridge':
        raise ValueError('V255 requires the complete17-lesson V253 course')
    result=[]
    for stage in source:
        cfg=replace(CornerRewardConfig(**vars(stage.cfg)),**REWARD_CHANGES)
        validate_config(cfg)
        result.append(replace(stage,cfg=cfg,
            name=stage.name.replace('fplc_v253_','fplc_v255_',1),
            notes='V255 reward-only: existing third-preparation cost smoothly '
            'ramps1->8 for the near-side20->35mm startup corner, unchanged '
            'for nominal/mirror/<=20mm. No sample or loss weighting change. '
            'Late bounded horizontal flight quality52->100/s restores '
            'velocity pressure lost when late event fines were reduced. '
            'Keep all17 original domains/gates,conv_len/full1,72/393/7, '
            'physics/control/reset/DR and uncapped1024x128 PPO. Continue '
            'latest full V253/own actor/critic/Adam without rollback/reset.'))
    return result


def validate_training_contract(settings):
    from fixed_plane_v253_preimpact import validate_training_contract as previous
    if settings.advance_retention_stage_name!=RETENTION_STAGE:
        raise ValueError('V255 retains nominal absolute validation')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name':'fplc_v253_s01_workspace_alignment'}))


def validate_resume_payload(payload,*,requested_start_stage,resume_curriculum_state,
                            reset_optimizer,reset_critic,allow_obs_dim_migration):
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None or not {'pi','v'}.issubset(payload.get('params',{}))
            or tuple(payload.get(k) for k in ['obs_dim','critic_obs_dim','act_dim'])!=(72,393,7)
            or payload.get('args',{}).get('max_stage_updates')!=-1):
        raise ValueError('V255 requires full uncapped72/393/7 actor/critic/Adam')
    source=payload.get('args',{}).get('curriculum_profile')
    if source=='fixed_plane_limit_cycle_qvel_rmp_v253':
        expected=str(payload.get('stage_name','')).replace('fplc_v253_','fplc_v255_',1)
        if resume_curriculum_state or requested_start_stage!=expected or int(payload.get('stage_index',0))<9:
            raise ValueError('V255 preserves current lesson and starts fresh reward statistics')
    elif source==PROFILE:
        if resume_curriculum_state:
            if requested_start_stage!=payload.get('stage_name'):
                raise ValueError('V255 restores exact causal own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload,next_stage=requested_start_stage)
    else:
        raise ValueError('V255 rejects another source or from-scratch launch')
