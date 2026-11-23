"""Reward earlier observable reference preparation in weak near-side startup."""
from dataclasses import dataclass,replace
import math
from typing import Any,Mapping,Sequence
import jax.numpy as jnp
from fixed_plane_v279_energy_progress import EnergyProgressConfig

PROFILE='fixed_plane_limit_cycle_qvel_rmp_v284_reference_prepare'
PARENT_PROFILE='fixed_plane_limit_cycle_qvel_rmp_v283_primary_completion'
START_STAGE='fplc_v281_s12_near_y_100mm'
RETENTION_STAGE='fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS=frozenset({'v284_reference_prepare_weight_per_s'})

@dataclass(frozen=True)
class ReferencePreparationConfig(EnergyProgressConfig):
    """Default-off reward only; no new plant, observation or runtime state."""
    v284_reference_prepare_weight_per_s: float=0.


def validate_config(cfg: ReferencePreparationConfig) -> None:
    from fixed_plane_v279_energy_progress import validate_config as previous
    previous(cfg)
    w=cfg.v284_reference_prepare_weight_per_s
    if not math.isfinite(w) or not 0.<=w<=1000.:
        raise ValueError('V284 reference reward rate must be finite in[0,1000]/s')
    if w>0. and (not cfg.v225_enabled or cfg.v279_energy_progress_weight<=0.):
        raise ValueError('V284 requires the preserved near-side physical energy course')


def preparation_penalty(*,previous_hits,pending,in_contact,ball_z_m,racket_z_m,
        ball_vz_m_s,racket_vz_m_s,gravity_m_s2,ball_radius_m,reference_z_lead_m,
        initial_y_m,group,weight_per_s,dt_s):
    """Dense (N,) weak-flight intent cost, before the delayed plant response.

    The target reference-Z lead band is30--60mm during50--180ms moving-racket
    ballistic TTC. Physical first bootstrap, all nominal/mirror and late
    behavior are unchanged. This is a negative preparation cost, never a
    physical hit or survival credit. Outcome/height/XY/safety gates remain.
    """
    g=jnp.maximum(jnp.abs(jnp.asarray(gravity_m_s2)),1e-6)
    bv=jnp.asarray(ball_vz_m_s);rv=jnp.asarray(racket_vz_m_s)
    gap=jnp.asarray(ball_z_m)-jnp.asarray(racket_z_m)-jnp.asarray(ball_radius_m)
    relative=bv-rv
    ttc=(relative+jnp.sqrt(jnp.maximum(relative**2+2.*g*jnp.maximum(gap,0.),0.)))/g
    incoming=jnp.asarray(ball_z_m)+bv**2/(2.*g)
    weak=jnp.clip((1.380-incoming)/.080,0.,1.)
    fraction=jnp.clip((jnp.asarray(initial_y_m)-.050)/.015,0.,1.)
    scale=fraction*fraction*(3.-2.*fraction)
    active=((jnp.asarray(group)==1)&(jnp.asarray(previous_hits)>=1)
            &(jnp.asarray(previous_hits)<=2)&(~jnp.asarray(pending,dtype=bool))
            &(~jnp.asarray(in_contact,dtype=bool))&(bv<-.05)&(gap>0.)
            &(ttc>=.050)&(ttc<=.180)&(weak>0.)&(scale>0.)&(weight_per_s>0.))
    lead=jnp.asarray(reference_z_lead_m)
    error=(jnp.maximum(.030-lead,0.)+jnp.maximum(lead-.060,0.))/.020
    cost=-weight_per_s*dt_s*scale*weak*(jnp.sqrt(1.+error**2)-1.)
    metrics=dict(active=active,reference_z_lead_m=jnp.where(active,lead,0.),
        ttc_s=jnp.where(active,ttc,0.),incoming_apex_m=jnp.where(active,incoming,0.),
        weak_fraction=jnp.where(active,weak,0.))
    return jnp.where(active,cost,0.),metrics


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source)!=17 or source[11].name!=START_STAGE:
        raise ValueError('V284 requires the complete identical17-lesson V283 course')
    result=[]
    for i,s in enumerate(source):
        cfg=ReferencePreparationConfig(**vars(s.cfg),v284_reference_prepare_weight_per_s=400. if i>=11 else 0.)
        validate_config(cfg)
        result.append(replace(s,cfg=cfg,notes=('V284 S12+ reward-only weak near startup reference preparation: '
            '400/s pseudo-Huber cost outside30--60mm integrated reference-Z lead during '
            '50--180ms moving-racket TTC, falling pre-third low-energy flight, smooth '
            'Y50--65mm activation. Incoming apex>=1.38m, first bootstrap, nominal/mirror '
            'and late phases unchanged. No control residual or commanded hit credit. '
            'Every17 physical lesson/primary gate/V281 ID,72/393/7,physics/reset/DR '
            'and numerical1024x128 PPO remain. Restore latest full state and actual '
            'physical primary history; old reward rows retain provenance. Fresh native/'
            'nominal validation,all minibatches,no update cap or forced advancement.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v283_primary_completion import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str,Any],*,requested_start_stage: str,
        resume_curriculum_state: bool,reset_optimizer: bool,reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration or payload.get('opt') is None
            or not {'pi','v'}.issubset(payload.get('params',{}))
            or tuple(payload.get(k) for k in ('obs_dim','critic_obs_dim','act_dim'))!=(72,393,7)
            or payload.get('args',{}).get('max_stage_updates')!=-1):
        raise ValueError('V284 requires complete uncapped72/393/7 state')
    source=payload.get('args',{}).get('curriculum_profile')
    if source==PARENT_PROFILE:
        u=int(payload.get('stage_update',0));tail=payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage!=START_STAGE
                or payload.get('stage_name')!=START_STAGE or payload.get('stage_index')!=12
                or u!=50 or payload.get('step')!=3179413504 or int(payload['opt'].t)!=785694
                or not isinstance(tail,list) or not tail or int(tail[-1]['stage_update'])!=u
                or tail[-1]['stage_name']!=START_STAGE):
            raise ValueError('V284 pins latest V283 S12u50 full state and unchanged physical primary history')
    elif source==PROFILE:
        if resume_curriculum_state:
            if requested_start_stage!=payload.get('stage_name'):
                raise ValueError('V284 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload),next_stage=requested_start_stage)
    else:
        raise ValueError('V284 rejects another source or from-scratch launch')
