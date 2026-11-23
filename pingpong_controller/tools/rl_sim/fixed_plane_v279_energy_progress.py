"""State-consistent startup energy progress, with unchanged physical hit gates."""
from dataclasses import dataclass, replace
import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import jax.numpy as jnp
from fixed_plane_v277_second_reserve import SecondReserveConfig

PROFILE='fixed_plane_limit_cycle_qvel_rmp_v279_energy_progress'
PARENT_PROFILE='fixed_plane_limit_cycle_qvel_rmp_v278_all_samples'
START_STAGE='fplc_v279_s11_near_y_080mm'
RETENTION_STAGE='fplc_v279_s01_workspace_alignment'
EXTRA_FIELDS=frozenset({'v279_energy_progress_weight'})


@dataclass(frozen=True)
class EnergyProgressConfig(SecondReserveConfig):
    """Default-off reward replacement; no physical or observed state extension."""
    v279_energy_progress_weight: float=0.


def validate_config(cfg: EnergyProgressConfig) -> None:
    from fixed_plane_v277_second_reserve import validate_config as previous
    previous(cfg)
    w=cfg.v279_energy_progress_weight
    if not math.isfinite(w) or not 0.<=w<=1024.:
        raise ValueError('V279 progress weight must be finite in[0,1024]')
    if w>0. and (not cfg.v225_enabled or cfg.v241_physical_apex_huber_weight<=0.):
        raise ValueError('V279 requires the preserved physical-outcome domain')


def startup_energy_correction(*,release_event,phase,incoming_z_m,incoming_vz_m_s,
        gravity_m_s2,outgoing_apex_z_m,initial_y_m,group,old_outcome_cost,
        old_weak_cost,old_preparation_cost,preparation_phase,
        physical_huber_weight,physical_huber_delta_m,progress_weight):
    """Replace hidden-ordinal startup targets with actual energy progress.

    All physical arrays have shape(N,). Only falling post-first physical
    releases in the original startup are eligible. The potential is flat in
    the unchanged1.40--1.45m steady band, and positive progress credit is
    prohibited above its ceiling. No contact, counter, sample or gate changes.
    """
    fraction=jnp.clip((jnp.asarray(initial_y_m)-.050)/.015,0.,1.)
    scale=jnp.where((jnp.asarray(group)==1)&(progress_weight>0),fraction*fraction*(3.-2.*fraction),0.)
    startup=(jnp.asarray(phase)>=1)&(jnp.asarray(phase)<=2)
    event=jnp.asarray(release_event,dtype=bool)&startup&(jnp.asarray(incoming_vz_m_s)<-.05)&(scale>0.)
    gravity=jnp.maximum(jnp.abs(jnp.asarray(gravity_m_s2)),1e-6)
    before=jnp.asarray(incoming_z_m)+jnp.asarray(incoming_vz_m_s)**2/(2.*gravity)
    after=jnp.asarray(outgoing_apex_z_m)
    lower=jnp.clip(before+.080,1.240,1.400)
    error=jnp.maximum(lower-after,0.)+jnp.maximum(after-1.450,0.)
    delta=physical_huber_delta_m
    cost=-physical_huber_weight*jnp.where(error<=delta,error**2,2.*delta*error-delta**2)
    def potential(apex):
        distance=jnp.maximum(1.400-apex,0.)+jnp.maximum(apex-1.450,0.)
        return jnp.exp(-.5*jnp.square(distance/.150))
    progress=progress_weight*(potential(after)-potential(before))
    progress=jnp.where(after>1.450,jnp.minimum(progress,0.),progress)
    event_correction=jnp.where(event,scale*(cost+progress-jnp.asarray(old_outcome_cost)-jnp.asarray(old_weak_cost)),0.)
    prepare=(jnp.asarray(preparation_phase)>=1)&(jnp.asarray(preparation_phase)<=2)
    cancellation=jnp.where(prepare,-scale*jnp.asarray(old_preparation_cost),0.)
    correction=event_correction+cancellation
    metrics=dict(event_active=event,progress_credit=jnp.where(event,scale*progress,0.),
        band_lower_m=jnp.where(event,lower,0.),incoming_apex_m=jnp.where(event,before,0.),
        net_outcome_reward=jnp.where(event,scale*(cost+progress),0.),preparation_cancellation=cancellation)
    return correction,metrics


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source)!=17 or source[10].name!='fplc_v278_s11_near_y_080mm':
        raise ValueError('V279 requires the complete17-stage V278 course')
    result=[]
    for i,stage in enumerate(source):
        cfg=EnergyProgressConfig(**vars(stage.cfg))
        if i>=10:cfg=replace(cfg,v279_energy_progress_weight=512.)
        validate_config(cfg)
        result.append(replace(stage,cfg=cfg,name=stage.name.replace('fplc_v278_','fplc_v279_',1),
            notes=('V279 reward-only hard-near startup energy progress. S11+ '
                   '50--65mm smooth activation; falling post-first early physical '
                   'releases share an incoming-apex+80mm band floor clipped1.24--1.40m, '
                   'upper1.45m, bounded512 energy-potential progress credit. Replace '
                   'old ordinal-based physical/weak costs and cancel old startup '
                   'preparation costs in this cohort. First bootstrap, nominal/mirror '
                   'and all late rewards/gates/physics/DR/reset/control remain. '
                   'All17 domains,72/393/7,1024x128 and numerical PPO unchanged; '
                   'learn every minibatch without rejection/scaling/rollback, no cap.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v278_all_samples import validate_training_contract as previous
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V279 requires unchanged nominal retention')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name':'fplc_v278_s01_workspace_alignment'}))


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V279 requires complete uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_index') != 11
                or payload.get('stage_name') != 'fplc_v278_s11_near_y_080mm'
                or payload.get('stage_update') != 105 or payload.get('step') != 3114008576
                or int(payload['opt'].t) != 769726):
            raise ValueError('V279 pins the full V278 S11u105 and fresh reward history')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V279 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V279 rejects another source or from-scratch launch')
