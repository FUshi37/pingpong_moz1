"""Opt-in full-cycle racket-local return forecast used only for scalar rewards."""
from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any

from fixed_plane_wind_dynamics_guidance import DRAG_FACTOR_KG_M

MODE = 'clean10_dynamics_reference_center_negy_v9'
BIRTH_MODE = 'clean10_dynamics_reference_center_negy_v7'
BIRTH_SHA256 = '32e922ae6527edd83f3091ea1fdf86701c4519fcb24228552a39361eced25f86'
HORIZON_S = 1.2
SAMPLES = 64


def cycle_contract() -> dict:
    """Keep forecast costs separate from every native gate/reference cache."""
    return dict(revision='native_plane_cycle_release_precision_v1', horizon_s=HORIZON_S,
        samples=SAMPLES, midpoint_picard_iterations=5,
        return_geometry='next descending crossing of current periodic racket plane; native local sphere-center XY',
        apex_target_m=1.425, return_sigma_m={'325': .0055, '326': .0065}, apex_sigma_m=.008,
        cost_per_component=4., huber_cap_per_component=8.,
        wind='two interleaved unchanged32-point causal forecasts; no future random samples',
        source=dict(sha256=BIRTH_SHA256,update=14600,true_step=3523346432,Adam_t=969775,stage_index=15),
        phase='retain mature preview/balance clock8192 with birth14500; costs enabled after original shelter/ramp',
        preserved='old guidance, all physical/gate telemetry and caches, observations,control,physics,PPO,sampling,courses')


@lru_cache(maxsize=16)
def _config_type(parent: type) -> type:
    return make_dataclass('WindCycleReleaseConfig', [
        ('wind_cycle_release_precision', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_cycle_config(cfg: Any, enabled: bool) -> Any:
    """Add one default-off reward flag; never combine two release recipes."""
    if not enabled:
        return cfg
    if (not getattr(cfg,'wind_precontact_center',False)
            or not getattr(cfg,'wind_precontact_balance',False)
            or getattr(cfg,'wind_release_precision',False)):
        raise ValueError('Cycle release requires its distinct balanced-preview recipe')
    return _config_type(type(cfg))(**vars(cfg))


def cycle_wind_samples(state: Any, cfg: Any, now_s: Any, *, xp: Any) -> Any:
    """Return [N,64,3] causal samples without changing the old helper/globals."""
    from fixed_plane_wind_dynamics_guidance import expected_wind_samples
    duration=xp.full_like(now_s,HORIZON_S)
    quarter=duration/(4.*32.)
    left=expected_wind_samples(state,cfg,now_s,duration,lead_s=-quarter,xp=xp)
    right=expected_wind_samples(state,cfg,now_s,duration,lead_s=quarter,xp=xp)
    return xp.stack((left,right),axis=-2).reshape((*left.shape[:-2],SAMPLES,3))


def forecast_cycle(start_m: Any, velocity_m_s: Any, racket_pos_m: Any,
                   racket_rotation: Any, plane_offset_m: Any, wind_m_s: Any,
                   mass_kg: Any, gravity_m_s2: Any, *, xp: Any,
                   drag_factor_kg_m: float = DRAG_FACTOR_KG_M,
                   force_limit_n: float = .020) -> dict:
    """Predict native local miss/apex/return time [N], not a policy observation.

    Position/velocity inputs are [N,3], rotation [N,3,3], wind [N,64,3], and
    physical parameters [N]. The periodic plane uses the current racket pose,
    rather than extrapolating an upward stroke over an entire flight.
    """
    from fixed_plane_wind_dynamics_guidance import _loop
    if wind_m_s.shape[-2:] != (SAMPLES,3):
        raise ValueError('Cycle forecast requires64 causal 3-D wind samples')
    dtype=start_m.dtype
    dt=HORIZON_S/SAMPLES
    normal=racket_rotation[..., :,2]
    g=xp.broadcast_to(xp.asarray(gravity_m_s2,dtype=dtype),start_m.shape[:-1])
    mass=xp.broadcast_to(xp.asarray(mass_kg,dtype=dtype),start_m.shape[:-1])
    gravity=xp.stack((xp.zeros_like(g),xp.zeros_like(g),-g),axis=-1)

    def acceleration(velocity):
        relative=velocity-wind_m_s
        force=drag_factor_kg_m*(xp.linalg.norm(velocity,axis=-1,keepdims=True)*velocity
            -xp.linalg.norm(relative,axis=-1,keepdims=True)*relative)
        force=xp.where(xp.all(wind_m_s==0.,axis=-1,keepdims=True),0.,force)
        force=force*xp.minimum(1.,force_limit_n/xp.maximum(xp.linalg.norm(force,axis=-1,keepdims=True),1e-12))
        return force/xp.maximum(mass[...,None,None],1e-6)+gravity[...,None,:]

    def refine(_,velocity):
        acc=acceleration(velocity)
        return velocity_m_s[...,None,:]+dt*(xp.cumsum(acc,axis=-2)-.5*acc)

    mid=_loop(5,refine,xp.broadcast_to(velocity_m_s[...,None,:],wind_m_s.shape),xp)
    acc=acceleration(mid)
    prefix=xp.cumsum(acc,axis=-2)
    times=xp.arange(1,SAMPLES+1,dtype=dtype)*dt
    positions=start_m[...,None,:]+velocity_m_s[...,None,:]*times[:,None]+dt**2*(xp.cumsum(prefix,axis=-2)-.5*prefix)
    velocities=velocity_m_s[...,None,:]+dt*prefix
    positions=xp.concatenate((start_m[...,None,:],positions),axis=-2)
    velocities=xp.concatenate((velocity_m_s[...,None,:],velocities),axis=-2)
    gaps=xp.sum((positions-racket_pos_m[...,None,:])*normal[...,None,:],axis=-1)-plane_offset_m[...,None]
    crossing=(gaps[...,:-1]>0.)&(gaps[...,1:]<=0.)
    index=xp.argmax(crossing,axis=-1)

    def vector(values):
        return xp.take_along_axis(values,index[...,None,None],axis=-2)[...,0,:]

    gap=xp.take_along_axis(gaps,index[...,None],axis=-1)[...,0]
    v0=vector(velocities);p0=vector(positions);a0=vector(acc)
    vn=xp.sum(v0*normal,axis=-1);an=xp.sum(a0*normal,axis=-1)
    discriminant=xp.maximum(vn**2-2.*an*gap,0.)
    tau=xp.clip(2.*gap/xp.maximum(-vn+xp.sqrt(discriminant),1e-9),0.,dt)
    ttc=index.astype(dtype)*dt+tau
    contact=p0+v0*tau[...,None]+.5*a0*tau[...,None]**2
    local=xp.einsum('...ji,...j->...i',racket_rotation,contact-racket_pos_m)
    closing=-xp.sum((v0+a0*tau[...,None])*normal,axis=-1)
    before=positions[...,:-1,:];before_v=velocities[...,:-1,:]
    apex_tau=xp.clip(-before_v[...,2]/xp.minimum(acc[...,2],-1e-6),0.,dt)
    apex_candidates=before[...,2]+before_v[...,2]*apex_tau+.5*acc[...,2]*apex_tau**2
    pre_return=xp.arange(SAMPLES,dtype=dtype)*dt<=ttc[...,None]
    apex=xp.max(xp.where(pre_return,apex_candidates,-xp.inf),axis=-1)
    error=xp.linalg.norm(local[...,:2],axis=-1)
    plane_error=xp.abs(xp.sum((contact-racket_pos_m)*normal,axis=-1)-plane_offset_m)
    valid=(xp.any(crossing,axis=-1)&(normal[...,2]>.05)&(mass>0.)&(g>0.)
           &(xp.sum(velocity_m_s*normal,axis=-1)>0.)&(closing>.1)&(ttc>0.)
           &xp.isfinite(error)&xp.isfinite(apex)&xp.isfinite(plane_error))
    raw=dict(return_error_m=error,return_local_x_m=local[...,0],return_local_y_m=local[...,1],
             apex_error_m=apex-1.425,ttc_s=ttc,plane_error_m=plane_error)
    return dict(valid=valid,**{k:xp.where(valid,v,xp.zeros_like(v)) for k,v in raw.items()})


def cycle_terms(metrics: dict, prediction: dict, *, gain: Any, reward_steps: Any,
                version: int, xp: Any) -> dict:
    """Reuse unchanged nonpositive costs with reward-only forecast validity."""
    from fixed_plane_wind_release_precision import release_precision_terms
    checked={**metrics,'dynamics/reference_valid':prediction['valid'].astype(metrics['dynamics/release_event'].dtype)}
    original=release_precision_terms(checked,prediction,gain=gain,reward_steps=reward_steps,version=version,xp=xp)
    result={k.replace('reward/release_','reward/cycle_release_').replace('release_precision/','cycle_release_precision/'):v for k,v in original.items()}
    active=result['cycle_release_precision/valid']
    result['cycle_release_precision/return_time_s_sum']=active*prediction['ttc_s']
    result['cycle_release_precision/plane_error_m_sum']=active*prediction['plane_error_m']
    width=.0055 if version==325 else .0065
    result['cycle_release_precision/return_saturated_events']=active*(prediction['return_error_m']/width>=8.5)
    result['cycle_release_precision/apex_saturated_events']=active*(xp.abs(prediction['apex_error_m'])/.008>=8.5)
    return result


def environment_terms(env: Any, core: Any, metrics: dict, *, gain: Any,
                      reward_steps: Any) -> dict:
    """Read actual geometry/state; only return new scalar rewards/diagnostics."""
    import jax.numpy as jnp
    rotation=core.data.site_xmat[:,env.racket_site_id].reshape((-1,3,3))
    site=core.data.site_xpos[:,env.racket_site_id]
    normal=rotation[:,:,2]
    offset=(jnp.sum((core.data.geom_xpos[:,env.racket_geom_id]-site)*normal,axis=-1)
        +core.model.geom_size[:,env.racket_geom_id,int(env.cfg.fixed_contact_racket_surface_size_axis)]
        +core.model.geom_size[:,env.ball_geom_id,0])
    prediction=forecast_cycle(core.data.qpos[:,env.ball_qadr:env.ball_qadr+3],
        core.data.qvel[:,env.ball_vadr:env.ball_vadr+3],site,rotation,offset,
        cycle_wind_samples(core,env.cfg.wind_config,core.step_count*env.dt,xp=jnp),
        core.model.body_mass[:,env.ball_body_id],jnp.abs(core.dr_gravity_z),xp=jnp,
        drag_factor_kg_m=env.cfg.wind_config.drag_factor_kg_m,force_limit_n=env.cfg.wind_config.force_limit_n)
    terms=cycle_terms(metrics,prediction,gain=gain,reward_steps=reward_steps,
        version=env.cfg.wind_center_precision_version,xp=jnp)
    active=terms['cycle_release_precision/valid']
    for label,mask in (('clean',core.wind_mode==0),('wind',core.wind_mode>0)):
        weight=active*mask
        terms[f'cycle_release_precision/{label}_events']=weight
        terms[f'cycle_release_precision/{label}_return_error_m_sum']=weight*prediction['return_error_m']
        terms[f'cycle_release_precision/{label}_apex_error_m_sum']=weight*prediction['apex_error_m']
        terms[f'cycle_release_precision/{label}_apex_error_sq_sum']=weight*prediction['apex_error_m']**2
    return terms


def source_resume_context(payload: dict, sha256: str) -> dict:
    """Authenticate the actual clean-passing S16 parent and own mature phase."""
    from fixed_plane_wind_balanced_precontact import source_resume_context as balanced_context
    mode=payload.get('args',{}).get('wind_training_mix')
    if mode==BIRTH_MODE:
        if (sha256!=BIRTH_SHA256 or payload.get('global_update')!=14600
                or payload.get('stage_index')!=15 or payload.get('stage_converged',False)):
            raise ValueError('Unreviewed cycle-release parent')
    elif mode==MODE:
        cfg=payload.get('env_cfg',{})
        if (cfg.get('wind_cycle_release_precision') is not True or cfg.get('wind_release_precision',False)
                or payload.get('global_update',0)<14600 or payload.get('precontact_reward_birth_update')!=14500):
            raise ValueError('Changed cycle-release recipe or phase')
    else:
        raise ValueError('Only pinned v7 parent or own cycle-release checkpoints may resume')
    checked={**payload,'args':{**payload['args'],'wind_training_mix':BIRTH_MODE}}
    context=balanced_context(checked,sha256)
    if context!=dict(birth_update=14500,policy_steps=8192):
        raise ValueError('Cycle-release source must retain its mature reward phase')
    return context
