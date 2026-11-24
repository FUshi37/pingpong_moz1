"""Opt-in periodic-stroke motion limits and independent impact/height precision.

References are privileged training targets, never policy inputs. All physical
quantities and raw motion logs retain the world/base frame of the native task.
"""
from collections import namedtuple
from dataclasses import field, make_dataclass
from functools import lru_cache
import math
from typing import Any


REFERENCE_FIELDS = tuple('reference_motion/' + key for key in (
    'active_samples', 'steady_excess_vxy_sq_sum', 'recurrent_excess_vxy_sum',
    'recurrent_excess_vxy_sq_sum', 'steady_excess_omega_sq_sum',
    'steady_speed_ref_sq_sum', 'steady_omega_ref_sq_sum')) + (
    'reference_quality/center_sq_sum_m2', 'reference_quality/apex_error_sq_sum_m2')
SPEED_REF_CAP_M_S = .18
OMEGA_REF_CAP_RAD_S = 1.
CONTACT_RESTITUTION = .592


def reference_contract() -> dict:
    return dict(revision='periodic_stroke_reference_constraints_negy_v1',
        reference='periodic desired incoming/outgoing; normal-only racket stroke',
        effective_restitution=CONTACT_RESTITUTION,
        calibration='nominal CPU contact inverse; advisory reference, not changed physics or hardware evidence',
        speed_reference_cap_m_s=SPEED_REF_CAP_M_S,
        omega_reference_cap_rad_s=OMEGA_REF_CAP_RAD_S,
        rotation='angle between consecutive physical-edge reference normals / impact interval >=0.15s',
        cache='physical edge; reused unchanged at native delayed hit confirmation; reset on domain change',
        wind_motion='max(raw magnitude - reference magnitude,0); same residual in reward/gates/guard/ranking',
        clean_motion='original raw limits; references exactly zero',
        residual_limits=dict(xy_rms_m_s=.055,xy_mean_m_s=.05,omega_rms_rad_s=.28),
        raw_safety_limits=dict(xy_rms_m_s=.235,omega_rms_rad_s=1.28),
        precision=dict(apex_target_m=1.425,apex_sigma_m=.010,apex_huber_rate_per_s=16.,
            apex_huber_cap=4.,extra_band_miss_cost=8.,apex_error_rms_limit_m=.010,
            center_sigma_m=.004,center_huber_rate_per_s=12.,center_huber_cap=4.,
            center_rms_limit_m={'325':.007,'326':.008},interval_cap_s=1.2),
        timing='wind height uses actual apex and saved release gain; clean retains native apex definition',
        preserved='all original longevity/height-band/mean-center/ball-motion gates; no forced progression',
        scope='all32 stages; constant,slow,mixed wind to4m/s;Y<=0;five-hit shelter;BC0;clean10')


@lru_cache(maxsize=16)
def _config_type(parent: type) -> type:
    return make_dataclass('WindReferenceConstraintsConfig',[
        ('wind_reference_constraints',bool,field(default=True)),
        ('wind_reference_constraints_version',int,field(default=325))],
        bases=(parent,),frozen=True)


def with_reference_config(cfg: Any, enabled: bool, version: int):
    if not enabled:
        return cfg
    if version not in (325,326):
        raise ValueError('Reference constraints are scoped to V325/V326')
    if getattr(cfg,'wind_dynamics_guidance',False):
        # The .592 reference is calibrated only for this fixed contact model.
        # Stored inactive contact-DR intervals must not be mistaken for samples.
        checks=(not cfg.dr_randomize_contact,
            tuple(cfg.dr_ball_mass_range)==(.004,.004),
            tuple(cfg.dr_ball_normalized_inertia_range)==(.4,.4),
            tuple(cfg.ball_racket_pair_solref)==(.006836,.228086),
            cfg.wind_config.max_speed_m_s<=4.,
            cfg.wind_config.model=='quadratic_wind_increment_v1',
            cfg.wind_config.wind_y_nonpositive)
        if not all(checks):
            raise ValueError('Periodic stroke reference requires its calibrated fixed physics and Y<=0 wind')
    return _config_type(type(cfg))(**vars(cfg),wind_reference_constraints=True,
                                  wind_reference_constraints_version=version)


def motion_reference(incoming, outgoing, valid, windy, *, xp: Any) -> dict:
    """Bounded normal-stroke family for (...,3) desired periodic velocities.

    n follows the collision impulse. Restitution constrains the normal stroke
    speed; its horizontal projection is a reference, not a necessary minimum.
    Using desired incoming avoids rewarding a policy for inventing drift.
    """
    delta = outgoing-incoming
    norm = xp.linalg.norm(delta,axis=-1)
    finite = xp.all(xp.isfinite(incoming)&xp.isfinite(outgoing),axis=-1)
    active = valid & windy & finite & (norm>1e-6)
    normal = delta/xp.maximum(norm[...,None],1e-6)
    normal = xp.where(active[...,None],normal,xp.asarray([0.,0.,1.]))
    normal_speed = xp.sum((outgoing+CONTACT_RESTITUTION*incoming)*normal,axis=-1)/(
        1.+CONTACT_RESTITUTION)
    speed = xp.abs(normal_speed)*xp.linalg.norm(normal[...,:2],axis=-1)
    return dict(normal=normal,speed_m_s=xp.where(active,
        xp.clip(speed,0.,SPEED_REF_CAP_M_S),0.),valid=active)


class ReferenceState(namedtuple('WindReferenceStateBase',
        'core speed_ref omega_ref normal_ref impact_time_s domain_s valid')):
    __slots__ = ()

    def __getattr__(self,name):
        return getattr(self.core,name)

    def _replace(self,**kwargs):
        own={k:v for k,v in kwargs.items() if k in self._fields}
        native={k:v for k,v in kwargs.items() if k not in self._fields}
        if native:
            own['core']=own.get('core',self.core)._replace(**native)
        return super()._replace(**own)


def initialize_reference(core, *, xp: Any) -> ReferenceState:
    zero=xp.zeros_like(core.step_count,dtype=xp.float32)
    normal=xp.broadcast_to(xp.asarray([0.,0.,1.],dtype=xp.float32),zero.shape+(3,))
    return ReferenceState(core,zero,zero,normal,zero-1.,core.wind_sampled_s,
                          xp.zeros_like(zero,dtype=xp.bool_))


def cache_reference(state, *, normal,speed,valid,edge,time_s,domain_s,xp: Any):
    same=state.domain_s==domain_s
    age=time_s-state.impact_time_s
    rotation=xp.arccos(xp.clip(xp.sum(normal*state.normal_ref,axis=-1),-1.,1.))
    omega=xp.where(same&state.valid&valid&(age>=.15),
        xp.minimum(rotation/xp.maximum(age,.15),OMEGA_REF_CAP_RAD_S),0.)
    current_valid=xp.where(edge,valid,state.valid&same)
    return state._replace(
        speed_ref=xp.where(current_valid,xp.where(edge,speed,state.speed_ref),0.),
        omega_ref=xp.where(current_valid,xp.where(edge,omega,state.omega_ref),0.),
        normal_ref=xp.where(edge[...,None],normal,state.normal_ref),
        impact_time_s=xp.where(edge,time_s,state.impact_time_s),domain_s=domain_s,
        valid=current_valid)


def aligned_motion_terms(metrics: dict,speed_ref,omega_ref,gain, *, xp: Any) -> dict:
    hit=metrics['hit_event_count']
    steady=xp.maximum(metrics['rapid_steady_steady_events'],1.)
    speed=xp.sqrt(xp.maximum(metrics['hit_racket_vxy_sq_sum'],0.)/xp.maximum(hit,1.))
    omega=xp.sqrt(xp.maximum(metrics['rapid_steady_steady_racket_omega_sq_sum'],0.)/steady)
    excess=xp.maximum(speed-speed_ref,0.)
    omega_excess=xp.maximum(omega-omega_ref,0.)
    center=metrics['hit_contact_center_dist_sum']/xp.maximum(hit,1.)
    quality=xp.exp(-.5*((center/.0055)**2+(excess/.055)**2+(omega_excess/.28)**2))
    valid=(metrics['terminated']<.5)&(metrics['dynamics/recent']>.5)
    interval=xp.clip(metrics['dynamics/hit_interval_s'],0.,1.2)
    margin=xp.clip(metrics['dynamics/z_margin_m']/.025,0.,1.)
    event=gain*xp.where(valid,interval*hit*margin,0.)
    return {'reward/dynamics_hit':4.*event*quality,
        'reward/continuation_hit_quality':12.*event*quality,
        'reward/gate_racket_xy':20.*event*xp.exp(-.5*(xp.maximum(excess-.045,0.)/.020)**2)}


def precision_costs(metrics: dict, *,height_gain,center_gain,height_interval_s,xp: Any) -> dict:
    live=metrics['terminated']<.5
    hit=metrics['hit_event_count']
    apex_count=metrics['wind_height/apex_events']
    apex=metrics['wind_height/apex_sum_m']/xp.maximum(apex_count,1.)
    center=metrics['hit_contact_center_dist_sum']/xp.maximum(hit,1.)
    def huber(x):
        return .5*xp.minimum(xp.abs(x),1.)**2+xp.maximum(xp.abs(x)-1.,0.)
    return {
        'reward/reference_height_dispersion':-height_gain*xp.where(live,
            16.*xp.clip(height_interval_s,0.,1.2)*apex_count*xp.minimum(huber((apex-1.425)/.010),4.),0.),
        'reward/reference_height_tail':-height_gain*xp.where(live,
            8.*xp.maximum(metrics['wind_height/band_events']-metrics['wind_height/in_band_events'],0.),0.),
        'reward/reference_center_precision':-center_gain*xp.where(live,
            12.*xp.clip(metrics['dynamics/hit_interval_s'],0.,1.2)*hit*xp.minimum(huber(center/.004),4.),0.)}


def reference_quality_summary(sums: dict) -> dict:
    from fixed_plane_wind_flight_outcomes import flight_quality_summary
    q=flight_quality_summary(sums)
    for name,numerator,denominator,root in (
        ('racket_vxy_excess_rms_m_s','reference_motion/steady_excess_vxy_sq_sum','rapid_steady_steady_events',True),
        ('recurrent_vxy_excess_m_s','reference_motion/recurrent_excess_vxy_sum','recurrent_hit_event_count',False),
        ('recurrent_vxy_excess_rms_m_s','reference_motion/recurrent_excess_vxy_sq_sum','recurrent_hit_event_count',True),
        ('racket_omega_excess_rms_rad_s','reference_motion/steady_excess_omega_sq_sum','rapid_steady_steady_events',True),
        ('reference_vxy_rms_m_s','reference_motion/steady_speed_ref_sq_sum','rapid_steady_steady_events',True),
        ('reference_omega_rms_rad_s','reference_motion/steady_omega_ref_sq_sum','rapid_steady_steady_events',True),
        ('center_rms_m','reference_quality/center_sq_sum_m2','hit_event_count',True),
        ('apex_error_rms_m','reference_quality/apex_error_sq_sum_m2','wind_height/apex_events',True)):
        if sums.get(denominator,0.)>0. and numerator in sums:
            value=sums[numerator]/sums[denominator]
            q[name]=math.sqrt(max(value,0.)) if root else value
    if 'apex_error_rms_m' in q and 'apex_m' in q:
        q['apex_std_m']=math.sqrt(max(q['apex_error_rms_m']**2-(q['apex_m']-1.425)**2,0.))
    return q


def gate_quality(q: dict) -> dict:
    result=dict(q)
    for raw,excess in (('racket_vxy_rms_m_s','racket_vxy_excess_rms_m_s'),
        ('recurrent_vxy_m_s','recurrent_vxy_excess_m_s'),
        ('recurrent_vxy_rms_m_s','recurrent_vxy_excess_rms_m_s'),
        ('racket_omega_rms_rad_s','racket_omega_excess_rms_rad_s')):
        if excess in q:
            if raw in q:
                result.setdefault('raw_'+raw,q[raw])
            result[raw]=q[excess]
    return result


def make_reference_gate(base,version: int):
    """Add tail gates; retain raw metrics in logs and physical safety ceilings."""
    def build(original,clean=False):
        def check(q):
            limits={'apex_error_rms_m':.010,'center_rms_m':.007 if version==325 else .008}
            if not clean:
                limits.update(racket_vxy_excess_rms_m_s=.055,recurrent_vxy_excess_m_s=.05,
                    recurrent_vxy_excess_rms_m_s=.055,racket_omega_excess_rms_rad_s=.28,
                    reference_vxy_rms_m_s=SPEED_REF_CAP_M_S+1e-7,
                    reference_omega_rms_rad_s=OMEGA_REF_CAP_RAD_S+1e-7)
            if any(not math.isfinite(q.get(k,math.inf)) or q[k]>limit for k,limit in limits.items()):
                return False
            if not clean and any(not math.isfinite(q.get('raw_'+key,q.get(key,math.inf)))
                    or q.get('raw_'+key,q.get(key,math.inf))>limit for key,limit in (
                    ('racket_vxy_rms_m_s',.235),('racket_omega_rms_rad_s',1.28))):
                return False
            return original(q if clean else gate_quality(q))
        return check
    gate=build(base)
    gate.for_wind=lambda name:build(getattr(base,'for_wind',lambda _:base)(name),name=='clean')
    return gate


def reference_checkpoint_score(score: tuple,q: dict,version: int) -> tuple:
    violations=[max(q.get(k,limit*10.)/limit-1.,0.) for k,limit in (
        ('apex_error_rms_m',.010),('center_rms_m',.007 if version==325 else .008))]
    return (*score[:2],min(score[2],-max(violations)),score[3]-sum(violations),*score[4:])


def reference_environment(inner_class):
    class ReferenceEnv:
        def __init__(self,inner,reward_resume_policy_steps=0,precontact_resume_policy_steps=0):
            self.inner=inner
            normalized=getattr(inner.cfg,'wind_normalized_group_priority',False)
            precontact=getattr(inner.cfg,'wind_precontact_center',False)
            steady_braking = getattr(inner.cfg, 'wind_steady_braking', False)
            confirmed_launch_braking = getattr(inner.cfg, 'wind_confirmed_launch_braking', False)
            center_tail = getattr(inner.cfg, 'wind_center_tail', False)
            calm_release = getattr(inner.cfg, 'wind_calm_release', False)
            early_recovery = getattr(inner.cfg, 'wind_early_recovery', False)
            calm_return = getattr(inner.cfg, 'wind_calm_return_height', False)
            impact_normal = getattr(inner.cfg, 'wind_impact_normal_precision', False)
            cubic_recovery = getattr(inner.cfg, 'wind_cubic_recovery', False)
            clean_joint_quality = getattr(inner.cfg, 'wind_clean_joint_quality_priority', False)
            if normalized and precontact:
                raise ValueError('Distinct reward recipes cannot be stacked')
            if (normalized or precontact or steady_braking or confirmed_launch_braking
                    or calm_release or early_recovery or calm_return or impact_normal
                    or cubic_recovery or clean_joint_quality):
                if not 0 <= reward_resume_policy_steps <= 8192:
                    raise ValueError('Invalid reward-only continuation clock')
                self.reward_resume_policy_steps=reward_resume_policy_steps
            elif reward_resume_policy_steps:
                raise ValueError('Historical reward does not use a continuation offset')
            if precontact or confirmed_launch_braking or center_tail:
                if not 0 <= precontact_resume_policy_steps <= 8192:
                    raise ValueError('Invalid precontact reward continuation clock')
                self.precontact_resume_policy_steps=precontact_resume_policy_steps
            elif precontact_resume_policy_steps:
                raise ValueError('Historical reward does not use a precontact offset')

        def __getattr__(self,name):
            return getattr(self.inner,name)

        def attach_training_state(self,core):
            import jax.numpy as jnp
            if isinstance(core,ReferenceState):
                return core
            core=self.inner.attach_training_state(core)
            return initialize_reference(core,xp=jnp)

        def reset(self,keys):
            import jax.numpy as jnp
            core,obs=self.inner.reset(keys)
            return initialize_reference(core,xp=jnp),obs

        def reset_done(self,state,obs,done,keys):
            import jax
            import jax.numpy as jnp
            core,obs=self.inner.reset_done(state.core,obs,done,keys)
            fresh=initialize_reference(core,xp=jnp)
            select=lambda new,old:jnp.where(done.reshape((done.shape[0],)+(1,)*(new.ndim-1)),new,old)
            fields=jax.tree.map(select,tuple(fresh)[1:],tuple(state)[1:])
            return ReferenceState(core,*fields),obs

        def get_critic_obs(self,state,obs):
            return self.inner.get_critic_obs(state.core,obs)

        def step(self,state,action):
            import jax.numpy as jnp
            from fixed_plane_wind_pose import shelter_gain
            from fixed_plane_wind_gate_precision import group_reward_metrics
            core,obs,reward,done,m=self.inner.step(state.core,action)
            if not getattr(self.cfg,'wind_dynamics_guidance',False):
                m.update({key:jnp.zeros_like(reward) for key in REFERENCE_FIELDS})
                return state._replace(core=core),obs,reward,done,m
            before=state.core
            windy=before.wind_mode>0
            gain=shelter_gain(before,self.dt,xp=jnp)
            incoming=jnp.stack([m[f'dynamics/target_incoming_v{k}_m_s'] for k in 'xyz'],axis=-1)
            outgoing=jnp.stack([m[f'dynamics/target_v{k}_m_s'] for k in 'xyz'],axis=-1)
            ref=motion_reference(incoming,outgoing,m['dynamics/reference_valid']>.5,windy,xp=jnp)
            cached=cache_reference(state,normal=ref['normal'],speed=ref['speed_m_s'],valid=ref['valid'],
                edge=m['raw_physical_contact_edge']>.5,time_s=core.step_count*self.dt,
                domain_s=before.wind_sampled_s,xp=jnp)
            aligned=aligned_motion_terms(m,cached.speed_ref,cached.omega_ref,gain*windy,xp=jnp)
            previous=reward
            delta=sum(value-m[key] for key,value in aligned.items())
            reward=reward+delta
            m.update(aligned)
            same=before.tracker.domain_s==before.wind_sampled_s
            elapsed=jnp.where((before.release_time>0.)&same,
                core.step_count*self.dt-before.release_time,0.)
            costs=precision_costs(m,height_gain=jnp.where(windy,before.tracker.gain,gain),
                center_gain=gain,height_interval_s=jnp.where(windy,2.*elapsed,
                    m['dynamics/hit_interval_s']),xp=jnp)
            reward=reward+sum(costs.values())
            m.update(costs)
            reward_steps=core.total_env_steps
            confirmed_launch_braking = getattr(self.cfg, 'wind_confirmed_launch_braking', False)
            if (getattr(self.cfg,'wind_normalized_group_priority',False)
                    or getattr(self.cfg,'wind_precontact_center',False)
                    or getattr(self.cfg, 'wind_steady_braking', False)
                    or getattr(self.cfg, 'wind_calm_return_height', False)
                    or getattr(self.cfg, 'wind_impact_normal_precision', False)
                    or getattr(self.cfg, 'wind_cubic_recovery', False)
                    or getattr(self.cfg, 'wind_clean_joint_quality_priority', False)
                    or confirmed_launch_braking):
                reward_steps=reward_steps+self.reward_resume_policy_steps
            if getattr(self.cfg, 'wind_mature_precision_reward', False):
                from fixed_plane_wind_resume_precision_clock import reward_steps as mature_steps
                reward_steps = mature_steps(reward_steps, enabled=True)
            if getattr(self.cfg,'wind_center_precision_reward',False):
                from fixed_plane_wind_center_precision_reward import center_reward_terms
                m['counterfactual/pre_center_reward']=reward
                center_terms=center_reward_terms(m,clean=~windy,gain=gain,
                    total_env_steps=reward_steps,
                    version=self.cfg.wind_center_precision_version,xp=jnp)
                reward=reward+center_terms['reward/center_mean_rms_credit']
                m.update(center_terms)
                if getattr(self.cfg,'wind_clean_apex_preservation',False):
                    from fixed_plane_wind_center_precision_reward import clean_apex_reward_terms
                    m['counterfactual/pre_clean_apex_reward']=reward
                    apex_terms=clean_apex_reward_terms(m,clean=~windy,gain=gain,
                        total_env_steps=reward_steps,xp=jnp,
                        rate_per_s=120. if getattr(self.cfg,'wind_clean_apex_priority',False) else 40.)
                    reward=reward+apex_terms['reward/clean_apex_preservation_credit']
                    m.update(apex_terms)
            if getattr(self.cfg, 'wind_clean_joint_quality_priority', False):
                from fixed_plane_wind_clean_joint_quality import extra_clean_joint_quality, add_credit
                m['counterfactual/pre_clean_joint_reward'] = reward
                original_joint_credit = m['reward/gate_clean_quality']
                joint_terms = extra_clean_joint_quality(
                    original_joint_credit, clean=~windy, reward_steps=reward_steps, xp=jnp)
                joint_delta = joint_terms['delta']
                reward = add_credit(reward, joint_delta, xp=jnp)
                m['reward/gate_clean_quality'] = add_credit(
                    original_joint_credit, joint_delta, xp=jnp)
                m.update({
                    'reward/clean_joint_priority_delta': joint_delta,
                    'clean_joint_priority/warmup': joint_terms['warmup'],
                    'clean_joint_priority/original_credit': original_joint_credit,
                    'clean_joint_priority/replacement_credit': m['reward/gate_clean_quality'],
                })
            if getattr(self.cfg, 'wind_calm_release', False):
                from fixed_plane_wind_calm_release import calm_release_credit
                credit = calm_release_credit(
                    m, calm=~windy, shelter_gain=gain, release_elapsed_s=elapsed,
                    phase_steps=core.total_env_steps + self.reward_resume_policy_steps, xp=jnp,
                )
                m['reward/calm_vertical_release'] = credit
                reward = jnp.where(credit > 0., reward + credit, reward)
            if getattr(self.cfg, 'wind_early_recovery', False):
                from fixed_plane_wind_early_recovery import recovery_active, recovery_cost
                active = recovery_active(
                    confirmed_hits=before.wind_startup_hit_count, recent=m['dynamics/recent'] > .5,
                    recovery_phase=m['dynamics/recovery_phase'] > .5,
                    pending_before=before.pending_hit, pending_after=core.pending_hit, done=done,
                )
                cost = recovery_cost(
                    m['racket_z'], core.prev_racket_vel[:, 2],
                    self.cfg.fixed_contact_racket_z_bounds_m, windy=windy, active=active, gain=gain,
                    phase_steps=core.total_env_steps + self.reward_resume_policy_steps,
                    dt_s=self.dt, xp=jnp,
                )
                m['reward/early_racket_recovery'] = cost
                reward = jnp.where(cost < 0., reward + cost, reward)
            if getattr(self.cfg, 'wind_calm_return_height', False):
                from fixed_plane_wind_calm_return import return_active, return_height_cost
                active = return_active(
                    confirmed_hits=before.wind_startup_hit_count,
                    recent=m['dynamics/recent'] > .5,
                    recovery_phase=m['dynamics/recovery_phase'] > .5,
                    pending_before=before.pending_hit, pending_after=core.pending_hit, done=done,
                )
                cost = return_height_cost(
                    m['racket_z'], m['dynamics/target_racket_z_m'], calm=~windy,
                    active=active, gain=gain,
                    reward_steps=core.total_env_steps + self.reward_resume_policy_steps,
                    dt_s=self.dt, xp=jnp,
                )
                m['reward/calm_return_height'] = cost
                reward = jnp.where(cost < 0., reward + cost, reward)
            if getattr(self.cfg, 'wind_impact_normal_precision', False):
                from fixed_plane_wind_impact_normal_precision import (
                    impact_normal_active, impact_normal_cost,
                )
                count = m['impact/count']
                target_tilt = m['impact/target_tilt_deg_sum'] / jnp.maximum(count, 1.)
                active = impact_normal_active(
                    edge=count > .5, confirmed_hits=before.wind_startup_hit_count,
                    recent=m['dynamics/recent'] > .5, done=done,
                    reference_valid=m['dynamics/reference_valid'] > .5,
                    target_tilt_deg=target_tilt,
                    available_tilt_deg=jnp.rad2deg(jnp.min(
                        self.inner.wind_pose_limits(core)[1], axis=-1)), xp=jnp,
                )
                cost = impact_normal_cost(
                    m['impact/normal_error_deg_sum'] / jnp.maximum(count, 1.),
                    active=active, gain=gain, interval_s=m['dynamics/hit_interval_s'],
                    reward_steps=core.total_env_steps + self.reward_resume_policy_steps,
                    xp=jnp,
                )
                m['reward/impact_normal_precision'] = cost
                m['impact_precision/active_count'] = active.astype(jnp.float32)
                reward = jnp.where(cost < 0., reward + cost, reward)
            if getattr(self.cfg, 'wind_cubic_recovery', False):
                from fixed_plane_wind_cubic_recovery import cubic_return_correction
                actual_elapsed = core.step_count * self.dt - core.release_time
                correction = cubic_return_correction(
                    m, incoming, outgoing, period_s=core.expected_period,
                    elapsed_s=actual_elapsed, windy=windy, gain=gain,
                    reference_valid=(core.reference_valid & same
                        & (m['dynamics/reference_valid'] > .5)
                        & (m['dynamics/normal_reference_feasible'] > .5)),
                    reward_steps=reward_steps, dt_s=self.dt,
                    bounds_m=self.cfg.fixed_contact_racket_z_bounds_m, xp=jnp,
                )
                cubic_delta = correction['reward/cubic_return_delta']
                reward = jnp.where(cubic_delta != 0., reward + cubic_delta, reward)
                m['reward/dynamics_posthit_return'] = correction[
                    'cubic_recovery/replacement_return_reward']
                m.update(correction)
            if getattr(self.cfg, 'wind_center_tail', False):
                from fixed_plane_wind_center_tail import center_tail_cost
                tail_cost = center_tail_cost(
                    m, clean=~windy, gain=gain,
                    phase_steps=core.total_env_steps + self.precontact_resume_policy_steps,
                    xp=jnp,
                )
                m['reward/center_tail'] = tail_cost
                reward = jnp.where(tail_cost < 0., reward + tail_cost, reward)
            if getattr(self.cfg, 'wind_measured_precision', False):
                from fixed_plane_wind_measured_precision import measured_precision_terms
                m['counterfactual/pre_measured_precision_reward'] = reward
                measured_costs = measured_precision_terms(
                    m, windy=windy, center_gain=gain, height_gain=before.tracker.gain,
                    center_interval_s=m['dynamics/hit_interval_s'], height_interval_s=2. * elapsed,
                    reward_steps=core.total_env_steps, version=self.cfg.wind_center_precision_version,
                    xp=jnp,
                )
                delta_measured = sum(measured_costs.values())
                reward = jnp.where(delta_measured < 0., reward + delta_measured, reward)
                m.update(measured_costs)
            if getattr(self.cfg, 'wind_steady_braking', False) or confirmed_launch_braking:
                from fixed_plane_wind_steady_braking import active_recovery, steady_braking_cost
                braking_active = active_recovery(
                    confirmed_hits=before.wind_startup_hit_count,
                    recent=m['dynamics/recent'] > .5, contact=core.prev_contact,
                    precontact=m['dynamics/precontact_geometry'] > .5,
                    terminated=m['terminated'] > .5,
                )
                braking_steps = reward_steps
                if confirmed_launch_braking:
                    from fixed_plane_wind_confirmed_launch_braking import after_launch_confirmation
                    braking_active = after_launch_confirmation(
                        braking_active, before.pending_hit, core.pending_hit,
                    )
                    braking_steps = core.total_env_steps + self.precontact_resume_policy_steps
                extra_braking = steady_braking_cost(
                    m['dynamics/braking_cost'], windy=windy, active=braking_active,
                    gain=gain, reward_steps=braking_steps, dt_s=self.dt, xp=jnp,
                )
                m['reward/steady_racket_braking'] = extra_braking
                reward = jnp.where(extra_braking < 0., reward + extra_braking, reward)
            hit=m['hit_event_count']; steady=m['rapid_steady_steady_events']
            recurrent=m['recurrent_hit_event_count'];apex=m['wind_height/apex_events']
            def rms(field,count):
                return jnp.sqrt(jnp.maximum(m[field],0.)/jnp.maximum(count,1.))
            steady_excess=jnp.maximum(rms('rapid_steady_steady_racket_vxy_sq_sum',steady)-cached.speed_ref,0.)
            recurrent_excess=jnp.maximum(rms('recurrent_hit_racket_vxy_sq_sum',recurrent)-cached.speed_ref,0.)
            omega_excess=jnp.maximum(rms('rapid_steady_steady_racket_omega_sq_sum',steady)-cached.omega_ref,0.)
            m.update({
                'reference_motion/active_samples':jnp.ones_like(reward),
                'reference_motion/steady_excess_vxy_sq_sum':steady*steady_excess**2,
                'reference_motion/recurrent_excess_vxy_sum':recurrent*recurrent_excess,
                'reference_motion/recurrent_excess_vxy_sq_sum':recurrent*recurrent_excess**2,
                'reference_motion/steady_excess_omega_sq_sum':steady*omega_excess**2,
                'reference_motion/steady_speed_ref_sq_sum':steady*cached.speed_ref**2,
                'reference_motion/steady_omega_ref_sq_sum':steady*cached.omega_ref**2,
                'reference_quality/center_sq_sum_m2':m['hit_contact_center_dist_sum']**2/jnp.maximum(hit,1.),
                'reference_quality/apex_error_sq_sum_m2':apex*(
                    m['wind_height/apex_sum_m']/jnp.maximum(apex,1.)-1.425)**2,
                'reference_motion/reward_alignment_delta':delta,
                'counterfactual/previous_reward':previous})
            if getattr(self.cfg,'wind_clean_group_priority',False):
                from fixed_plane_wind_clean_group_priority import priority_reward
                m['counterfactual/pre_clean_group_priority_reward']=reward
                reward,priority=priority_reward(reward,clean=~windy,
                    total_env_steps=core.total_env_steps,xp=jnp)
                m.update(priority)
            if getattr(self.cfg,'wind_normalized_group_priority',False):
                from fixed_plane_wind_normalized_priority import normalized_reward
                m['counterfactual/pre_normalized_priority_reward']=reward
                reward,priority=normalized_reward(reward,clean=~windy,
                    total_env_steps=reward_steps,xp=jnp)
                m.update(priority)
            if getattr(self.cfg,'wind_precontact_center',False):
                from fixed_plane_wind_precontact_center import environment_credit
                m['counterfactual/pre_precontact_center_reward']=reward
                terms=environment_credit(self,core,m,clean=~windy,gain=gain,
                    reward_steps=core.total_env_steps+self.precontact_resume_policy_steps)
                credit=terms['reward/precontact_center_credit']
                reward=jnp.where(credit>0.,reward+credit,reward)
                m.update(terms)
            if getattr(self.cfg,'wind_cycle_release_precision',False):
                from fixed_plane_wind_cycle_release import environment_terms
                m['counterfactual/pre_cycle_release_reward']=reward
                cycle_terms=environment_terms(self,core,m,gain=gain,
                    reward_steps=core.total_env_steps+self.precontact_resume_policy_steps)
                reward=reward+cycle_terms['reward/cycle_release_return_precision_cost']+cycle_terms['reward/cycle_release_apex_precision_cost']
                m.update(cycle_terms)
            if getattr(self.cfg,'wind_release_precision',False):
                from fixed_plane_wind_release_precision import environment_terms
                m['counterfactual/pre_release_precision_reward']=reward
                release_terms=environment_terms(self,core,m,gain=gain,
                    reward_steps=core.total_env_steps+self.precontact_resume_policy_steps)
                reward=reward+release_terms['reward/release_return_precision_cost']+release_terms['reward/release_apex_precision_cost']
                m.update(release_terms)
            if getattr(self.cfg,'wind_precontact_balance',False):
                from fixed_plane_wind_balanced_precontact import balanced_reward
                m['counterfactual/pre_precontact_balance_reward']=reward
                reward,balance=balanced_reward(reward,clean=~windy,
                    reward_steps=core.total_env_steps+self.precontact_resume_policy_steps,xp=jnp)
                m.update(balance)
            m.update(group_reward_metrics(reward,windy,m,xp=jnp))
            m['reward/total']=reward
            return cached._replace(core=core),obs,reward,done,m

    def factory(*args,**kwargs):
        reward_resume_policy_steps=kwargs.pop('reward_resume_policy_steps',0)
        precontact_resume_policy_steps=kwargs.pop('precontact_resume_policy_steps',0)
        inner=inner_class(*args,**kwargs)
        return (ReferenceEnv(inner,reward_resume_policy_steps,precontact_resume_policy_steps)
                if getattr(inner.cfg,'wind_reference_constraints',False) else inner)
    return factory
