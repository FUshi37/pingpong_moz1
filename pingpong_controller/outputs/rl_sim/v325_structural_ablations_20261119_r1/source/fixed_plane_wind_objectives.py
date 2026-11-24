"""Per-metric outcome costs and sampled-training regression protection."""
from __future__ import annotations

from dataclasses import dataclass
import math

REVISION = 'wind4_pose20_outcome_quality_20261113'
WEIGHTS = dict(center=4., band=32., apex=4., racket_xy=3., ball_xy=3.,
               angular=2., jump=2., recurrent_mean=2., recurrent_rms=2., ball_rms=2.)


def quality_limits(version: int) -> dict[str, float]:
    if version not in (325, 326):
        raise ValueError('Balanced quality is scoped to V325/V326')
    return dict(racket_vxy_rms_m_s=.055, racket_omega_rms_rad_s=.28,
        center_m=.0055 if version==325 else .0065,
        adjacent_jump_m=.012 if version==325 else .015,
        recurrent_vxy_m_s=.05, recurrent_vxy_rms_m_s=.055,
        ball_vxy_m_s=.035 if version==325 else .030,
        ball_vxy_rms_m_s=.045 if version==325 else .035,
        in_band=.99 if version==325 else .98)


def objective_contract(version: int) -> dict:
    return dict(revision=REVISION, limits=quality_limits(version), weights=WEIGHTS,
        apex_band_m=[1.40,1.45], apex_scale_m=.025, per_component_cap=4.,
        positive_reward_added='separate bounded4-point late apex target',
        ball_velocity_budget='physical velocity; per-wind flight budget; clean unchanged', first_nine_lessons_unchanged=True,
        guard_warmup_updates=64, guard_required_streak=16,
        guard_upper_limit_multiplier=1.5, guard_band_margin=.04,
        checkpoint_order='stage,all-quality,worst-violation,total-violation,survival,length')


def objective_terms(metrics: dict, *, version: int, scale, xp, ball_allowance=0.) -> dict:
    """Price every outcome violation separately; no credit for mere motion."""
    q=quality_limits(version)
    def cost(name, value, event, limit):
        excess=xp.maximum(value/xp.maximum(limit,1e-12)-event,0.)
        return -scale*WEIGHTS[name]*xp.where(event>0.,xp.minimum(excess+excess**2,4.),0.)
    hit=metrics['hit_event_count']; steady=metrics['rapid_steady_steady_events']
    recurrent=metrics['recurrent_hit_event_count']
    terms={
        'center':cost('center',metrics['hit_contact_center_dist_sum'],hit,q['center_m']),
        'racket_xy':cost('racket_xy',metrics['rapid_steady_steady_racket_vxy_sq_sum'],steady,q['racket_vxy_rms_m_s']**2),
        'angular':cost('angular',metrics['rapid_steady_steady_racket_omega_sq_sum'],steady,q['racket_omega_rms_rad_s']**2),
        'ball_xy':cost('ball_xy',metrics['hit_vxy_sum'],hit,q['ball_vxy_m_s']+ball_allowance),
        'ball_rms':cost('ball_rms',metrics['hit_vxy_sq_sum'],hit,(q['ball_vxy_rms_m_s']+ball_allowance)**2),
        'recurrent_mean':cost('recurrent_mean',metrics['recurrent_hit_racket_vxy_sum'],recurrent,q['recurrent_vxy_m_s']),
        'recurrent_rms':cost('recurrent_rms',metrics['recurrent_hit_racket_vxy_sq_sum'],recurrent,q['recurrent_vxy_rms_m_s']**2),
        'jump':cost('jump',metrics['v80_late_pair_jump_sum_m'],metrics['v80_late_pair_event'],q['adjacent_jump_m']),
        'band':-scale*WEIGHTS['band']*xp.maximum(metrics['v92_late_contact_event']-metrics['v92_in_1p4_1p45_event'],0.),
    }
    apex=metrics['rapid_steady_steady_apex_sum_z_m']/xp.maximum(steady,1.)
    excess=xp.maximum(xp.maximum(1.40-apex,apex-1.45),0.)/.025
    terms['apex']=-scale*WEIGHTS['apex']*xp.where(steady>0.,xp.minimum(excess+excess**2,4.),0.)
    return {'reward/wind_quality_'+name:value for name,value in terms.items()}


def quality_violation_score(quality: dict, version: int, wind_m_s: float = 0.) -> tuple[float,float]:
    """Worst metric precedes average cost, so one collapse cannot be hidden."""
    if quality.get('steady_events',0.)<100:
        return (1e6,1e6)
    violations=[]
    from fixed_plane_wind_strong import wind_quality_limits
    for name,limit in wind_quality_limits(version,wind_m_s).items():
        value=quality.get(name,math.nan)
        if not math.isfinite(value):
            return (1e6,1e6)
        ratio=(1.-value)/(1.-limit) if name=='in_band' else value/limit
        violations.append(max(ratio-1.,0.))
    apex=quality.get('apex_m',math.nan)
    if not math.isfinite(apex):
        return (1e6,1e6)
    violations.append(max(1.40-apex,apex-1.45,0.)/.025)
    return max(violations),sum(violations)


@dataclass
class RegressionGuard:
    """Stop sustained severe sampled-training degradation; never graduate it."""
    version: int
    streak: int = 0
    wind_m_s: float = 0.

    def add(self, quality: dict, *, stage_updates: int, window_size: int,
            per_wind: dict | None = None) -> dict:
        severe=[]
        if stage_updates>=64 and window_size==32:
            for prefix,q in [('',quality),*((name+'/',q) for name,q in (per_wind or {}).items())]:
                if q.get('steady_events',0.)<100:
                    continue
                from fixed_plane_wind_strong import wind_quality_limits
                limits=wind_quality_limits(self.version,0. if prefix=='clean/' else self.wind_m_s)
                for name,limit in limits.items():
                    value=q.get(name,math.nan)
                    bad=(not math.isfinite(value) or
                         (value<limit-.04 if name=='in_band' else value>1.5*limit))
                    if bad:severe.append(prefix+name)
                apex=q.get('apex_m',math.nan)
                if not math.isfinite(apex) or not 1.385<=apex<=1.465:
                    severe.append(prefix+'apex_m')
        self.streak=self.streak+1 if severe else 0
        return dict(stop=self.streak>=16,streak=self.streak,severe_metrics=severe,
                    scope='rolling sampled training, not independent validation')
