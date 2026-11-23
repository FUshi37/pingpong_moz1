"""Independent goal shaping; candidate weights do not redefine task success."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import jax.numpy as jnp

from .ball_flight import ballistic_apex


@dataclass(frozen=True)
class RewardContract:
    """Per-release terms and per-second rates, approved for calibration only."""

    hit_bonus: float = 2.
    apex_xy_weight: float = 1.
    apex_z_weight: float = 1.
    apex_xy_scale_m: float = .10
    apex_z_scale_m: float = .05
    precision_bonus: float = 1.
    precision_xy_scale_m: float = .02
    precision_z_scale_m: float = .02
    steady_horizontal_weight: float = .25
    steady_horizontal_scale_m_s: float = .05
    near_goal_scale_m: float = .04
    survival_per_s: float = 1.
    failure_cost: float = 10.
    smooth_action_per_s: float = .02
    projection_per_s: float = .10
    approach_per_s: float = .20
    approach_xy_scale_m: float = .10
    approach_horizon_s: float = .15

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0. for v in asdict(self).values()):
            raise ValueError('Reward coefficients must be finite and nonnegative')
        if any(v<=0. for k,v in asdict(self).items() if 'scale' in k or k=='approach_horizon_s'):
            raise ValueError('Reward scales and approach horizon must be positive')


def release_reward(position_m, velocity_m_s, goal_xyz_m, accepted_release, *, contract):
    """Settle once per unique valid release; masks come from task accounting.

    Log-distance costs retain a direction at remote targets. The additional
    local bonus sharpens the two-centimeter neighborhood without being the
    sole learning signal. Flight/apex success is still measured separately.
    """
    c=contract
    apex,_,valid=ballistic_apex(position_m,velocity_m_s)
    delta=apex-goal_xyz_m
    xy2=jnp.sum(delta[...,:2]**2,axis=-1); z2=delta[...,2]**2
    near=jnp.exp(-.5*(xy2+z2)/c.near_goal_scale_m**2)
    terms=dict(hit=jnp.full_like(xy2,c.hit_bonus),
        apex_xy=-c.apex_xy_weight*jnp.log1p(xy2/c.apex_xy_scale_m**2),
        apex_z=-c.apex_z_weight*jnp.log1p(z2/c.apex_z_scale_m**2),
        precision=c.precision_bonus*jnp.exp(-.5*(xy2/c.precision_xy_scale_m**2+z2/c.precision_z_scale_m**2)),
        steady_horizontal=-c.steady_horizontal_weight*near*jnp.log1p(
            jnp.sum(velocity_m_s[...,:2]**2,axis=-1)/c.steady_horizontal_scale_m_s**2))
    terms={key:jnp.where(accepted_release & valid,value,0.) for key,value in terms.items()}
    return dict(terms,total=sum(terms.values()))


def control_reward(accepted_action, previous_accepted_action, clipped_action, *,
                   active, physical_failure, dt_s, contract):
    """Charge failure on its first active transition; padding has zero reward."""
    c=contract
    terms=dict(survival=jnp.where(physical_failure,0.,c.survival_per_s*dt_s),
        failure=-c.failure_cost*physical_failure.astype(jnp.float32),
        action_smooth=-c.smooth_action_per_s*dt_s*jnp.mean((accepted_action-previous_accepted_action)**2,axis=-1),
        projection=-c.projection_per_s*dt_s*jnp.mean((clipped_action-accepted_action)**2,axis=-1))
    terms={key:jnp.where(active,value,0.) for key,value in terms.items()}
    return dict(terms,total=sum(terms.values()))


def approach_reward(ball_position_m,ball_velocity_m_s,racket_position_m,*,active,dt_s,contract):
    """Small descending-flight interception cost for trajectories without a hit.

    This is a shaping estimate at the current site height, not a contact or a
    target ball impulse. It cannot earn a positive hit/goal reward.
    """
    c=contract; g=9.81
    dz=ball_position_m[...,2]-racket_position_m[...,2]
    vz=ball_velocity_m_s[...,2]
    time=(vz+jnp.sqrt(jnp.maximum(vz*vz+2*g*dz,1e-12)))/g
    eligible=active & (dz>0.) & (vz<0.) & (time>0.) & (time<c.approach_horizon_s)
    offset=ball_position_m[...,:2]+time[...,None]*ball_velocity_m_s[...,:2]-racket_position_m[...,:2]
    taper=jnp.square(jnp.maximum(1.-time/c.approach_horizon_s,0.))
    value=-c.approach_per_s*dt_s*taper*jnp.log1p(jnp.sum(offset**2,axis=-1)/c.approach_xy_scale_m**2)
    return jnp.where(eligible,value,0.)
