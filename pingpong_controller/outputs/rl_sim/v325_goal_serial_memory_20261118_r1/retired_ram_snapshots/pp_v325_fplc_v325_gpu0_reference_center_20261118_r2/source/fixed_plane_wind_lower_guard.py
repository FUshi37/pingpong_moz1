"""Opt-in early feedback for downward racket retreats near the lower boundary."""
from typing import Any


def lower_guard_contract() -> dict:
    return dict(revision='wind_lower_stop_preview_v1', preview_s=.200,
        reserve_m=.025, velocity_scale_m_s=.10, squared_cap=4., weight_per_s=8.,
        onset_multiplier='1+2exp(-max(confirmed_hits-5,0)/3)',
        scope='wind-only downward motion; no extra upper-bound or upward-hit cost',
        interpretation='soft constant-velocity preview, not a certified stopping distance')


def lower_stop_cost(z_m, vz_m_s, lower_m: float, *, xp: Any):
    safe_downward_m_s = xp.maximum(z_m - lower_m - .025, 0.) / .200
    excess = xp.maximum(-vz_m_s - safe_downward_m_s, 0.) / .10
    return xp.minimum(excess ** 2, 4.)


def lower_stop_reward(z_m, vz_m_s, lower_m: float, *, gain, confirmed_hits,
                      terminated, dt_s: float, xp: Any):
    onset = 1. + 2. * xp.exp(-xp.maximum(confirmed_hits - 5., 0.) / 3.)
    active = (confirmed_hits >= 5) & (terminated < .5)
    return -gain * xp.where(active, 8. * dt_s * onset * lower_stop_cost(
        z_m, vz_m_s, lower_m, xp=xp), 0.)
