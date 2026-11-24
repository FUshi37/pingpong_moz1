"""Opt-in impact-motion and post-impact return feedback for windy episodes."""
from typing import Any


def recovery_contract(*, full_flight: bool = False) -> dict:
    contract = dict(revision='wind_impact_motion_return_v1',
        angular_weight_per_late_hit=1., angular_scale_rad_s=.28, angular_cap_per_hit=9.,
        angular='native late-confirmed full angular squared speed; no orientation restriction',
        return_phase='30--240ms after confirmed hit, detached, upward ball,>=5hits',
        target_vz='clip(-4*(racket_z-midpoint),-.15,.15)m/s',
        return_scale_m_s=.10, return_weight_per_s=4., return_squared_cap=4.,
        onset_multiplier='same1+2exp(-max(confirmed_hits-5,0)/3)',
        scope='wind gain only; clean and sheltered rewards unchanged',
        unchanged='theoretical3D output/normal,measured height,physics,control,PPO,gates')
    if full_flight:
        contract.update(revision='wind_impact_motion_return_v2',
            return_phase='>=30ms after actual physical release, detached, outside causal precontact; includes descending flight and weak releases',
            clock='FlightState.release_time, not last counted confirmation')
    return contract


def recovery_phase(age_s, ball_vz_m_s, contact, confirmed_hits, *, xp: Any):
    return ((age_s > .030) & (age_s < .240) & (ball_vz_m_s > 0.)
            & ~contact & (confirmed_hits >= 5))


def full_flight_recovery_phase(since_release_s, contact, precontact, confirmed_hits, *, xp: Any):
    return ((since_release_s > .030) & ~contact & ~precontact & (confirmed_hits >= 5))


def safety_terms(metrics: dict, *, gain, dt_s: float, xp: Any) -> dict:
    events = metrics['rapid_steady_steady_events']
    angular = xp.minimum(metrics['rapid_steady_steady_racket_omega_sq_sum'] / .28 ** 2,
                         9. * events)
    target_vz = xp.clip(-4. * (metrics['racket_z'] - metrics['dynamics/target_racket_z_m']),
                        -.15, .15)
    error = (metrics['dynamics/racket_vz_m_s'] - target_vz) / .10
    onset = 1. + 2. * xp.exp(-xp.maximum(metrics['dynamics/confirmed_hits'] - 5., 0.) / 3.)
    returning = 4. * dt_s * onset * metrics['dynamics/recovery_phase'] * xp.minimum(error ** 2, 4.)
    valid = metrics['terminated'] < .5
    return {'reward/dynamics_hit_angular': -gain * xp.where(valid, angular, 0.),
            'reward/dynamics_posthit_return': -gain * xp.where(valid, returning, 0.)}
