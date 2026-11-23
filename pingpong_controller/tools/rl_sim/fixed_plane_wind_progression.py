"""Population-based pose exposure and training-only progress windows.

Episodes remain 1200 seconds. Shorter segments are sampled from those same
episodes; they do not reset physics or certify final 1200-second survival.
"""
from __future__ import annotations

import math

REVISION = 'wind4_full20_time_exposure_20261114'
POSE_UPDATES = 128
FAST_SOURCE_SHA256 = 'bd5e61a588482d3594d62de5270ee5f058d30fd0faecbca611366a21c54430d7'
EXPOSURE_SOURCE_SHA256 = '2706bafe7a8ddec958ce988b7845bc5c3db8b598f664b385e0d85f362e172774'
PRECISION_SOURCE_SHA256 = '8e9e415b6ff6de799f10dd231226aa62b7c6efa7549fe79420cbea746b386f67'


def progress_segment_seconds(kind: str, speed_m_s: float, *, final: bool = False) -> float:
    if final:
        return 1200.
    if kind == 'pose':
        return 60.
    if kind == 'constant':
        return 120. if speed_m_s <= 1.2 else 180. if speed_m_s <= 2.5 else 300.
    if kind in ('slow', 'mixed'):
        return 120. if speed_m_s <= .6 else 180. if speed_m_s <= 1.2 else 300.
    raise ValueError('Unknown wind training segment kind')


def growth_hold_updates(lesson) -> int:
    return 4 if getattr(lesson, 'fast_pose_growth', False) else 16


def pose_growth_status(*, version, lesson, gate, rolling, per_wind,
                       latest, latest_per_wind) -> dict:
    """Use unchanged rolling quality floors plus a rapid severe-regression veto."""
    names = ('clean', 'constant', 'slow', 'gust')
    active = [name for name, weight in zip(names, lesson.mode_weights) if weight > 0.]
    population_gate = lambda name: getattr(gate, 'for_wind', lambda _: gate)(name)
    exposing = getattr(lesson, 'fast_pose_growth', False) and lesson.pose_ramp_updates > 0
    def readiness(q):
        # Temporary exposure tolerance only. Original metrics and the actual
        # graduation gate remain untouched; NaN continues to fail the gate.
        return {**q, 'in_band': q.get('in_band', math.nan) + .02} if exposing else q
    rolling_ok = gate(readiness(rolling)) and all(population_gate(name)(readiness(per_wind.get(name, {})))
                                      for name in active)
    severe = []
    if getattr(lesson, 'fast_pose_growth', False):
        from fixed_plane_wind_strong import wind_quality_limits
        for name, q in [('all', latest), *((name, latest_per_wind.get(name, {})) for name in active)]:
            if q.get('steady_events', 0.) < 100:
                continue
            limits = wind_quality_limits(version, 0. if name == 'clean' else lesson.max_speed_m_s)
            for key, limit in limits.items():
                value = q.get(key, math.nan)
                if (not math.isfinite(value) or
                        (value < limit - .03 if key == 'in_band' else value > 1.5*limit)):
                    severe.append(name + '/' + key)
            if not 1.385 <= q.get('apex_m', math.nan) <= 1.465:
                severe.append(name + '/apex_m')
        latest_ok = not severe
    else:
        latest_ok = gate(latest) and all(population_gate(name)(latest_per_wind.get(name, {}))
                                        for name in active)
    return dict(passed=bool(rolling_ok and latest_ok), rolling_passed=bool(rolling_ok),
                exposure_band_tolerance_pp=2. if exposing else 0., latest_severe_metrics=severe)


def rebase_pose_steps(steps: int, *, old_updates: int, new_updates: int) -> int:
    if (type(steps) is not int or steps < 0 or old_updates < 1 or new_updates < 1
            or steps > old_updates*128):
        raise ValueError('Invalid pose clock rebase')
    return round(steps*new_updates/old_updates)


def progression_contract() -> dict:
    from fixed_plane_wind_precision import precision_contract
    return dict(revision=REVISION, pose_exposure_updates=POSE_UPDATES,
        pose_authority='128 actual updates to20deg; quality gates graduation, not intermediate authority',
        precision_reward=precision_contract(),quality_regression='hold current opt-in lesson and continue learning; never graduate it',
        pose_growth_hold=4, pose_quality='rolling32 global and each active wind population; exposure band2pp tolerance only',
        pose_exposure_band_margin_pp=2., graduation_quality_floors_unchanged=True,
        rapid_veto='physical termination or latest severe regression: upper1.5x, band minus3pp',
        native_episode_s=1200., pose_segment_s=60., intermediate_segment_s=[120.,180.,300.],
        final_segment_s=1200., graduation_hold=16, independent_validation=False,
        source_migration='pinned3515 complete learner; rebase credited clock preserving angle')
