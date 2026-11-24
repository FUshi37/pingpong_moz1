"""V325/V326 progress from sampled training windows, without frozen tests."""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
import math

import numpy as np

from fixed_plane_v321_long_episode import QUALITY_FIELDS, quality_summary
from fixed_plane_v322_condition_gates import CONDITION_NAMES, condition_gate_settings
from fixed_plane_v325_wind import MODE_NAMES

PROTOCOL = 'wind_training_hold_gates'
QUALITY_UPDATES = 32
HOLD_UPDATES = 16


def gate_contract(version: int) -> dict:
    """Record training evidence separately from independent policy acceptance."""
    if version not in (325, 326):
        raise ValueError('Training-only gates are scoped to V325/V326')
    return dict(basis='sampled_training_segments', quality_updates=QUALITY_UPDATES,
                hold_updates=HOLD_UPDATES, minimum_segments=1024,
                minimum_segments_per_wind=64, segment_capacity=4096,
                minimum_hits_per_s=1.2, original_stage_durations=True,
                original_quality_and_longevity_floors=True,
                m13_all_conditions=version == 326, independent_validation=False)


class TrainingSegments:
    """Count full-duration segments without resetting physical episodes.

    Every physical failure closes a failed segment. Long successful episodes
    keep producing fresh outcomes, preventing stale early failures from
    blocking progression indefinitely. Incomplete time-limit segments are
    censored, never promoted to success. All arrays are [time, environment].
    """

    def __init__(self, seconds: float, dt_s: float, n_envs: int, *,
                 capacity: int = 4096, grouped: bool = False, quality_fields=None):
        if (not math.isfinite(seconds) or not math.isfinite(dt_s)
                or seconds <= 0. or dt_s <= 0. or n_envs < 1 or capacity < 1):
            raise ValueError('Invalid training-segment dimensions')
        self.steps = round(seconds / dt_s)
        if self.steps < 1:
            raise ValueError('Training segment must contain a physical policy step')
        self.n_envs, self.grouped = n_envs, grouped
        self.quality_fields = QUALITY_FIELDS if quality_fields is None else tuple(quality_fields)
        self.age_steps = np.zeros(n_envs, np.int64)
        self.hits = np.zeros(n_envs, np.float64)
        self.modes = np.full(n_envs, -1, int)
        self.conditions = np.full(n_envs, -1, int)
        self.outcomes = deque(maxlen=capacity)
        self.qualities = deque(maxlen=QUALITY_UPDATES)
        self.completed_segments = 0
        self.censored_segments = 0

    @property
    def quality_updates(self) -> int:
        return len(self.qualities)

    def add(self, *, hits, done, terminated, recent, modes, conditions=None,
            quality_metrics: Mapping[str, np.ndarray] | None = None) -> None:
        hits = np.asarray(hits, np.float64)
        done, terminated, recent = (np.asarray(x, bool)
                                    for x in (done, terminated, recent))
        raw_modes = np.asarray(modes)
        if (hits.ndim != 2 or hits.shape[1] != self.n_envs
                or any(x.shape != hits.shape for x in (done, terminated, recent, raw_modes))
                or not np.isfinite(hits).all() or (hits < 0.).any()
                or not np.isin(raw_modes, range(4)).all()
                or np.any(terminated & ~done)):
            raise ValueError('Invalid training-segment transitions')
        modes = raw_modes.astype(int)
        conditions = (np.zeros_like(modes) if conditions is None else np.asarray(conditions))
        if (conditions.shape != hits.shape
                or not np.isin(conditions, range(5)).all()
                or (not self.grouped and np.any(conditions != 0))):
            raise ValueError('Invalid training condition labels')
        conditions = conditions.astype(int)
        if quality_metrics is not None:
            values = {name: np.asarray(quality_metrics[name], np.float64)
                      for name in self.quality_fields}
            if any(x.shape != hits.shape or not np.isfinite(x).all() for x in values.values()):
                raise ValueError('Invalid per-wind quality metrics')
            nominal = conditions == 0
            self.qualities.append({name: {field: float(value[nominal & (modes == index)].sum())
                                         for field, value in values.items()}
                                   for index, name in enumerate(MODE_NAMES)})
        for t in range(hits.shape[0]):
            continuing = self.age_steps > 0
            changed = (self.modes != modes[t]) | (self.conditions != conditions[t])
            if np.any(continuing & changed):
                raise ValueError('Training segment label changed without a reset')
            self.modes[:] = modes[t]
            self.conditions[:] = conditions[t]
            self.age_steps += 1
            self.hits += hits[t]
            full = self.age_steps >= self.steps
            selected = full | terminated[t]
            for lane in np.flatnonzero(selected):
                self.outcomes.append((min(1., self.age_steps[lane] / self.steps),
                    float(self.hits[lane]), bool(full[lane] and not terminated[t, lane]
                                               and recent[t, lane]),
                    int(modes[t, lane]), int(conditions[t, lane])))
            self.completed_segments += int(selected.sum())
            self.censored_segments += int((done[t] & ~terminated[t] & ~full).sum())
            reset = full | done[t]
            self.age_steps[reset] = 0
            self.hits[reset] = 0.

    @staticmethod
    def _summary(rows: np.ndarray) -> dict:
        if not len(rows):
            return {'n': 0}
        return dict(n=len(rows), len_fraction=float(rows[:, 0].mean()),
                    hits_mean=float(rows[:, 1].mean()), survival=float(rows[:, 2].mean()))

    def _rows(self) -> np.ndarray:
        return np.asarray(self.outcomes, np.float64).reshape((-1, 5))

    def summary(self) -> dict:
        return self._summary(self._rows())

    def wind_summary(self) -> dict:
        rows = self._rows()
        return {name: self._summary(rows[(rows[:, 3] == index) & (rows[:, 4] == 0)])
                for index, name in enumerate(MODE_NAMES)}

    def condition_summary(self) -> dict | None:
        if not self.grouped:
            return None
        rows = self._rows()
        return {name: self._summary(rows[rows[:, 4] == index])
                for index, name in enumerate(CONDITION_NAMES)}

    def wind_quality_summary(self) -> dict:
        summarize = quality_summary
        if 'wind_height/active_samples' in self.quality_fields:
            from fixed_plane_wind_flight_outcomes import flight_quality_summary
            summarize = flight_quality_summary
        if 'reference_motion/active_samples' in self.quality_fields:
            from fixed_plane_wind_reference_constraints import reference_quality_summary
            summarize = reference_quality_summary
        return {name: summarize({field: sum(q[name][field] for q in self.qualities)
                                      for field in self.quality_fields})
                for name in MODE_NAMES}


def training_gate_decision(*, window_size: int, quality: dict, stage_age_s: float,
                           lesson, origin, outcome: dict, wind: dict,
                           wind_quality: dict, quality_gate: Callable[[dict], bool],
                           conditions: dict | None = None,
                           group_limits: Mapping[str, Mapping[str, float]] | None = None) -> dict:
    """Apply original numerical floors to current-domain training evidence."""
    failed = []
    if window_size != QUALITY_UPDATES:
        failed.append('rolling32')
    if not math.isfinite(stage_age_s) or stage_age_s < lesson.seconds:
        failed.append('stage_duration')
    if not quality_gate(quality):
        failed.append('quality')
    if outcome.get('n', 0) < 1024:
        failed.append('fewer_than1024_segments')

    def check(name: str, measured: dict, min_n: int, survival: float, length: float) -> None:
        if measured.get('n', 0) < min_n:
            failed.append(name + '/population')
            return
        for field, threshold in (('survival', survival), ('len_fraction', length),
                                 ('hits_mean', 1.2 * lesson.seconds)):
            value = measured.get(field, math.nan)
            if not math.isfinite(value) or value < threshold:
                failed.append(name + '/' + field)

    for name, weight in zip(MODE_NAMES, lesson.mode_weights):
        if weight <= 0.:
            continue
        limits = (group_limits['clean' if name == 'clean' else 'wind']
                  if group_limits is not None else {})
        check('wind/' + name, wind.get(name, {}), 64,
              limits.get('min_survival', origin.min_survival),
              limits.get('min_length_fraction', origin.min_length_fraction))
        population_gate=getattr(quality_gate,'for_wind',lambda _:quality_gate)(name)
        if not population_gate(wind_quality.get(name, {})):
            failed.append('wind/' + name + '/quality')
    if conditions is not None:
        for name, limits in condition_gate_settings().items():
            check('condition/' + name, conditions.get(name, {}), limits['min_n'],
                  limits['min_survival'], limits['min_length_fraction'])
    result = dict(passed=not failed, failures=failed, protocol=PROTOCOL)
    if group_limits is not None:
        names = [name for name, weight in zip(MODE_NAMES, lesson.mode_weights) if weight > 0.]
        result['groups'] = {}
        for group, selected in (('clean', ['clean'] if 'clean' in names else []),
                                ('wind', [name for name in names if name != 'clean'])):
            failures = [failure for failure in failed
                        if any(failure.startswith('wind/' + name + '/') for name in selected)]
            result['groups'][group] = dict(passed=not failures, failures=failures,
                populations=selected, limits=dict(group_limits[group]))
    return result


def advance_hold(previous: int, decision: dict, *, stopping: bool = False) -> tuple[int, bool]:
    hold = previous + 1 if decision['passed'] else 0
    return hold, bool(not stopping and decision['passed'] and hold >= HOLD_UPDATES)


def progression_decision(version: int, hold: int, eligible: bool,
                         legacy_validator: Callable[[], bool] | None) -> bool:
    if not eligible or hold < HOLD_UPDATES:
        return False
    if version in (325, 326):
        return True
    if legacy_validator is None:
        raise ValueError('Historical profile requires its original validator')
    return bool(legacy_validator())


def completion_metadata(stage: int, count: int, *, converged: bool) -> dict:
    return dict(training_stage_passed=bool(converged),
                training_course_completed=bool(converged and stage == count - 1),
                final_acceptance_passed=False, independent_evaluation=None,
                convergence_basis=PROTOCOL)
