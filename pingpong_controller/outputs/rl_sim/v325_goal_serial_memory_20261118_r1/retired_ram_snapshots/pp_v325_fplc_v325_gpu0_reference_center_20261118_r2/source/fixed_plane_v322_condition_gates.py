"""Opt-in V322 admission by initial condition; all failures remain counted."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import replace

import numpy as np

from fixed_plane_v321_long_episode import EndpointWindow

CONDITION_NAMES = ('nominal', 'near_0_50', 'near_50_80', 'near_80_100', 'mirror')
# Frozen from the original M11 native sampled-action experiment, before
# consulting candidate outcomes. See FIXED_CONDITION_GATES.json in
# outputs/rl_sim/v322_grouped_conditions_20261031_r1. These are task gates,
# never an online reference-model comparison or an adaptive threshold.
CONDITION_SURVIVAL_FLOORS = (.93, .90, .70, .50, .90)


def condition_gate_settings() -> dict:
    """Return independent copies of the fixed deployment-priority gates."""
    return {name: dict(min_n=64, min_survival=floor,
            min_length_fraction=.95 if name == 'nominal' else floor)
            for name, floor in zip(CONDITION_NAMES, CONDITION_SURVIVAL_FLOORS)}


def required_conditions(stage_index: int) -> tuple[str, ...]:
    """Primary longevity first; final graduation still requires every bucket."""
    if stage_index not in range(4):
        raise ValueError('V322 has exactly four lessons')
    return CONDITION_NAMES if stage_index == 3 else ('nominal',)


def centered_acceptance_config(cfg):
    """Final frozen check of deployment-like +/-1mm centered placement only.

    This explicit evaluation control never changes the training distribution.
    Noncentral robustness is separately required by all training group gates.
    """
    return replace(cfg, v225_nominal_fraction=1., v225_mirror_fraction=0.,
                   racket_launch_xy_jitter=min(cfg.racket_launch_xy_jitter, .001))


def calibrate_condition_gates(results: list[dict]) -> dict:
    """Apply the preregistered baseline-only Wilson95/five-point rule.

    This is experiment design, never an adaptive training comparison against
    another actor. Persist the resulting numeric gates before training.
    """
    rows = [r for r in results if r['model'] == 'M11'
            and r['scenario'] == 'native' and r['mode'] == 'sampled']
    if len(rows) != 1:
        raise ValueError('Expected one original native sampled baseline')
    settings = {}
    for name in CONDITION_NAMES:
        point = rows[0]['groups'].get(name, {})
        n = point.get('n', 0)
        if n < 32:
            raise ValueError(f'Insufficient baseline samples for {name}: {n}')
        if name == 'nominal':
            settings[name] = dict(min_n=64, min_survival=.93, min_length_fraction=.95)
            continue
        p = point['survival']
        z = 1.959963984540054
        lower = (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1+z*z/n)
        floor = max(.50, math.floor((lower + 1e-12)*20)/20)
        settings[name] = dict(min_n=64, min_survival=floor, min_length_fraction=floor)
    return settings


def condition_ids(initial_group, initial_y_m):
    """Map native episode groups and saved initial world-Y offsets to buckets.

    Inputs have the same arbitrary shape, including (steps, environments).
    A nominal episode keeps ID0 even after its ball moves toward the robot.
    """
    groups = np.asarray(initial_group)
    offsets = np.asarray(initial_y_m)
    if (groups.shape != offsets.shape or not np.isfinite(offsets).all()
            or not np.isin(groups, (0, 1, 2)).all()):
        raise ValueError('Invalid native initial-condition labels')
    near = np.where(offsets < .05, 1, np.where(offsets < .08, 2, 3))
    return np.where(groups == 0, 0, np.where(groups == 2, 4, near))


class ConditionEndpointWindow:
    """One deadline/failure outcome per episode under its original condition."""
    def __init__(self, seconds: float, dt_s: float, capacity: int = 4096):
        self.window = EndpointWindow(seconds, dt_s, capacity)
        self.labels = deque(maxlen=capacity)

    def add(self, lengths, hits, done, recent, initial_group, initial_y_m,
            *, terminated):
        arrays = [np.asarray(v) for v in
                  (lengths, hits, done, recent, terminated)]
        ids = condition_ids(initial_group, initial_y_m)
        if any(a.shape != ids.shape for a in arrays):
            raise ValueError('Condition labels must match every transition')
        selected = ((arrays[0] == self.window.steps)
                    | (arrays[2] & (arrays[0] < self.window.steps)))
        self.labels.extend(ids[selected].tolist())
        self.window.add(*arrays[:4], terminated=arrays[4])

    def summary(self):
        values = np.asarray(self.window.outcomes, dtype=float)
        labels = np.asarray(self.labels)
        result = {}
        for index, name in enumerate(CONDITION_NAMES):
            selected = values[labels == index]
            result[name] = ({'n': len(selected),
                'len_fraction': float(selected[:, 0].mean()),
                'hits_mean': float(selected[:, 1].mean()),
                'survival': float(selected[:, 2].mean())} if len(selected) else {'n': 0})
        return result


def condition_gates_passed(summaries: dict, settings: dict, *, required=None) -> bool:
    """Require each preregistered condition; never promote from overall mean."""
    if set(settings) != set(CONDITION_NAMES):
        raise ValueError('Every native condition requires a declared gate')
    names = CONDITION_NAMES if required is None else tuple(required)
    if not names or not set(names).issubset(CONDITION_NAMES):
        raise ValueError('Invalid required conditions')
    for name in names:
        limits = settings[name]
        measured = summaries.get(name, {})
        if measured.get('n', 0) < limits['min_n']:
            return False
        for field, threshold in (('survival', limits['min_survival']),
                                 ('len_fraction', limits['min_length_fraction'])):
            value = measured.get(field, float('nan'))
            if not math.isfinite(value) or value < threshold:
                return False
    return True
