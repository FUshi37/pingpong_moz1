"""Preregistered action diagnostics for the user-authorized WMJ-V1 pilot."""

import numpy as np


def counterfactual_actions(raw_actions, cases):
    """Replay saved raw pulses; clipping remains inside the original controller."""
    raw = np.asarray(raw_actions)
    if raw.ndim != 3 or raw.shape[0] != 128 or raw.shape[2] != 7:
        raise ValueError('Expected 128 controls with seven raw actions')
    lanes = [row['lane'] for row in cases]
    if len(set(lanes)) != len(lanes) or any(i < 0 or i >= raw.shape[1] for i in lanes):
        raise ValueError('Invalid or duplicate source lanes')
    base = raw[:, lanes].copy()
    pulse = np.zeros_like(base)
    for i, row in enumerate(cases):
        if (row['pulse_ms'] not in (10, 20) or not 0 <= row['joint'] < 7
                or not any(np.isclose(row['epsilon'], x) for x in (.005, .01, .02))):
            raise ValueError('Unregistered saved action pulse')
        pulse[:row['pulse_ms'] // 5, i, row['joint']] = row['epsilon']
    return dict(base=base, plus=base + pulse, minus=base - pulse)


def action_gate(parent, candidate):
    """Diagnostic admission only; preserve unresolved/missing-event denominators."""
    rows = {}
    for name in ('outgoing_velocity_m_s', 'actual_apex_m'):
        old, new = parent[name], candidate[name]
        count = new['effect_exceeds_repeat_cases']
        rate = new['resolved_direction_agreement_fraction']
        parent_rate = old['resolved_direction_agreement_fraction']
        rows[name] = dict(informative=count, fraction=rate,
            parent_informative=old['effect_exceeds_repeat_cases'], parent_fraction=parent_rate,
            passed=bool(count >= 100 and count >= .9 * old['effect_exceeds_repeat_cases']
                and rate is not None and np.isfinite(rate) and rate >= .90
                and parent_rate is not None and rate >= parent_rate - .05))
    return dict(passed=all(row['passed'] for row in rows.values()), metrics=rows,
        minimum_informative=100, minimum_parent_count_fraction=.9,
        minimum_agreement=.90, maximum_fraction_drop=.05,
        full_world_qualification=False)
