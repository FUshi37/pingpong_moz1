"""Task-scaled execution checks; solver working arrays remain diagnostics."""

from __future__ import annotations

import math


def assess_execution_difference(summary: dict, *, position_atol=1e-5,
                                velocity_atol=1e-4, other_observable_atol=1e-4) -> dict:
    """Assess a complete leaf comparison without requiring identical float bits.

    Generalized position/velocity include their native SI angular/linear units.
    Tolerances are small compared with the registered 2 cm / 0.05 m/s task
    budgets. This is a numerical smoke check, not a long-run task certificate.
    """
    failures, observable_changes, diagnostics = [], [], []
    changes = summary.get('changes', [])
    if summary.get('changed_leaves') != len(changes):
        raise ValueError('A complete leaf difference report is required')
    for row in changes:
        if 'reason' in row or not math.isfinite(row.get('max_abs_diff', math.nan)):
            failures.append(row)
            continue
        path, dtype = row.get('path', ''), row.get('dtype', '')
        # Solver iterations, accelerations, forces, factorizations and warm
        # starts are logged. Actual integrated motion is checked separately.
        internal = '.data.' in path and not any(token in path for token in (
            '.data.qpos', '.data.qvel', '.data.site_xpos', '.data.site_xmat'))
        if internal:
            diagnostics.append(row)
            continue
        if not (dtype.startswith('float') or dtype.startswith('complex')):
            failures.append(dict(row, reason='discrete_state_or_event_changed'))
            continue
        if any(token in path for token in ('qpos', 'site_xpos', 'site_xmat', "['q_after']", "['q_before']", 'arm_cmd_q')):
            limit = position_atol
        elif any(token in path for token in ('qvel', "['dq_after']", "['dq_before']")):
            limit = velocity_atol
        else:
            limit = other_observable_atol
        item = dict(row, absolute_tolerance=limit)
        observable_changes.append(item)
        if row['max_abs_diff'] > limit:
            failures.append(item)
    return dict(status='PASS' if not failures else 'FAIL', failures=failures,
        observable_changes=observable_changes, solver_diagnostics=diagnostics,
        position_absolute_tolerance=position_atol, velocity_absolute_tolerance=velocity_atol,
        other_observable_absolute_tolerance=other_observable_atol,
        bitwise_float_equality_required=False, scope='numerical_smoke_not_task_success')
