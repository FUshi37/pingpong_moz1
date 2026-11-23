"""Ordinal release comparison with event-count changes and missing cases visible."""

import numpy as np

from .fit_robot import summarize_errors


def compare_releases(native_release, model_release, native_p, native_v, model_p, model_v, case_ids):
    """Compare each event at its own clock, without nearest-outcome matching.

    Ordinal matching is diagnostic when release counts differ. Its errors and
    the subset with equal counts are both reported; neither hides missing or
    extra events. A nonfinite predicted outcome remains an explicit failure.
    """
    cases, errors, apex_errors, stable_errors = [], [], [], []
    missing = extra = nonfinite = 0
    for lane, case_id in enumerate(case_ids):
        actual = np.flatnonzero(native_release[:, lane])
        predicted = np.flatnonzero(model_release[:, lane])
        same = len(actual) == len(predicted)
        missing += max(len(actual) - len(predicted), 0)
        extra += max(len(predicted) - len(actual), 0)
        pairs = []
        for a, p in zip(actual, predicted):
            vectors = [np.asarray(x[t, lane]) for x, t in
                       ((native_p, a), (native_v, a), (model_p, p), (model_v, p))]
            finite = all(np.isfinite(x).all() for x in vectors)
            nonfinite += not finite
            if finite:
                pa, va, pp, vp = vectors
                ta, tp = max(float(va[2]), 0.) / 9.81, max(float(vp[2]), 0.) / 9.81
                apex_a = pa + va * ta + np.array([0., 0., -.5 * 9.81 * ta**2])
                apex_p = pp + vp * tp + np.array([0., 0., -.5 * 9.81 * tp**2])
                error = float(np.linalg.norm(vp - va))
                apex = float(np.linalg.norm(apex_p - apex_a))
            else:
                error = apex = float('nan')
            errors.append(error); apex_errors.append(apex)
            if same:
                stable_errors.append(error)
            pairs.append(dict(native_substep=int(a), model_substep=int(p), time_shift_ms=int(p-a),
                finite=finite, velocity_error_m_s=error if finite else None,
                analytic_apex_error_m=apex if finite else None))
        cases.append(dict(case_id=case_id, native_releases=len(actual), model_releases=len(predicted),
                          same_release_count=same, pairs=pairs))
    return dict(alignment='ordinal_in_window;unequal_counts_are_ambiguous_not_success',
        paired_events=len(errors), missing_events=missing, extra_events=extra, nonfinite_pairs=nonfinite,
        velocity_error_m_s=summarize_errors([np.asarray(errors)]),
        analytic_apex_error_m=summarize_errors([np.asarray(apex_errors)]),
        same_count_velocity_error_m_s=summarize_errors([np.asarray(stable_errors)]), cases=cases)
