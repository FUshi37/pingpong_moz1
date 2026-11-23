"""Read-only value diagnostics by source, contact phase and failure distance."""

from __future__ import annotations

import numpy as np


PHASES = ('pre_first_release', 'in_contact_at_start', 'release_control',
          'ascending_after_release', 'descending_after_release')


def failure_horizon(valid: np.ndarray, failed: np.ndarray) -> tuple:
    """Only a first physical failure completes a Monte Carlo return."""
    valid, failed = np.asarray(valid, bool), np.asarray(failed, bool)
    if valid.ndim != 2 or valid.shape != failed.shape:
        raise ValueError('Expected matching time/lane masks')
    if (np.any(failed & ~valid) or np.any(failed.sum(axis=0) > 1)
            or np.any(valid != np.logical_and.accumulate(valid, axis=0))):
        raise ValueError('Expected one continuous valid prefix per source')
    end = np.argmax(failed, axis=0)
    has_failure = failed.any(axis=0)
    if np.any(has_failure & (valid.sum(axis=0) != end+1)):
        raise ValueError('No samples after a first physical failure are allowed')
    complete = valid & has_failure[None, :]
    remaining = np.where(complete, end[None, :]-np.arange(len(valid))[:, None], -1)
    return complete, remaining.astype(np.int16)


def contact_phase(contact: np.ndarray, releases: np.ndarray, vz_m_s: np.ndarray) -> np.ndarray:
    """Label start-state contact and current release; no future outcomes used."""
    previous = np.cumsum(releases, axis=0)-releases
    phase = np.where(previous == 0, 0, np.where(vz_m_s >= 0., 3, 4))
    phase = np.where(contact, 1, phase)
    return np.where(releases > 0, 2, phase).astype(np.int8)


def compact_regression(prediction: np.ndarray, target: np.ndarray,
                       sources: np.ndarray | None = None) -> dict:
    """Report bias and discrimination separately; optionally balance roots."""
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    if prediction.shape != target.shape or prediction.ndim != 1:
        raise ValueError('Expected equal one-dimensional value arrays')
    if not (np.isfinite(prediction).all() and np.isfinite(target).all()):
        raise ValueError('Nonfinite diagnostic values')
    if not target.size:
        return dict(count=0)
    error = prediction-target
    variance = np.var(target)
    spread = np.std(prediction)*np.std(target)
    result = dict(count=int(target.size), bias=float(error.mean()),
        rmse=float(np.sqrt(np.mean(error**2))), prediction_mean=float(prediction.mean()),
        target_mean=float(target.mean()), prediction_std=float(prediction.std()),
        target_std=float(target.std()),
        explained_variance=float(1.-np.var(error)/variance) if variance > 1e-12 else None,
        correlation=float(np.mean((prediction-prediction.mean())*(target-target.mean()))/spread)
        if spread > 1e-12 else None)
    if sources is not None:
        sources = np.asarray(sources)
        if sources.shape != target.shape:
            raise ValueError('Source identity must accompany every value')
        unique = np.unique(sources)
        means = [error[sources == source].mean() for source in unique]
        squares = [(error[sources == source]**2).mean() for source in unique]
        result.update(source_count=len(unique), source_balanced_bias=float(np.mean(means)),
                      source_balanced_rmse=float(np.sqrt(np.mean(squares))))
    return result


def stratified_summary(rows: dict, arms: tuple = ('ema001', 'ema005')) -> dict:
    """Censored trajectories have lambda labels but never complete MC labels."""
    size = len(rows['source'])
    if any(np.asarray(value).shape != (size,) for value in rows.values()):
        raise ValueError('Every diagnostic stream must have one entry per valid row')
    complete = np.asarray(rows['complete'], bool)

    def summarize(mask):
        result = {}
        for arm in arms:
            result[arm] = {}
            pairs = [('online', 'mc'), ('target', 'mc'), ('lambda', 'mc'),
                     ('online', 'lambda'), ('target', 'lambda')]
            for prediction, target in pairs:
                selected = mask & complete if target == 'mc' else mask
                target_key = target if target == 'mc' else arm+'_'+target
                result[arm][prediction+'_vs_'+target] = compact_regression(
                    rows[arm+'_'+prediction][selected], rows[target_key][selected],
                    rows['source'][selected])
        return result

    distance = rows['remaining']
    bins = dict(terminal=distance == 0, controls_1_10=(distance >= 1) & (distance <= 10),
        controls_11_30=(distance >= 11) & (distance <= 30),
        controls_31_64=(distance >= 31) & (distance <= 64),
        controls_65_127=distance >= 65, censored=distance < 0)
    return dict(valid_rows=size, source_count=len(np.unique(rows['source'])),
        complete_sources=len(np.unique(rows['source'][complete])),
        censored_rows=int((~complete).sum()),
        overall=summarize(np.ones(size, bool)),
        phase={name:summarize(rows['phase'] == i) for i, name in enumerate(PHASES)},
        time_to_failure={name:summarize(mask) for name, mask in bins.items()},
        region={str(int(region)):summarize(rows['region'] == region)
                for region in np.unique(rows['region'])})
