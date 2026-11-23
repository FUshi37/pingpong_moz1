"""Four independent, reproducible CPU samplers with explicit draw probabilities.

Probabilities are marginal probabilities after the fixed-count mixture is
shuffled. Strata/events balance independent sources rather than the number of
overlapping windows stored for one source. No sampler changes a frozen task
set or manufactures uncertainty, contact labels, or missing seed categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import numpy as np


class SamplingShortage(ValueError):
    """A mandatory bucket needs more real collection before it can be sampled."""


@dataclass(frozen=True)
class Draw:
    indices: np.ndarray
    probability: np.ndarray
    component: np.ndarray
    report: dict


def _counts(total: int, weights) -> np.ndarray:
    weights = np.asarray(weights, np.float64)
    if (total < 1 or weights.ndim != 1 or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1.)):
        raise ValueError('Positive population and normalized mixture weights are required')
    ideal = weights*total
    counts = np.floor(ideal).astype(int)
    remaining = total-int(counts.sum())
    order = np.argsort(-(ideal-counts), kind='stable')
    counts[order[:remaining]] += 1
    return counts


def _shuffle(rng, indices, components, distribution, report) -> Draw:
    indices = np.asarray(indices, np.int64)
    components = np.asarray(components)
    order = rng.permutation(len(indices))
    return Draw(indices[order], np.asarray(distribution)[indices[order]], components[order], report)


def _balanced(rng, population: int, count: int):
    if population < 1:
        raise SamplingShortage('No eligible strata remain')
    full, tail = divmod(count, population)
    return np.concatenate((np.tile(np.arange(population), full), rng.permutation(population)[:tail]))


def sample_collection_regions(rng, n_envs: int, *, coverage, normalized_error,
                              cost_s, uncertainty=None, uncertainty_calibrated=False,
                              error_source='validation') -> Draw:
    """P_collect: half global strata, half error/coverage per measured second.

    normalized_error combines the registered outcome/action-effect budgets.
    At startup it is zero and unseen regions receive the coverage prior.
    Missing timings use the same positive pilot-cost prior for every region.
    An ensemble term is permitted only after calibration against real errors.
    """
    if error_source not in ('train', 'validation', 'coverage_only'):
        raise ValueError('Final test data cannot guide collection')
    coverage, error, cost = [np.asarray(v, np.float64) for v in (coverage, normalized_error, cost_s)]
    if (coverage.ndim != 1 or not len(coverage) or error.shape != coverage.shape
            or cost.shape != coverage.shape or not all(np.isfinite(a).all() for a in (coverage, error, cost))
            or np.any(coverage < 0) or np.any(error < 0) or np.any(cost <= 0)):
        raise ValueError('Finite nonnegative coverage/error and positive regional cost are required')
    score = (error+1./(1.+coverage))/cost
    if uncertainty is not None:
        if not uncertainty_calibrated:
            raise ValueError('Ensemble uncertainty must be calibrated against real prediction error')
        uncertainty = np.asarray(uncertainty, np.float64)
        if uncertainty.shape != score.shape or not np.isfinite(uncertainty).all() or np.any(uncertainty < 0):
            raise ValueError('Invalid calibrated uncertainty')
        score += uncertainty/cost
    weak_probability = score/score.sum()
    global_count, weak_count = _counts(n_envs, [.5, .5])
    global_ids = _balanced(rng, len(score), global_count)
    weak_ids = rng.choice(len(score), weak_count, p=weak_probability)
    mixture = global_count/n_envs/len(score) + weak_count/n_envs*weak_probability
    return _shuffle(rng, np.concatenate((global_ids, weak_ids)),
        ['global']*global_count+['weak']*weak_count, mixture,
        dict(sampler='P_collect', counts={'global': int(global_count), 'weak': int(weak_count)},
             uncertainty_used=uncertainty is not None, error_source=error_source,
             region_probability=mixture.tolist(), weight_rule='(error+coverage_prior+calibrated_uncertainty)/cost'))


def sample_model_windows(rng, rows: list[dict], *,
                         bucket_counts=None) -> Draw:
    """P_model: draw independent events uniformly, then a window within each.

    Rows contain an exclusive bucket and independent_event_id. Multiple
    overlapping windows from one impact never multiply its sampling weight.
    Bootstrap replicas may provide their own row population before this draw.
    """
    counts = dict(valid_contact=128, miss_failure=64, robot_identification=38, free_flight=26)
    if bucket_counts is not None:
        counts = dict(bucket_counts)
    if sum(counts.values()) != 256 or any(c < 0 or not isinstance(c, int) for c in counts.values()):
        raise ValueError('P_model must contain 256 declared windows')
    if any(row.get('split') != 'train' for row in rows):
        raise ValueError('Model fitting samplers require the train partition')
    by_bucket = {name: defaultdict(list) for name in counts}
    for i, row in enumerate(rows):
        if row['bucket'] not in by_bucket:
            raise ValueError('Unknown primary model-window bucket')
        event = row.get('independent_event_id')
        if not isinstance(event, str) or not event:
            raise ValueError('An independent event/acquisition identity is required')
        by_bucket[row['bucket']][event].append(i)
    missing = [name for name, count in counts.items() if count and not by_bucket[name]]
    if missing:
        raise SamplingShortage(f'P_model requires collection for: {", ".join(missing)}')
    probability = np.zeros(len(rows), np.float64)
    selected, components = [], []
    for name, count in counts.items():
        events = list(by_bucket[name].values())
        if not count:
            continue
        for event in events:
            probability[event] = count/256/len(events)/len(event)
        event_indices = rng.integers(len(events), size=count)
        selected.extend(int(rng.choice(events[i])) for i in event_indices)
        components.extend([name]*count)
    return _shuffle(rng, selected, components, probability,
        dict(sampler='P_model', counts=counts,
             independent_events={key: len(value) for key, value in by_bucket.items()},
             reallocation=None, replacement=True))


def sample_branch_starts(rng, rows: list[dict], *, minimum_precontact_lead_s: float) -> Draw:
    """P_start: 128 precontact, 64 target-switch and 64 recovery full seeds.

    The caller supplies the registered action-to-contact lead budget; it must
    be checked by actual action-effect branches rather than summing unlike
    control delays and treating that sum as a measured response time.
    """
    if not np.isfinite(minimum_precontact_lead_s) or minimum_precontact_lead_s <= 0:
        raise ValueError('A positive explicit precontact lead time is required')
    if any(row.get('split') != 'train' for row in rows):
        raise ValueError('Imagination branch seeds require the train partition')
    counts = dict(precontact=128, goal_switch=64, recoverable_disturbance=64)
    categories = {}
    for category, count in counts.items():
        eligible = [i for i, row in enumerate(rows) if row['category'] == category
                    and (category != 'precontact'
                         or row['time_to_contact_s'] >= minimum_precontact_lead_s)]
        if not eligible:
            raise SamplingShortage(f'P_start requires collection for: {category}')
        categories[category] = eligible
    probability = np.zeros(len(rows))
    selected, components = [], []
    for category, count in counts.items():
        eligible = categories[category]
        probability[eligible] = count/256/len(eligible)
        selected.extend(rng.choice(eligible, count))
        components.extend([category]*count)
    return _shuffle(rng, selected, components, probability,
        dict(sampler='P_start', counts=counts, minimum_precontact_lead_s=minimum_precontact_lead_s,
             reallocation=None))


def sample_goals(rng, total: int, *, success_rate, improvement, observations,
                 statistics_source='training_diagnostic') -> Draw:
    """P_goal: global/frontier/review 50/30/20, with explicit cold-start fallback."""
    if statistics_source not in ('training_diagnostic', 'validation'):
        raise ValueError('Final test tasks cannot guide goal sampling')
    success, slope, count = [np.asarray(x, np.float64) for x in (success_rate, improvement, observations)]
    if (success.ndim != 1 or not len(success) or slope.shape != success.shape or count.shape != success.shape
            or not all(np.isfinite(a).all() for a in (success, slope, count))
            or np.any((success < 0) | (success > 1)) or np.any(count < 0)):
        raise ValueError('Goal statistics must be finite, matched and physically valid')
    global_count, frontier_count, review_count = _counts(total, [.5, .3, .2])
    frontier = np.flatnonzero((count > 0) & (success >= .2) & (success <= .8) & (slope > 0))
    review = np.flatnonzero((count > 0) & (success >= .9))
    reallocated = {}
    if not len(frontier):
        reallocated['frontier'] = int(frontier_count)
        global_count += frontier_count
        frontier_count = 0
    if not len(review):
        reallocated['review'] = int(review_count)
        global_count += review_count
        review_count = 0
    selected = list(_balanced(rng, len(success), global_count))
    components = ['global']*global_count
    probability = np.full(len(success), global_count/total/len(success))
    for name, eligible, size in (('frontier', frontier, frontier_count), ('review', review, review_count)):
        if size:
            selected.extend(rng.choice(eligible, size))
            components.extend([name]*size)
            probability[eligible] += size/total/len(eligible)
    return _shuffle(rng, selected, components, probability,
        dict(sampler='P_goal', reallocated_to_global=reallocated, statistics_source=statistics_source,
             counts=dict(global_=int(global_count), frontier=int(frontier_count), review=int(review_count)),
             frozen_evaluation_set_modified=False))
