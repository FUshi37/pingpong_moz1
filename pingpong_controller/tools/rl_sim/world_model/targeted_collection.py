"""Independent spatial proposals for the bounded supplemental acquisition."""

import numpy as np

from .sampling import _balanced, _counts, _shuffle


def missing_regions(requested, available):
    """Return sorted JSON-safe region identifiers absent from actual sources."""
    return sorted(set(map(int, requested)) - set(map(int, available)))


def acquisition_groups(seed, source_lanes):
    """Keep every clone of a complete warm source in its original partition."""
    return [dict(root_seed_family=f'wmj_targeted_{seed}_{int(source)}',
                 session=f'wmj_targeted_{seed}_{int(source)}',
                 episode=f'wmj_targeted_{seed}_lane_{lane}')
            for lane, source in enumerate(source_lanes)]


def independent_targets(point, lower, upper, *, seed, region_id):
    """Two fresh proposals near a source-grid node, inside unchanged task limits.

    The10mm inward proposal offset is an initialization search choice, not an
    added collision clearance or a changed admissible workspace.
    """
    lower, upper, point = (np.asarray(x, np.float64) for x in (lower, upper, point))
    rng = np.random.default_rng(np.random.SeedSequence([seed, region_id]))
    center = np.clip(point, lower + .010, upper - .010)
    return np.clip(center + rng.uniform(-.003, .003, (2, 3)), lower, upper)


def targeted_draw(rng, count, *, region_ids, focus_region_ids):
    """Half uniform over admitted regions, half over declared gap/near/edge regions."""
    region_ids = np.asarray(region_ids, int)
    focus = np.flatnonzero(np.isin(region_ids, focus_region_ids))
    if not len(region_ids) or len(np.unique(region_ids)) != len(region_ids) or not len(focus):
        raise ValueError('Distinct admitted regions and a nonempty registered focus are required')
    global_count, focus_count = _counts(count, [.5, .5])
    global_ids = _balanced(rng, len(region_ids), global_count)
    focused = focus[_balanced(rng, len(focus), focus_count)]
    probability = np.full(len(region_ids), global_count / count / len(region_ids))
    probability[focus] += focus_count / count / len(focus)
    return _shuffle(rng, np.r_[global_ids, focused],
        ['global'] * global_count + ['focus'] * focus_count, probability,
        dict(counts={'global': int(global_count), 'focus': int(focus_count)},
             region_ids=region_ids.tolist(), focus_region_ids=region_ids[focus].tolist(),
             region_probability=probability.tolist(), uncertainty_used=False,
             priority_source='approved_gap_near_side_and_XY_boundary_strata;not_invented_model_errors'))


def warm_fallback(rng, admitted, requested_regions):
    """Keep valid lanes; copy full legal states for invalid setup proposals only."""
    admitted = np.asarray(admitted, bool)
    regions = np.asarray(requested_regions, int)
    if admitted.ndim != 1 or regions.shape != admitted.shape or not admitted.any():
        raise ValueError('At least one admitted warm lane is required')
    mapping = np.arange(len(regions))
    available = np.flatnonzero(admitted)
    records = []
    for lane in np.flatnonzero(~admitted):
        same_region = available[regions[available] == regions[lane]]
        candidates = same_region if len(same_region) else available
        selected = int(rng.choice(candidates))
        mapping[lane] = selected
        records.append(dict(lane=int(lane), source_lane=selected,
            requested_region=int(regions[lane]), actual_region=int(regions[selected]),
            requested_region_missing=not bool(len(same_region)),
            conditional_source_probability=1. / len(candidates),
            scope='invalid_initialization_only;not_selected_by_policy_or_model_outcome'))
    return mapping, regions[mapping], records
