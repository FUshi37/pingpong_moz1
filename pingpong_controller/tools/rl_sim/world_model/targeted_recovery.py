"""Resume the same admitted acquisition; keep provenance and replay tolerances."""

import json

import numpy as np

from .artifacts import read_array_archive
from .contracts import authenticate, config_hash, repository_root
from .generators import correlated_action_sequences, generator_assignments, sample_contact_lead_controls
from .state_adapter import pack_numeric_state, restore_numeric_state
from .targeted_collection import acquisition_groups, targeted_draw, warm_fallback


def compare_replay_arrays(first, second):
    """Keep integers exact and allow FP32 arithmetic roundoff in floating leaves."""
    if set(first) != set(second):
        raise ValueError('Replay layouts differ')
    rows = []
    for key, left in first.items():
        left, right = np.asarray(left), np.asarray(second[key])
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError('Replay array layout differs')
        exact = np.array_equal(left, right)
        if left.dtype.kind in 'biu':
            passed = exact
            error = float(np.max(np.abs(left.astype(float)-right.astype(float)), initial=0.))
        else:
            error = float(np.max(np.abs(left-right), initial=0.))
            passed = bool(np.all(np.isfinite(left)) and np.all(np.isfinite(right))
                          and np.allclose(left, right, rtol=1e-5, atol=1e-6))
        if not exact:
            rows.append(dict(path=key, maximum_absolute_difference=error if np.isfinite(error) else None,
                             passed=passed))
    return dict(passed=all(row['passed'] for row in rows), exact=not rows,
                floating_atol=1e-6, floating_rtol=1e-5, discrete_exact=True, differences=rows)


def restore_prepared(cfg, template, context, regions, focus, region_rows, report):
    """Recover authenticated control0 and the same precomputed action schedule.

    This path requires zero committed chunks. It does not perform a new draw,
    warmup or geometry search; deterministic RNG reconstruction is checked
    against the saved setup metadata before its actions are used.
    """
    refs = cfg['targeted_supplement']['prepared_setup']
    root = repository_root()
    paths = {key: authenticate(root, ref) for key, ref in refs.items()}
    old = json.loads(paths['result'].read_text())
    if old['schema'] != 'wmj_v1_independent_targeted_collection' or old['status'] != 'INCOMPLETE':
        raise ValueError('Only an interrupted acquisition setup can be recovered')
    if list((paths['result'].parent/'dataset').glob('chunk_*')):
        raise ValueError('Committed partial acquisitions require a chunk continuation')
    snapshot = json.loads(paths['config'].read_text())
    for key in ('physics', 'control', 'domain', 'collection'):
        if config_hash(snapshot[key]) != config_hash(cfg[key]):
            raise ValueError('Prepared acquisition contract changed')
    settings = cfg['targeted_supplement']; previous = snapshot['targeted_supplement']
    for key in ('seed', 'n_envs', 'controls', 'chunk_controls', 'catalog_reference',
                'near_side_y_lower_m', 'XY_boundary_width_m', 'missing_source_regions'):
        if settings[key] != previous[key]:
            raise ValueError('Prepared sampling settings changed')
    if context != old['context_sha256']:
        raise ValueError('Prepared native context changed')
    trial = json.loads(paths['trial'].read_text())
    admission = json.loads(paths['admission'].read_text())
    initial = json.loads(paths['initial_geometry'].read_text())
    manifest = json.loads(paths['seed'].read_text())
    arrays = read_array_archive(paths['seed'].parent/manifest['archive_file'], manifest['archive'])
    state = restore_numeric_state(template, arrays, manifest, context_sha256=context)
    _, restored_manifest = pack_numeric_state(state, context_sha256=context)
    if restored_manifest['leaves'] != manifest['leaves']:
        raise ValueError('Restored initial state differs from the authenticated arrays')

    rng = np.random.default_rng(settings['seed']); n = settings['n_envs']
    draw = targeted_draw(rng, n, region_ids=regions, focus_region_ids=focus)
    requested = regions[draw.indices]
    for index, row in enumerate(region_rows):
        # Exactly the original per-region candidate-choice RNG consumption.
        rng.permutation(len(row['candidates']))
    admitted = np.array([row['status'] == 'ADMITTED' for row in admission['rows']])
    mapping, effective, substitutions = warm_fallback(rng, admitted, requested)
    assigned = generator_assignments(rng, n, cfg)
    recipe = cfg['collection']['generator_execution']
    actions, modes = correlated_action_sequences(rng, 128, n,
        amplitude_range=recipe['identification_normalized_amplitude_range'])
    mask = assigned == 'boundaries_failure'
    if mask.any():
        actions[:, mask], modes[mask] = correlated_action_sequences(rng, 128, int(mask.sum()),
            amplitude_range=recipe['boundary_normalized_amplitude_range'])
    lead = sample_contact_lead_controls(rng, cfg, n)
    for actual, name in ((requested, 'requested_regions'), (effective, 'effective_regions'),
                         (assigned, 'generator'), (modes, 'excitation_mode'), (lead, 'preview_controls')):
        if not np.array_equal(actual, np.asarray(trial[name])):
            raise ValueError('Recovered action schedule sampling differs at '+name)
    if substitutions != trial['substitutions']:
        raise ValueError('Recovered complete source mapping changed')
    initial_valid = np.array([row['admitted'] for row in initial['rows']])
    groups = [dict(group, warm_source_lane=int(mapping[lane]), region_id=int(effective[lane]),
        requested_region_id=int(requested[lane]), generator=str(assigned[lane]),
        initial_geometry_admitted=bool(initial_valid[lane]),
        contact_offset_class=str(trial['proposal']['offset_class'][lane]))
        for lane, group in enumerate(acquisition_groups(settings['seed'], mapping))]
    report.update(prepared_setup=refs, warmup_admission=admission['summary'],
        recovered_same_acquisition=True, additional_geometry_searches=0,
        recovered_seed_reference=dict(manifest=str(paths['seed']), sha256=refs['seed']['sha256']))
    goal = state.pop('goal')
    return state, goal, groups, actions, assigned, effective, initial_valid
