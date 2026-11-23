"""Offline selection of independent, complete training-partition sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .artifacts import read_array_archive
from .contracts import file_hash


def initial_candidates(groups: list, initial_valid, contact_entries,
                       *, trial: int, minimum_lead_s: float) -> tuple:
    """Use actual first contact only as precontact category/lead metadata.

    No imagined outcome, return, hit count or policy score selects a source.
    A source starts at its actual control0 with honest reset-history validity.
    """
    entries = np.asarray(contact_entries, bool)
    valid = np.asarray(initial_valid, bool)
    n = len(groups)
    if entries.ndim != 2 or entries.shape[1] != n or valid.shape != (n,):
        raise ValueError('Initial labels and time-major contact events differ')
    train = np.array([g['split'] == 'train' for g in groups])
    observed = entries.any(axis=0)
    lead = np.where(observed, (entries.argmax(axis=0) + 1) * .001, -1.)
    eligible = train & valid & observed & (lead >= minimum_lead_s)
    rows = [dict(
        trial=trial, lane=int(i), region_id=int(groups[i]['region_id']),
        root_seed_family=groups[i]['root_seed_family'], split='train',
        time_to_contact_s=float(lead[i]), source_control=0,
        generator=groups[i].get('generator', 'unspecified'))
        for i in np.flatnonzero(eligible)]
    counts = dict(total=n, train=int(train.sum()), eligible=int(eligible.sum()),
                  train_invalid_initial=int((train & ~valid).sum()),
                  train_no_observed_contact=int((train & ~observed).sum()),
                  train_contact_lead_too_short=int((train & observed & (lead < minimum_lead_s)).sum()),
                  exclusion_counts_may_overlap=True)
    return rows, counts


def balanced_selection(rows: list, *, count: int, seed: int) -> tuple:
    """Round-robin regions, sampling unique source families within each region."""
    if any(row['split'] != 'train' for row in rows):
        raise ValueError('Only train sources may enter the training bank')
    if len({row['root_seed_family'] for row in rows}) != len(rows):
        raise ValueError('Source root families must be unique')
    if not 0 < count <= len(rows):
        raise ValueError(f'Training seed shortage: requested{count}, available{len(rows)}')
    rng = np.random.default_rng(seed)
    regions = np.unique([row['region_id'] for row in rows])
    buckets = {int(region): [row for row in rows if row['region_id'] == region]
               for region in regions}
    sizes = {key: len(value) for key, value in buckets.items()}
    for values in buckets.values():
        rng.shuffle(values)
    order = rng.permutation(regions).tolist()
    chosen = []
    while len(chosen) < count:
        for region in order:
            if buckets[region]:
                chosen.append(buckets[region].pop())
            if len(chosen) == count:
                break
    quotas = {key: sum(row['region_id'] == key for row in chosen) for key in sizes}
    selected = [dict(row, conditional_inclusion_probability=quotas[row['region_id']] / sizes[row['region_id']])
                for row in chosen]
    return selected, dict(
        method='region_round_robin_uniform_within_region_without_replacement',
        regions=len(regions), available=len(rows), selected=len(selected),
        region_available=sizes, region_selected=quotas,
        probability_scope='conditional_on_recorded_region_quota', duplicated_families=0)


def incoming_admitted(audit: dict) -> bool:
    """An incoming ball may move laterally; a vertical-drop test is irrelevant."""
    return bool(audit['self_collision']['status'] == 'PASS'
                and audit['initial_ball_clearance']['status'] == 'PASS'
                and audit['bounds'])


def read_initial_pose(reference: dict, *, context_sha256: str) -> tuple:
    """Authenticate native full qpos/qvel, retaining the unmodified initial ball."""
    path = Path(reference['manifest'])
    if file_hash(path) != reference['sha256']:
        raise ValueError('Initial native pose manifest changed')
    manifest = json.loads(path.read_text())
    if manifest['schema'] != 'wmj_numeric_seed_v1' or manifest['context_sha256'] != context_sha256:
        raise ValueError('Initial native pose context changed')
    lookup = {row['path']: (f'leaf_{i:06d}', row) for i, row in enumerate(manifest['leaves'])}
    keys = ["['native'].data.qpos", "['native'].data.qvel"]
    arrays = read_array_archive(path.parent / manifest['archive_file'], manifest['archive'],
                                names={lookup[key][0] for key in keys})
    result = []
    for key in keys:
        name, expected = lookup[key]
        value = arrays[name]
        if (list(value.shape) != expected['shape'] or value.dtype.str != expected['dtype']
                or hashlib.sha256(value.tobytes()).hexdigest() != expected['sha256']):
            raise ValueError('Initial native pose integrity failed')
        result.append(value)
    return tuple(result)
