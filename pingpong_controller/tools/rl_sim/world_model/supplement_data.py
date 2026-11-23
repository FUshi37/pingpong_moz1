"""Read direct full-support supplement chunks without a duplicate replay overlay."""

import json
from pathlib import Path

import numpy as np

from .artifacts import read_array_archive
from .contracts import authenticate, config_hash, repository_root
from .fit_robot import dataset_identity
from .support_data import POSE_STREAMS, SUPPORT_STREAMS
from .trajectory_data import TrajectoryDataset


def diagnostic_batches(groups, *, width=64):
    """Select all eligible train/validation sources, never final-test outcomes."""
    if width < 1 or any(row['split'] not in ('train', 'validation', 'test') for row in groups):
        raise ValueError('Invalid width or unregistered source partition')
    excluded = dict(test=[], invalid_initial_geometry=[])
    selected = []
    for lane, row in enumerate(groups):
        if row['split'] == 'test':
            excluded['test'].append(lane)
        elif not row['initial_geometry_admitted']:
            excluded['invalid_initial_geometry'].append(lane)
        else:
            selected.append(lane)
    batches = []
    for start in range(0, len(selected), width):
        useful = selected[start:start+width]
        padded = useful + [useful[0]] * (width-len(useful))
        batches.append(dict(lanes=np.asarray(padded, int), useful=len(useful)))
    return batches, excluded


class SupplementReader:
    """Authenticate a complete single acquisition and expose coherent trial pairs."""

    def __init__(self, cfg, reference, budget):
        path = authenticate(repository_root(), reference)
        self.root = path.parent
        self.report = json.loads(path.read_text())
        if (self.report['schema'] != 'wmj_v1_independent_targeted_collection'
                or self.report['status'] != 'TARGETED_COLLECTION_COMPLETE'
                or self.report['teacher_transitions'] != 131072
                or self.report['frozen_validation_states_used']):
            raise ValueError('A complete independent1024x128 supplement is required')
        self.dataset = TrajectoryDataset.open(self.root / 'dataset', budget)
        if (self.dataset.identity['control_sha256'] != config_hash(cfg['control'])
                or self.dataset.identity['contact_sha256'] != config_hash(cfg['physics'])):
            raise ValueError('Supplement physics/control differs from the frozen world')
        names = list(self.dataset.chunks)
        if names != ['r0000', 'r0001']:
            raise ValueError('Exactly two chronological64-control chunks are required')
        first = self.dataset.chunks[names[0]]
        self.records = {}
        for half, name in enumerate(names):
            row = self.dataset.chunks[name]
            required = {'substep__' + k for k in (*POSE_STREAMS, *SUPPORT_STREAMS)}
            if ((row['steps'], row['lanes']) != (64, 1024)
                    or row['context_sha256'] != self.report['context_sha256']
                    or row['groups'] != first['groups']
                    or not required.issubset(row['archive']['arrays'])):
                raise ValueError('Supplement labels, source grouping or complete context changed')
            self.records[name] = dict(record=row, first_seed_ref=first['seed_ref'],
                                      replay_controls_before_chunk=64*half)
        self.identity = dataset_identity(self.dataset)

    def read(self, chunk_id, *, names):
        row = self.records[chunk_id]['record']
        return read_array_archive(self.dataset.root / f'chunk_{chunk_id}.npz', row['archive'], names=names)
