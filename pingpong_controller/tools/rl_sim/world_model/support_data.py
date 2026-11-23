"""Authenticated full-chain robot batches from coherent native replay overlays."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

from .artifacts import read_array_archive
from .contracts import config_hash, file_hash
from .fit_robot import dataset_identity
from .model_data import ROBOT_STREAMS, sample_robot_windows
from .replay_overlay import decode_overlay
from .trajectory_data import TrajectoryDataset


POSE_STREAMS = ('racket_geom_position_before_m', 'racket_geom_rotation_before',
    'racket_site_position_before_m', 'racket_linear_velocity_before_m_s',
    'racket_angular_velocity_before_rad_s')
SUPPORT_STREAMS = ('support_q_before', 'support_dq_before',
                   'support_q_after', 'support_dq_after')


class SupportReplayReader:
    """Bounded CPU component reader; every overlay and base file is authenticated."""

    def __init__(self, root, budget, *, diagnostic=False, max_bytes=256*1024**2):
        self.root = Path(root).absolute()
        self.report = json.loads((self.root/'result.json').read_text())
        allowed = ('REPLAY_COMPLETE', 'DIAGNOSTIC_COMPLETE') if diagnostic else ('REPLAY_COMPLETE',)
        if self.report['schema'] != 'wmj_support_replay_v1' or self.report['status'] not in allowed:
            raise ValueError('A completed support replay is required; partial data cannot qualify fitting')
        collection = Path(self.report['source_collection'])
        if file_hash(collection/'result.json') != self.report['source_result_sha256']:
            raise ValueError('Source collection report changed')
        self.dataset = TrajectoryDataset.open(collection/'dataset', budget)
        if dataset_identity(self.dataset) != self.report['source_dataset_sha256']:
            raise ValueError('Replay source dataset changed')
        self.records = {}
        for row in self.report['chunks']:
            name = row['chunk_id']
            path = self.root/f'chunk_{name}.json'
            if file_hash(path) != row['sha256']:
                raise ValueError('Support replay metadata changed')
            record = json.loads(path.read_text())
            base_path = self.dataset.root/f'chunk_{name}.npz'
            if (record['record'] != self.dataset.chunks[name]
                    or record['base_archive_path'] != str(base_path.absolute())
                    or record['base_archive'] != self.dataset.chunks[name]['archive']
                    or record['source_dataset_sha256'] != self.report['source_dataset_sha256']
                    or record['base_record_sha256'] != file_hash(base_path.with_suffix('.json'))):
                raise ValueError('Replay overlay base identity mismatch')
            self.records[name] = record
        if not diagnostic and len(self.records) != len(self.dataset.chunks):
            raise ValueError('Support replay omitted required acquisition chunks')
        if not self.records or len(self.records) % 2:
            raise ValueError('Support replay requires complete continuous trial pairs')
        self.identity = config_hash(dict(source=self.report['source_dataset_sha256'],
                                        overlays=self.report['chunks']))
        self.max_bytes, self.cache = max_bytes, OrderedDict()

    def read(self, chunk_id, *, names=None):
        record = self.records[chunk_id]
        if names is None:
            names = {'substep__'+name for name in (*ROBOT_STREAMS, *SUPPORT_STREAMS, *POSE_STREAMS)}
            names.update('state__'+name for name in
                         ('q_rad', 'dq_rad_s', 'support_q_rad', 'support_dq_rad_s', 'time_us'))
        names = frozenset(names)
        key = chunk_id, names
        base_path, path = Path(record['base_archive_path']), self.root/record['archive_file']
        identity = tuple((p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns)
                         for p in (base_path, path, self.root/f'chunk_{chunk_id}.json'))
        if key in self.cache:
            prior, arrays, size = self.cache.pop(key)
            if prior != identity:
                raise ValueError('Immutable support input changed after loading')
            self.cache[key] = prior, arrays, size
            return arrays
        if not names or not names.issubset(record['layout']):
            raise ValueError('Unknown support replay streams')
        base_names = {name for name in names if record['layout'][name]['encoding'] in ('base', 'xor')}
        overlay_names = {name for name in names if record['layout'][name]['encoding'] != 'base'}
        base = read_array_archive(base_path, record['base_archive'], names=base_names,
                                  release_cache=False) if base_names else {}
        overlay = read_array_archive(path, record['archive'], names=overlay_names,
                                     release_cache=False) if overlay_names else {}
        arrays = decode_overlay(base, overlay, record['layout'], names=names)
        size = sum(value.nbytes for value in arrays.values())
        used = sum(item[2] for item in self.cache.values())
        while self.cache and used+size > self.max_bytes:
            _, (_, _, removed) = self.cache.popitem(last=False)
            used -= removed
        if size <= self.max_bytes:
            for value in arrays.values():
                value.flags.writeable = False
            self.cache[key] = identity, arrays, size
        return arrays


def support_chunk(arrays):
    """Return continuous 1 kHz streams with arm7/support6 joint states."""
    result = {}
    for name in (*ROBOT_STREAMS, *POSE_STREAMS):
        value = arrays['substep__'+name]
        if name in ('q_before', 'dq_before', 'q_after', 'dq_after'):
            value = np.concatenate((value, arrays['substep__support_'+name]), axis=-1)
        if value.shape[1] != 5:
            raise ValueError('Expected five native 1 ms substeps')
        result[name] = value.reshape((-1,)+value.shape[2:])
    return result


def support_statistics_update(inputs, targets, arrays, record):
    """Training-only normalization preserves the 35/14 arm-prefix layouts."""
    data = support_chunk(arrays)
    valid = data['robot_model_valid'] & np.asarray([g['split'] == 'train' for g in record['groups']])[None, :]
    q, dq = data['q_before'], data['dq_before']
    features = np.concatenate((q[..., :7], dq[..., :7], data['delayed_q_command']-q[..., :7],
        data['delayed_dq_command'], data['delayed_ddq_command'], q[..., 7:], dq[..., 7:]), axis=-1)
    delta_q, delta_dq = data['q_after']-q-dq*.001, data['dq_after']-dq
    residual = np.concatenate((delta_q[..., :7], delta_dq[..., :7],
                               delta_q[..., 7:], delta_dq[..., 7:]), axis=-1)
    inputs.update(features, valid, split='train')
    targets.update(residual, valid, split='train')


def sample_support_windows(rng, data, record, **kwargs):
    """Sample true native pose/twist labels at exactly the selected joint times."""
    batch, refs = sample_robot_windows(rng, data, record, **kwargs)
    ticks = refs['start_control'][None, :]*5+np.arange(kwargs['controls']*5)[:, None]
    for name in POSE_STREAMS:
        batch[name] = data[name][ticks, refs['lane'][None, :]].copy()
    return batch, refs
