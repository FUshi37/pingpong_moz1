"""CPU teacher chunks, stable group partitions and training-only statistics.

State streams have shape (T+1,N,...), action streams (T,N,...), and physical
substeps (T,5,N,...). A full seed at the chunk's first state plus recorded
actions reconstructs controller/camera/history state at an interior window.
The index stores that seed's manifest locator; loading/restoring the full seed
must still authenticate it through state_adapter. No history is fabricated
from the shorter supervised window or duplicated for every transition.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass, field
import json
from pathlib import Path
import re

import numpy as np

from .artifacts import read_array_archive, write_array_archive_unique, write_json_unique
from .datasets import SplitRegistry
from .resources import StorageBudget

ACTION_STREAMS = frozenset(('raw_normalized', 'clipped_normalized',
                            'projected_normalized', 'accepted_normalized'))
PURPOSES = frozenset(('model_fit', 'normalization', 'model_validation',
                      'active_query', 'final_evaluation'))


def _identifier(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_]+', value):
        raise ValueError('Chunk and stream names must be simple identifiers')


@dataclass
class TrajectoryDataset:
    root: Path
    budget: StorageBudget
    identity: dict
    registry: SplitRegistry
    chunks: dict
    cache_max_bytes: int = 256*1024**2
    _cache: OrderedDict = field(default_factory=OrderedDict, repr=False)

    @classmethod
    def create(cls, root: Path, budget: StorageBudget, *, identity: dict,
               split_version: str) -> TrajectoryDataset:
        """Create a new independently named collection; existing data is immutable."""
        for key in ('config_sha256', 'geometry_sha256', 'control_sha256', 'contact_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', identity.get(key, '')):
                raise ValueError(f'Missing dataset identity: {key}')
        if not re.fullmatch('[0-9a-f]{40}', identity.get('source_commit', '')):
            raise ValueError('Missing source commit')
        root = root.absolute()
        if root.exists():
            raise FileExistsError(f'Existing dataset is protected: {root}')
        header = dict(schema='wmj_teacher_dataset_v1', origin='mjx_teacher',
                      identity=identity, split_version=split_version,
                      policy_dt_us=5000, physics_dt_us=1000,
                      history_storage='complete_start_seed_and_continuous_samples')
        write_json_unique(root/'dataset.json', header, budget)
        return cls(root, budget, copy.deepcopy(identity), SplitRegistry(split_version), {})

    @classmethod
    def open(cls, root: Path, budget: StorageBudget) -> TrajectoryDataset:
        """Rebuild assignments from committed chunk manifests, not loose NPZ files."""
        root = root.absolute()
        if not root.resolve().is_relative_to(budget.root.resolve()):
            raise ValueError('Dataset must remain inside its budget root')
        header = json.loads((root/'dataset.json').read_text())
        if header['schema'] != 'wmj_teacher_dataset_v1' or header['origin'] != 'mjx_teacher':
            raise ValueError('Unknown dataset schema/source')
        dataset = cls(root, budget, header['identity'], SplitRegistry(header['split_version']), {})
        records = [json.loads(path.read_text()) for path in root.glob('chunk_*.json')]
        # Assignment order matters when later roots join an existing session.
        # Preserve committed append order, not a filename's lexical order.
        for record in sorted(records, key=lambda item: item['append_index']):
            if record['append_index'] != len(dataset.chunks):
                raise ValueError('Missing or duplicate committed chunk index')
            _identifier(record['chunk_id'])
            if record['identity'] != dataset.identity:
                raise ValueError('Chunk configuration/physics identity mismatch')
            for group in record['groups']:
                split = dataset.registry.assign(**{k: group[k] for k in
                    ('root_seed_family', 'episode', 'session')})
                if split != group['split']:
                    raise ValueError('Recorded group split changed')
            if record['chunk_id'] in dataset.chunks:
                raise ValueError('Duplicate chunk identity')
            dataset.chunks[record['chunk_id']] = record
        return dataset

    def append_chunk(self, chunk_id: str, *, states: dict, actions: dict,
                     substeps: dict, valid, groups: list[dict], seed_ref: dict,
                     context_sha256: str) -> dict:
        """Commit one continuous chunk; resets require a new seed/episode chunk.

        Action meanings are distinct: raw actor/teacher output, external clip,
        projection result, and the accepted action used in the observation.
        For V324 the last two can coincide; do not substitute measured dq.
        Each group is an independent acquisition session, not the whole run.
        """
        _identifier(chunk_id)
        if chunk_id in self.chunks:
            raise FileExistsError(f'Existing chunk is protected: {chunk_id}')
        valid = np.asarray(valid)
        if valid.ndim != 2 or valid.dtype != bool or not all(valid.shape):
            raise ValueError('Transition validity must be a nonempty (T,N) boolean mask')
        steps, lanes = valid.shape
        if len(groups) != lanes:
            raise ValueError('Every lane needs root/session/episode grouping')
        if not re.fullmatch('[0-9a-f]{64}', context_sha256):
            raise ValueError('An authenticated episode context hash is required')
        if (not isinstance(seed_ref.get('manifest'), str)
                or not re.fullmatch('[0-9a-f]{64}', seed_ref.get('sha256', ''))):
            raise ValueError('A complete starting seed manifest locator/hash is required')
        if not {'q_rad', 'time_us', 'ball_source_time_us'}.issubset(states):
            raise ValueError('State streams need measured q and explicit sample clocks')
        if not ACTION_STREAMS.issubset(actions):
            raise ValueError('Raw/clipped/projected/accepted action streams are all required')
        if not substeps:
            raise ValueError('Five physical substeps are required for contact labeling')
        arrays = {'valid': valid}
        for category, streams, prefix in (
                ('state', states, (steps+1, lanes)), ('action', actions, (steps, lanes)),
                ('substep', substeps, (steps, 5, lanes))):
            for name, value in streams.items():
                _identifier(name)
                array = np.asarray(value)
                if array.shape[:len(prefix)] != prefix:
                    raise ValueError(f'{category}/{name} has incompatible time/lane shape')
                arrays[f'{category}__{name}'] = array
        for name in ACTION_STREAMS:
            if arrays[f'action__{name}'].shape != (steps, lanes, 7):
                raise ValueError('Action streams must follow RightArm-0..6 ordering')
        times = arrays['state__time_us']
        sources = arrays['state__ball_source_time_us']
        if (times.shape != (steps+1, lanes) or times.dtype.kind not in 'iu'
                or not np.all(np.diff(times.astype(np.int64), axis=0) == 5000)):
            raise ValueError('State timestamps must advance continuously by 5 ms')
        if (sources.shape != times.shape or sources.dtype.kind not in 'iu'
                or np.any(sources > times)):
            raise ValueError('Ball-source timestamps must never use future samples')
        registry = copy.deepcopy(self.registry)
        grouped = []
        for group in groups:
            split = registry.assign(**{k: group[k] for k in
                ('root_seed_family', 'episode', 'session')})
            grouped.append({**group, 'split': split})
        archive = write_array_archive_unique(self.root/f'chunk_{chunk_id}.npz', arrays, self.budget)
        record = dict(schema='wmj_teacher_chunk_v1', chunk_id=chunk_id,
            append_index=len(self.chunks),
            identity=self.identity, context_sha256=context_sha256, seed_ref=seed_ref,
            groups=grouped, steps=steps, lanes=lanes, transitions=steps*lanes,
            valid_transitions=int(valid.sum()), archive=archive)
        # An interrupted write can leave an unindexed archive. It stays as
        # evidence, consumes the budget, and is never silently used for fitting.
        write_json_unique(self.root/f'chunk_{chunk_id}.json', record, self.budget)
        self.registry = registry
        self.chunks[chunk_id] = record
        return record

    def read_window(self, chunk_id: str, *, lane: int, start: int, length: int,
                    split: str, purpose: str) -> dict:
        """Read one CPU window; restore its full seed and replay to start separately."""
        record = self.chunks[chunk_id]
        if purpose not in PURPOSES:
            raise ValueError('An explicit data access purpose is required')
        if not 0 <= lane < record['lanes'] or start < 0 or length < 1 or start+length > record['steps']:
            raise ValueError('Window would cross the continuous chunk boundary')
        group = record['groups'][lane]
        if split != group['split']:
            raise ValueError('Requested split disagrees with the stable group assignment')
        if split == 'test' and purpose != 'final_evaluation':
            raise ValueError('Final test data cannot be used for fitting or active queries')
        if purpose in ('model_fit', 'normalization') and split != 'train':
            raise ValueError('Fitting and normalization require the train split')
        arrays = self._read_chunk_arrays(chunk_id)
        window = dict(states={}, actions={}, substeps={})
        for key, value in arrays.items():
            if key.startswith('state__'):
                window['states'][key[7:]] = value[start:start+length+1, lane].copy()
            elif key.startswith('action__'):
                window['actions'][key[8:]] = value[start:start+length, lane].copy()
            elif key.startswith('substep__'):
                window['substeps'][key[9:]] = value[start:start+length, :, lane].copy()
        window.update(valid=arrays['valid'][start:start+length, lane].copy(),
            origin='mjx_teacher', group=copy.deepcopy(group), seed_ref=copy.deepcopy(record['seed_ref']),
            context_sha256=record['context_sha256'], replay_control_steps_before_window=start)
        return window

    def _read_chunk_arrays(self, chunk_id: str) -> dict:
        """Keep only a bounded CPU chunk cache; window copies never retain it."""
        path = self.root/f'chunk_{chunk_id}.npz'
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        if chunk_id in self._cache:
            previous, arrays, count = self._cache.pop(chunk_id)
            if previous != identity:
                raise ValueError('An immutable dataset chunk changed after loading')
            self._cache[chunk_id] = (identity, arrays, count)
            return arrays
        arrays = read_array_archive(path, self.chunks[chunk_id]['archive'])
        count = sum(value.nbytes for value in arrays.values())
        if count <= self.cache_max_bytes:
            used = sum(item[2] for item in self._cache.values())
            while self._cache and used+count > self.cache_max_bytes:
                _, (_, _, removed) = self._cache.popitem(last=False)
                used -= removed
            for value in arrays.values():
                value.flags.writeable = False
            self._cache[chunk_id] = (identity, arrays, count)
        return arrays

    def read_sequence(self, chunk_ids: list[str], *, lane: int, start: int,
                      length: int, split: str, purpose: str) -> dict:
        """Join a continuous episode across chunks for up to 128-step audits.

        Storage boundaries must not become physical resets or truncate model
        validation. Matching episode/context identities and exact saved shared
        endpoints are required; no interpolation fills gaps between chunks.
        The first full seed and recorded actions remain the reconstruction path.
        """
        if not chunk_ids or length < 1:
            raise ValueError('A nonempty continuous sequence is required')
        windows, segments, remaining = [], [], length
        first_record = self.chunks[chunk_ids[0]]
        if not 0 <= lane < first_record['lanes']:
            raise ValueError('Unknown lane')
        first_group = first_record['groups'][lane]
        for index, chunk_id in enumerate(chunk_ids):
            if not remaining:
                break
            record = self.chunks[chunk_id]
            if not 0 <= lane < record['lanes']:
                raise ValueError('Lane count changed inside an episode')
            group = record['groups'][lane]
            if any(group[k] != first_group[k] for k in ('root_seed_family', 'episode', 'session', 'split')):
                raise ValueError('Cannot concatenate different episodes or acquisition groups')
            if record['context_sha256'] != first_record['context_sha256']:
                raise ValueError('Physical context changed inside an episode')
            begin = start if index == 0 else 0
            take = min(remaining, record['steps']-begin)
            window = self.read_window(chunk_id, lane=lane, start=begin, length=take,
                                      split=split, purpose=purpose)
            if windows:
                for category in ('states', 'actions', 'substeps'):
                    if window[category].keys() != windows[0][category].keys():
                        raise ValueError('Continuous stream layout changed')
                for name, value in window['states'].items():
                    # Older collection files store prior-step native_done as
                    # a diagnostic and initialize it to zero at a chunk start.
                    # It is not a physical state or WMJ termination signal.
                    if not name.endswith('_diagnostic') and not np.array_equal(windows[-1]['states'][name][-1], value[0]):
                        raise ValueError(f'Chunks do not share a continuous endpoint: {name}')
            windows.append(window)
            segments.append(dict(chunk_id=chunk_id, start=begin, length=take))
            remaining -= take
        if remaining:
            raise ValueError('Not enough continuous recorded steps for the requested window')
        result = {k: copy.deepcopy(v) for k, v in windows[0].items()
                  if k not in ('states', 'actions', 'substeps', 'valid')}
        for category in ('states', 'actions', 'substeps'):
            result[category] = {name: np.concatenate([
                window[category][name][1:] if category == 'states' and i else window[category][name]
                for i, window in enumerate(windows)], axis=0) for name in windows[0][category]}
        result['valid'] = np.concatenate([window['valid'] for window in windows])
        result['storage_segments'] = segments
        return result


class TrainNormalizer:
    """Merge finite valid training samples with stable population statistics.

    These statistics are for learned dynamics features. The released actor
    prefix retains its explicit encoding and no running normalization.
    """

    def __init__(self, features: int):
        if features < 1:
            raise ValueError('At least one feature is required')
        self.count = 0
        self.mean = np.zeros(features, np.float64)
        self.m2 = np.zeros(features, np.float64)

    def update(self, values, valid_mask, *, split: str) -> None:
        if split != 'train':
            raise ValueError('Normalization statistics must use train data only')
        values, mask = np.asarray(values), np.asarray(valid_mask)
        if values.shape[:-1] != mask.shape or values.shape[-1] != len(self.mean) or mask.dtype != bool:
            raise ValueError('Normalization feature/mask shape mismatch')
        selected = values[mask].astype(np.float64)
        if not np.isfinite(selected).all():
            raise ValueError('Valid training samples must be finite')
        if not len(selected):
            return
        count = len(selected)
        mean = selected.mean(axis=0)
        delta = mean-self.mean
        total = self.count+count
        self.m2 += np.sum((selected-mean)**2, axis=0) + delta**2*self.count*count/total
        self.mean += delta*count/total
        self.count = total

    def to_dict(self) -> dict:
        if self.count == 0:
            raise ValueError('Cannot normalize without valid training data')
        return dict(schema='wmj_train_normalization_v1', source='train', count=self.count,
                    mean=self.mean.tolist(), variance=(self.m2/self.count).tolist())
