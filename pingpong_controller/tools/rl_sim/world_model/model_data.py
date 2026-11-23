"""Bounded CPU batches for supervised dynamics modules, with stable splits."""

from __future__ import annotations

import numpy as np
from collections import OrderedDict

from .artifacts import read_array_archive


ROBOT_STREAMS = ('q_before','dq_before','q_after','dq_after',
                 'delayed_q_command','delayed_dq_command','delayed_ddq_command','robot_model_valid')


class RobotChunkReader:
    """Bounded component-only CPU cache; preserve shared read cache for reuse."""

    def __init__(self, dataset, max_bytes=256*1024**2):
        self.dataset = dataset
        self.max_bytes = max_bytes
        self.cache = OrderedDict()

    def read(self, chunk_id):
        path = self.dataset.root/f'chunk_{chunk_id}.npz'
        stat = path.stat()
        identity = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if chunk_id in self.cache:
            previous, arrays, size = self.cache.pop(chunk_id)
            if previous != identity:
                raise ValueError('Immutable robot input changed after loading')
            self.cache[chunk_id] = (identity, arrays, size)
            return arrays
        names = {'substep__'+name for name in ROBOT_STREAMS}
        names.update('state__'+name for name in ('q_rad', 'dq_rad_s', 'time_us'))
        arrays = read_array_archive(path, self.dataset.chunks[chunk_id]['archive'],
                                    names=names, release_cache=False)
        size = sum(value.nbytes for value in arrays.values())
        used = sum(item[2] for item in self.cache.values())
        while self.cache and used+size > self.max_bytes:
            _, (_, _, removed) = self.cache.popitem(last=False)
            used -= removed
        if size <= self.max_bytes:
            for value in arrays.values():
                value.flags.writeable = False
            self.cache[chunk_id] = (identity, arrays, size)
        return arrays


def robot_chunk(arrays):
    """Flatten control/substep axes while retaining the independent lane axis."""
    streams = {}
    for key in ROBOT_STREAMS:
        value = arrays['substep__'+key]
        if value.shape[1] != 5:
            raise ValueError('Expected exactly five native physics substeps')
        streams[key] = value.reshape((-1,)+value.shape[2:])
    return streams


def robot_statistics_update(inputs, targets, arrays, record):
    """Merge only finite valid training rows into module normalization."""
    data = robot_chunk(arrays)
    train = np.asarray([g['split']=='train' for g in record['groups']])
    valid = data['robot_model_valid'] & train[None,:]
    q,dq = data['q_before'],data['dq_before']
    features = np.concatenate((q,dq,data['delayed_q_command']-q,
                               data['delayed_dq_command'],data['delayed_ddq_command']),axis=-1)
    residual = np.concatenate((data['q_after']-q-dq*.001,data['dq_after']-dq),axis=-1)
    inputs.update(features,valid,split='train')
    targets.update(residual,valid,split='train')


def sample_robot_windows(rng, data, record, *, controls, batch_size, split):
    """Sample independent acquisitions and valid control-aligned windows.

    This sampler is for the separately fitted robot execution module. The
    composed world trainer retains its explicit event-balanced P_model mix.
    Final test trajectories are inaccessible to this fitting/validation API.
    """
    if split not in ('train','validation'):
        raise ValueError('Final test data are reserved for final model evaluation')
    if not 1<=controls<=record['steps'] or batch_size<1:
        raise ValueError('Robot window exceeds the continuous storage chunk')
    lanes = np.flatnonzero([g['split']==split for g in record['groups']])
    if not len(lanes):
        raise ValueError('No acquisitions in the requested split')
    possible = record['steps']-controls+1
    # Label validity is independent of task success. Numeric/physical masks
    # select usable execution labels; failed juggling remains training data.
    starts = np.arange(possible)*5
    index = starts[:,None]+np.arange(controls*5)[None,:]
    valid = data['robot_model_valid'][index[:,:,None],lanes[None,None,:]].all(axis=1)
    eligible = [np.flatnonzero(valid[:,i]) for i in range(len(lanes))]
    usable = np.flatnonzero([len(item)>0 for item in eligible])
    if not len(usable):
        raise ValueError('No complete valid robot execution windows')
    selected = rng.choice(usable,size=batch_size)
    chosen_lanes = lanes[selected]
    chosen_start = np.asarray([rng.choice(eligible[i]) for i in selected])
    time = chosen_start[None,:]*5+np.arange(controls*5)[:,None]

    def take(name):
        return data[name][time,chosen_lanes[None,:]].copy()

    batch = {key:take(key) for key in ('q_before','dq_before','q_after','dq_after')}
    batch['commands'] = {key:take(name) for key,name in
        (('q','delayed_q_command'),('dq','delayed_dq_command'),('ddq','delayed_ddq_command'))}
    batch['valid'] = take('robot_model_valid')
    refs = dict(lane=chosen_lanes,start_control=chosen_start,
        region_id=np.asarray([record['groups'][i]['region_id'] for i in chosen_lanes]),
        root_seed_family=[record['groups'][i]['root_seed_family'] for i in chosen_lanes])
    return batch,refs
