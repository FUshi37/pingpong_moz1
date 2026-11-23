"""Position-only augmentation of coherent native contact data; no new replay."""

from __future__ import annotations

import numpy as np

from .native_contact_data import prepare_native_contact_batch, read_native_contact_trial


def read_position_contact_trial(reader, first_id, second_id):
    """Append only the actual same-interval delayed position target."""
    data, record = read_native_contact_trial(reader, first_id, second_id)
    targets = []
    for name in (first_id, second_id):
        value = reader.read(name, names={'substep__delayed_q_command'})['substep__delayed_q_command']
        targets.append(value.reshape((-1,)+value.shape[2:]))
    data['delayed_q_command'] = np.concatenate(targets)
    if data['delayed_q_command'].shape != data['q_before'].shape[:-1]+(7,):
        raise ValueError('Position targets are not aligned with the recorded joint state')
    return data, record


def prepare_position_contact_batch(unused_chain, data):
    """No future q, joint dq, target dq/ddq or solver qacc enters the inputs."""
    batch = prepare_native_contact_batch(None, data)
    batch['racket'].update(joint_q_rad=data['q_before'],
                           delayed_q_command_rad=data['delayed_q_command'])
    return batch


def position_statistics_update(normalizer, arrays, record):
    """Merge current position features on valid training lanes only."""
    q = np.concatenate((arrays['substep__q_before'], arrays['substep__support_q_before']), axis=-1)
    target = arrays['substep__delayed_q_command']
    if q.shape[-1] != 13 or target.shape != q.shape[:-1]+(7,):
        raise ValueError('Position statistics require native arm7/support6 and delayed target7')
    features = np.concatenate((q, target-q[..., :7]), axis=-1)
    train = np.array([group['split'] == 'train' for group in record['groups']])
    valid = arrays['substep__ball_model_valid'] & train
    normalizer.update(features, valid, split='train')


def collect_position_statistics(reader, guard):
    """Bounded CPU reads; source identity and fixed split come from the reader."""
    from .trajectory_data import TrainNormalizer
    normalizer = TrainNormalizer(20)
    names = {'substep__'+name for name in ('q_before', 'support_q_before',
             'delayed_q_command', 'ball_model_valid')}
    for name, entry in reader.records.items():
        guard.check()
        position_statistics_update(normalizer, reader.read(name, names=names), entry['record'])
        print(f'[NORMALIZE] {name} train_substeps={normalizer.count}', flush=True)
    return normalizer.to_dict()
