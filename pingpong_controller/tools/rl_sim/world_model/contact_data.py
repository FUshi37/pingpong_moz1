"""Bounded real contact windows, event-balanced sampling and physical labels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import read_array_archive
from .contact_dynamics import ContactState
from .contact_model import world_spin
from .kinematics import forward_kinematics


CONTACT_STREAMS = (
    'q_before', 'dq_before', 'q_after', 'dq_after',
    'ball_qpos_before', 'ball_qpos_after', 'ball_qvel_before', 'ball_qvel_after',
    'ball_model_valid', 'contact_overlap', 'event_contact_entry',
    'event_contact_release', 'event_controlled_release', 'event_impact_id',
    'event_contact_duration_steps',
)


def check_continuity(first: dict, second: dict) -> None:
    """Keep exact physical endpoints, allowing native quaternion renormalization."""
    for name in ('q', 'dq', 'ball_qpos', 'ball_qvel'):
        left, right = first[name+'_after'][-1], second[name+'_before'][0]
        if name == 'ball_qpos':
            # The native final forward normalizes the ball quaternion. The
            # observed FP32 discrepancy is 1.19e-7, with position unchanged.
            orientation_error = np.minimum(np.linalg.norm(left[:, 3:]-right[:, 3:], axis=-1),
                                           np.linalg.norm(left[:, 3:]+right[:, 3:], axis=-1))
            continuous = np.array_equal(left[:, :3], right[:, :3]) and np.all(orientation_error <= 1e-6)
        else:
            continuous = np.array_equal(left, right)
        if not continuous:
            raise ValueError(f'Physical state changed between contact chunks: {name}')


def read_contact_trial(dataset, first_id: str, second_id: str):
    """Read one authentic continuous 128-control trial; retain no GPU dataset."""
    first, second = (dataset.chunks[key] for key in (first_id, second_id))
    if (first['context_sha256'] != second['context_sha256']
            or first['groups'] != second['groups']
            or second['append_index'] != first['append_index']+1):
        raise ValueError('Contact windows require matching continuous trial/context')
    chunks = []
    for key in (first_id, second_id):
        arrays = read_array_archive(dataset.root/f'chunk_{key}.npz',
            dataset.chunks[key]['archive'], names={'substep__'+s for s in CONTACT_STREAMS},
            release_cache=False)
        chunks.append({name: value.reshape((-1,)+value.shape[2:])
            for full_name, value in arrays.items() for name in [full_name.removeprefix('substep__')]})
    check_continuity(*chunks)
    return ({name: np.concatenate([chunk[name] for chunk in chunks]) for name in CONTACT_STREAMS},
            {**first, 'steps': first['steps']+second['steps'], 'trial_chunks': [first_id, second_id]})


def contact_window_rows(data: dict, record: dict, *, controls: int, split: str) -> list[dict]:
    """Enumerate mutually exclusive physical windows, balanced later by event.

    Contact has first priority. Robot-identification lanes retain their own
    bucket. A nonrobot trial without any actual overlap supplies miss windows;
    the remaining noncontact windows are free flight. Invalid physics never
    becomes a task-failure label. Geometric touch is retained even with zero
    impulse; the separate controlled-release label identifies useful hits.
    """
    if split not in ('train', 'validation'):
        raise ValueError('Final test trajectories cannot guide contact fitting')
    if not 1 <= controls <= record['steps']:
        raise ValueError('Contact window exceeds the continuous trial')
    length = controls*5
    starts = np.arange(record['steps']-controls+1)*5
    rows = []
    for lane, group in enumerate(record['groups']):
        if group['split'] != split:
            continue
        valid = data['ball_model_valid'][:, lane]
        overlap = data['contact_overlap'][:, lane]
        cumulative = np.r_[0, np.cumsum(valid)]
        allowed = starts[cumulative[starts+length]-cumulative[starts] == length]
        root = group['root_seed_family']
        for start in allowed:
            contact_at = np.flatnonzero(overlap[start:start+length])
            if len(contact_at):
                bucket = 'valid_contact'
                event = int(data['event_impact_id'][start+contact_at[0], lane])
            elif group['generator'] == 'robot_identification':
                bucket, event = 'robot_identification', 0
            elif not overlap.any():
                bucket, event = 'miss_failure', 0
            else:
                bucket, event = 'free_flight', 0
            rows.append(dict(lane=lane, start_substep=int(start), bucket=bucket,
                independent_event_id=f'{root}:{bucket}:{event}', split=split,
                root_seed_family=root, region_id=group['region_id']))
    return rows


def take_contact_windows(data: dict, rows: list[dict], indices, *, controls: int):
    selected = [rows[int(i)] for i in indices]
    lanes = np.asarray([row['lane'] for row in selected])
    times = (np.asarray([row['start_substep'] for row in selected])[None, :]
             +np.arange(controls*5)[:, None])
    return ({key: value[times, lanes[None, :]].copy() for key, value in data.items()}, selected)


def prepare_contact_batch(chain, data: dict):
    """Derive both poses and twist from measured q/dq; convert ball spin to world."""
    shape = data['q_before'].shape[:-1]
    fk = jax.vmap(lambda q: forward_kinematics(chain, q))(data['q_before'].reshape(-1, 7))
    velocity = data['dq_before'].reshape(-1, 7)
    linear = jnp.einsum('nij,nj->ni', fk.geom_jacobian, velocity).reshape(shape+(3,))
    angular = jnp.einsum('nij,nj->ni', fk.angular_jacobian, velocity).reshape(shape+(3,))
    overlap = data['contact_overlap']
    entry, release = data['event_contact_entry'], data['event_contact_release']
    previous = (overlap & ~entry) | release
    current_age = data['event_contact_duration_steps'].astype(jnp.float32)*.001
    previous_age = jnp.where(overlap, jnp.maximum(current_age-.001, 0.),
                             jnp.where(release, current_age, 0.))

    def state(suffix, contact, age):
        qpos, qvel = data['ball_qpos_'+suffix], data['ball_qvel_'+suffix]
        return ContactState(qpos[..., :3], qvel[..., :3], world_spin(qpos[..., 3:], qvel[..., 3:]),
                            contact, age)

    return dict(before=state('before', previous, previous_age),
        after=state('after', overlap, jnp.where(overlap, current_age, 0.)),
        racket=dict(geom_position_m=fk.geom_position.reshape(shape+(3,)),
            geom_rotation_world=fk.geom_rotation.reshape(shape+(3, 3)),
            racket_linear_velocity_m_s=linear, racket_angular_velocity_rad_s=angular),
        valid=data['ball_model_valid'], entry=entry, release=release,
        controlled_release=data['event_controlled_release'])
