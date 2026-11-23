"""Contact windows conditioned on true native pose/twist from support replays."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .contact_data import CONTACT_STREAMS, check_continuity
from .contact_dynamics import ContactState
from .contact_model import world_spin
from .support_data import POSE_STREAMS, SUPPORT_STREAMS


def read_native_contact_trial(reader, first_id, second_id):
    """Join one canonical acquisition, never mix original and replayed physics."""
    first, second = (reader.records[key] for key in (first_id, second_id))
    if (first['record']['groups'] != second['record']['groups']
            or first['first_seed_ref'] != second['first_seed_ref']
            or (first['replay_controls_before_chunk'], second['replay_controls_before_chunk']) != (0, 64)):
        raise ValueError('Native contact labels require one continuous trial')
    chunks = []
    for key in (first_id, second_id):
        arrays = reader.read(key, names={'substep__'+name for name in
                            (*CONTACT_STREAMS, *SUPPORT_STREAMS, *POSE_STREAMS)})
        chunk = {}
        for name in (*CONTACT_STREAMS, *POSE_STREAMS):
            value = arrays['substep__'+name]
            if name in ('q_before', 'dq_before', 'q_after', 'dq_after'):
                value = np.concatenate((value, arrays['substep__support_'+name]), axis=-1)
            chunk[name] = value.reshape((-1,)+value.shape[2:])
        chunks.append(chunk)
    check_continuity(*chunks)
    return ({name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]},
            {**first['record'], 'steps': first['record']['steps']+second['record']['steps'],
             'trial_chunks': [first_id, second_id]})


def prepare_native_contact_batch(unused_chain, data):
    """Keep native robot conditioning; only initial ball state seeds autoregression."""
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
        racket=dict(geom_position_m=data['racket_geom_position_before_m'],
            geom_rotation_world=data['racket_geom_rotation_before'],
            racket_linear_velocity_m_s=data['racket_linear_velocity_before_m_s'],
            racket_angular_velocity_rad_s=data['racket_angular_velocity_before_rad_s']),
        valid=data['ball_model_valid'], entry=entry, release=release,
        controlled_release=data['event_controlled_release'])
