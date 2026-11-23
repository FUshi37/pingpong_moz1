"""Coherent command-conditioned physics windows; future states are labels only."""

from __future__ import annotations

import numpy as np

from .position_contact_data import read_position_contact_trial


def read_physics_trial(reader, first_id: str, second_id: str) -> tuple[dict, dict]:
    """Read the already approved execution command dq/ddq, retaining contact39."""
    data, record = read_position_contact_trial(reader, first_id, second_id)
    names = ('delayed_dq_command', 'delayed_ddq_command')
    pieces = {name:[] for name in names}
    for chunk in (first_id, second_id):
        arrays = reader.read(chunk, names={'substep__'+name for name in names})
        for name in names:
            values = arrays['substep__'+name]
            pieces[name].append(values.reshape((-1,)+values.shape[2:]))
    for name,values in pieces.items():
        data[name] = np.concatenate(values)
        if data[name].shape != data['delayed_q_command'].shape:
            raise ValueError('Execution commands do not share the physical state clock')
    return data, record
