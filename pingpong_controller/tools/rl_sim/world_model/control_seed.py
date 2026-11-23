"""Restore authentic controller memory without loading an entire MJX seed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from recovered_rmp_jax import JointRMPState

from .artifacts import read_array_archive
from .contracts import file_hash
from .world_control import RMP_CONTEXT_FIELDS, control_from_native


def read_control_seed(reference: dict, *, context_sha256: str, lanes):
    """Select lanes only after authenticating every requested numerical leaf."""
    path = Path(reference['manifest'])
    if file_hash(path) != reference['sha256']:
        raise ValueError('Full seed manifest changed')
    metadata = json.loads(path.read_text())
    if metadata['schema'] != 'wmj_numeric_seed_v1' or metadata['context_sha256'] != context_sha256:
        raise ValueError('Full seed context changed')
    names = ('arm_cmd_q', 'prev_action', 'action_scale_mult', 'action_latency_steps',
        'rmp_output_delay_steps', 'rmp_output_q_delay_buffer', 'rmp_output_qd_delay_buffer',
        'rmp_output_qdd_delay_buffer', *('rmp_'+key for key in RMP_CONTEXT_FIELDS))
    required = {"['native']."+name for name in names}
    required.update("['native'].rmp_state."+name for name in JointRMPState._fields)
    lookup = {row['path']:(f'leaf_{i:06d}', row) for i,row in enumerate(metadata['leaves'])}
    if not required.issubset(lookup):
        raise ValueError('Full seed omitted required controller memory')
    arrays = read_array_archive(path.parent/metadata['archive_file'], metadata['archive'],
                                names={lookup[key][0] for key in required})
    values = {}
    for key in required:
        name, expected = lookup[key]
        value = arrays[name]
        if (list(value.shape) != expected['shape'] or value.dtype.str != expected['dtype']
                or hashlib.sha256(value.tobytes()).hexdigest() != expected['sha256']):
            raise ValueError(f'Controller seed leaf changed: {key}')
        values[key.removeprefix("['native'].")] = jnp.asarray(value)
    rmp = JointRMPState(**{name:values['rmp_state.'+name] for name in JointRMPState._fields})
    native = SimpleNamespace(**{name:values[name] for name in names}, rmp_state=rmp)
    state, context = control_from_native(native)
    selected = np.asarray(lanes)
    if (selected.ndim != 1 or selected.dtype.kind not in 'iu' or len(selected) < 1
            or np.any(selected < 0) or np.any(selected >= native.arm_cmd_q.shape[0])):
        raise ValueError('Controller seed selection must contain valid lane indices')
    return jax.tree.map(lambda value:value[selected], (state, context))
