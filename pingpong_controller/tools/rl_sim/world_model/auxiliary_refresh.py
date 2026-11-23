"""Explicit auxiliary-only refresh with inherited scales and optimizer memory."""

import json
from pathlib import Path

import jax
import numpy as np

from .artifacts import read_array_archive
from .contracts import authenticate, file_hash, repository_root
from .fit_solver_auxiliary import family_pool, sample_pool
from .neural import init_adam, init_mlp
from .solver_auxiliary import WIDTHS, scale_from_statistics
from .state_adapter import restore_numeric_state


def family_ids(groups):
    """Identify common warm-source forks without treating clones as independent."""
    names = sorted({row['root_seed_family'] for row in groups})
    indexes = {name: index for index, name in enumerate(names)}
    partitions = {}
    for row in groups:
        name, split = row['root_seed_family'], row['split']
        if name in partitions and partitions[name] != split:
            raise ValueError('Common source family crosses a data partition')
        partitions[name] = split
    return np.array([indexes[row['root_seed_family']] for row in groups], np.int32)


def validation_batches(groups, *, width=64):
    """Keep every validation lane, marking padding only for fixed GPU shapes."""
    if width < 1 or any(row['split'] not in ('train','validation','test') for row in groups):
        raise ValueError('Invalid source partition or batch width')
    lanes = [i for i,row in enumerate(groups) if row['split']=='validation']
    batches = []
    for start in range(0,len(lanes),width):
        selected = lanes[start:start+width]
        batches.append(dict(lanes=np.array(selected+[selected[0]]*(width-len(selected)),int),
                            useful=len(selected)))
    return batches


def refresh_indexes(rng, old, new, *, batch_size=512):
    """Equal domains, each half contact/noncontact; sample families uniformly."""
    if batch_size < 4 or batch_size % 4:
        raise ValueError('Batch must divide equally among four source/contact bins')
    return tuple(np.concatenate([sample_pool(rng, family_pool(rows,
        (rows['split'] == 0) & mask), batch_size//4)
        for mask in (rows['overlap'], ~rows['overlap'])]) for rows in (old, new))


def restore_auxiliary(cfg, reference):
    """Load only the pinned parent or a hash-pinned descendant of that parent."""
    path = authenticate(repository_root(), reference)
    metadata = json.loads(path.read_text())
    parent = cfg['auxiliary_refresh']['parent_checkpoint']
    if (reference['sha256'] != parent['sha256']
            and metadata.get('parent_checkpoint') != parent):
        raise ValueError('Auxiliary candidate is not descended from the pinned parent')
    if metadata['schema'] != 'wmj_solver_auxiliary_checkpoint_v1' or metadata['width'] != list(WIDTHS):
        raise ValueError('Auxiliary47/128/128/6 checkpoint contract changed')
    for name in ('solver_auxiliary.py', 'neural.py'):
        if metadata['source_sha256'][name] != file_hash(Path(__file__).parent/name):
            raise ValueError('Auxiliary inference source changed')
    scale = scale_from_statistics(**metadata['normalization'])
    parameters = init_mlp(jax.random.PRNGKey(0), WIDTHS, zero_output=True)
    restored = restore_numeric_state(dict(parameters=parameters, optimizer=init_adam(parameters), scale=scale),
        read_array_archive(path.parent/metadata['archive_file'], metadata['archive']),
        metadata['state'], context_sha256=metadata['model_signature'])
    parent_path = authenticate(repository_root(), parent)
    if metadata['normalization'] != json.loads(parent_path.read_text())['normalization']:
        raise ValueError('Auxiliary refresh must preserve parent normalization exactly')
    return restored, metadata
