"""Immutable complete warmed native populations shared by frozen evaluations."""

import json
from pathlib import Path

from .artifacts import write_json_unique, write_array_archive_unique, read_array_archive
from .contracts import authenticate, config_hash, repository_root
from .state_adapter import pack_numeric_state, restore_numeric_state, numeric_tree_digest


def save_warmed_seed(path,state,admission,context,budget):
    """Freeze actual robot/model/controller/RNG values and offline admission."""
    path=Path(path).absolute()
    arrays,numeric=pack_numeric_state(state,context_sha256=config_hash(context))
    array_path=path.with_suffix('.npz')
    archive=write_array_archive_unique(array_path,arrays,budget)
    payload=dict(schema='wmj_v1_frozen_warmed_seed',context=context,numeric=numeric,
        array_reference=dict(path=str(array_path),sha256=archive['sha256']),archive=archive,
        admission=admission,numeric_sha256=numeric_tree_digest(state),
        warmup_repeated_on_restore=False,policy_outputs_used=False)
    digest=write_json_unique(path,payload,budget)
    return dict(path=str(path),sha256=digest)


def load_warmed_seed(reference,template,context):
    """Reject changed cases/physics/control; restore all shared queue phases."""
    payload=json.loads(authenticate(repository_root(),reference).read_text())
    if payload['schema']!='wmj_v1_frozen_warmed_seed' or payload['context']!=context:
        raise ValueError('Frozen warmed population context changed')
    path=authenticate(repository_root(),payload['array_reference'])
    arrays=read_array_archive(path,payload['archive'])
    state=restore_numeric_state(template,arrays,payload['numeric'],context_sha256=config_hash(context))
    if numeric_tree_digest(state)!=payload['numeric_sha256']:
        raise ValueError('Warmed state numerical identity changed')
    return state,payload['admission']
