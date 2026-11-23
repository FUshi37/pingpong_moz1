"""Authenticated opt-in full-prior experiment checkpoints and frozen inputs."""

from __future__ import annotations

import json
from pathlib import Path

import jax

from .artifacts import read_array_archive
from .contracts import config_hash, file_hash
from .neural import init_adam
from .physics_components import load_physics_components
from .normal_checkpoint import load_auxiliary
from .state_adapter import restore_numeric_state


SCHEMA = 'wmj_contact_tangent_coupled_checkpoint_v1'
INFERENCE_SOURCES = ('tangent_contact.py', 'tangent_coupling.py', 'tangent_checkpoint.py',
                     'normal_contact.py', 'normal_coupling.py', 'normal_checkpoint.py',
                     'solver_auxiliary.py', 'neural.py')


def model_signature(cfg, reader, structure):
    return config_hash(dict(structure=structure, dataset=reader.identity,
        physics=cfg['physics'], geometry=reader.report['geometry']))


def load_components(cfg, reader, *, robot_checkpoint, contact_checkpoint, auxiliary_checkpoint):
    metadata = json.loads(contact_checkpoint.read_text())
    parent = contact_checkpoint
    if metadata['schema'] == SCHEMA:
        reference = metadata['parent_contact_reference']
        parent = Path(reference['path'])
        if file_hash(parent) != reference['sha256']:
            raise ValueError('Original39-D parent changed')
        if (metadata['dataset_sha256'] != reader.identity
                or metadata['physical_contract'] != reader.dataset.identity
                or metadata['model_signature'] != model_signature(cfg, reader, metadata['structure'])
                or metadata['frozen_robot_reference']['sha256'] != file_hash(robot_checkpoint)
                or metadata['frozen_auxiliary_reference']['sha256'] != file_hash(auxiliary_checkpoint)):
            raise ValueError('Full-prior checkpoint context/frozen-component mismatch')
        for name in INFERENCE_SOURCES:
            if metadata['source_sha256'][name] != file_hash(Path(__file__).parent/name):
                raise ValueError(f'Full-prior checkpoint source changed: {name}')
    parameters, options, references = load_physics_components(cfg, reader,
        robot_checkpoint=robot_checkpoint, contact_checkpoint=parent)
    auxiliary, auxiliary_ref = load_auxiliary(cfg, auxiliary_checkpoint)
    if metadata['schema'] == SCHEMA:
        parent_meta = json.loads(parent.read_text())
        expected = dict(parent_meta['structure'], tangent_auxiliary=metadata['structure']['tangent_auxiliary'])
        if metadata['structure'] != expected or metadata['fit_recipe'] != parent_meta['fit_recipe']:
            raise ValueError('Full-prior experiment changed an unrelated model/optimizer contract')
        restored = restore_numeric_state(dict(parameters=parameters['contact'], optimizer=init_adam(parameters['contact'])),
            read_array_archive(contact_checkpoint.parent/metadata['archive_file'], metadata['archive']),
            metadata['state'], context_sha256=metadata['model_signature'])
        parameters['contact'] = restored['parameters']
        references['contact'] = dict(path=str(contact_checkpoint.absolute()), sha256=file_hash(contact_checkpoint),
            model_signature=metadata['model_signature'], optimizer_updates=metadata['optimizer_updates'])
    return parameters, options, references, auxiliary, auxiliary_ref
