"""Pilot snapshots and explicit969-D frozen actor exports; no default promotion."""

import json
from pathlib import Path

from .contracts import authenticate, config_hash, file_hash, repository_root
from .imagination_checkpoint import runtime_signature, restore_checkpoint
from .migration import expand_actor_history, load_v321_actor, _validate_actor


def pilot_signature(cfg, evidence):
    core = runtime_signature(cfg, model_references=evidence['model_references'],
        geometry_sha256=evidence['geometry_sha256'], critic_layout=evidence['critic_layout'])
    names = ('pilot_runtime.py', 'pilot_learning.py', 'pilot_checkpoint.py',
             'train_pilot.py', 'course_learner.py', 'course_rollout.py', 'course_env.py')
    return config_hash(dict(core=core, training_bank=evidence['training_bank'],
        sources={name: file_hash(Path(__file__).parent / name) for name in names},
        stage='S01', maximum_actor_updates=20, native_interval=10))


def load_pilot_actor(cfg, reference):
    payload = json.loads(authenticate(repository_root(), reference).read_text())
    meta = payload['metadata']
    if (meta['kind'] != 'bounded_S01_pilot_actor' or meta['actor_updates'] not in (10, 20)
            or meta['candidate_checkpoint'] != cfg['bounded_imagination_pilot']['candidate_checkpoint']
            or meta['teacher_transitions_used_for_learning'] != 0):
        raise ValueError('Unexpected pilot actor or transition provenance')
    template = dict(actor=expand_actor_history(load_v321_actor(repository_root(), cfg['lineage']['actor_source'])))
    state = restore_checkpoint(reference, template, signature=payload['runtime_signature'])
    _validate_actor(state['actor'], 969)
    return state['actor'], meta
