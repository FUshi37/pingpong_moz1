"""Unique full numerical snapshots, authenticated before exact learner resume."""

from __future__ import annotations

import json
from pathlib import Path

from .artifacts import read_array_archive, write_array_archive_unique, write_json_unique
from .contracts import config_hash, file_hash
from .state_adapter import pack_numeric_state, restore_numeric_state


def runtime_signature(cfg,*,model_references,geometry_sha256,critic_layout):
    """Bind behavior, frozen models and actual feature slices, excluding run logs."""
    implementation=cfg['imagination_implementation']
    return config_hash(dict(physics=cfg['physics'],control=cfg['control'],policy=cfg['policy'],
        observation=cfg['observation'],task=cfg['task'],domain=cfg['domain'],
        imagination=cfg['imagination'],reward=implementation['reward'],critic=critic_layout,
        model_references=model_references,geometry_sha256=geometry_sha256,
        inference_sources={name:file_hash(Path(__file__).parent/name) for name in
            ('imagined_env.py','imagined_rewards.py','critic_features.py','actor_critic.py',
             'imagined_rollout.py','persistent_pool.py','full_prior_world.py')}))


def save_checkpoint(path,state,*,signature,budget,metadata):
    """Save actor/critic/target/optimizers/pool/RNG/counters as one checked tree."""
    path=Path(path)
    if path.exists() or path.with_suffix('.npz').exists():
        raise FileExistsError('Existing learner evidence cannot be overwritten')
    arrays,manifest=pack_numeric_state(state,context_sha256=signature)
    archive=write_array_archive_unique(path.with_suffix('.npz'),arrays,budget)
    payload=dict(schema='wmj_v1_imagination_checkpoint',runtime_signature=signature,
        state=manifest,archive=archive,archive_file=path.with_suffix('.npz').name,metadata=metadata)
    digest=write_json_unique(path,payload,budget)
    return dict(path=str(path.absolute()),sha256=digest)


def restore_checkpoint(reference,template,*,signature):
    path=Path(reference['path'])
    if file_hash(path)!=reference['sha256']:
        raise ValueError('Learner checkpoint manifest changed')
    payload=json.loads(path.read_text())
    if payload['schema']!='wmj_v1_imagination_checkpoint' or payload['runtime_signature']!=signature:
        raise ValueError('Learner behavior/model/critic schema differs from checkpoint')
    arrays=read_array_archive(path.parent/payload['archive_file'],payload['archive'])
    return restore_numeric_state(template,arrays,payload['state'],context_sha256=signature)
