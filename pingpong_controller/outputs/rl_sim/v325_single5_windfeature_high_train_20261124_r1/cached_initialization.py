"""Restore actual pristine native trajectories using allocation-free templates."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np


def shape_views(tree):
    import jax
    return jax.tree.map(lambda item:np.broadcast_to(np.zeros((),dtype=item.dtype),item.shape),tree)


def physical_config_matches(cfg, recorded):
    return json.loads(json.dumps(asdict(cfg)))==recorded


def runner_template(env, ppo, keys, rng):
    import jax
    state_shape,obs_shape=jax.eval_shape(env.reset,keys)
    critic_shape=jax.eval_shape(env.get_critic_obs,state_shape,obs_shape)
    state,obs,critic=shape_views((state_shape,obs_shape,critic_shape))
    return ppo.RunnerState(state,obs,critic,np.zeros(rng.shape,dtype=rng.dtype),
                           np.zeros(env.n_envs,np.float32),np.zeros(env.n_envs,np.int32))


def restore_cached(plan, env, learner, template):
    from training_snapshot import load_snapshot,state_digest
    cache=plan['cached_initialization']
    def verify(path,expected):
        path=Path(path)
        if path.is_symlink():raise ValueError('Cached evidence must be a regular file')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:
                raise ValueError('Cached evidence hash mismatch')
        return path
    snapshot=verify(cache['snapshot'],cache['snapshot_sha256'])
    evidence=verify(cache['preflight'],cache['preflight_sha256'])
    donor=json.loads(evidence.read_text())
    if not physical_config_matches(env.cfg,donor['physical_cfg']):
        raise ValueError('Cached trajectory physics/reward configuration changed')
    with ZipFile(snapshot) as archive:
        manifest=json.loads(archive.read('manifest.json'))
    contract=manifest['contract']
    expected=dict(source_sha256=plan['source_sha256'],actor_critic_action=[118,453,7],
        seed=plan['seed'],n_envs=1024,n_steps=128,pose_limit_deg=5.,pose_ramp_s=0.,
        rehearsal=plan['rehearsal'],onset=plan['onset'],wind_domain=plan['wind_domain'],
        timing=plan['timing'])
    for key,value in expected.items():
        if contract[key]!=value:raise ValueError('Cached trajectory contract mismatch: '+key)
    restored,runner,aux,metadata=load_snapshot(snapshot,learner,template,contract)
    if metadata['actual_updates']!=0 or aux:
        raise ValueError('Cached initialization contains trained state or convergence history')
    if state_digest(restored)!=state_digest(learner):
        raise ValueError('Cached learner is not the exact pinned parent and Adam')
    if state_digest(runner)!=cache['expected_runner_digest']:
        raise ValueError('Cached native trajectory digest mismatch')
    if not (np.asarray(runner.env_state.wind_mode)==0).all():
        raise ValueError('Calm foundation cache contains wind')
    receipt={**donor['rehearsal_stagger_receipt'],
        'loaded_from_authenticated_pre_PPO_state':True,
        'snapshot_sha256':cache['snapshot_sha256'],'new_warmup_rollouts':0,
        'parent_actor_critic_and_all_Adam_exact':True}
    return restored,runner,metadata,receipt
