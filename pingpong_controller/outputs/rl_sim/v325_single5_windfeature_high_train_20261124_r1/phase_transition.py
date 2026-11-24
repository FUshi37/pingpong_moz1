"""Explicit sampling-phase transition; retain learner, reset episode history."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import jax
import jax.numpy as jnp


@jax.jit
def frozen_backbone_equal(params, reference):
    checks = [jnp.all(params['log_std'] == reference['log_std'])]
    for name, layer in params['pi'].items():
        for key, value in layer.items():
            expected = reference['pi'][name][key]
            if name == 'l1' and key == 'w':
                checks += [jnp.all(value[:114] == expected[:114]),
                           jnp.all(value[116:] == expected[116:])]
            else:
                checks.append(jnp.all(value == expected))
    return jnp.all(jnp.stack(checks))


def verify_phase_parent(plan):
    branch = plan['phase_transition']
    for field in ('model', 'full_state', 'parent_plan', 'decision'):
        path = Path(branch[field])
        if path.is_symlink() or not path.is_file():
            raise ValueError('Phase parent must be a pinned regular file: ' + field)
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != branch[field + '_sha256']:
                raise ValueError('Phase parent hash mismatch: ' + field)
    decision = json.loads(Path(branch['decision']).read_text())
    if not decision['accepted'] or not decision['preservation_passed'] or decision['failures']:
        raise ValueError('Phase parent did not pass original short preservation checks')
    old = json.loads(Path(branch['parent_plan']).read_text())
    for key in ('source_sha256', 'pose', 'observation', 'onset', 'wind_domain',
                'timing', 'reward_revision', 'ppo', 'full_episode_outcome_capacity'):
        if plan[key] != old[key]:
            raise ValueError('Wind-only phase changed preserved contract: ' + key)
    if (plan['training_phase'] != 'wind_adaptation' or plan['BC'] != 0
            or not plan['feature_optimizer']['permanent_wind_rows_only']
            or plan['rehearsal']['n_envs'] != 0
            or plan['direct_trainable_actor_parameters'] != 564):
        raise ValueError('Invalid wind-only phase configuration')
    with ZipFile(branch['full_state']) as archive:
        manifest = json.loads(archive.read('manifest.json'))
    if (manifest['metadata']['actual_updates'] != branch['actual_updates']
            or manifest['contract']['plan_sha256'] != branch['parent_plan_sha256']):
        raise ValueError('Wrong source update or native plan')
    return manifest['contract'], dict(
        qualified_update=branch['actual_updates'],
        source_model_sha256=branch['model_sha256'],
        source_full_sha256=branch['full_state_sha256'],
        actor_critic_and_all_Adam_retained=True,
        new_physical_episodes_and_convergence_history=True,
        source_calm1200_qualification_pending=True,
        scope='Isolated wind-only input adaptation while the unchanged calm base is checked',
        no_new_actor_branch=True, no_BC=True)
