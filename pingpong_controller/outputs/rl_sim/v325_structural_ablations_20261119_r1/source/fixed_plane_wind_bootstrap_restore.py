"""Pinned diagnostic actor migration with explicit optimizer-only ancestry."""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

import numpy as np

SOURCE_S9 = '12b963df7721179585b3b768344fdc97d912797053db27e3911a6433294b35d2'
BOOTSTRAP_FILES = {
    '8d242e6e62379878004f40ce921a119c5178eb531f1696e62ec5bf5cf5f31433':
        'pingpong_controller/outputs/rl_sim/v325_final_convergence_repair_20261114_r1/'
        'actor_bootstrap_recovery_r4/actor_candidate.pkl',
}
EXPLORATION = dict(min_log_std=math.log(.005), max_log_std=math.log(.02))


def learner_settings_for_stack(stack):
    from train_fixed_plane_ball_obs_noise import learner_settings
    result = learner_settings(stack.parent['args'])
    if getattr(stack, 'actor_bootstrap', None):
        result.update(EXPLORATION)
    recipe = getattr(stack, 'bootstrap_online_recipe', None)
    if recipe:
        result['actor_anchor_kl_coef'] = recipe['actor_anchor_kl_coef']
        if 'exploration_max_std' in recipe:
            result['max_log_std'] = math.log(recipe['exploration_max_std'])
        if 'learning_rate' in recipe:
            result['learning_rate'] = recipe['learning_rate']
    return result


def _verify_frozen_actor(payload, origin, numeric_digest):
    import jax
    for key in ('teacher_pi', 'residual_action_scale', 'wind_pose_bound_gate_v1'):
        if numeric_digest(payload['params'][key]) != numeric_digest(origin['params'][key]):
            raise ValueError('Frozen startup actor or authority gate changed')
        for moment in (payload['opt'].m, payload['opt'].v):
            if any(np.any(np.asarray(v) != 0.) for v in jax.tree.leaves(moment[key])):
                raise ValueError('Frozen startup actor has nonzero optimizer momentum')


def normalize_bootstrap_checkpoint(stack, payload, resume):
    """Authenticate the source; keep params/Adam and true simulation step exact."""
    from fixed_plane_v323_ball_obs_noise import digest_file
    from train_fixed_plane_ball_obs_noise import numeric_digest
    from validate_juggle_mjx_ppo import load_checkpoint
    from fixed_plane_wind_history import expand_reference_params
    import jax
    stack.actor_bootstrap = None
    stack.actor_bootstrap_offset = 0
    stack.actor_bootstrap_requires_probe = False
    stack.spec = stack.default_spec
    stack.bootstrap_online_recipe = None
    stack.bootstrap_stage_anchor = None
    stack.bootstrap_anchor_stage = None
    stack.online_recipe_migration = None
    direct = payload.get('projection_bootstrap')
    saved = payload.get('wind_contract', {}).get('actor_bootstrap')
    marker = 'wind_pose_bound_gate_v1' in payload.get('params', {})
    if not direct and not saved:
        if getattr(stack, 'requested_online_recipe', None):
            raise ValueError('Recovery online recipe requires the pinned actor source')
        if marker:
            raise ValueError('Authority-gated actor has no audited bootstrap ancestry')
        return payload
    if stack.spec.version != 325 or resume is None or not marker:
        raise ValueError('This actor bootstrap is pinned to V325 only')
    source_sha = digest_file(resume) if direct else saved.get('artifact_sha256')
    if source_sha not in BOOTSTRAP_FILES:
        raise ValueError('Unreviewed actor bootstrap source')
    source_path = stack.repo_root / BOOTSTRAP_FILES[source_sha]
    if not source_path.is_file() or digest_file(source_path) != source_sha:
        raise ValueError('Pinned actor bootstrap artifact is missing or changed')
    origin = payload if direct else load_checkpoint(source_path)
    recipe = origin['projection_bootstrap']
    s9_path = Path(recipe['source_checkpoint'])
    if not s9_path.is_file() or digest_file(s9_path) != SOURCE_S9:
        raise ValueError('Pinned S9 bootstrap parent changed')
    s9 = load_checkpoint(s9_path)
    expected = expand_reference_params(s9['params'])
    steps = recipe.get('optimizer_auxiliary_steps')
    if (type(steps) is not int or steps <= 0
            or recipe.get('source_sha256') != SOURCE_S9
            or recipe.get('source_true_step') != s9['step']
            or recipe.get('source_update') != s9['global_update']
            or recipe.get('source_adam_t') != int(s9['opt'].t)
            or recipe.get('true_simulation_steps_added') != 0
            or int(origin['opt'].t) != int(s9['opt'].t) + steps
            or origin['step'] != s9['step'] or origin['global_update'] != s9['global_update']
            or numeric_digest(origin['params']) != recipe['params_digest']
            or numeric_digest(origin['opt']) != recipe['optimizer_digest']
            or numeric_digest(origin['params']['teacher_pi']) != numeric_digest(expected['pi'])
            or not np.array_equal(np.asarray(origin['params']['residual_action_scale']), np.full(7, 2.))
            or float(origin['params']['wind_pose_bound_gate_v1']) != .25):
        raise ValueError('Actor bootstrap optimizer/source accounting mismatch')
    for key in ('v', 'log_std'):
        if numeric_digest(origin['params'][key]) != numeric_digest(expected[key]):
            raise ValueError('Actor bootstrap changed the source critic or exploration tensor')
        for old, new in ((s9['opt'].m, origin['opt'].m), (s9['opt'].v, origin['opt'].v)):
            if numeric_digest(new[key]) != numeric_digest(expand_reference_params(old)[key]):
                raise ValueError('Actor bootstrap changed source critic/exploration moments')
    for name in ('dataset_obs', 'dataset_actions', 'training_obs', 'training_actions'):
        path = Path(recipe[name])
        if not path.is_file() or digest_file(path) != recipe[name + '_sha256']:
            raise ValueError('Actor bootstrap demonstration artifact changed')
    info = dict(artifact_sha256=source_sha, source_s9_sha256=SOURCE_S9,
        extra_optimizer_steps=steps, format='authority_gated_residual_v1',
        exploration=dict(EXPLORATION), simulation_steps_added=0)
    if saved is not None and saved != info:
        raise ValueError('Stored actor bootstrap ancestry changed')
    _verify_frozen_actor(payload, origin, numeric_digest)
    stack.actor_bootstrap = info
    stack.actor_bootstrap_offset = steps
    from fixed_plane_wind_online_recipe import configure_recipe, prepare_recipe_migration
    payload = prepare_recipe_migration(stack, payload, resume)
    configure_recipe(stack, payload)
    stack.actor_bootstrap_requires_probe = bool(direct or
        (payload.get('stage_index') == 9 and not payload.get('stage_converged', False)))
    if not direct:
        return payload
    # This is an explicit S9 -> new S10 migration. No simulated steps, ramp
    # ticks, convergence history or completed-stage claims are manufactured.
    from train_fixed_plane_wind import course_config
    from fixed_plane_wind_training_gates import PROTOCOL, gate_contract, completion_metadata
    from fixed_plane_wind_reward import repair_contract
    from fixed_plane_wind_pose import pose_contract
    from fixed_plane_wind_history import history_contract
    from fixed_plane_wind_objectives import objective_contract
    stage = 9
    contract = dict(payload['wind_contract'])
    contract.update(course=[asdict(s) for s in stack.spec.course],
        reward_repair=repair_contract(), pose_curriculum=pose_contract(),
        observed_history=history_contract(), balanced_objectives=objective_contract(325),
        validation_protocol=PROTOCOL, training_gates=gate_contract(325),
        startup_shelter=dict(confirmed_hits=5, ramp_s=1.), actor_bootstrap=info)
    if stack.bootstrap_online_recipe:
        contract['bootstrap_online_recipe'] = stack.bootstrap_online_recipe
    result = {**payload, 'args': {**payload['args'], **learner_settings_for_stack(stack)},
        'env_cfg': asdict(course_config(stack, stack.spec.course[stage])),
        'stage_index': stage, 'stage_name': stack.spec.course[stage].name,
        'stage_update': 0, 'pose_stage_updates': 0, 'pose_ramp_progress_steps': 0,
        'stage_converged': False, 'last_summary': {}, 'wind_contract': contract,
        **completion_metadata(stage, len(stack.spec.course), converged=False)}
    return result
