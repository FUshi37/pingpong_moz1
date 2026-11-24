#!/usr/bin/env python3
"""Private V323/V324 driver; preserve each frozen parent's native training path."""
from __future__ import annotations

import argparse
import ast
from collections import deque
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from fixed_plane_v323_ball_obs_noise import (
    EXECUTION_VERSION, NOISE_FIELDS, append_noise_critic_state, canonical,
    digest_file, make_noise_config, spec_for, validate_parent_payload,
)
from fixed_plane_v321_long_episode import (
    CheckpointRetention, EndpointWindow, QUALITY_FIELDS, quality_summary,
)
from fixed_plane_noise_validation_v2 import (
    PROTOCOL as PAIRED_VALIDATION_V2, profile_for, protocol_contract,
)
from fixed_plane_noise_training_v3 import (
    PROTOCOL as TRAINING_QUALITY_V3, TrainingQualityWindow,
    completion_metadata, protected_checkpoint_name,
    protocol_contract as training_protocol_contract, validate_episode,
)

SOURCE_FILES = (
    'train_fixed_plane_ball_obs_noise.py', 'train_fixed_plane_v323.py',
    'train_fixed_plane_v324.py', 'fixed_plane_v323_ball_obs_noise.py',
    'fixed_plane_v324_ball_obs_noise.py', 'fixed_plane_v323_noise_backport.py',
    'train_fixed_plane_v321.py', 'train_fixed_plane_v322.py',
    'fixed_plane_v321_long_episode.py', 'fixed_plane_v322_long_episode.py',
    'fixed_plane_v322_condition_gates.py', 'fixed_plane_v322_late_cycle_reward.py',
    'fixed_plane_v322_quality_repair.py', 'ball_position_noise_ar1.py',
    'train_juggle_mjx_ppo.py', 'bounded_validation_memory.py',
    'training_memory_scope.py', 'mjx_line_search_schedule.py',
    'mjx_topology_execution.py',
    'fixed_plane_noise_validation_v2.py',
    'fixed_plane_noise_training_v3.py',
)
PAIR_POLICY = 'core_exact_diagnostic_fp32_v2'
# These compact batch means do not enter PPO, curriculum graduation or safety
# decisions. GPU fusion/reduction rounding is bounded separately; complete
# state, learning tensors, losses and every other metric remain byte-exact.
DIAGNOSTIC_ROUNDING_FIELDS = (
    'ddq_post_comp_safe_latest', 'ddq_servo_target',
    'dq_post_comp_safe_latest', 'dq_ref_latest', 'dq_servo_target',
    'fixed_contact_pre_hit_launch_command_vz_m_s', 'q_cmd_nominal',
    'q_servo_target', 'qvel_reference_error_norm', 'rmp_effective_target_error_norm',
    'rmp_target_error_norm', 'reward/apex_soft_penalty',
)


def load_stack(repo: Path, source: Path, version: int) -> SimpleNamespace:
    """Authenticate native dynamics, selected parent and all replay inputs."""
    spec = spec_for(version)
    module = importlib.import_module(f'train_fixed_plane_v{version - 2}')
    helpers = importlib.import_module(f'fixed_plane_v{version - 2}_long_episode')
    if 'mjx_juggle_env' in sys.modules:
        raise ValueError('Load each frozen training stack in a fresh Python process')
    ppo, ancestor, native_cfg, xml, snapshot = module.load_stack(
        repo, source / 'train_juggle_mjx_ppo.py')
    from validate_juggle_mjx_ppo import load_checkpoint
    native = importlib.import_module('mjx_juggle_env')
    if digest_file(Path(native.__file__)) != spec.native_env_sha256:
        raise ValueError('Wrong frozen native environment')
    parent_path = repo / spec.parent_checkpoint
    validate_parent_payload(parent := load_checkpoint(parent_path),
                            digest_file(parent_path), spec)
    saved_contract = parent[f'v{version - 2}_contract']
    adapter_hashes = {Path(name).name: value
                      for name, value in saved_contract['source_sha256'].items()}
    required_adapters = ['fixed_plane_v321_long_episode.py']
    if version == 324:
        required_adapters += ['fixed_plane_v322_long_episode.py',
            'fixed_plane_v322_condition_gates.py', 'fixed_plane_v322_late_cycle_reward.py']
    for name in required_adapters:
        if digest_file(source / name) != adapter_hashes.get(name):
            raise ValueError(f'Authenticated parent reward/gate adapter changed: {name}')
    for name, expected in saved_contract['frozen_parent_source_sha256'].items():
        if digest_file(snapshot / name) != expected:
            raise ValueError(f'Frozen parent source changed: {name}')
    for name, expected in saved_contract['replay_sha256'].items():
        if digest_file(Path(name)) != expected:
            raise ValueError(f'Parent replay input changed: {name}')
    native_cfg = helpers.make_config(native_cfg)
    if canonical(asdict(native_cfg)) != canonical(parent['env_cfg']):
        raise ValueError('Selected parent differs from its exact native configuration')
    cfg = make_noise_config(native_cfg)
    base = native.MjxJuggleEnv
    if version == 323:
        from fixed_plane_v323_noise_backport import make_m12_noise_base
        base = make_m12_noise_base(base)
        env_class = module.environment_class(base)
    else:
        from fixed_plane_v323_noise_backport import make_m13_noise_base
        base = make_m13_noise_base(base)
        env_class = module.environment_class(base,
            reward_preset='late_center_precision', grouped_conditions=True)
        if numeric_digest(ancestor['stage_actor_anchor_reference_params']) != numeric_digest(
                parent['stage_actor_anchor_reference_params']):
            # Array values are also authenticated by the pinned parent SHA;
            # retain the serialized M13 anchor rather than inventing a new one.
            raise ValueError('M13 actor replay anchor changed')
    return SimpleNamespace(spec=spec, ppo=ppo, parent=parent, cfg=cfg,
        native_cfg=native_cfg, xml=xml, snapshot=snapshot,
        env_class=env_class, module=module, helpers=helpers,
        parent_path=parent_path, saved_contract=saved_contract)


def numeric_digest(tree: Any) -> str:
    """Hash every finite numeric leaf's type, shape and raw bytes for pairing."""
    import hashlib
    import jax
    digest = hashlib.sha256()
    leaves, _ = jax.tree_util.tree_flatten_with_path(jax.device_get(tree))
    for path, leaf in leaves:
        digest.update(jax.tree_util.keystr(path).encode())
        value = np.asarray(leaf)
        if value.dtype.kind not in 'biufc' or not np.isfinite(value).all():
            raise FloatingPointError('Nonfinite or nonnumeric paired execution leaf')
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def compare_numeric_leaves(reference: Any, candidate: Any) -> dict:
    """Report raw-byte differences one leaf at a time with bounded host RAM."""
    import jax
    left, _ = jax.tree_util.tree_flatten_with_path(reference)
    right, _ = jax.tree_util.tree_flatten_with_path(candidate)
    changes = []
    if len(left) != len(right):
        return dict(changed_leaves=1, changes=[dict(reason='leaf_count',
            reference=len(left), candidate=len(right))])
    for (path_a, value_a), (path_b, value_b) in zip(left, right):
        name = jax.tree_util.keystr(path_a)
        a, b = np.asarray(value_a), np.asarray(value_b)
        if (name != jax.tree_util.keystr(path_b)
                or a.shape != b.shape or a.dtype != b.dtype):
            changes.append(dict(path=name, reason='layout',
                reference_shape=list(a.shape), candidate_shape=list(b.shape),
                reference_dtype=str(a.dtype), candidate_dtype=str(b.dtype)))
        elif a.tobytes() != b.tobytes():
            difference = np.abs(a.astype(np.float64)-b.astype(np.float64))
            changes.append(dict(path=name, shape=list(a.shape), dtype=str(a.dtype),
                max_abs_diff=float(difference.max()) if difference.size else 0.,
                unequal_values=int(np.count_nonzero(a != b))))
    return dict(changed_leaves=len(changes), changes=changes)


def execution_binding(stack: SimpleNamespace, source: Path,
                      source_checkpoint: Path, seed: int,
                      s1_eval_retry_updates: int = 1000,
                      validation_protocol: str = 'legacy') -> dict:
    """Bind a paired CUDA smoke result to this exact model/config/source stack."""
    import jax
    import mujoco
    from mujoco.mjx._src import support, solver
    from mjx_topology_execution import make_collect
    # This factory checks the locked versions, native source hashes and
    # projection source without applying any patch or starting simulation.
    make_collect(lambda: None, 'project')
    schedule = (protocol_contract(stack.spec.version)['retry_updates']
        if validation_protocol == PAIRED_VALIDATION_V2 else
        [0, 0] if validation_protocol == TRAINING_QUALITY_V3 else
        validation_retry_schedule(stack.spec.version, s1_eval_retry_updates))
    return dict(profile=profile_for(stack.spec, validation_protocol), execution_version=EXECUTION_VERSION,
        checkpoint_sha256=digest_file(source_checkpoint),
        env_cfg=asdict(stack.cfg), xml_sha256=digest_file(stack.xml),
        native_source_sha256=stack.saved_contract['frozen_parent_source_sha256'],
        source_sha256={name: digest_file(source / name) for name in SOURCE_FILES},
        jax_version=jax.__version__, mujoco_version=mujoco.__version__,
        support_sha256=digest_file(Path(support.__file__)),
        solver_sha256=digest_file(Path(solver.__file__)),
        gpu_uuid=stack.spec.gpu_uuid, seed=seed,
        validation_protocol=validation_protocol, validation_retry_updates=schedule,
        batch=[1024, 128], adam_per_update=32)


def validate_execution_audit(report: dict, binding: dict) -> None:
    """Recheck pairing facts; a declared pass flag alone is never accepted."""
    digests = report.get('legacy_digests', [])
    if binding.get('validation_protocol') == PAIRED_VALIDATION_V2:
        from fixed_plane_noise_validation_v2 import validate_evaluator_audit
        profile = binding.get('profile', '')
        version = next((v for v in (323, 324)
                        if profile == profile_for(spec_for(v), PAIRED_VALIDATION_V2)), None)
        if version is None:
            raise ValueError('Unregistered paired validation profile')
        validate_evaluator_audit(report.get('validation_v2_audit', {}), version)
    if (not compatible_execution_binding(report, binding)
            or report.get('backend') != 'gpu'
            or len(digests) != 3
            or any(not isinstance(s, str) or not re.fullmatch('[a-f0-9]{64}', s)
                   for s in digests)
            or len(report.get('topology_v2_digests', [])) != 3
            or any(not isinstance(s, str) or not re.fullmatch('[a-f0-9]{64}', s)
                   for s in report.get('topology_v2_digests', []))
            or report.get('legacy_adam_deltas') != [32] * 3
            or report.get('topology_v2_adam_deltas') != [32] * 3):
        raise ValueError('A matching three-update paired CUDA execution audit is required')
    if digests == report['topology_v2_digests']:
        return
    core = report.get('legacy_core_digests', [])
    if (report.get('admission_policy', report.get('pair_policy')) != PAIR_POLICY or len(core) != 3
            or any(not isinstance(s, str) or not re.fullmatch('[a-f0-9]{64}', s) for s in core)
            or core != report.get('topology_v2_core_digests')
            or len(report.get('diagnostic_rounding_samples', [])) != 3):
        raise ValueError('Execution pair requires byte-exact core state and learner data')
    for sample in report['diagnostic_rounding_samples']:
        if set(sample) != set(DIAGNOSTIC_ROUNDING_FIELDS):
            raise ValueError('Execution pair diagnostic field set changed')
        for name, values in sample.items():
            a = np.asarray(values['legacy'], dtype=np.float32)
            b = np.asarray(values['topology_v2'], dtype=np.float32)
            if (a.shape != (128,) or b.shape != a.shape
                    or not np.isfinite(a).all() or not np.isfinite(b).all()):
                raise ValueError(f'Invalid compact diagnostic samples: {name}')
            if not diagnostic_close(name, a, b):
                raise ValueError(f'Compact diagnostic exceeds float32 rounding bound: {name}')


def diagnostic_close(name: str, reference: np.ndarray, candidate: np.ndarray) -> bool:
    """Bound diagnostic cancellation at unit scale and the fixed5ms difference.

    Relative ULP alone is inappropriate for nearly cancelling reference
    errors. Acceleration diagnostics additionally divide a velocity
    difference by dt. No core state/learning or gate tensor uses this bound.
    """
    epsilon = np.finfo(np.float32).eps
    absolute = 4. * epsilon
    if name in ('ddq_post_comp_safe_latest', 'ddq_servo_target'):
        absolute /= .005
    return bool(np.allclose(reference, candidate, rtol=4.*epsilon,
                            atol=absolute, equal_nan=False))


def compatible_execution_binding(report: dict, current: dict) -> bool:
    """Reuse measured GPU data only across audited validation/storage edits."""
    previous = report.get('binding')
    if canonical(previous) == canonical(current):
        return True
    if not isinstance(previous, dict) or not report.get('qualification_driver_source'):
        return False
    name = 'train_fixed_plane_ball_obs_noise.py'
    old_sources = previous.get('source_sha256', {})
    new_sources = current.get('source_sha256', {})
    if (canonical({k:v for k,v in previous.items() if k != 'source_sha256'})
            != canonical({k:v for k,v in current.items() if k != 'source_sha256'})
            or {k:v for k,v in old_sources.items() if k != name}
            != {k:v for k,v in new_sources.items() if k != name}):
        return False
    reference = Path(report['qualification_driver_source'])
    driver = Path(__file__)
    if (not reference.is_file() or digest_file(reference) != old_sources.get(name)
            or digest_file(driver) != new_sources.get(name)):
        return False

    def math_source(path):
        tree = ast.parse(path.read_text())
        names = {'load_stack', 'restore_training_state', 'make_train_functions',
                 'main', 'run_training'}
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in functions} != names:
            raise ValueError('Incomplete qualified driver')
        # Checkpoint selection/serialization is the sole permitted normal
        # path difference. The actual collect/update/advancement loop stays
        # exact; all environment, PPO and supporting-source hashes stay exact.
        for node in functions:
            if node.name == 'run_training':
                node.body = [item for item in node.body
                             if not (isinstance(item, ast.FunctionDef) and item.name == 'save')]
        return ast.dump(ast.Module(body=functions, type_ignores=[]), include_attributes=False)

    return math_source(reference) == math_source(driver)


def checkpoint_selection_score(version: int, stage: int, quality_ok: bool,
                               endpoint: dict, conditions: dict | None,
                               ongoing_s_mean: float) -> tuple:
    """Use the parent duration/cohort ranking with the same5-save pruning."""
    if version == 323:
        return (stage, float(quality_ok), endpoint.get('survival', 0.),
                endpoint.get('len_fraction', 0.), ongoing_s_mean)
    from fixed_plane_v322_condition_gates import condition_gate_settings
    settings = condition_gate_settings()
    conditions = conditions or {}
    primary = conditions.get('nominal', {})
    margin = min(min(conditions.get(name, {}).get('survival', 0.) / limits['min_survival'],
        conditions.get(name, {}).get('len_fraction', 0.) / limits['min_length_fraction'])
        for name, limits in settings.items())
    return (stage, float(quality_ok), margin, primary.get('survival', 0.),
            primary.get('len_fraction', 0.), ongoing_s_mean)


def validation_retry_schedule(version: int,
                              s1_eval_retry_updates: int = 1000) -> tuple[int, int]:
    """Opt in to the requested V323 short-test cadence; preserve other lessons."""
    spec_for(version)
    if (s1_eval_retry_updates not in (100, 1000)
            or (version != 323 and s1_eval_retry_updates != 1000)):
        raise ValueError('Only V323 S1 may opt in to the 100-update retry interval')
    return s1_eval_retry_updates, 1000


def validation_due(*, hold: int, update_idx: int, last_eval_update: int,
                   retry_updates: int) -> bool:
    """Require the unchanged hold gate and the complete retest interval."""
    return hold >= 16 and update_idx-last_eval_update >= retry_updates


def core_packet(packet: Any, prefix: str) -> Any:
    """Keep all learning data and metrics except the explicit diagnostic means."""
    state, runner, transitions, losses = packet
    excluded = {prefix + name for name in DIAGNOSTIC_ROUNDING_FIELDS}
    metrics = {name: value for name, value in transitions.metrics.items()
               if name not in excluded}
    return state, runner, transitions._replace(metrics=metrics), losses


def restore_training_state(stack: SimpleNamespace,
                           resume: Path | None) -> tuple[Any, int, int, int]:
    """Restore all optimizer leaves; migrate only the initial critic suffix."""
    import jax
    import jax.numpy as jnp
    from validate_juggle_mjx_ppo import load_checkpoint
    payload = stack.parent if resume is None else load_checkpoint(resume)
    protocol = getattr(stack, 'validation_protocol', 'legacy')
    if protocol == TRAINING_QUALITY_V3 and resume is None:
        raise ValueError('Training-quality mode requires a complete noisy S2 checkpoint')
    stage_index = 0
    update_idx = 0
    if resume is not None:
        contract = payload.get('ball_observation_noise_contract', {})
        allowed_profiles = {profile_for(stack.spec, protocol)}
        if protocol == PAIRED_VALIDATION_V2:
            allowed_profiles.add(stack.spec.profile)
        if protocol == TRAINING_QUALITY_V3:
            allowed_profiles.add(profile_for(stack.spec, PAIRED_VALIDATION_V2))
        saved_profile = payload.get('args', {}).get('curriculum_profile')
        if (saved_profile not in allowed_profiles
                or contract.get('parent_sha256') != stack.spec.parent_sha256
                or canonical(contract.get('course')) != canonical(
                    [asdict(s) for s in stack.spec.course])
                or canonical(payload.get('env_cfg')) != canonical(asdict(stack.cfg))
                or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim'))
                    != (72, 396, 7)
                or payload.get('opt') is None):
            raise ValueError('Only the exact same noise course can resume')
        if saved_profile == profile_for(stack.spec, TRAINING_QUALITY_V3):
            if (protocol != TRAINING_QUALITY_V3
                    or contract.get('validation_protocol') != TRAINING_QUALITY_V3
                    or canonical(contract.get('training_gate_contract')) != canonical(
                        training_protocol_contract(stack.spec.version))):
                raise ValueError('A training-only profile requires its exact saved contract')
        elif (saved_profile != stack.spec.profile
                and (contract.get('validation_protocol') != PAIRED_VALIDATION_V2
                     or canonical(contract.get('validation_gate_contract'))
                     != canonical(protocol_contract(stack.spec.version)))):
            raise ValueError('A revised gate profile requires its explicit saved protocol')
        for name, value in learner_settings(stack.parent['args']).items():
            if payload['args'].get(name) != value:
                raise ValueError(f'Noise resume learner mismatch: {name}')
        stage_index = payload['stage_index']
        if (not isinstance(stage_index, int) or not 0 <= stage_index < 2
                or payload['stage_name'] != stack.spec.course[stage_index].name):
            raise ValueError('Noise resume requires an explicit registered stage')
        if protocol == TRAINING_QUALITY_V3 and (stage_index != 1
                or payload.get('stage_converged', False)
                or payload.get('training_acceptance', {}).get('qualified', False)):
            raise ValueError('Training-quality continuation requires an unfinished noisy S2')
        stage_index += int(payload.get('stage_converged', False))
        update_idx = int(payload['global_update'])
        if stage_index >= len(stack.spec.course):
            raise ValueError('The noise continuation already completed all lessons')
        if (update_idx < 0
                or payload['step'] != stack.spec.parent_step + update_idx * 131072
                or int(payload['opt'].t) != stack.spec.parent_adam_t + update_idx * 32):
            raise ValueError('Noise resume optimizer/update provenance mismatch')
    state = stack.ppo.TrainState(jax.tree.map(jnp.asarray, payload['params']),
                                 jax.tree.map(jnp.asarray, payload['opt']))
    if resume is None:
        state = append_noise_critic_state(state)
    if (state.params['pi']['l1']['w'].shape[0] != 72
            or state.params['v']['l1']['w'].shape[0] != 396
            or state.params['log_std'].shape != (7,)):
        raise ValueError('Noise continuation requires the exact 72/396/7 network layout')
    for moments in (state.opt.m, state.opt.v):
        if jax.tree.structure(moments) != jax.tree.structure(state.params):
            raise ValueError('Noise continuation requires complete matching Adam trees')
        for parameter, moment in zip(jax.tree.leaves(state.params), jax.tree.leaves(moments)):
            if parameter.shape != moment.shape or parameter.dtype != moment.dtype:
                raise ValueError('Noise continuation Adam leaf layout mismatch')
    numeric_digest(state)
    return state, int(payload['step']), stage_index, update_idx


def learner_settings(args: dict) -> dict:
    """Record all parameters of this driver's unchanged parent PPO objective."""
    names = ('n_envs', 'n_steps', 'update_epochs', 'minibatch_size', 'gamma',
        'gae_lambda', 'learning_rate', 'clip_range', 'vf_coef', 'ent_coef',
        'max_grad_norm', 'min_log_std', 'max_log_std', 'target_kl',
        'time_limit_bootstrap', 'counterfactual_supervision_coef',
        'counterfactual_vxy_weight_mode', 'actor_anchor_kl_coef',
        'actor_anchor_replay_kl_coef', 'actor_anchor_replay_max_kl',
        'residual_l2_coef', 'failure_focus_hit_threshold', 'failure_focus_weight',
        'failure_focus_tail_steps', 'failure_focus_balanced_mass')
    return {name: args[name] for name in names}


def make_train_functions(stack: SimpleNamespace, env: Any,
                         args: argparse.Namespace, mode: str) -> tuple[Any, Any]:
    """Keep each parent's BC, anchor and failure-focus terms exactly intact."""
    import jax
    import jax.numpy as jnp
    ppo = stack.ppo
    replay_obs = np.load(args.counterfactual_replay_obs)
    replay_actions = np.load(args.counterfactual_replay_actions)
    if replay_obs.shape[1] != 72 or replay_actions.shape != (len(replay_obs), 7):
        raise ValueError('Wrong parent BC replay dimensions')
    kwargs = {name: getattr(args, name) for name in (
        'n_steps', 'update_epochs', 'minibatch_size', 'gamma', 'gae_lambda',
        'learning_rate', 'clip_range', 'vf_coef', 'ent_coef', 'max_grad_norm',
        'min_log_std', 'max_log_std', 'target_kl', 'time_limit_bootstrap')}
    prefix = f'v{stack.spec.version - 2}'
    metrics = (f'{prefix}/recent_hit',)
    if stack.spec.version == 324:
        metrics += ('v322/initial_group', 'v322/initial_y_m')
        kwargs.update(reference_params=jax.tree.map(jnp.asarray,
            stack.parent['stage_actor_anchor_reference_params']),
            actor_anchor_replay_obs=jnp.asarray(np.load(args.actor_anchor_replay_obs)),
            **{name: getattr(args, name) for name in (
                'actor_anchor_kl_coef', 'actor_anchor_replay_kl_coef',
                'actor_anchor_replay_max_kl', 'residual_l2_coef',
                'failure_focus_hit_threshold', 'failure_focus_weight',
                'failure_focus_tail_steps', 'failure_focus_balanced_mass')})
    collect, update = ppo.make_train_fns(env=env, **kwargs,
        compact_rollout_metrics=True, full_rollout_metric_names=metrics,
        conditional_reset_done=True, apply_every_minibatch=True,
        counterfactual_replay_obs=jnp.asarray(replay_obs),
        counterfactual_replay_actions=jnp.asarray(replay_actions),
        counterfactual_supervision_coef=args.counterfactual_supervision_coef,
        counterfactual_vxy_weight_mode=args.counterfactual_vxy_weight_mode)
    if mode == 'topology_v2':
        if env.model.opt.iterations != 100 or env.model.opt.ls_iterations != 50:
            raise ValueError('Execution V2 requires unchanged 100/50 solver caps')
        from mjx_topology_execution import make_collect
        collect, _ = make_collect(collect, 'project')
    elif mode != 'legacy':
        raise ValueError('Unsupported execution mode')
    return collect, update


def paired_execution_smoke(stack: SimpleNamespace, env: Any, args: argparse.Namespace,
                           train_state: Any, runner: Any,
                           binding: dict, runtime: Path,
                           stop: list[bool] | None = None) -> None:
    """Run both real three-update arms from identical reset/learner/RNG state."""
    import jax
    from bounded_validation_memory import release_caches
    report = dict(binding=binding, backend=jax.default_backend(), pair_policy=PAIR_POLICY)
    baseline_packets = []
    baseline_diagnostics = []
    for mode in ('legacy', 'topology_v2'):
        if stop is not None and stop[0]:
            raise InterruptedError('Execution qualification cancelled')
        release_caches()
        collect, update = make_train_functions(stack, env, args, mode)
        state, live = train_state, runner
        signatures, deltas, core_signatures = [], [], []
        differences = []
        for index in range(3):
            if stop is not None and stop[0]:
                raise InterruptedError('Execution qualification cancelled')
            previous_t = int(state.opt.t)
            live, transitions = collect(state.params, live)
            state, losses = update(state, live, transitions)
            deltas.append(int(state.opt.t) - previous_t)
            signatures.append(numeric_digest((state, live, transitions, losses)))
            packet = (state, live, transitions, losses)
            core_signatures.append(numeric_digest(core_packet(
                packet, stack.ppo.COMPACT_METRIC_MEAN_PREFIX)))
            diagnostics = {name: np.asarray(transitions.metrics[
                stack.ppo.COMPACT_METRIC_MEAN_PREFIX + name]).tolist()
                for name in DIAGNOSTIC_ROUNDING_FIELDS}
            if mode == 'legacy':
                baseline_packets.append(packet)
                baseline_diagnostics.append(diagnostics)
            else:
                differences.append(compare_numeric_leaves(baseline_packets[index], packet))
                report.setdefault('diagnostic_rounding_samples', []).append({
                    name: dict(legacy=baseline_diagnostics[index][name],
                               topology_v2=diagnostics[name])
                    for name in DIAGNOSTIC_ROUNDING_FIELDS})
            print(f'[V{stack.spec.version} qualification] arm={mode} update={index+1}/3 '
                  f'Adam+{deltas[-1]} signature={signatures[-1]}', flush=True)
            del transitions
        report[f'{mode}_digests'] = signatures
        report[f'{mode}_adam_deltas'] = deltas
        report[f'{mode}_core_digests'] = core_signatures
        if differences:
            report['numeric_differences'] = differences
        (runtime / 'execution_pair_attempt.json').write_text(json.dumps(report, indent=2))
        del collect, update, state, live
    if getattr(args, 'validation_protocol', 'legacy') == PAIRED_VALIDATION_V2:
        from fixed_plane_noise_validation_v2 import audit_evaluator
        del baseline_packets, baseline_diagnostics, packet, diagnostics
        release_caches()
        report['validation_v2_audit'] = audit_evaluator(stack, args, train_state, runtime, stop=stop)
    validate_execution_audit(report, binding)
    (runtime / 'execution_pair_audit.json').write_text(json.dumps(report, indent=2))
    print(f'[V{stack.spec.version}] paired CUDA smoke: 3 core-exact updates / '
          '96 Adam steps; declared diagnostic reductions within float32 bounds',
          flush=True)


def main(version: int) -> None:
    """Prepare without a GPU, qualify execution explicitly, or train manually."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--save-dir', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--seed', type=int, default=2026110400 + version - 320)
    parser.add_argument('--s1-eval-retry-updates', type=int, choices=(100, 1000),
                        default=1000)
    parser.add_argument('--validation-protocol', choices=(
                        'legacy', PAIRED_VALIDATION_V2, TRAINING_QUALITY_V3),
                        default='legacy')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--execution-audit', type=Path)
    opts = parser.parse_args()
    spec = spec_for(version)
    retry_schedule = (protocol_contract(version)['retry_updates']
        if opts.validation_protocol == PAIRED_VALIDATION_V2 else
        [0, 0] if opts.validation_protocol == TRAINING_QUALITY_V3 else
        validation_retry_schedule(version, opts.s1_eval_retry_updates))
    if opts.preflight:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    elif (opts.runtime is None or opts.save_dir is None or opts.run_id is None
            or not re.fullmatch('[a-zA-Z0-9_-]+', opts.run_id)
            or opts.runtime.resolve().parent != Path('/dev/shm')
            or os.environ.get('PP_MEMORY_SCOPE_ACTIVE') != 'YES'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != spec.gpu_uuid):
        raise ValueError('Use the isolated version-specific manual launcher')
    source = Path(__file__).resolve().parent
    stack = load_stack(opts.repo_root.resolve(), source, version)
    stack.validation_protocol = opts.validation_protocol
    train_state, global_step, stage_index, update_idx = restore_training_state(stack, opts.resume)
    source_checkpoint = opts.resume or stack.parent_path
    binding = execution_binding(stack, source, source_checkpoint, opts.seed,
                                opts.s1_eval_retry_updates, opts.validation_protocol)
    if opts.preflight:
        print(json.dumps(dict(profile=profile_for(spec, opts.validation_protocol), parent=spec.parent_model,
            parent_checkpoint=str(stack.parent_path), parent_sha256=spec.parent_sha256,
            source_checkpoint=str(source_checkpoint), true_step=global_step,
            Adam_t=int(train_state.opt.t), actor_critic_action=[72, 396, 7],
            course=[asdict(s) for s in spec.course],
            seed=opts.seed,
            noise={k: getattr(stack.cfg, k) for k in NOISE_FIELDS},
            learner=learner_settings(stack.parent['args']),
            validation_retry_updates=retry_schedule,
            validation_protocol=opts.validation_protocol,
            validation_gate_contract=(protocol_contract(version)
                if opts.validation_protocol == PAIRED_VALIDATION_V2 else None),
            training_gate_contract=(training_protocol_contract(version)
                if opts.validation_protocol == TRAINING_QUALITY_V3 else None),
            training_episode_s=stack.cfg.horizon_sec,
            execution_version=EXECUTION_VERSION, training_started=False), indent=2))
        return
    import jax
    import jax.numpy as jnp
    from bounded_validation_memory import release_caches
    if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
        raise ValueError('The selected physical CUDA GPU is required')
    if not opts.smoke:
        if opts.execution_audit is None:
            raise ValueError('Run the paired --smoke before formal noise training')
        validate_execution_audit(json.loads(opts.execution_audit.read_text()), binding)
    args = argparse.Namespace(**stack.parent['args'])
    args.curriculum_profile = profile_for(spec, opts.validation_protocol)
    args.save_dir, args.seed, args.xml = opts.save_dir, opts.seed, stack.xml
    args.wandb_name = args.wandb_id = opts.run_id
    args.wandb_mode, args.wandb_resume = 'online', 'never'
    args.resume_from, args.resume_start_stage = source_checkpoint, spec.course[stage_index].name
    args.resume_curriculum_state = False
    args.reset_optimizer_on_resume = args.reset_critic_on_resume = False
    args.apply_every_minibatch, args.conditional_reset_done = True, True
    args.total_update_limit, args.max_stage_updates = None, -1
    args.max_stages, args.save_every_updates = 2, 100
    args.mjx_execution_mode = 'topology_v2'
    args.s1_eval_retry_updates = retry_schedule[0]
    args.validation_protocol = opts.validation_protocol
    args.training_entrypoint = f'train_fixed_plane_v{version}.py'
    startup_lease, stop_flag = None, None
    if opts.validation_protocol in (PAIRED_VALIDATION_V2, TRAINING_QUALITY_V3):
        from fixed_plane_noise_validation_v2 import ValidationMemoryLease
        stop_flag = [False]
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop_flag.__setitem__(0, True))
        startup_lease = ValidationMemoryLease(version)
        if not startup_lease.acquire(stop_flag):
            return
    env = stack.env_class(stack.xml, n_envs=1024, cfg=stack.cfg)
    if (env.obs_dim, env.critic_obs_dim, env.act_dim, env.max_steps) != (72, 396, 7, 240000):
        raise ValueError('Noise continuation layout/horizon mismatch')
    if opts.validation_protocol == TRAINING_QUALITY_V3:
        validate_episode(stack.cfg.horizon_sec, env.dt, env.max_steps)
    rng, reset_key = jax.random.split(jax.random.PRNGKey(opts.seed))
    state, obs = jax.jit(env.reset)(jax.random.split(reset_key, env.n_envs))
    runner = stack.ppo.RunnerState(state, obs, env.get_critic_obs(state, obs), rng,
        jnp.zeros(env.n_envs), jnp.zeros(env.n_envs, dtype=jnp.int32))
    jax.block_until_ready((runner, train_state))
    release_caches()
    if opts.smoke:
        try:
            paired_execution_smoke(stack, env, args, train_state, runner, binding, opts.runtime,
                                   stop=stop_flag)
        finally:
            if startup_lease is not None:
                startup_lease.release()
        return
    run_training(stack, env, args, opts, train_state, runner,
                 global_step, stage_index, update_idx, binding,
                 startup_lease=startup_lease, stop_flag=stop_flag)


def run_training(stack: SimpleNamespace, env: Any, args: argparse.Namespace,
                 opts: argparse.Namespace, train_state: Any, runner: Any,
                 global_step: int, stage_index: int, update_idx: int, binding: dict,
                 *, startup_lease: Any = None, stop_flag: list[bool] | None = None) -> None:
    """Train with fresh noise-domain gates; never inherit a clean-domain pass."""
    import jax
    import jax.numpy as jnp
    import wandb
    from bounded_validation_memory import release_caches
    from fixed_plane_v322_condition_gates import (
        ConditionEndpointWindow, condition_gate_settings, condition_gates_passed,
    )
    spec = stack.spec
    grouped = spec.version == 324
    paired_validation = opts.validation_protocol == PAIRED_VALIDATION_V2
    training_only = opts.validation_protocol == TRAINING_QUALITY_V3
    retry_schedule = (protocol_contract(spec.version)['retry_updates'] if paired_validation
        else [0, 0] if training_only
        else validation_retry_schedule(spec.version, opts.s1_eval_retry_updates))
    collect, update = make_train_functions(stack, env, args, 'topology_v2')
    contract = dict(profile=profile_for(spec, opts.validation_protocol), parent_model=spec.parent_model,
        parent_sha256=spec.parent_sha256, parent_step=spec.parent_step,
        course=[asdict(s) for s in spec.course],
        noise={k: getattr(stack.cfg, k) for k in NOISE_FIELDS},
        learner=learner_settings(stack.parent['args']), binding=binding,
        parent_contract=stack.saved_contract,
        execution_qualification='three paired real updates, not convergence evidence',
        validation_retry_updates=retry_schedule,
        validation_protocol=opts.validation_protocol,
        validation_gate_contract=protocol_contract(spec.version) if paired_validation else None,
        resume_history='fresh rolling windows and reset episodes; actor/critic/Adam/time retained',
        acceptance='fresh noisy-domain endpoints and rolling32/hold16 quality; frozen parent evaluator',
        hardware_coverage=False)
    if training_only:
        contract.update(training_gate_contract=training_protocol_contract(spec.version),
            acceptance='event-weighted rolling32 quality/hold16; training-only completion',
            training_episode_s=1200., independent_evaluation_required=False,
            source_checkpoint_sha256=binding['checkpoint_sha256'])
    (opts.runtime / 'contract.json').write_text(json.dumps(contract, indent=2))
    opts.save_dir.mkdir(parents=True, exist_ok=False)
    run = wandb.init(project='pingpong-mjx', id=opts.run_id, name=opts.run_id,
        resume='never', mode='online', dir=str(opts.runtime),
        settings=wandb.Settings(x_disable_stats=True),
        config={**contract, 'env_cfg': asdict(stack.cfg), 'seed': opts.seed,
                'command': sys.argv, 'devices': [str(d) for d in jax.devices()]},
        tags=[f'v{spec.version}', spec.parent_model, '1200s', 'camera-ar1-1x', EXECUTION_VERSION])
    run.define_metric('update')
    run.define_metric('*', step_metric='update')
    artifact = wandb.Artifact(f'{opts.run_id}-source', type='source')
    artifact.add_dir(str(opts.runtime / 'source'))
    artifact.add_file(str(opts.runtime / 'contract.json'))
    artifact.add_file(str(opts.runtime / 'command.sh'))
    artifact.add_file(str(opts.execution_audit))
    run.log_artifact(artifact)
    stop = stop_flag if stop_flag is not None else [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    retention = CheckpointRetention(opts.save_dir)
    row, evaluation = {}, None
    training_gate = TrainingQualityWindow() if training_only else None
    training_gate_report = None
    prefix = f'v{spec.version - 2}'
    entry_update, hold = update_idx, 0
    endpoint = EndpointWindow(spec.course[stage_index].seconds, env.dt)
    conditions = (ConditionEndpointWindow(spec.course[stage_index].seconds, env.dt)
                  if grouped else None)
    quality_window = deque(maxlen=32)
    last_eval_update = -1000
    memory_lease = startup_lease
    trim_startup_cache = startup_lease is not None

    def save(converged: bool = False, *, training_qualified: bool = False,
             interrupted: bool = False):
        lesson = spec.course[stage_index]
        args.resume_start_stage = lesson.name
        args.advance_eval_steps = round((lesson.seconds if paired_validation
            else 60. if grouped else lesson.seconds) / env.dt)
        args.advance_eval_n_envs = args.advance_eval_min_episodes = (
            1024 if paired_validation else 16 if grouped else 128)
        if training_only:
            args.advance_eval_steps = 0
            args.advance_eval_n_envs = args.advance_eval_min_episodes = 0
        score = checkpoint_selection_score(spec.version, stage_index,
            stack.helpers.quality_passed(row.get('quality', {})),
            row.get('endpoint', {}), row.get('conditions'),
            float(np.asarray(runner.running_length).mean()) * env.dt)
        path = opts.save_dir / (f'converged_s{stage_index + 1}_{lesson.name}.pkl'
            if converged else f'periodic_u{update_idx:08d}.pkl')
        if training_only and (training_qualified or interrupted):
            path = opts.save_dir / protected_checkpoint_name(update_idx,
                qualified=training_qualified)
        if path.exists():
            raise FileExistsError(path)
        extra = dict(stage_index=stage_index, stage_name=lesson.name,
            stage_update=update_idx-entry_update, global_update=update_idx,
            stage_converged=converged, ball_observation_noise_contract=contract,
            last_summary=row, independent_evaluation=evaluation,
            selection_score=score, selection_kind='mixed_policy_training_rank')
        if paired_validation:
            extra['validation_protocol'] = opts.validation_protocol
            extra['final_acceptance_passed'] = bool(stage_index == 1
                and evaluation and evaluation.get('final_acceptance_passed'))
        if training_only:
            extra.update(validation_protocol=TRAINING_QUALITY_V3,
                final_acceptance_passed=False, deterministic_certified=False,
                training_quality_gate=training_gate_report,
                checkpoint_kind=('training_qualified' if training_qualified else
                    'interrupted' if interrupted else 'periodic'))
            if training_qualified:
                extra['training_acceptance'] = completion_metadata(
                    training_gate_report, update_idx)
        if grouped:
            extra['stage_actor_anchor_reference_params'] = stack.parent[
                'stage_actor_anchor_reference_params']
        stack.ppo.save_checkpoint(path, train_state, args, env, global_step, extra=extra)
        if not converged and not training_qualified and not interrupted:
            retention.register(path, score)

    try:
        while not stop[0]:
            started = time.monotonic()
            previous_t = int(train_state.opt.t)
            runner, transitions = collect(train_state.params, runner)
            train_state, losses = update(train_state, runner, transitions)
            loss = {k: float(v) for k, v in jax.device_get(losses).items()}
            if (int(train_state.opt.t)-previous_t != 32
                    or not all(np.isfinite(v) for v in loss.values())):
                raise FloatingPointError('Noise continuation requires all32 finite Adam steps')
            update_idx += 1
            global_step += 131072
            done, terminated = (np.asarray(v, bool) for v in (
                transitions.done, transitions.terminated))
            length, hits = map(np.asarray, (transitions.episode_length, transitions.hit_count))
            recent = np.asarray(transitions.metrics[f'{prefix}/recent_hit']) > .5
            endpoint.add(length, hits, done, recent, terminated=terminated)
            if grouped:
                conditions.add(length, hits, done, recent,
                    np.asarray(transitions.metrics['v322/initial_group']),
                    np.asarray(transitions.metrics['v322/initial_y_m']), terminated=terminated)
            means = {k.removeprefix(stack.ppo.COMPACT_METRIC_MEAN_PREFIX): float(v)
                for k, v in jax.device_get(jax.tree.map(jnp.mean, transitions.metrics)).items()
                if k.startswith(stack.ppo.COMPACT_METRIC_MEAN_PREFIX)}
            if means[f'{prefix}/finite'] != 1. or not all(np.isfinite(v) for v in means.values()):
                raise FloatingPointError('Nonfinite noise-domain simulation metrics')
            if set(QUALITY_FIELDS) - set(means):
                raise ValueError('Missing native quality metric channels')
            quality_window.append(means)
            quality = quality_summary({k: sum(m[k] for m in quality_window)*131072
                                       for k in QUALITY_FIELDS})
            outcome = endpoint.summary()
            lesson = spec.course[stage_index]
            primary_ok = (outcome.get('survival', 0.) >= lesson.min_survival
                          and outcome.get('len_fraction', 0.) >= lesson.min_length_fraction)
            grouped_summary = conditions.summary() if grouped else None
            if grouped:
                primary_ok = condition_gates_passed(grouped_summary, condition_gate_settings())
            eligible = (len(quality_window) == 32 and outcome['n'] >= 1024
                and (update_idx-entry_update)*.64 >= lesson.seconds
                and stack.helpers.quality_passed(quality) and primary_ok
                and outcome.get('hits_mean', 0.)/lesson.seconds >= 1.2)
            if paired_validation:
                from fixed_plane_noise_validation_v2 import validation_trigger_ready
                eligible = validation_trigger_ready(len(quality_window), quality,
                    (update_idx-entry_update)*.64, lesson.seconds)
            if training_only:
                training_gate_report = training_gate.add(quality)
                eligible = training_gate_report['eligible']
            hold = hold + 1 if eligible else 0
            row = dict(update=update_idx, stage=stage_index+1, quality=quality,
                       endpoint=outcome, conditions=grouped_summary)
            flat = dict(loss, update=update_idx, global_step=global_step,
                stage=stage_index+1, **{'convergence/hold': hold,
                    'convergence/eligible': int(eligible),
                    'sps': 131072/(time.monotonic()-started)})
            if paired_validation:
                flat['validation/readiness_hold'] = flat.pop('convergence/hold')
                flat['validation/ready'] = flat.pop('convergence/eligible')
            if training_only:
                flat['training_quality/hold'] = flat.pop('convergence/hold')
                flat['training_quality/eligible'] = flat.pop('convergence/eligible')
                flat['training_quality/qualified'] = int(training_gate_report['qualified'])
            for namespace, values in (('quality', quality), ('endpoint', outcome), ('metric', means)):
                flat.update({f'{namespace}/{k}': v for k, v in values.items()})
            if grouped:
                flat.update({f'condition/{name}/{k}': v
                    for name, summary in grouped_summary.items() for k, v in summary.items()})
            if done.any():
                flat.update({'completed/mean_hits': float(hits[done].mean()),
                    'completed/mean_s': float(length[done].mean()*env.dt)})
                flat.update({f'completed/hit{k}_rate': float((hits[done] >= k).mean())
                             for k in (1, 3, 12)})
            run.log(flat)
            hold_label = ('training_quality_hold' if training_only else
                          'ready_hold' if paired_validation else 'hold')
            print(f'[V{spec.version}] S{stage_index+1}/2 u={update_idx} '
                f'P={outcome.get("survival", "pending")} quality={quality} '
                f'{hold_label}={hold}/16 Adam+32 execution={EXECUTION_VERSION}', flush=True)
            del transitions
            if memory_lease is not None:
                if trim_startup_cache:
                    # Drop reset/first-trace caches once. Recompile the hot
                    # collector/update under the same slot before releasing it.
                    release_caches()
                    trim_startup_cache = False
                    print(f'[V{spec.version} memory slot] initialization caches released; '
                          'recompiling hot PPO once before releasing the slot', flush=True)
                else:
                    memory_lease.release()
                    memory_lease = None
                    print(f'[V{spec.version} memory slot] hot PPO ready; slot released', flush=True)
            if update_idx % 100 == 0:
                save()
            if training_only and training_gate_report['qualified']:
                save(training_qualified=True)
                (opts.runtime/'training_completion.json').write_text(json.dumps(
                    dict(profile=contract['profile'], update=update_idx,
                        training_episode_s=1200., quality=quality,
                        acceptance=completion_metadata(training_gate_report, update_idx)),
                    indent=2))
                print(f'[V{spec.version}] training-quality complete at u={update_idx}; '
                      '1200s training episodes retained; no frozen acceptance claimed', flush=True)
                break
            if not training_only and validation_due(hold=hold, update_idx=update_idx,
                    last_eval_update=last_eval_update,
                    retry_updates=retry_schedule[stage_index]):
                jax.block_until_ready((runner, train_state))
                release_caches()
                if paired_validation:
                    from fixed_plane_noise_validation_v2 import ValidationMemoryLease
                    memory_lease = ValidationMemoryLease(spec.version)
                    if not memory_lease.acquire(stop):
                        memory_lease = None
                        break
                # Keep each parent's independent evaluator and numerical gates.
                # M13's original cheap final evaluator uses centered +/-1mm
                # controls; all five native groups remain in training gates.
                if paired_validation:
                    from fixed_plane_noise_validation_v2 import make_context, evaluate_policy
                    context = make_context(stack, args)
                    evaluation = evaluate_policy(context, train_state.params, stage_index, stop)
                    (opts.runtime/f'validation_u{update_idx:08d}.json').write_text(
                        json.dumps(evaluation, indent=2))
                    validation_log = {'validation/passed': int(evaluation['passed']),
                        'validation/deterministic_ready': int(evaluation['deterministic_ready']),
                        'validation/final_acceptance_passed': int(evaluation['final_acceptance_passed'])}
                    for mode, measured in evaluation['modes'].items():
                        summary = measured.get('all', {})
                        for key in ('survival', 'len_fraction', 'hits_mean'):
                            if key in summary:
                                validation_log[f'validation/{mode}/{key}'] = summary[key]
                        for key,value in summary.get('quality', {}).items():
                            validation_log[f'validation/{mode}/quality/{key}'] = value
                        for name, cohort in measured.get('conditions', {}).items():
                            for key in ('n', 'survival', 'len_fraction'):
                                validation_log[f'validation/{mode}/condition/{name}/{key}'] = cohort.get(key, 0.)
                    run.log({'update': update_idx, **validation_log})
                    del context
                else:
                    eval_cfg = stack.cfg
                    if grouped:
                        from fixed_plane_v322_condition_gates import centered_acceptance_config
                        eval_cfg = centered_acceptance_config(eval_cfg)
                    eval_env = stack.env_class(stack.xml, n_envs=(16 if grouped else 128), cfg=eval_cfg)
                    evaluator = stack.module.build_evaluator(eval_env, stack.ppo,
                        args.min_log_std, args.max_log_std)
                    eval_lesson = (stack.helpers.evaluation_lesson_for('deployment_groups')
                                   if grouped else lesson)
                    evaluation = stack.module.evaluate(eval_env, evaluator, train_state.params,
                        eval_lesson, opts.seed+100000+update_idx, stop, run.log)
                    del evaluator, eval_env
                print(f'[V{spec.version} eval result] u={update_idx} '
                      f'retry_updates={retry_schedule[stage_index]} '
                      f'result={json.dumps(evaluation)}', flush=True)
                last_eval_update = update_idx
                release_caches()
                if evaluation and evaluation['passed']:
                    save(converged=True)
                    stage_index += 1
                    if stage_index == 2:
                        break
                    entry_update, hold = update_idx, 0
                    quality_window.clear()
                    endpoint = EndpointWindow(spec.course[stage_index].seconds, env.dt)
                    conditions = (ConditionEndpointWindow(spec.course[stage_index].seconds, env.dt)
                                  if grouped else None)
                    # Start a fresh noisy population at this new acceptance
                    # deadline; retain the live runner, weights and optimizer.
            temperature = float(subprocess.check_output(['nvidia-smi', '-i', spec.gpu_uuid,
                '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], text=True))
            if temperature >= 78:
                stop[0] = True
        if stop[0] and stage_index < 2 and (training_only or update_idx % 100):
            save(interrupted=training_only)
    finally:
        if memory_lease is not None:
            memory_lease.release()
        retention.prune()
        run.finish()
