"""Paired Critic EMA continuation with one identical frozen-world trajectory."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .actor_critic import frozen_value, init_critic, lambda_returns
from .audit_full_prior_world import release_cpu_heap
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .critic_fit import make_update, update_session, prepare_fit_state, unchanged_actor_digest
from .critic_targets import lambda_mixture_oracle
from .imagination_checkpoint import save_checkpoint, restore_checkpoint
from .imagination_learner import init_learner, commit_policy_windows, critic_training_arrays
from .imagined_rollout import window_batch
from .native_entry_collision import write_compact_json
from .persistent_pool import init_pool, count_as_int
from .recover_reward_direction import write_gzip_json
from .reward_direction import make_prepare
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest, pack_numeric_state, restore_numeric_state
from .train_critic_fit import (take, storage_reservation, make_forward, check_window,
                               numerical_pair, restore_gate, evaluate, compare)
from .train_pose_trial import phase_budget


def failure_counts(transition: Any) -> dict:
    """Count outside predicates only at actual first physical failures."""
    failed = np.asarray(transition.terminated, bool)
    return {key.removeprefix('bound_'): int((failed & ~np.asarray(value, bool)).sum())
            for key, value in transition.task_events.items() if key.startswith('bound_')}


def shared_digest(state: dict) -> str:
    learner = state['learner']
    return numeric_tree_digest((learner.pool, learner.sampling_rng, learner.goal_rng,
        learner.actor, learner.actor_optimizer, learner.actor_updates, learner.critic_updates,
        state['reset_bank'], state['validation_bank'], state['current_source_indices'],
        state['fit_transactions']))


def check_plan(cfg: dict) -> tuple:
    setting = cfg['critic_ema_pair']
    plan = json.loads(authenticate(repository_root(), setting['plan']).read_text())
    source = json.loads(authenticate(repository_root(), plan['source_plan']).read_text())
    if (not setting['authorized'] or setting['default_enabled'] or not plan['authorized']
            or plan['arms'] != dict(ema001=.01, ema005=.05)
            or plan['maximum_updates_per_arm'] != 512 or plan['parent_critic_updates'] != 632
            or plan['parent_fit_transactions'] != 128 or plan['recipe'] != cfg['imagination']
            or plan['actor_updates'] or plan['world_updates']
            or plan['total_cap_bytes'] != cfg['resources']['max_new_bytes']):
        raise ValueError('EMA pairing authorization or recipe differs')
    for name, digest in plan['preserved_sources'].items():
        if file_hash(Path(__file__).parent/name) != digest:
            raise ValueError('Prior implementation changed: '+name)
    if file_hash(Path(__file__).parent/'contracts.py') != plan['budget_validator_sha256']:
        raise ValueError('Budget validator changed')
    for key in ('physics', 'control'):
        if config_hash(cfg[key]) != plan[key+'_sha256']:
            raise ValueError('Physics/control contract changed')
    if config_hash(cfg['imagination_implementation']['reward']['coefficients']) != plan['default_reward_sha256']:
        raise ValueError('Default reward changed')
    authenticate(repository_root(), plan['prerequisite'])
    return plan, source


def paired_result(groups: dict, criteria: dict) -> dict:
    result = {}
    for mode in groups['ema001']:
        slow, fast = groups['ema001'][mode], groups['ema005'][mode]
        assert slow['common_trajectory_sha256'] == fast['common_trajectory_sha256']
        online = 1-fast['online']['realized']['rmse']/slow['online']['realized']['rmse']
        ema = 1-fast['ema']['realized']['rmse']/slow['ema']['realized']['rmse']
        bias = abs(fast['online']['terminal']['bias'])-abs(slow['online']['terminal']['bias'])
        result[mode] = dict(online_rmse_reduction_fraction=online, ema_rmse_reduction_fraction=ema,
            online_terminal_abs_bias_difference=bias,
            relative_gates=dict(online=online >= criteria['minimum_online_rmse_reduction_fraction'],
                ema=ema >= criteria['minimum_ema_rmse_reduction_fraction'],
                terminal=bias <= criteria['maximum_terminal_abs_bias_regression']))
    return result


def run(cfg: dict, args: Any, guard: Any, report: dict, plan: dict, source: dict) -> None:
    data = make_prepare()(cfg, args, guard, report, source)
    train, held = take(data['initial'], slice(0, 128)), take(data['initial'], slice(128, 160))
    keys = jax.random.split(jax.random.PRNGKey(source['seed']+1), 128)
    learner = init_learner(data['actor'], init_critic(jax.random.PRNGKey(0), 2602),
                           init_pool(train, keys), seed=source['seed']+2)
    template = dict(learner=learner, reset_bank=train, validation_bank=held,
        current_source_indices=jnp.arange(128, dtype=jnp.int32), fit_transactions=jnp.array(0, jnp.int32))
    parent = restore_checkpoint(plan['parent_checkpoint'], template, signature=plan['parent_signature'])
    manifest = json.loads(authenticate(repository_root(), plan['parent_checkpoint']).read_text())
    if (manifest['metadata']['plan'] != plan['parent_plan'] or int(parent['fit_transactions']) != 128
            or int(parent['learner'].critic_updates) != 632
            or int(parent['learner'].critic_optimizer.step) != 632
            or int(parent['learner'].actor_updates) != 30):
        raise ValueError('Expected complete Critic632 parent state')
    if numeric_tree_digest(parent['learner'].actor) != numeric_tree_digest(data['actor']):
        raise ValueError('Frozen Actor differs from parent')
    frozen_actor = unchanged_actor_digest(parent['learner'])
    frozen_world = numeric_tree_digest((data['robot'], data['auxiliary']))
    signatures = {name: config_hash(dict(plan=report['plan'], rate=rate,
        sources=report['source_sha256'], models=report['model_references']))
        for name, rate in plan['arms'].items()}
    arms = {name: parent for name in plan['arms']}
    if bool(args.resume_slow) != bool(args.resume_fast):
        raise ValueError('Both complete paired checkpoints are required for resume')
    if args.resume_slow:
        report['resumed_from'] = {}
        for name, path in zip(plan['arms'], (args.resume_slow, args.resume_fast)):
            ref = dict(path=str(path.resolve()), sha256=file_hash(path))
            payload = json.loads(path.read_text())
            if payload['metadata']['plan'] != report['plan'] or payload['metadata']['arm'] != name:
                raise ValueError('Paired checkpoint belongs to another arm or phase')
            arms[name] = restore_checkpoint(ref, template, signature=signatures[name])
            report['resumed_from'][name] = ref
    if shared_digest(arms['ema001']) != shared_digest(arms['ema005']):
        raise ValueError('Initial pools, counters or sampling RNG differ')
    start = int(arms['ema001']['fit_transactions'])
    if not 128 <= start <= 256:
        raise ValueError('Paired continuation exceeds the fixed update budget')
    for state in arms.values():
        if (int(state['learner'].critic_updates) != 632+4*(start-128)
                or unchanged_actor_digest(state['learner']) != frozen_actor):
            raise ValueError('Unexpected continuation counters or Actor')
    report.update(parent_full_state_sha256=numeric_tree_digest(parent),
        initial_shared_state_sha256=shared_digest(arms['ema001']),
        initial_arm_full_sha256={name:numeric_tree_digest(v) for name,v in arms.items()},
        runtime_signatures=signatures, frozen_actor_before_sha256=frozen_actor,
        frozen_world_before_sha256=frozen_world, initial_phase_transactions=start-128,
        common_window_reused=True, actual_added_critic_updates_per_arm=4*(start-128))
    reservation = storage_reservation(parent)
    estimate = int(np.ceil(manifest['archive']['stored_bytes']*1.2))
    reserve = estimate+reservation['single_archive_writer_bytes']+4*2**20
    report['two_checkpoint_preflight'] = dict(reservation,
        first_archive_estimate_with20percent_headroom=estimate, report_and_log_reserve=4*2**20,
        **guard.budget.check(additional_bytes=reserve))
    forward = make_forward(data)
    jax.clear_caches(); release_cpu_heap()
    restored = restore_checkpoint(plan['parent_checkpoint'], template, signature=plan['parent_signature'])
    gate = dict(parent_disk_restore=restore_gate(parent, restored))
    prepared, _ = prepare_fit_state(arms['ema001'])
    repeat, _ = prepare_fit_state(arms['ema005'])
    before = {name:numeric_tree_digest(value) for name,value in arms.items()}
    guard.check()
    w = forward(prepared['learner'].pool.environment, prepared['learner'].pool.exploration_rng,
                std=.03, horizon=64)
    jax.block_until_ready(w); check_window(w)
    again = forward(repeat['learner'].pool.environment, repeat['learner'].pool.exploration_rng,
                    std=.03, horizon=64)
    jax.block_until_ready(again); gate['paired_no_update_forward'] = numerical_pair(w, again)
    x, y = critic_training_arrays([w], prepared['learner'].target_critic)
    oracle = lambda_mixture_oracle(window_batch(w), np.asarray(frozen_value(
        prepared['learner'].target_critic, w.transitions.next_critic_features)), .99)
    np.testing.assert_allclose(y, oracle[np.asarray(w.transitions.valid)], atol=5e-5, rtol=1e-5)
    gate['lambda_oracle_max_abs'] = float(np.max(np.abs(y-oracle[np.asarray(w.transitions.valid)])))
    committed = dict(prepared, learner=commit_policy_windows(prepared['learner'], [w]))
    arrays, state_manifest = pack_numeric_state(committed, context_sha256=signatures['ema001'])
    after = restore_numeric_state(committed, arrays, state_manifest, context_sha256=signatures['ema001'])
    gate['post_window_restore'] = restore_gate(committed, after)
    assert before == {name:numeric_tree_digest(value) for name,value in arms.items()}
    gate.update(status='PAIRED_EMA_NO_UPDATE_PASS', actor_updates=0, critic_updates=0,
                world_updates=0, initial_checkpoint=plan['parent_checkpoint'])
    write_compact_json(args.output_dir/'gate.json', gate, guard.budget)
    report['admission'] = dict(path=str(args.output_dir/'gate.json'), sha256=file_hash(args.output_dir/'gate.json'))
    print('[EMA_PAIR] full restore and no-update gate PASS', flush=True)
    del restored, prepared, repeat, again, w, x, y, committed, arrays, after
    release_cpu_heap()

    windows, common_targets = {}, {}
    eval_keys = jax.random.split(jax.random.PRNGKey(source['seed']+3), 32)
    for mode, std in plan['evaluation']['modes'].items():
        guard.check()
        window = forward(parent['validation_bank'], eval_keys, std=std, horizon=128)
        jax.block_until_ready(window); check_window(window)
        windows[mode] = jax.device_get(window)
        common_targets[mode] = np.asarray(lambda_returns(window_batch(window),
            frozen_value(parent['learner'].target_critic, window.transitions.next_critic_features), lambda_value=.99))
    baseline = evaluate(parent['learner'], windows, common_targets, source)
    report['baseline'] = baseline
    report['evaluations'] = {str(4*(start-128)): {
        name:evaluate(v['learner'], windows, common_targets, source) for name,v in arms.items()}}
    write_gzip_json(args.output_dir/'evaluation_initial.json.gz', report['evaluations'][str(4*(start-128))], guard.budget)
    update_fns = {name:make_update(dict(plan['recipe'], target_critic_ema=rate)) for name,rate in plan['arms'].items()}
    report['updates'] = []; saved = start

    def save_pair(interrupted: bool = False):
        nonlocal saved
        report['checkpoints'] = {}
        for name, state in arms.items():
            count = 4*(int(state['fit_transactions'])-128)
            ref = save_checkpoint(args.output_dir/f'{name}_critic_added{count:03d}.json', state,
                signature=signatures[name], budget=guard.budget,
                metadata=dict(kind='paired_critic_ema_learner', arm=name, rate=plan['arms'][name],
                    plan=report['plan'], parent=plan['parent_checkpoint'], interrupted=interrupted,
                    actor_updates=30, critic_updates=int(state['learner'].critic_updates),
                    new_critic_updates=count, new_actor_updates=0, world_updates=0, teacher_learning_rows=0,
                    model_references=report['model_references'],
                    imagined_valid_total=count_as_int(state['learner'].pool.imagined_transition_words)))
            restored = restore_checkpoint(ref, state, signature=signatures[name])
            report['checkpoints'][name] = dict(reference=ref, restore=restore_gate(state, restored))
        saved = int(arms['ema001']['fit_transactions'])

    try:
        for transaction in range(start+1, 257):
            guard.check(); began = time.monotonic()
            prepared = {}; sampling = {}
            for name, value in arms.items():
                prepared[name], sampling[name] = prepare_fit_state(value)
            if shared_digest(prepared['ema001']) != shared_digest(prepared['ema005']):
                raise ValueError('Paired state or sampling diverged before rollout')
            p = prepared['ema001']['learner']; forward_start = time.monotonic()
            window = forward(p.pool.environment, p.pool.exploration_rng, std=.03, horizon=64)
            jax.block_until_ready(window); check_window(window)
            forward_s = time.monotonic()-forward_start
            digest = numeric_tree_digest((window.critic_inputs, window.transitions,
                window.raw_action, window.environment, window.exploration_rng))
            outcomes = {}; candidates = {}
            # Each arm sees exactly the same input array and rewards; only its
            # own EMA enters its detached lambda targets.
            for name in plan['arms']:
                x, y = critic_training_arrays([window], prepared[name]['learner'].target_critic)
                committed = commit_policy_windows(prepared[name]['learner'], [window])
                optimizer_start = time.monotonic()
                learner, metrics = update_session(committed, x, y, update_fns[name],
                    authorized=True, initial_updates=632, maximum_updates=512, padded_rows=8192)
                if (int(learner.critic_updates) != 632+4*(transaction-128)
                        or int(learner.critic_optimizer.step) != int(learner.critic_updates)
                        or unchanged_actor_digest(learner) != frozen_actor):
                    raise ValueError('Critic-only paired counters or frozen Actor changed')
                candidates[name] = dict(prepared[name], learner=learner,
                    fit_transactions=jnp.asarray(transaction, jnp.int32))
                outcomes[name] = dict(learning=metrics, optimizer_wall_s=time.monotonic()-optimizer_start,
                    input_sha256=numeric_tree_digest(x), target_sha256=numeric_tree_digest(y), valid_rows=len(y))
            if (outcomes['ema001']['input_sha256'] != outcomes['ema005']['input_sha256']
                    or shared_digest(candidates['ema001']) != shared_digest(candidates['ema005'])):
                raise ValueError('Paired inputs, sampling RNG or forward endpoints diverged')
            arms = candidates
            tr = window.transitions
            row = dict(transaction=transaction-128, added_critic_updates_per_arm=4*(transaction-128),
                valid_rows=len(y), common_window_sha256=digest, sampling=sampling['ema001'],
                actual_physical_failures=int(np.asarray(tr.terminated).sum()),
                time_limits=int(np.asarray(tr.truncated).sum()), model_failures=int(np.asarray(tr.model_truncated).sum()),
                failure_causes_nonexclusive=failure_counts(tr),
                releases=int(np.asarray(tr.task_events['valid_release']).sum()),
                successful_goals=int(np.asarray(tr.task_events['goal_success']).sum()),
                reward_terms={k:float(np.asarray(v).sum()) for k,v in tr.reward_terms.items()},
                shared_forward_wall_s=forward_s, arms=outcomes, wall_s=time.monotonic()-began,
                teacher_learning_rows=0, paired_exact=True)
            report['updates'].append(row)
            report['actual_added_critic_updates_per_arm'] = 4*(transaction-128)
            write_compact_json(args.output_dir/f'update_{transaction-128:03d}.json', row, guard.budget)
            added = 4*(transaction-128)
            if added in plan['evaluation']['checkpoints']:
                result = {name:evaluate(value['learner'], windows, common_targets, source) for name,value in arms.items()}
                report['evaluations'][str(added)] = result
                write_gzip_json(args.output_dir/f'evaluation_{added:03d}.json.gz', result, guard.budget)
            if (transaction-128) == 1 or transaction % 8 == 0:
                losses = {name:float(np.mean(v['learning']['loss'])) for name,v in outcomes.items()}
                print(f'[EMA_PAIR] updates={added}/512perarm valid={len(y)} losses={losses} seconds={row["wall_s"]:.2f}', flush=True)
            del window, tr, x, y, prepared, committed, candidates
        save_pair()
        final = report['evaluations']['512']
        report['within_arm_comparison'] = {name:compare(baseline, final[name], plan['evaluation']['criteria']) for name in plan['arms']}
        report['paired_comparison'] = paired_result(final, plan['paired_criteria'])
        report['status'] = 'PAIRED_EMA_512_PER_ARM_COMPLETE'
    except Exception:
        if int(arms['ema001']['fit_transactions']) > saved and not report.get('checkpoints'):
            save_pair(interrupted=True)
        raise
    finally:
        report['frozen_actor_after_sha256'] = {name:unchanged_actor_digest(v['learner']) for name,v in arms.items()}
        report['frozen_world_after_sha256'] = numeric_tree_digest((data['robot'], data['auxiliary']))
        if any(v != frozen_actor for v in report['frozen_actor_after_sha256'].values()) or report['frozen_world_after_sha256'] != frozen_world:
            raise ValueError('Frozen Actor or world changed')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--resume-slow', type=Path)
    parser.add_argument('--resume-fast', type=Path)
    args = parser.parse_args(); cfg = load_config(); plan, source = check_plan(cfg)
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, plan['max_worker_wall_s'], startup_reserve_bytes=2**20) as guard:
        phase_budget(guard, plan); args.output_dir.mkdir(parents=True, exist_ok=False)
        launcher = Path(__file__).parent.parent/'run_wmj_v1_critic_ema.sh'
        report = dict(schema='wmj_v1_critic_ema_pair', status='INCOMPLETE', training_version='WMJ-V1',
            plan=cfg['critic_ema_pair']['plan'], config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            arguments=sys.argv, source_commit=plan['source_commit'],
            launcher=dict(path=str(launcher), sha256=file_hash(launcher)),
            actor_updates=0, world_updates=0, default_replaced=False, wandb=None,
            versions={k:importlib.metadata.version(k) for k in ('jax','jaxlib','mujoco','numpy')},
            source_sha256={p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
        write_gzip_json(args.output_dir/'config_snapshot.json.gz', cfg, guard.budget)
        write_gzip_json(args.output_dir/'implementation.json.gz', dict(
            source=Path(__file__).read_text(), protocol=authenticate(repository_root(), plan['protocol']).read_text()), guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            run(cfg, args, guard, report, plan, source)
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_gzip_json(args.output_dir/'result.json.gz', report, guard.budget)


if __name__ == '__main__':
    main()
