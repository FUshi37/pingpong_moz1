"""Bounded existing-Critic fitting with a frozen Actor and structured world."""

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
from .course_env import course_critic_layout
from .course_goals import build_goal_table
from .course_rollout import course_policy_window, validate_course_window
from .critic_fit import (restore_learning, make_update, update_session,
                         prepare_fit_state, unchanged_actor_digest)
from .critic_targets import lambda_mixture_oracle, regression_stats, activation_summary
from .imagination_checkpoint import save_checkpoint, restore_checkpoint
from .imagination_learner import init_learner, commit_policy_windows, critic_training_arrays
from .imagined_rollout import window_batch
from .native_entry_collision import write_compact_json
from .paired_precision import random32_clone
from .persistent_pool import init_pool, count_as_int
from .recover_reward_direction import write_gzip_json
from .reward_direction import make_prepare
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest, pack_numeric_state, restore_numeric_state
from .train_pose_trial import phase_budget
from .velocity_reward import VelocityRewardContract, candidate_course_step


def take(tree: Any, indices: Any) -> Any:
    return jax.tree.map(lambda x: x[indices], tree)


def storage_reservation(state: dict) -> dict:
    leaves = jax.tree.leaves(state)
    raw = sum(np.asarray(v).nbytes for v in leaves)
    return dict(raw_bytes=raw, leaves=len(leaves),
                single_archive_writer_bytes=raw+raw//100+len(leaves)*8192+65536)


def check_plan(cfg: dict) -> dict:
    setting = cfg['critic_only_fit']
    plan = json.loads(authenticate(repository_root(), setting['plan']).read_text())
    if (not setting['authorized'] or setting['default_enabled'] or not plan['authorized']
            or plan['maximum_critic_updates'] != 512 or plan['logical_envs'] != 128
            or plan['horizon'] != 64 or plan['actor_updates'] or plan['world_updates']
            or plan['total_cap_bytes'] != cfg['resources']['max_new_bytes']
            or plan['recipe'] != cfg['imagination']):
        raise ValueError('Critic-only authorization or recipe differs')
    for key in ('physics', 'control'):
        if config_hash(cfg[key]) != plan[key+'_sha256']:
            raise ValueError('Preserved contract changed')
    if config_hash(cfg['imagination_implementation']['reward']['coefficients']) != plan['default_reward_sha256']:
        raise ValueError('Default reward changed')
    for name, digest in plan['preserved_sources'].items():
        if file_hash(Path(__file__).parent/name) != digest:
            raise ValueError('Preserved implementation changed: '+name)
    if file_hash(Path(__file__).parent/'contracts.py') != plan['budget_validator_revision']['after_sha256']:
        raise ValueError('Confirmed exact-budget extension changed')
    assert len(set(plan['train_indices'])) == 128 and len(set(plan['heldout_indices'])) == 32
    assert not set(plan['train_indices']) & (set(plan['heldout_indices']) | set(plan['excluded_indices']))
    assert not set(plan['heldout_indices']) & set(plan['excluded_indices'])
    roots = [row['root_seed_family'] for row in plan['source_rows']]
    assert len(set(roots)) == 160
    authenticate(repository_root(), plan['prerequisite'])
    return plan


def make_forward(data: dict):
    initial = data['initial']
    goals = np.unique(np.asarray(initial.task.goal_xyz), axis=0)
    table = build_goal_table(goals, static_admitted=np.ones(len(goals), bool))
    step = candidate_course_step(data['runtime'], VelocityRewardContract(True, 1.23, .43))
    function = random32_clone(course_policy_window, dict(course_step=step))
    options = dict(data['options'], control_contract=data['control'],
        episode_contract=data['episode'], reward_contract=data['reward'], stage_id='S01')

    def forward(state, keys, std, horizon):
        return function(data['actor'], data['robot'], data['auxiliary'], state, keys,
            table, horizon=horizon, exploration_std=std, rematerialize=False, **options)
    return jax.jit(forward, static_argnames=('std', 'horizon'))


def check_window(window: Any) -> None:
    validate_course_window(window)
    if np.any(np.asarray(window.transitions.model_truncated)):
        raise FloatingPointError('Untrusted world trajectory; stop before Critic update')
    for a in jax.tree.leaves(window):
        if not np.isfinite(np.asarray(a)).all():
            raise FloatingPointError('Nonfinite model rollout; no fitting permitted')


def numerical_pair(first: Any, second: Any) -> dict:
    worst = 0.
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype.kind == 'f':
            worst = max(worst, float(np.max(np.abs(a-b))) if a.size else 0.)
            if not np.allclose(a, b, atol=1e-5, rtol=1e-6):
                raise ValueError('Restored model rollout differs beyond FP32 tolerance')
        elif not np.array_equal(a, b):
            raise ValueError('Restored rollout changed a discrete state')
    return dict(maximum_float_abs=worst, discrete_exact=True, passed=True)


def restore_gate(state: dict, restored: dict) -> dict:
    if numeric_tree_digest(state) != numeric_tree_digest(restored):
        raise ValueError('Full learner/reset banks/RNG changed on restore')
    a, ar = prepare_fit_state(state); b, br = prepare_fit_state(restored)
    if ar != br or numeric_tree_digest(a) != numeric_tree_digest(b):
        raise ValueError('Restore changed the next source draw, queue or exploration RNG')
    return dict(full_numeric_state_exact=True, next_source_and_rng_exact=True)


def evaluate(session: Any, windows: dict, parent_targets: dict, plan: dict) -> dict:
    """Use common fixed trajectories; terminal Monte Carlo labels have no EMA."""
    report = {}
    for mode, window in windows.items():
        batch = window_batch(window)
        valid = np.asarray(batch.valid, bool)
        failure = np.asarray(window.transitions.terminated, bool)
        complete = valid & failure.any(axis=0)[None, :]
        # Only physically failed lanes count in this realized-return statistic.
        # Truncated/live lanes remain reported with their lambda targets below.
        mc = np.asarray(lambda_returns(batch, jnp.zeros_like(batch.reward), lambda_value=1.))
        x = np.asarray(window.critic_inputs)
        online = np.asarray(frozen_value(session.critic, window.critic_inputs))
        ema = np.asarray(frozen_value(session.target_critic, window.critic_inputs))
        own = np.asarray(lambda_returns(batch,
            frozen_value(session.target_critic, window.transitions.next_critic_features), lambda_value=.99))
        common = parent_targets[mode]
        rows = []
        for lane, source in enumerate(plan['heldout_indices']):
            failed = np.flatnonzero(failure[:, lane])
            end = int(failed[0]) if failed.size else None
            rows.append(dict(source_index=source, valid_rows=int(valid[:, lane].sum()),
                failure_control=None if end is None else end+1,
                online_start=float(online[0, lane]), ema_start=float(ema[0, lane]),
                realized_start=float(mc[0, lane]) if end is not None else None,
                terminal_online=float(online[end, lane]) if end is not None else None,
                terminal_ema=float(ema[end, lane]) if end is not None else None,
                terminal_return=float(mc[end, lane]) if end is not None else None))
        report[mode] = dict(valid_rows=int(valid.sum()), failed_sources=int(failure.any(axis=0).sum()),
            censored_sources=int((~failure.any(axis=0)).sum()), sources=rows,
            online=dict(realized=regression_stats(online[complete], mc[complete]),
                terminal=regression_stats(online[failure], mc[failure]),
                fixed_parent_lambda=regression_stats(online[valid], common[valid]),
                own_lambda=regression_stats(online[valid], own[valid]),
                activations=activation_summary(session.critic, x[valid])),
            ema=dict(realized=regression_stats(ema[complete], mc[complete]),
                terminal=regression_stats(ema[failure], mc[failure]),
                fixed_parent_lambda=regression_stats(ema[valid], common[valid]),
                own_lambda=regression_stats(ema[valid], own[valid]),
                activations=activation_summary(session.target_critic, x[valid])),
            common_trajectory_sha256=numeric_tree_digest((window.critic_inputs, window.transitions)),
            origin='imagination_only', kl=None, actor_changed=False)
    return report


def compare(baseline: dict, final: dict, criteria: dict) -> dict:
    result = {}
    for mode, row in final.items():
        old = baseline[mode]
        assert row['common_trajectory_sha256'] == old['common_trajectory_sha256']
        online_ratio = row['online']['realized']['rmse']/old['online']['realized']['rmse']
        ema_ratio = row['ema']['realized']['rmse']/old['ema']['realized']['rmse']
        ev = row['online']['realized']['explained_variance']
        gates = dict(sufficient_failed_sources=row['failed_sources'] >= criteria['minimum_failed_sources'],
            online_rmse=online_ratio <= criteria['online_realized_rmse_ratio_max'],
            ema_rmse=ema_ratio <= criteria['ema_realized_rmse_ratio_max'],
            online_variance=ev is not None and ev >= criteria['online_realized_explained_variance_min'],
            terminal_bias=abs(row['online']['terminal']['bias']) <= criteria['online_terminal_abs_bias_max'])
        result[mode] = dict(online_realized_rmse_ratio=online_ratio,
            ema_realized_rmse_ratio=ema_ratio, gates=gates, diagnostic_pass=all(gates.values()))
    return result


def run(cfg: dict, args: Any, guard: Any, report: dict, plan: dict) -> None:
    data = make_prepare()(cfg, args, guard, report, plan)
    layout = course_critic_layout(data['initial'])
    if layout['dimension'] != 2602:
        raise ValueError('Critic input layout changed')
    train, held = take(data['initial'], slice(0, 128)), take(data['initial'], slice(128, 160))
    keys = jax.random.split(jax.random.PRNGKey(plan['seed']+1), 128)
    template = init_learner(data['actor'], init_critic(jax.random.PRNGKey(0), 2602),
                            init_pool(train, keys), seed=plan['seed']+2)
    session = restore_learning(plan['parent_learner'], template)
    if numeric_tree_digest(session.actor) != numeric_tree_digest(data['actor']):
        raise ValueError('Parent Actor differs from registered frozen u30 export')
    state = dict(learner=session, reset_bank=train, validation_bank=held,
                 current_source_indices=jnp.arange(128, dtype=jnp.int32),
                 fit_transactions=jnp.asarray(0, jnp.int32))
    frozen_actor = unchanged_actor_digest(session)
    frozen_world = numeric_tree_digest((data['robot'], data['auxiliary']))
    sig = config_hash(dict(plan=report['plan'], sources=report['source_sha256'],
                          layout=layout, model_references=report['model_references']))
    report.update(runtime_signature=sig, parent_learning_sha256=numeric_tree_digest(
        {k:getattr(session, k) for k in session._fields if k != 'pool'}),
        frozen_actor_before_sha256=frozen_actor, frozen_world_before_sha256=frozen_world)
    if args.resume:
        ref = dict(path=str(args.resume.resolve()), sha256=file_hash(args.resume))
        manifest = json.loads(args.resume.read_text())
        if manifest['metadata']['plan'] != report['plan']:
            raise ValueError('Resume belongs to another Critic-only trial')
        state = restore_checkpoint(ref, state, signature=sig)
        if unchanged_actor_digest(state['learner']) != frozen_actor:
            raise ValueError('Resume changed the frozen Actor')
        report['resumed_from'] = ref
    initial_transactions = int(state['fit_transactions'])
    if int(state['learner'].critic_updates) != 120+4*initial_transactions:
        raise ValueError('Checkpoint transaction/update counters disagree')
    reservation = storage_reservation(state)
    # Reserve final raw writer, checkpoint manifest and all remaining small logs.
    guard.budget.check(additional_bytes=reservation['single_archive_writer_bytes']+4*2**20)
    forward = make_forward(data)
    jax.clear_caches(); release_cpu_heap()
    initial_hash = numeric_tree_digest(state)
    if args.resume:
        ref = report['resumed_from']
    else:
        ref = save_checkpoint(args.output_dir/'initial_learner.json', state, signature=sig,
            budget=guard.budget, metadata=dict(kind='critic_only_initial', plan=report['plan'],
            actor_updates=30, critic_updates=120, teacher_learning_rows=0))
    restored = restore_checkpoint(ref, state, signature=sig)
    gate = restore_gate(state, restored)
    prepared, sampling = prepare_fit_state(state)
    repeated, _ = prepare_fit_state(restored)
    del restored
    guard.check()
    window = forward(prepared['learner'].pool.environment,
        prepared['learner'].pool.exploration_rng, std=.03, horizon=64)
    jax.block_until_ready(window); check_window(window)
    other = forward(repeated['learner'].pool.environment,
        repeated['learner'].pool.exploration_rng, std=.03, horizon=64)
    jax.block_until_ready(other)
    gate['restored_forward'] = numerical_pair(window, other)
    x, y = critic_training_arrays([window], prepared['learner'].target_critic)
    oracle = lambda_mixture_oracle(window_batch(window), np.asarray(frozen_value(
        prepared['learner'].target_critic, window.transitions.next_critic_features)), .99)
    np.testing.assert_allclose(y, oracle[np.asarray(window.transitions.valid)], atol=5e-5, rtol=1e-5)
    gate['lambda_oracle_max_abs'] = float(np.max(np.abs(y-oracle[np.asarray(window.transitions.valid)])))
    committed = dict(prepared, learner=commit_policy_windows(prepared['learner'], [window]))
    arrays, manifest = pack_numeric_state(committed, context_sha256=sig)
    replay = restore_numeric_state(committed, arrays, manifest, context_sha256=sig)
    gate['post_window_full_restore'] = restore_gate(committed, replay)
    if numeric_tree_digest(state) != initial_hash:
        raise ValueError('No-update gate changed the starting learner')
    gate.update(status='CRITIC_ONLY_NO_UPDATE_PASS', initial_checkpoint=ref,
        valid_critic_rows=len(y), actor_updates=0, critic_updates=0, world_updates=0,
        storage_preflight=dict(reservation, **guard.budget.check(
            additional_bytes=reservation['single_archive_writer_bytes']+4*2**20)))
    write_compact_json(args.output_dir/'gate.json', gate, guard.budget)
    report['admission'] = dict(path=str(args.output_dir/'gate.json'), sha256=file_hash(args.output_dir/'gate.json'))
    print('[CRITIC_FIT] no-update and full restore admission PASS', flush=True)
    del window, other, x, y, committed, replay, arrays, repeated, prepared
    release_cpu_heap()

    windows, parent_targets = {}, {}
    eval_keys = jax.random.split(jax.random.PRNGKey(plan['seed']+3), 32)
    for mode, std in plan['validation']['modes'].items():
        guard.check()
        w = forward(state['validation_bank'], eval_keys, std=std, horizon=128)
        jax.block_until_ready(w); check_window(w)
        windows[mode] = jax.device_get(w)
        parent_targets[mode] = np.asarray(lambda_returns(window_batch(w),
            frozen_value(session.target_critic, w.transitions.next_critic_features), lambda_value=.99))
    baseline = evaluate(session, windows, parent_targets, plan)
    report['evaluations'] = {str(4*initial_transactions): evaluate(state['learner'], windows, parent_targets, plan)}
    write_gzip_json(args.output_dir/'evaluation_initial.json.gz', report['evaluations'][str(4*initial_transactions)], guard.budget)
    report['updates'] = []
    update_fn = make_update(plan['recipe'])
    saved = initial_transactions

    def save_current(interrupted: bool = False):
        nonlocal saved
        added = int(state['fit_transactions'])*4
        path = args.output_dir/f'learner_critic_added{added:03d}.json'
        ref = save_checkpoint(path, state, signature=sig, budget=guard.budget,
            metadata=dict(kind='critic_only_learner', plan=report['plan'], interrupted=interrupted,
                actor_updates=30, critic_updates=int(state['learner'].critic_updates),
                new_actor_updates=0, new_critic_updates=added, world_updates=0,
                teacher_learning_rows=0, model_references=report['model_references'],
                imagined_valid_transitions=count_as_int(state['learner'].pool.imagined_transition_words)))
        restored = restore_checkpoint(ref, state, signature=sig)
        report['final_checkpoint'] = dict(reference=ref, restore=restore_gate(state, restored))
        saved = int(state['fit_transactions'])

    try:
        for transaction in range(initial_transactions+1, 129):
            guard.check(); started = time.monotonic()
            prepared, sampling = prepare_fit_state(state)
            w = forward(prepared['learner'].pool.environment,
                prepared['learner'].pool.exploration_rng, std=.03, horizon=64)
            jax.block_until_ready(w); check_window(w)
            x, y = critic_training_arrays([w], prepared['learner'].target_critic)
            committed = commit_policy_windows(prepared['learner'], [w])
            trained, info = update_session(committed, x, y, update_fn,
                authorized=True, initial_updates=120, maximum_updates=512, padded_rows=8192)
            state = dict(prepared, learner=trained, fit_transactions=jnp.asarray(transaction, jnp.int32))
            if unchanged_actor_digest(trained) != frozen_actor:
                raise ValueError('Critic-only transaction changed Actor/optimizer state')
            if int(trained.critic_optimizer.step) != int(trained.critic_updates):
                raise ValueError('Critic Adam/counter mismatch')
            events = w.transitions.task_events
            row = dict(transaction=transaction, added_critic_updates=4*transaction,
                actual_critic_updates=int(trained.critic_updates), valid_rows=len(y),
                physical_failures=int(np.asarray(w.transitions.terminated).sum()),
                releases=int(np.asarray(events['valid_release']).sum()),
                successful_goals=int(np.asarray(events['goal_success']).sum()),
                reward_terms={k:float(np.asarray(v).sum()) for k,v in w.transitions.reward_terms.items()},
                physical_failure_bounds={k:int(np.asarray(v).sum()) for k,v in events.items() if k.startswith('bound_')},
                learning=info, sampling=sampling, teacher_learning_rows=0,
                wall_s=time.monotonic()-started)
            report['updates'].append(row)
            write_compact_json(args.output_dir/f'update_{transaction:03d}.json', row, guard.budget)
            report.update(actual_added_critic_updates=4*transaction, actual_critic_updates=int(trained.critic_updates))
            if 4*transaction in plan['validation']['checkpoints']:
                result = evaluate(trained, windows, parent_targets, plan)
                report['evaluations'][str(4*transaction)] = result
                write_gzip_json(args.output_dir/f'evaluation_{4*transaction:03d}.json.gz', result, guard.budget)
            if transaction == 1 or transaction % 8 == 0:
                print(f'[CRITIC_FIT] updates={4*transaction}/512 valid={len(y)} '
                      f'loss={np.mean(info["loss"]):.5f} seconds={row["wall_s"]:.2f}', flush=True)
            del w, x, y, prepared, committed, trained
        save_current()
        report['comparison'] = compare(baseline, report['evaluations']['512'], plan['validation']['criteria'])
        report['status'] = 'CRITIC_ONLY_512_UPDATES_COMPLETE'
    except Exception:
        if int(state['fit_transactions']) > saved:
            save_current(interrupted=True)
        raise
    finally:
        report['frozen_actor_after_sha256'] = unchanged_actor_digest(state['learner'])
        report['frozen_world_after_sha256'] = numeric_tree_digest((data['robot'], data['auxiliary']))
        if (report['frozen_actor_after_sha256'] != frozen_actor
                or report['frozen_world_after_sha256'] != frozen_world):
            raise ValueError('Frozen Actor/world parameters changed')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(); cfg = load_config(); plan = check_plan(cfg)
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, plan['max_worker_wall_s'], startup_reserve_bytes=2**20) as guard:
        phase_budget(guard, plan)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        report = dict(schema='wmj_v1_critic_only_fit', status='INCOMPLETE', training_version='WMJ-V1',
            plan=cfg['critic_only_fit']['plan'], config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            actor_updates=0, world_updates=0, default_replaced=False, wandb=None,
            arguments=sys.argv, source_commit=plan['source_commit'],
            launcher=dict(path=str(Path(__file__).parent.parent/'run_wmj_v1_critic_fit.sh'),
                sha256=file_hash(Path(__file__).parent.parent/'run_wmj_v1_critic_fit.sh')),
            versions={k:importlib.metadata.version(k) for k in ('jax', 'jaxlib', 'mujoco', 'numpy')},
            source_sha256={p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
        write_gzip_json(args.output_dir/'config_snapshot.json.gz', cfg, guard.budget)
        write_gzip_json(args.output_dir/'implementation.json.gz',
            dict(sources={n:(Path(__file__).parent/n).read_text() for n in ('critic_fit.py', 'train_critic_fit.py')},
                protocol=authenticate(repository_root(), plan['protocol']).read_text()), guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            run(cfg, args, guard, report, plan)
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_gzip_json(args.output_dir/'result.json.gz', report, guard.budget)


if __name__ == '__main__':
    main()
