"""Authorized FP32 pose-margin trial, at most ten Actor and forty Critic updates."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from .actor_critic import frozen_value
from .artifacts import write_json_unique
from .audit_full_prior_world import release_cpu_heap
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .course_learner import compact_critic_arrays, commit_course_windows
from .imagination_checkpoint import restore_checkpoint, save_checkpoint
from .module_gradient_tools import private_clone
from .paired_precision import (compact_window, learning_state, make_gradient,
    prepare_session, setup)
from .pose_trial import phase_plan, signature, verify_gate, with_pose_reward
from .pose_margin import PoseMarginContract
from .persistent_pool import count_as_int
from .pilot_learning import optimize_pilot
from .resources import StorageBudget
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest, pack_numeric_state, restore_numeric_state
from .train_imagination import gradient_summary
from .train_pilot import gradient_window as original_gradient_window


gradient_window = private_clone(original_gradient_window,
                               overrides={'compact_course_window': compact_window})


def phase_budget(guard, plan):
    guard.budget = StorageBudget(guard.budget.root, min(guard.budget.max_bytes,
        plan['phase_start_used_bytes'] + plan['max_phase_bytes']), plan['min_free_bytes'])
    guard.budget.check()


def assert_learning_fp32(session):
    for name in ('actor', 'critic', 'target_critic', 'actor_optimizer', 'critic_optimizer'):
        for leaf in jax.tree.leaves(getattr(session, name)):
            value = np.asarray(leaf)
            if value.dtype.kind == 'f' and value.dtype != np.float32:
                raise ValueError(f'{name} left the FP32 learning boundary')
            if not np.isfinite(value).all():
                raise FloatingPointError(f'{name} is nonfinite')


def restore_checks(session, restored, bank):
    if numeric_tree_digest(session) != numeric_tree_digest(restored):
        raise ValueError('Full learner restore changed values or dtypes')
    next_a, draw_a = prepare_session(session, bank)
    next_b, draw_b = prepare_session(restored, bank)
    if numeric_tree_digest(next_a) != numeric_tree_digest(next_b) or draw_a != draw_b:
        raise ValueError('Saved state changed next sampling, pool or exploration RNG')
    keys = next_a.pool.exploration_rng
    noise = jax.vmap(lambda k: jax.random.normal(k, (7,), dtype=jnp.float32))
    if not np.array_equal(noise(keys), noise(next_b.pool.exploration_rng)):
        raise ValueError('Saved state changed the next FP32 exploration noise')
    return dict(full_state_exact=True, next_sampling_exact=True, next_noise_exact=True)


def validate(cfg, args, guard, report, session, bank, runtime, fn, sig):
    before = numeric_tree_digest(learning_state(session))
    if args.initial_checkpoint:
        checkpoint = dict(path=str(args.initial_checkpoint.resolve()), sha256=file_hash(args.initial_checkpoint))
        payload = json.loads(args.initial_checkpoint.read_text())
        meta = payload['metadata']
        if (meta['kind'] != 'paired_precision_no_update_learner' or meta['mode'] != args.mode
                or meta['plan'] != cfg['paired_precision_trial']['plan']):
            raise ValueError('Reusable initial checkpoint belongs to another experiment')
        checkpoint_signature = payload['runtime_signature']
    else:
        checkpoint_signature = sig
        checkpoint = save_checkpoint(args.output_dir / 'initial_learner.json',
            dict(learner=session), signature=sig, budget=guard.budget,
            metadata=dict(kind='paired_precision_no_update_learner', mode=args.mode,
                          actor_updates=20, critic_updates=80, plan=cfg['paired_precision_trial']['plan']))
    restored = restore_checkpoint(checkpoint, dict(learner=session), signature=checkpoint_signature)['learner']
    report['initial_restore'] = restore_checks(session, restored, bank)
    report['initial_checkpoint'] = checkpoint
    del restored
    prepared, sampling = prepare_session(session, bank)
    gradient, windows, metrics = gradient_window(prepared, fn, args.microbatch, guard)
    inputs, targets = compact_critic_arrays(windows)
    if inputs.dtype != np.float32 or targets.dtype != np.float32:
        raise ValueError('Critic training interfaces must remain FP32')
    committed = commit_course_windows(prepared, windows)
    arrays, manifest = pack_numeric_state(committed, context_sha256=sig)
    restored = restore_numeric_state(committed, arrays, manifest, context_sha256=sig)
    report['post_window_restore'] = restore_checks(committed, restored, bank)
    if numeric_tree_digest(learning_state(session)) != before:
        raise ValueError('No-update check changed learning parameters or RNG')
    assert_learning_fp32(committed)
    report.update(status='POSE_TRIAL_NO_UPDATE_PASS', actor_updates=0, critic_updates=0,
        metrics=metrics, actor_gradient=gradient_summary(gradient), pool_sampling=sampling,
        valid_critic_rows=len(targets), critic_interfaces='float32',
        post_window_storage='lossless_in_memory_roundtrip',
        initial_storage='full_disk_checkpoint_roundtrip',
        initial_learning_unchanged=True)


def require_gates(cfg, args, evidence, sig):
    if len(args.gate or []) != 1:
        raise ValueError('Exactly one candidate no-update gate is required')
    path = args.gate[0]
    row = json.loads(path.read_text())
    verify_gate(row, cfg['pose_margin_trial']['plan'], evidence, sig,
                args.microbatch, cfg['pose_margin_trial']['contract'])
    return [dict(path=str(path.resolve()), sha256=file_hash(path))]


def train(cfg, args, guard, report, session, bank, runtime, fn, sig):
    report['gates'] = require_gates(cfg, args, report['runtime'], sig)
    # Unchanged archive writer reserves raw bytes, expansion and per-leaf headers.
    # Single-arm sequential exports plus 6 MiB for reports/evaluation/resource logs.
    raw = sum(np.asarray(x).nbytes for x in jax.tree.leaves(session))
    actor_raw = sum(np.asarray(x).nbytes for x in jax.tree.leaves(session.actor))
    writer_reserve = raw + raw // 100 + len(jax.tree.leaves(session)) * 8192 + 65536
    actor_reserve = actor_raw + actor_raw // 100 + len(jax.tree.leaves(session.actor)) * 8192 + 65536
    reserve = writer_reserve + actor_reserve + 6 * 2**20
    report['storage_preflight'] = dict(guard.budget.check(additional_bytes=reserve),
        raw_learner_bytes=raw, raw_actor_bytes=actor_raw,
        largest_single_writer_reservation=writer_reserve,
        report_evaluation_export_reservations_included=True)
    report.update(actual_actor_updates=20, actual_critic_updates=80, updates=[], checkpoints=[])
    write_json_unique(args.output_dir / 'prerun.json', report, guard.budget)
    saved = 20

    def save_current(interrupted=False):
        nonlocal saved
        update = int(session.actor_updates)
        meta = dict(mode=args.mode, actor_updates=update, critic_updates=int(session.critic_updates),
            new_actor_updates=update - 20, new_critic_updates=int(session.critic_updates) - 80,
            teacher_transitions_used_for_learning=0, interrupted=interrupted,
            model_references=report['runtime']['model_references'],
            parent=report['runtime']['parent_learner'], plan=report['plan'],
            candidate_contract=report['candidate_contract'],
            imagined_valid_transitions=count_as_int(session.pool.imagined_transition_words),
            elapsed_worker_s=guard.report()['wall_s'], default_model_replaced=False)
        actor = save_checkpoint(args.output_dir / f'actor_u{update:03d}.json',
            dict(actor=session.actor), signature=sig, budget=guard.budget,
            metadata=dict(meta, kind='pose_margin_trial_actor'))
        learner = save_checkpoint(args.output_dir / f'learner_u{update:03d}.json',
            dict(learner=session), signature=sig, budget=guard.budget,
            metadata=dict(meta, kind='pose_margin_trial_learner', actor_export=actor))
        restored = restore_checkpoint(learner, dict(learner=session), signature=sig)['learner']
        checks = restore_checks(session, restored, bank)
        report['checkpoints'].append(dict(actor=actor, learner=learner, checks=checks))
        saved = update

    try:
        for update in range(21, 31):
            guard.check(); began = time.monotonic()
            prepared, sampling = prepare_session(session, bank)
            gradient, windows, metrics = gradient_window(prepared, fn, args.microbatch, guard)
            inputs, targets = compact_critic_arrays(windows)
            committed = commit_course_windows(prepared, windows)
            with jax.enable_x64(False):
                online = np.asarray(frozen_value(session.critic, jnp.asarray(inputs)))
                variance = float(np.var(targets))
                metrics['explained_variance_before'] = (1. - float(np.var(targets - online)) / variance
                    if variance > 1e-12 else None)
                started = time.monotonic()
                candidate, learning = optimize_pilot(committed, gradient, inputs, targets,
                    cfg['imagination'], authorized=True)
            assert_learning_fp32(candidate)
            if (int(candidate.actor_updates) != update or int(candidate.critic_updates) != 4 * update
                    or int(candidate.actor_optimizer.step) != update
                    or int(candidate.critic_optimizer.step) != 4 * update):
                raise ValueError('Unexpected optimizer counts')
            session = candidate
            metrics.update(update=update, critic_updates=4 * update, pool_sampling=sampling,
                learning=learning, optimizer_wall_s=time.monotonic() - started,
                update_wall_s=time.monotonic() - began, actor_gradient=gradient_summary(gradient),
                imagined_valid_total=count_as_int(session.pool.imagined_transition_words),
                kl=None, optimizer_scope='pathwise_actor_critic_not_PPO', stage_advanced=False)
            report['updates'].append(metrics)
            report.update(actual_actor_updates=update, actual_critic_updates=4 * update)
            write_json_unique(args.output_dir / f'update_{update:03d}.json', metrics, guard.budget)
            print(f'[POSE] {args.mode} u={update}/30 critic={4*update} valid={metrics["valid"]} '
                  f'loss={metrics["loss"]:.6g} seconds={metrics["update_wall_s"]:.2f}', flush=True)
            del windows, inputs, targets, prepared, committed, candidate, gradient
            release_cpu_heap()
        save_current()
        report['status'] = 'POSE_TRIAL_TEN_UPDATES_COMPLETE_REQUIRES_NATIVE_EVALUATION'
    except Exception:
        if int(session.actor_updates) > saved:
            save_current(interrupted=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--phase', choices=('validate', 'train'), required=True)
    parser.set_defaults(mode='fp32')
    parser.add_argument('--microbatch', type=int, choices=(256,), default=256)
    parser.add_argument('--gate', type=Path, action='append')
    args = parser.parse_args()
    cfg = load_config(); plan = phase_plan(cfg)
    args.initial_checkpoint = authenticate(repository_root(), plan['initial_checkpoint'])
    paired_plan = json.loads(authenticate(repository_root(), plan['paired_plan']).read_text())
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, 43200., startup_reserve_bytes=2**20) as guard:
        phase_budget(guard, plan)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        sources = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_v1_pose_margin_trial', status='INCOMPLETE',
            training_version='WMJ-V1', mode=args.mode, phase=args.phase,
            plan=cfg['pose_margin_trial']['plan'], candidate_contract=plan['contract'], config_sha256=config_hash(cfg),
            gpu_uuid=args.gpu_uuid, selected_microbatch=args.microbatch,
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            source_sha256={p.name:file_hash(p) for p in sources},
            arguments={k:[str(x) for x in v] if isinstance(v,list) else str(v) if isinstance(v,Path) else v
                       for k,v in vars(args).items()},
            world_updates=0, promoted=False, default_model_replaced=False, wandb=None)
        write_json_unique(args.output_dir / 'implementation.json', dict(report=report,
            sources={n:(Path(__file__).parent/n).read_text() for n in
                     ('train_pose_trial.py', 'pose_trial.py', 'pose_margin.py')}), guard.budget)
        write_json_unique(args.output_dir / 'config_snapshot.json', cfg, guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            session, bank, table, runtime, options, evidence = setup(cfg, guard, args.mode, paired_plan)
            baseline = json.loads(authenticate(repository_root(), plan['baseline_gate']).read_text())
            for name in ('initial_session_sha256', 'inherited_learning_sha256', 'seed_bank_model_identity'):
                if evidence[name] != baseline['runtime'][name]:
                    raise ValueError('Initial state differs from the authenticated FP32 baseline: ' + name)
            report['baseline_initial_pairing'] = dict(exact=True, reference=plan['baseline_gate'])
            runtime = with_pose_reward(runtime, PoseMarginContract(**plan['contract']))
            rows = evidence.pop('source_rows')
            evidence['source_rows_manifest'] = dict(training_bank=evidence['training_bank'],
                row_count=len(rows), reconstructed_metadata_sha256=config_hash(rows),
                reconstruction='setup_pilot source_rows; full metadata retained in authenticated bank and earlier gates')
            sig = signature(cfg, evidence, args.microbatch)
            frozen = numeric_tree_digest((runtime.robot, runtime.auxiliary))
            report.update(runtime=evidence, runtime_signature=sig, frozen_world_before_sha256=frozen)
            jax.clear_caches(); release_cpu_heap()
            with jax.enable_x64(args.mode == 'mixed'):
                assert_learning_fp32(session)
                fn = make_gradient(runtime, table, options)
                operation = validate if args.phase == 'validate' else train
                operation(cfg, args, guard, report, session, bank, runtime, fn, sig)
            report['frozen_world_after_sha256'] = numeric_tree_digest((runtime.robot, runtime.auxiliary))
            if report['frozen_world_after_sha256'] != frozen:
                raise ValueError('Frozen world weights changed')
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_json_unique(args.output_dir / 'result.json', report, guard.budget)


if __name__ == '__main__':
    main()
