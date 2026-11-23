"""At most20 S01 pure-imagination Actor updates, split by frozen native evaluations."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from .actor_critic import average_gradients, frozen_value
from .artifacts import write_json_unique
from .audit_course_pool import summarize_windows
from .audit_full_prior_world import release_cpu_heap
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .course_learner import (make_course_gradient, compact_course_window,
    compact_critic_arrays, commit_course_windows)
from .course_rollout import validate_course_window
from .imagination_checkpoint import save_checkpoint, restore_checkpoint
from .imagination_learner import slice_environment
from .persistent_pool import count_as_int
from .pilot_checkpoint import pilot_signature
from .pilot_learning import optimize_pilot
from .pilot_runtime import validate_pilot, require_action_gate, setup_pilot, prepare_pilot_session
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .train_imagination import gradient_summary, tree_difference


def validate_gradient_gate(cfg):
    reference = cfg['bounded_imagination_pilot']['gradient_check']
    report = json.loads(authenticate(repository_root(), reference).read_text())
    if (report['status'] != 'CANDIDATE_GRADIENT_RECHECK_COMPLETE' or not report['pilot_gate_passed']
            or report['runtime']['model_references']['auxiliary']['sha256']
            != cfg['bounded_imagination_pilot']['candidate_checkpoint']['sha256']):
        raise ValueError('Candidate gradient admission did not pass')
    return report['selected_microbatch']


def gradient_window(session, fn, microbatch, guard):
    """One frozen logical batch, with compact host returns and rollout statistics."""
    gradients, windows, losses = [], [], []
    clipped = active_actions = 0
    rewards, reward_squares, terms = 0., 0., {}
    episodes = {'persistent': [], 'branch': []}
    failures = dict(ball=0, racket=0, orientation=0, joints=0)
    began = time.monotonic()
    for start in range(0, 1024, microbatch):
        guard.check()
        (loss, window), gradient = fn(session.actor, session.target_critic,
            slice_environment(session.pool.environment, start, start + microbatch),
            session.pool.exploration_rng[start:start + microbatch], jnp.zeros((64, microbatch, 3)))
        jax.block_until_ready((loss, window, gradient))
        gradient_summary(gradient)
        validate_course_window(window)
        valid = np.asarray(window.transitions.valid, bool)
        raw = np.asarray(window.raw_action)
        clipped += int((np.abs(raw[valid]) >= 1.).sum())
        active_actions += int(valid.sum()) * 7
        r = np.asarray(window.transitions.reward)[valid]
        rewards += float(r.sum()); reward_squares += float(np.square(r).sum())
        for key, value in window.transitions.reward_terms.items():
            terms[key] = terms.get(key, 0.) + float(np.asarray(value)[valid].sum())
        endpoint = jax.device_get(window.environment)
        ended = np.asarray(endpoint.terminated | endpoint.truncated | endpoint.model_truncated)
        for lane in np.flatnonzero(ended):
            kind = 'persistent' if start + lane < 768 else 'branch'
            episodes[kind].append(dict(logical_lane=int(start + lane),
                hits=int(endpoint.completed_goals[lane] * 16 + endpoint.task.hit_count[lane]),
                length_controls=int(endpoint.episode_steps[lane]),
                time_limit=bool(endpoint.truncated[lane]),
                model_failure=bool(endpoint.model_truncated[lane])))
        failed = np.asarray(window.transitions.terminated)
        for name in failures:
            failures[name] += int((failed & ~np.asarray(
                window.transitions.task_events['bound_' + name], bool)).sum())
        windows.append(compact_course_window(window, session.target_critic))
        gradients.append(gradient); losses.append(float(loss))
        del window
    summary = summarize_windows(windows)
    summary.update(loss=float(np.mean(losses)), reward_sum=rewards,
        reward_square_sum=reward_squares, reward_terms_sum=terms,
        clipped_action_fraction=clipped / max(active_actions, 1),
        completed_episodes={kind: dict(count=len(rows),
            hit1=sum(r['hits'] >= 1 for r in rows), hit3=sum(r['hits'] >= 3 for r in rows),
            hit12=sum(r['hits'] >= 12 for r in rows),
            mean_hits=float(np.mean([r['hits'] for r in rows])) if rows else None,
            mean_length_controls=float(np.mean([r['length_controls'] for r in rows])) if rows else None,
            rows=rows) for kind, rows in episodes.items()},
        physical_failure_bound_violations=failures,
        failure_bound_counts_may_overlap=True,
        gradient_and_returns_wall_s=time.monotonic() - began)
    return average_gradients(gradients, [microbatch] * len(gradients)), windows, summary


def train(cfg, args, guard, report):
    validate_pilot(cfg)
    report['action_check'] = require_action_gate(cfg)
    microbatch = validate_gradient_gate(cfg)
    report['gradient_check'] = cfg['bounded_imagination_pilot']['gradient_check']
    session, bank, table, robot, auxiliary, options, evidence = setup_pilot(cfg, guard)
    signature = pilot_signature(cfg, evidence)
    frozen = numeric_tree_digest((robot, auxiliary))
    if args.resume:
        reference = dict(path=str(args.resume.resolve()), sha256=file_hash(args.resume))
        manifest = json.loads(args.resume.read_text())
        if manifest['metadata']['kind'] != 'bounded_S01_pilot_learner':
            raise ValueError('Expected a bounded pilot learner checkpoint')
        session = restore_checkpoint(reference, dict(learner=session), signature=signature)['learner']
        report['parent_checkpoint'] = reference
        update = int(session.actor_updates)
        if update == 10:
            ref = cfg['bounded_imagination_pilot']['evaluations']['10']
            evaluation = json.loads(authenticate(repository_root(), ref).read_text())
            if (evaluation['status'] != 'COMPLETE'
                    or evaluation['actor_reference'] != manifest['metadata']['actor_export']):
                raise ValueError('The required frozen native evaluation is missing')
            report['preceding_native_evaluation'] = ref
    start = int(session.actor_updates)
    if not (start < args.stop_update <= 20) or args.stop_update not in (10, 20):
        raise ValueError('Bounded pilot update interval is invalid')
    if args.stop_update == 20 and start < 10:
        raise ValueError('Stop at10 for native validation before continuing')
    if int(session.critic_updates) != 4 * start or int(session.actor_optimizer.step) != start:
        raise ValueError('Checkpoint optimizer counts differ')
    # Reserve complete uncompressed snapshots for both boundaries plus exports
    # and logs. Authenticated immutable seed/model archives are referenced.
    raw_bytes = sum(np.asarray(x).nbytes for x in jax.tree.leaves(session))
    actor_bytes = sum(np.asarray(x).nbytes for x in jax.tree.leaves(session.actor))
    remaining_boundaries = 2 if start < 10 else 1
    reserve = remaining_boundaries * (raw_bytes + actor_bytes + 2 * 2**20) + 10 * 2**20
    storage = guard.budget.check(additional_bytes=reserve)
    report.update(runtime=evidence, runtime_signature=signature, initial_actor_updates=start,
        actual_actor_updates=start, actual_critic_updates=int(session.critic_updates),
        frozen_world_before_sha256=frozen, selected_microbatch=microbatch,
        storage_preflight=dict(storage, raw_learner_bytes=raw_bytes, raw_actor_bytes=actor_bytes),
        updates=[], checkpoints=[], default_model_replaced=False)
    write_json_unique(args.output_dir / 'prerun.json', report, guard.budget)
    jax.clear_caches(); release_cpu_heap()
    fn = make_course_gradient(robot, auxiliary, table, **options)
    saved = start

    def save_current(interrupted=False):
        nonlocal saved
        update = int(session.actor_updates)
        meta = dict(actor_updates=update, critic_updates=int(session.critic_updates),
            imagined_valid_transitions=count_as_int(session.pool.imagined_transition_words),
            teacher_transitions_used_for_learning=0, source_commit=report['source_commit'],
            candidate_checkpoint=cfg['bounded_imagination_pilot']['candidate_checkpoint'],
            model_references=evidence['model_references'], training_bank=evidence['training_bank'],
            config_sha256=config_hash(cfg), interrupted=interrupted,
            elapsed_worker_s=guard.report()['wall_s'], default_model_replaced=False)
        actor_export = None
        if update in (10, 20):
            actor_export = save_checkpoint(args.output_dir / f'actor_u{update:03d}.json',
                dict(actor=session.actor), signature=signature, budget=guard.budget,
                metadata=dict(meta, kind='bounded_S01_pilot_actor'))
        checkpoint = save_checkpoint(args.output_dir / f'learner_u{update:03d}.json',
            dict(learner=session), signature=signature, budget=guard.budget,
            metadata=dict(meta, kind='bounded_S01_pilot_learner', actor_export=actor_export))
        restored = restore_checkpoint(checkpoint, dict(learner=session), signature=signature)['learner']
        if numeric_tree_digest(restored) != numeric_tree_digest(session):
            raise ValueError('Complete learner restore changed values')
        next_a, draw_a = prepare_pilot_session(session, bank)
        next_b, draw_b = prepare_pilot_session(restored, bank)
        if numeric_tree_digest(next_a) != numeric_tree_digest(next_b) or draw_a != draw_b:
            raise ValueError('Restored pool sampling or RNG differs')
        report['checkpoints'].append(dict(update=update, learner=checkpoint, actor=actor_export,
            full_restore_exact=True, next_pool_sampling_exact=True))
        saved = update
        print(f'[SAVE] Actor={update} Critic={int(session.critic_updates)} checkpoint={checkpoint["path"]}', flush=True)

    try:
        for update in range(start + 1, args.stop_update + 1):
            guard.check()
            began = time.monotonic()
            prepared, sampling = prepare_pilot_session(session, bank)
            gradient, windows, metrics = gradient_window(prepared, fn, microbatch, guard)
            inputs, targets = compact_critic_arrays(windows)
            online = np.asarray(frozen_value(session.critic, jnp.asarray(inputs)))
            variance = float(np.var(targets))
            metrics['explained_variance_before'] = 1. - float(np.var(targets - online)) / variance if variance > 1e-12 else None
            committed = commit_course_windows(prepared, windows)
            optimization_started = time.monotonic()
            candidate, learning = optimize_pilot(committed, gradient, inputs, targets,
                cfg['imagination'], authorized=True)
            if int(candidate.actor_updates) != update or int(candidate.critic_optimizer.step) != 4 * update:
                raise ValueError('Unexpected optimizer increment')
            # Only a complete finite transaction replaces the current session.
            session = candidate
            metrics.update(update=update, critic_updates=4 * update, pool_sampling=sampling,
                learning=learning, optimizer_wall_s=time.monotonic() - optimization_started,
                update_wall_s=time.monotonic() - began, actor_gradient=gradient_summary(gradient),
                imagined_valid_total=count_as_int(session.pool.imagined_transition_words),
                source='TRAIN_MIXED_IMAGINATION', kl=None, entropy=None,
                optimizer_scope='pathwise_actor_critic_not_PPO', stage_advanced=False)
            report['updates'].append(metrics)
            report.update(actual_actor_updates=update, actual_critic_updates=4 * update)
            write_json_unique(args.output_dir / f'update_{update:03d}.json', metrics, guard.budget)
            print(f'[TRAIN] S01 u={update}/20 Critic={4*update} valid={metrics["valid"]} '
                  f'release={metrics["events"]["valid_release"]} loss={metrics["loss"]:.6g} '
                  f'seconds={metrics["update_wall_s"]:.2f}', flush=True)
            del windows, inputs, targets, prepared, candidate, committed, gradient
            release_cpu_heap()
        save_current()
        report['status'] = 'BOUNDED_PILOT_SEGMENT_COMPLETE_REQUIRES_FROZEN_NATIVE_EVALUATION'
    except Exception:
        if int(session.actor_updates) > saved:
            save_current(interrupted=True)
        raise
    finally:
        report['frozen_world_after_sha256'] = numeric_tree_digest((robot, auxiliary))
        if report['frozen_world_after_sha256'] != frozen:
            raise ValueError('Frozen world parameters changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--stop-update', type=int, choices=(10, 20), required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, 43200.) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        sources = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_v1_bounded_S01_imagination_pilot', status='INCOMPLETE',
            training_version='WMJ-V1', config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            source_sha256={p.name: file_hash(p) for p in sources}, world_updates=0,
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            world_qualified=False, promoted=False, wandb=None)
        write_json_unique(args.output_dir / 'implementation.json',
            dict(report=report, source={p.name: p.read_text() for p in sources}), guard.budget)
        write_json_unique(args.output_dir / 'config_snapshot.json', cfg, guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            train(cfg, args, guard, report)
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_json_unique(args.output_dir / 'result.json', report, guard.budget)


if __name__ == '__main__':
    main()
