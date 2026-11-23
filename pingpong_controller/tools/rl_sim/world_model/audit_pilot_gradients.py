"""Candidate-only full-batch gradient admission; no optimizer mutation."""

import argparse
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_json_unique
from .audit_course_pool import summarize_windows
from .audit_full_prior_world import release_cpu_heap
from .contracts import config_hash, file_hash, load_config
from .course_learner import make_course_gradient, course_gradient_window, compact_critic_arrays
from .course_rollout import course_actor_objective
from .imagination_learner import slice_environment, critic_gradient_batch
from .persistent_pool import init_pool
from .pilot_runtime import require_action_gate, setup_pilot, prepare_pilot_session
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .train_imagination import gradient_summary, timed, tree_difference


def audit(cfg, args, guard, report):
    report['action_check'] = require_action_gate(cfg)
    session, bank, table, robot, auxiliary, options, evidence = setup_pilot(cfg, guard)
    session, draw = prepare_pilot_session(session, bank)
    frozen = numeric_tree_digest((session, robot, auxiliary))
    report.update(runtime=evidence, pool_sampling=draw, frozen_before_sha256=frozen)
    jax.clear_caches()
    release_cpu_heap()
    fn = make_course_gradient(robot, auxiliary, table, **options)
    (gradient, windows, metrics), elapsed = timed(course_gradient_window,
        session, table, fn, stage_id='S01', microbatch=args.microbatch,
        horizon=64, compact=True)
    report['full_batch'] = dict(metrics, **gradient_summary(gradient), wall_s=elapsed,
        windows=summarize_windows(windows))
    inputs, targets = compact_critic_arrays(windows)
    picked = np.random.default_rng(cfg['bounded_imagination_pilot']['seed']).integers(len(inputs), size=8192)
    (critic_gradient, loss), elapsed = timed(critic_gradient_batch, session.critic,
        inputs[picked], targets[picked], microbatch=2048)
    report['critic_gradient'] = dict(**gradient_summary(critic_gradient), loss=loss,
        wall_s=elapsed, samples=8192)
    write_json_unique(args.output_dir / 'full_batch.json', report, guard.budget)
    del windows, inputs, targets
    fn.clear_cache()
    release_cpu_heap()
    guard.check()
    small = session._replace(pool=init_pool(slice_environment(session.pool.environment, 0, 4),
        session.pool.exploration_rng[:4]))
    (gradient4, _, _), _ = timed(course_gradient_window, small, table, fn,
        stage_id='S01', microbatch=4, horizon=64, compact=True)
    (gradient2, _, _), _ = timed(course_gradient_window, small, table, fn,
        stage_id='S01', microbatch=2, horizon=64, compact=True)
    difference = tree_difference(gradient4, gradient2)
    maximum = gradient_summary(gradient4)['maximum']
    report['microbatch'] = dict(max_abs=difference, maximum=maximum,
        passed=difference <= 1e-4 + .01 * maximum)
    direction = np.asarray(gradient4['out']['b'])
    norm = float(np.linalg.norm(direction))
    rows = []
    if norm > 1e-10:
        direction = jnp.asarray(direction / norm)

        @jax.jit
        def branch(offset):
            actor = dict(session.actor, out=dict(session.actor['out'],
                b=session.actor['out']['b'] + offset * direction))
            loss, window = course_actor_objective(actor, session.target_critic, robot,
                auxiliary, small.pool.environment, small.pool.exploration_rng, table,
                horizon=64, **options)
            return loss, (window.transitions.terminated,
                window.transitions.task_events['valid_release'],
                window.transitions.task_events['valid_apex'])

        for epsilon in (.0003, .001, .003, .01, .02):
            plus, _ = timed(branch, epsilon)
            minus, _ = timed(branch, -epsilon)
            delta = float((plus[0] - minus[0]) / (2 * epsilon))
            same = all(np.array_equal(a, b) for a, b in zip(plus[1], minus[1]))
            rows.append(dict(epsilon=epsilon, automatic_derivative=norm,
                central_difference=delta, same_recorded_event_ticks=same,
                relative_error=abs(delta - norm) / max(norm, abs(delta), 1e-8)))
    report['finite_difference'] = dict(rows=rows, passed=any(
        r['same_recorded_event_ticks'] and r['relative_error'] <= .1 for r in rows),
        qualification='recorded200Hz modes only;not full piecewise Jacobian certification')
    report['frozen_after_sha256'] = numeric_tree_digest((session, robot, auxiliary))
    if report['frozen_after_sha256'] != frozen:
        raise ValueError('A frozen parameter, optimizer or pool was changed')
    report.update(status='CANDIDATE_GRADIENT_RECHECK_COMPLETE',
        pilot_gate_passed=bool(report['microbatch']['passed']
            and report['finite_difference']['passed'] and report['full_batch']['norm'] > 0.),
        selected_microbatch=args.microbatch, all_gradients_finite=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--microbatch', type=int, choices=(64, 128, 256), default=256)
    args = parser.parse_args()
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, 3600.) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        sources = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_v1_candidate_gradient_recheck', status='INCOMPLETE',
            training_version='WMJ-V1', config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            source_sha256={p.name: file_hash(p) for p in sources},
            actor_updates=0, critic_updates=0, world_updates=0, world_qualified=False)
        write_json_unique(args.output_dir / 'implementation.json',
            dict(report=report, source={p.name: p.read_text() for p in sources}), guard.budget)
        write_json_unique(args.output_dir / 'config_snapshot.json', cfg, guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            audit(cfg, args, guard, report)
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_json_unique(args.output_dir / 'result.json', report, guard.budget)


if __name__ == '__main__':
    main()
