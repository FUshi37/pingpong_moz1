"""Frozen target/mask audit on the registered32sources and existing critics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from .actor_critic import ImaginedBatch, frozen_value
from .audit_full_prior_world import release_cpu_heap
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .course_env import course_critic_features, course_critic_layout
from .course_goals import build_goal_table
from .course_rollout import course_policy_window, validate_course_window
from .critic_targets import (lambda_mixture_oracle, checkpoint_critics_and_pool,
                             feature_summary, activation_summary, historical_learning,
                             stats, regression_stats)
from .imagination_learner import critic_training_arrays
from .native_entry_collision import write_compact_json
from .paired_precision import random32_clone
from .recover_reward_direction import write_gzip_json
from .reward_direction import make_prepare, score_terms
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .train_pose_trial import phase_budget
from .velocity_reward import diagnostic_course_step


def audit_window(window, networks, plan):
    """Check every current/next row in memory; publish all-source compact evidence."""
    tr = window.transitions
    valid = np.asarray(tr.valid)
    failure = np.asarray(tr.terminated)
    model_cut = np.asarray(tr.model_truncated)
    timeout = np.asarray(tr.truncated)
    discount = np.asarray(tr.discount)
    continuation = np.asarray(tr.trace_continue)
    expected_discount = np.where(valid & ~failure & ~model_cut, .9995, 0.)
    expected_trace = valid & ~failure & ~model_cut & ~timeout
    masks = dict(valid_rows=int(valid.sum()), physical_failures=int(failure.sum()),
        model_failures=int(model_cut.sum()), time_limits=int(timeout.sum()),
        discount_mismatches=int(np.sum(~np.isclose(discount, expected_discount, atol=1e-7))),
        trace_mismatches=int(np.sum(continuation != expected_trace)),
        validity_reactivation=int(np.sum(valid[1:] & ~valid[:-1])),
        failure_penalty_mismatches=int(np.sum(~np.isclose(np.asarray(tr.reward_terms['failure']),
                                                        -10. * failure, atol=1e-6))))
    if any(masks[key] for key in ('discount_mismatches', 'trace_mismatches',
                                  'validity_reactivation', 'failure_penalty_mismatches')):
        raise ValueError('Actual rollout mask/reward contract differs: ' + str(masks))
    inputs = np.asarray(window.critic_inputs)
    next_inputs = np.asarray(tr.next_critic_features)
    sequential = valid[1:] & valid[:-1]
    masks['consecutive_current_next_feature_max_abs'] = (
        float(np.max(np.abs(next_inputs[:-1][sequential]-inputs[1:][sequential])))
        if sequential.any() else 0.)
    evaluate = jax.jit(frozen_value)
    predictions = {label: {name: np.asarray(evaluate(params, window.critic_inputs))
                           for name, params in bundle.items()}
                   for label, bundle in networks.items()}
    next_values = {label: np.asarray(evaluate(bundle['target_critic'], tr.next_critic_features))
                   for label, bundle in networks.items()}
    old, candidate = score_terms(tr.reward_terms, plan['conditions']['rewards']['candidate_velocity'])
    result = dict(masks=masks, target_checks={}, row_digest=numeric_tree_digest(
        (inputs, next_inputs, tr.reward_terms, tr.discount, tr.trace_continue, tr.valid)),
        prediction_scales={label: {name: stats(value[valid]) for name, value in values.items()}
                           for label, values in predictions.items()})
    for horizon in plan['conditions']['windows']:
        result['target_checks'][str(horizon)] = {}
        take = valid[:horizon]
        for reward_name, rewards in (('old_apex', old), ('candidate_velocity', candidate)):
            reward_results = {}
            for label, bundle in networks.items():
                batch = ImaginedBatch(rewards[:horizon], tr.discount[:horizon],
                                      tr.trace_continue[:horizon], tr.valid[:horizon])
                values = next_values[label][:horizon]
                oracle = lambda_mixture_oracle(batch, values, .99)
                mc = lambda_mixture_oracle(batch, values, 1.)
                short_tr = jax.tree.map(lambda x: x[:horizon], tr)._replace(reward=rewards[:horizon])
                short = SimpleNamespace(transitions=short_tr, critic_inputs=window.critic_inputs[:horizon])
                actual_x, actual_y = critic_training_arrays([short], bundle['target_critic'], lambda_value=.99)
                if not np.array_equal(actual_x, inputs[:horizon][take]):
                    raise ValueError('Critic training selected wrong input rows')
                error = np.abs(actual_y - oracle[take])
                if not np.allclose(actual_y, oracle[take], atol=plan['checks']['atol'], rtol=plan['checks']['rtol']):
                    raise ValueError('Lambda target differs from independent n-step mixture')
                strata = dict(all=take, terminal=failure[:horizon],
                              last_window_step=take & (np.arange(horizon)[:, None] == horizon-1))
                report = dict(rows=len(actual_y), oracle_error=stats(error), targets=stats(actual_y),
                    target_digest=numeric_tree_digest(actual_y), regressions={}, sources=[])
                for name, prediction in predictions[label].items():
                    pred = prediction[:horizon]
                    report['regressions'][name] = {key: dict(
                        lambda_target=regression_stats(pred[mask], oracle[mask]),
                        finite_horizon_return=regression_stats(pred[mask], mc[mask]))
                        for key, mask in strata.items()}
                for lane, source in enumerate(plan['source_indices']):
                    failed = np.flatnonzero(failure[:horizon, lane])
                    edge = int(failed[0]) if failed.size else None
                    report['sources'].append(dict(source_index=source, valid_rows=int(take[:, lane].sum()),
                        first_failure_control=edge+1 if edge is not None else None,
                        initial_lambda_target=float(oracle[0, lane]), initial_finite_return=float(mc[0, lane]),
                        initial_online=float(predictions[label]['critic'][0, lane]),
                        initial_target=float(predictions[label]['target_critic'][0, lane]),
                        terminal_target=float(oracle[edge, lane]) if edge is not None else None,
                        terminal_online=float(predictions[label]['critic'][edge, lane]) if edge is not None else None,
                        terminal_ema=float(predictions[label]['target_critic'][edge, lane]) if edge is not None else None))
                reward_results[label] = report
            result['target_checks'][str(horizon)][reward_name] = reward_results
    evaluate.clear_cache()
    return result, inputs[valid]


def audit(cfg, args, guard, report, plan, source):
    data = make_prepare()(cfg, args, guard, report, source)
    layout = course_critic_layout(data['initial'])
    loaded = {name: checkpoint_critics_and_pool(plan['learner_'+name], data['initial'])
              for name in ('u20', 'u30')}
    networks = {name: {key: item[key] for key in ('critic', 'target_critic')}
                for name, item in loaded.items()}
    frozen = numeric_tree_digest((data['actor'], data['robot'], data['auxiliary'], networks))
    report['parameters_before_sha256'] = frozen
    details = dict(layout=layout, checkpoints={}, historical=historical_learning(plan), modes={})
    for name, item in loaded.items():
        pool = item['pool']
        features = np.asarray(jax.jit(course_critic_features)(pool))
        active = ~np.asarray(pool.terminated | pool.truncated | pool.model_truncated)
        details['checkpoints'][name] = dict(metadata=item['metadata'], counters=item['counters'],
            endpoint_rows=len(features), endpoint_active=int(active.sum()),
            endpoint_digest=numeric_tree_digest(pool), features=feature_summary(features, layout),
            activations={key: activation_summary(value, features) for key, value in networks[name].items()},
            scope=item['scope']+'; endpoint distribution includes terminal rows and is not the full training-input distribution')
    table = build_goal_table(np.unique(np.asarray(data['goals']), axis=0),
                             static_admitted=np.ones(len(np.unique(np.asarray(data['goals']), axis=0)), bool))
    keys = jax.random.split(jax.random.PRNGKey(source['seed']+2), 32)
    window_fn = random32_clone(course_policy_window,
        dict(course_step=diagnostic_course_step(data['runtime']), course_critic_features=course_critic_features))
    options = dict(stage_id='S01', control_contract=data['control'], episode_contract=data['episode'],
                   reward_contract=data['reward'], **data['options'])
    for mode, std in plan['conditions']['policy_modes'].items():
        guard.check()
        forward = jax.jit(lambda actor: window_fn(actor, data['robot'], data['auxiliary'],
            data['initial'], keys, table, horizon=128, exploration_std=std, **options))
        window = forward(data['actor']);jax.block_until_ready(window)
        validate_course_window(window)
        row, inputs = audit_window(window, networks, plan)
        row['features'] = feature_summary(inputs, layout)
        row['activations'] = {name: {key: activation_summary(value, inputs) for key, value in bundle.items()}
                              for name, bundle in networks.items()}
        details['modes'][mode] = row
        print('[CRITIC_AUDIT] '+mode+' '+json.dumps(row['masks']), flush=True)
        forward.clear_cache();del window;release_cpu_heap()
    report['parameters_after_sha256'] = numeric_tree_digest((data['actor'], data['robot'], data['auxiliary'], networks))
    if report['parameters_after_sha256'] != frozen:
        raise ValueError('Frozen parameters changed')
    path = args.output_dir/'details.json.gz'
    digest = write_gzip_json(path, details, guard.budget)
    report['details'] = dict(path=str(path), sha256=digest, encoding='gzip')
    for name, digest in plan['preserved_sources'].items():
        if file_hash(Path(__file__).parent/name) != digest:
            raise ValueError('Prior implementation changed: '+name)
    report.update(status='FROZEN_CRITIC_TARGET_AUDIT_COMPLETE', parameters_unchanged=True,
                  actor_updates=0, critic_updates=0, world_updates=0, optimizer_calls=0,
                  world_controls=8192, native_controls=0, default_replaced=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args();cfg = load_config();setting = cfg['critic_target_audit']
    plan = json.loads(authenticate(repository_root(), setting['plan']).read_text())
    source = json.loads(authenticate(repository_root(), plan['original_source_plan']).read_text())
    authenticate(repository_root(), plan['prerequisite'])
    if (not setting['authorized'] or setting['default_enabled'] or plan['max_phase_bytes'] != 512*1024
            or any(plan[key] for key in ('actor_updates', 'critic_updates', 'world_updates', 'optimizer_calls'))):
        raise ValueError('Only frozen Critic diagnostics are authorized')
    for key in ('physics', 'control'):
        if config_hash(cfg[key]) != plan[key+'_sha256']:
            raise ValueError('Preserved contract changed')
    if config_hash(cfg['imagination_implementation']['reward']['coefficients']) != plan['default_reward_sha256']:
        raise ValueError('Default reward changed')
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, plan['max_worker_wall_s'], startup_reserve_bytes=2**20) as guard:
        phase_budget(guard, plan);guard.budget.check(additional_bytes=plan['peak_write_reservation_bytes'])
        args.output_dir.mkdir(parents=True, exist_ok=False)
        report = dict(schema='wmj_v1_critic_target_audit', status='INCOMPLETE', training_version='WMJ-V1',
            plan=setting['plan'], config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            source_sha256={p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
        write_gzip_json(args.output_dir/'config_snapshot.json.gz', cfg, guard.budget)
        write_gzip_json(args.output_dir/'implementation.json.gz',
            {name:(Path(__file__).parent/name).read_text() for name in ('critic_targets.py', 'audit_critic_targets.py')}, guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('One CUDA GPU required')
            audit(cfg, args, guard, report, plan, source)
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_compact_json(args.output_dir/'result.json', report, guard.budget)


if __name__ == '__main__':
    main()
