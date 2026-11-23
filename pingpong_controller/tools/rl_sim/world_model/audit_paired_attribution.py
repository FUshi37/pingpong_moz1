"""Frozen full-Actor reward attribution and original-graph collision inspection."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import mujoco

from .artifacts import write_array_archive_unique, write_json_unique
from .attribution_common import population, source_report, apex_replay_deltas
from .audit_attribution_collision import original_policy_step
from .audit_full_prior_world import release_cpu_heap
from .audit_module_gradients import load_runtime
from .contracts import authenticate, file_hash, load_config, repository_root
from .course_goals import build_goal_table
from .evaluate_frozen import FAILURES, json_numeric
from .imagined_env import EpisodeContract
from .imagined_rewards import RewardContract
from .mixed_precision import make_runtime
from .paired_attribution import (OBJECTIVE_NAMES, ball_other_contacts, component_function,
    flat_tree, gradient_statistics, orientation_margin_deg, target_from_checkpoint)
from .paired_precision import load_actor
from .pilot_checkpoint import load_pilot_actor
from .resources import StorageBudget
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest


def plan_and_budget(cfg: dict, guard: Any) -> dict:
    ref = cfg['paired_frozen_attribution']['plan']
    plan = json.loads(authenticate(repository_root(), ref).read_text())
    if (not plan['authorized'] or plan['parameter_updates'] or plan['reward_changes']
            or plan['constraint_changes'] or plan['physics_changes'] or plan['new_scenarios']
            or plan['max_new_output_bytes'] != 2 * 2**20):
        raise ValueError('The bounded frozen attribution authorization changed')
    guard.budget = StorageBudget(guard.budget.root, min(guard.budget.max_bytes,
        plan['phase_start_used_bytes'] + plan['max_new_output_bytes']), plan['min_free_bytes'])
    check(guard)
    return plan


def check(guard: Any) -> None:
    guard.check()
    guard.budget.check(additional_bytes=128 * 1024)


def policies(cfg: dict, plan: dict) -> dict:
    return {name: (load_pilot_actor if name == 'u20' else load_actor)(cfg, reference)[0]
            for name, reference in plan['actors'].items()}


def shifted(actor: Any, delta: Any, alpha: float) -> Any:
    return jax.tree.map(lambda a, d: a + jnp.float32(alpha) * d, actor, delta)


def gradients(cfg: dict, args: Any, guard: Any, report: dict, plan: dict) -> None:
    source = json.loads(authenticate(repository_root(), plan['training_population_plan']).read_text())
    initial, _, robot, auxiliary, options, control, refs = load_runtime(cfg, guard, source)
    if source['source_indices'] != plan['source_indices']:
        raise ValueError('Independent training sources changed')
    actors = policies(cfg, plan)
    critics = {name: target_from_checkpoint(ref) for name, ref in plan['learners'].items()}
    before = numeric_tree_digest((actors, critics, robot, auxiliary))
    # These are the already-admitted goals from the authenticated training bank.
    goals = np.unique(np.asarray(initial.task.goal_xyz), axis=0)
    table = build_goal_table(goals, static_admitted=np.ones(len(goals), bool))
    keys = jax.random.split(jax.random.PRNGKey(plan['seed'] + 2), len(plan['source_indices']))
    episode = EpisodeContract.from_config(cfg)
    reward = RewardContract(**cfg['imagination_implementation']['reward']['coefficients'])
    report.update(model_references=refs, initial_state_sha256=numeric_tree_digest(initial),
        source_population=plan['training_population_plan'], source_indices=plan['source_indices'],
        source_ids=[r['source_id'] for r in source['source_rows']],
        parameters_before_sha256=before, objective_names=list(OBJECTIVE_NAMES), runs={},
        gradient_scope=plan['gradient_scope'], exploration_std=plan['exploration_std'],
        orientation_diagnostic_not_in_training_return=True)
    packed = {}; packed_shapes = {}

    def keep_modes(name: str, modes: Any) -> None:
        array = np.asarray(modes, bool)
        packed[name] = np.packbits(array.reshape(-1))
        packed_shapes[name] = list(array.shape)

    for item in plan['gradient_runs']:
        check(guard)
        name, mode = item['name'], item['precision']
        actor, critic = actors[item['actor']], critics[item['actor']]
        destination = 'u30_' + mode
        delta = jax.tree.map(lambda a, b: a - b, actors[destination], actors['u20'])
        with jax.enable_x64(mode == 'mixed'):
            runtime = make_runtime(mode, robot, auxiliary, options, control)
            fn = component_function(runtime, initial, table, keys, episode, reward,
                exploration_std=plan['exploration_std'])
            forward = jax.jit(fn)

            def objective(parameters, target, weights):
                values, aux = fn(parameters, target)
                return jnp.sum(jnp.mean(values, axis=0) * weights), (values, aux)

            derivative = jax.jit(jax.value_and_grad(objective, argnums=0, has_aux=True))
            started = time.monotonic()
            center = jax.device_get(forward(actor, critic))
            cold = time.monotonic() - started
            values, aux = center
            if not np.isfinite(values).all() or np.max(np.abs(aux['component_sum_error'])) > 1e-4:
                raise ValueError('Reward decomposition no longer reconstructs the original return')
            keep_modes(name + '_center', aux['modes'])
            parts = []; derivative_primal_delta = []; derivative_mode_changes = []
            for index, term in enumerate(OBJECTIVE_NAMES):
                check(guard)
                (_, (primal, traced)), gradient = jax.device_get(derivative(actor, critic,
                    jnp.eye(len(OBJECTIVE_NAMES), dtype=jnp.float32)[index]))
                parts.append(gradient)
                if index == 0:
                    derivative_values = np.asarray(primal)
                    keep_modes(name + '_derivative_primal', traced['modes'])
                derivative_primal_delta.append(float(np.max(np.abs(primal - values))))
                derivative_mode_changes.append(int(np.any(traced['modes'] != aux['modes'], axis=(0, 2)).sum()))
                print(f'[ATTR-GRAD] {name}/{term} norm={np.linalg.norm(flat_tree(gradient)):.6g}', flush=True)
            weights = jnp.asarray([1.] * 11 + [0.], jnp.float32)
            (_, _), total_gradient = jax.device_get(derivative(actor, critic, weights))
            summed = sum(flat_tree(x) for x in parts[:11])
            reconstruction = float(np.linalg.norm(summed - flat_tree(total_gradient))
                / max(np.linalg.norm(flat_tree(total_gradient)), 1e-12))
            # This is a frozen diagnostic: retain a failed numerical check and
            # its magnitude, rather than discarding the remaining attribution.
            whole = flat_tree(total_gradient)
            whole_norm = float(np.linalg.norm(whole))
            print(f'[ATTR-SUM] {name} relative_error={reconstruction:.6g}', flush=True)
            summary = gradient_statistics(parts, delta, reward_count=11)
            modes = np.asarray(aux['modes'])
            orientation_failed = np.any(modes[:, :, 1] & ~modes[:, :, 7], axis=0)
            result = dict(actor_reference=plan['actors'][item['actor']], critic_reference=plan['learners'][item['actor']],
                numeric_mode=mode, mean_components=values.mean(axis=0).tolist(),
                per_source_components=values.tolist(), full_actor_gradient=summary,
                derivative_primal_per_source_components=derivative_values.tolist(),
                gradient_sum_relative_error=reconstruction,
                gradient_sum_check_pass=reconstruction <= 1e-4,
                combined_objective_gradient_norm=whole_norm,
                combined_objective_dot_actual_update=float(whole @ flat_tree(delta)),
                summed_vs_combined_cosine=float(summed @ whole / max(np.linalg.norm(summed) * whole_norm, 1e-24)),
                forward_vs_derivative_primal_max=derivative_primal_delta,
                forward_vs_derivative_mode_changed_counts=derivative_mode_changes,
                valid_controls=np.asarray(aux['valid']).sum(axis=0).tolist(),
                hit_count=np.asarray(aux['hit_count']).tolist(), apex_count=np.asarray(aux['apex_count']).tolist(),
                terminated=np.asarray(aux['terminated']).tolist(), model_truncated=np.asarray(aux['model_truncated']).tolist(),
                orientation_failure_source_indices=np.flatnonzero(orientation_failed).tolist(),
                final_margin_deg=np.asarray(aux['final_margin_deg']).tolist(),
                clipped_action_fraction=float(aux['clipped_action_fraction']),
                actual_update_norm=float(np.linalg.norm(flat_tree(delta))),
                actual_update_max_abs=float(np.max(np.abs(flat_tree(delta)))),
                cold_forward_s=cold, finite_differences=[])
            expected = np.asarray(summary['dot_actual_update'])
            for alpha in plan['probe_multipliers']:
                check(guard)
                plus, pos = jax.device_get(forward(shifted(actor, delta, alpha), critic))
                minus, neg = jax.device_get(forward(shifted(actor, delta, -alpha), critic))
                changed = np.any((pos['modes'] != modes) | (neg['modes'] != modes), axis=(0, 2))
                fd = (plus - minus) / (2 * alpha)
                keep_modes(name + f'_p{int(alpha*100):03d}', pos['modes'])
                keep_modes(name + f'_m{int(alpha*100):03d}', neg['modes'])
                row = dict(multiplier=alpha, automatic_mean_components=expected.tolist(),
                    central_mean_components=fd.mean(axis=0).tolist(), per_source_central_components=fd.tolist(),
                    plus_mean_components=plus.mean(axis=0).tolist(), minus_mean_components=minus.mean(axis=0).tolist(),
                    changed_mode_source_indices=np.flatnonzero(changed).tolist(),
                    unchanged_mode_source_count=int((~changed).sum()),
                    no_all_1ms_mode_equality_claim=True)
                result['finite_differences'].append(row)
                print(f'[ATTR-FD] {name} multiplier={alpha} AD={expected[:11].sum():.6g} '
                      f'FD={fd.mean(axis=0)[:11].sum():.6g} changed={int(changed.sum())}', flush=True)
            report['runs'][name] = result
            forward.clear_cache(); derivative.clear_cache()
            del forward, derivative, parts, runtime
        jax.clear_caches(); release_cpu_heap()
    report['mode_archive'] = write_array_archive_unique(args.output_dir / 'modes.npz', packed, guard.budget)
    report['mode_archive_shapes_before_packbits'] = packed_shapes
    report['mode_columns'] = ['valid', 'terminated', 'model_truncated', 'release', 'apex',
                              'ball_bound', 'racket_bound', 'orientation_bound', 'joint_bound']
    report['parameters_after_sha256'] = numeric_tree_digest((actors, critics, robot, auxiliary))
    if report['parameters_after_sha256'] != before:
        raise ValueError('Frozen parameters changed')
    report.update(status='PAIRED_FROZEN_REWARD_GRADIENT_ATTRIBUTION_COMPLETE', parameters_unchanged=True)


def collisions(cfg: dict, args: Any, guard: Any, report: dict, plan: dict) -> None:
    from mjx_topology_execution import make_collect
    if jax.config.x64_enabled:
        raise ValueError('Native replay must retain the original FP32 runtime')
    p = population(cfg); env = p['env']; model = env.mj_model
    actors = policies(cfg, plan); before = numeric_tree_digest(actors)
    case = plan['collision']['case_id']
    lane = next(i for i, c in enumerate(p['cases']) if c['case_id'] == case)
    report.update(initial_native_sha256=numeric_tree_digest(p['initial']), case_id=case,
        case_lane=lane, original_population=128, known=int(p['known'].sum()), runs={},
        parameters_before_sha256=before, native_xml=cfg['authenticated_inputs'].get('xml'))
    reference_rotation = np.asarray(p['control'].projection.reference_rotation)
    limit = cfg['domain']['roll_pitch_limit_deg']

    def pose_info(native, index):
        rotation = np.asarray(native.data.site_xmat)[index, env.racket_site_id].reshape(3, 3)
        relative = reference_rotation.T @ rotation
        roll = float(np.rad2deg(np.arctan2(-relative[1, 2], relative[2, 2])))
        pitch = float(np.rad2deg(np.arcsin(np.clip(relative[0, 2], -1., 1.))))
        return dict(racket_position_m=np.asarray(native.data.site_xpos)[index, env.racket_site_id].tolist(),
                    roll_deg=roll, pitch_deg=pitch, margin_deg=limit - max(abs(roll), abs(pitch)))

    for name in plan['collision']['actor_names']:
        actor = actors[name]
        fn, identity = original_policy_step(p, actor, cfg)
        scoped, execution = make_collect(fn, mode='project'); step = jax.jit(scoped)
        keys = jax.random.split(jax.random.PRNGKey(plan['collision']['seed'] + 2), 128)
        state = (p['initial'], p['history'], p['task'], p['events'], keys)
        selected_events = []; all_apexes = []; contacts = []; failures = []; tail = deque(maxlen=20)
        started = time.monotonic()
        for tick in range(plan['collision']['max_controls']):
            if np.asarray(state[2].done).all():
                break
            check(guard)
            alive_before = not bool(np.asarray(state[2].done)[lane])
            out = step(*state, jnp.float32(plan['exploration_std'])); state = out[:5]
            trace = jax.device_get(out[-1])
            for sub, index in zip(*np.nonzero(trace['valid_apex'])):
                all_apexes.append(dict(case_id=p['cases'][index]['case_id'],
                    impact_id=int(trace['impact_id'][sub, index]), valid_apex=True,
                    apex_position_m=trace['apex_position_m'][sub, index].tolist()))
            if alive_before:
                q = np.asarray(out[0].data.qpos)
                tail.append(dict(control_tick=tick, elapsed_ms=int(out[2].elapsed_ms[lane]),
                    **pose_info(out[0], lane), ball_position_m=q[lane, env.ball_qadr:env.ball_qadr+3].tolist(),
                    ball_velocity_m_s=np.asarray(out[0].data.qvel)[lane, env.ball_vadr:env.ball_vadr+3].tolist(),
                    measured_q_rad=q[lane, p['adapter'].feedback_qadr].tolist(),
                    q_ref_rad=np.asarray(out[0].arm_cmd_q)[lane].tolist(),
                    accepted_action=np.asarray(out[0].prev_action)[lane].tolist()))
                mask = trace['valid_release'][:, lane] | trace['valid_apex'][:, lane] | (trace['failure_code'][:, lane] != 0)
                for sub in np.flatnonzero(mask):
                    row = {k:json_numeric(v[sub, lane]) for k, v in trace.items()}
                    row.update(control_tick=tick, substep=int(sub)); selected_events.append(row)
            failed = np.flatnonzero(np.any(trace['failure_code'] != 0, axis=0))
            for index in failed:
                code = int(np.max(trace['failure_code'][:, index]))
                failures.append(dict(case_id=p['cases'][index]['case_id'], code=code,
                    elapsed_ms=int(out[2].elapsed_ms[index]), **pose_info(out[0], index)))
            collided = np.flatnonzero(np.any(trace['failure_code'] == 2, axis=0))
            if len(collided):
                pairs = np.asarray(out[0].data.contact.geom); distances = np.asarray(out[0].data.contact.dist)
                for index in collided:
                    active = ball_other_contacts(pairs[index], distances[index],
                        ball_id=env.ball_geom_id, racket_id=env.racket_contact_geom_id)
                    for contact in active:
                        names = [dict(id=int(i), geom=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i)),
                            body=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])))
                            for i in pairs[index, contact]]
                        contacts.append(dict(case_id=p['cases'][index]['case_id'], control_tick=tick,
                            elapsed_ms=int(out[2].elapsed_ms[index]), geoms=names,
                            distance_m=float(distances[index, contact]), **pose_info(out[0], index)))
                    if not active:
                        contacts.append(dict(case_id=p['cases'][index]['case_id'], control_tick=tick,
                            elapsed_ms=int(out[2].elapsed_ms[index]), status='NO_ACTIVE_PAIR_RETAINED_AT_5MS_ENDPOINT'))
        old_ref = plan['collision']['references'][name.removeprefix('u30_')]
        old = json.loads(authenticate(repository_root(), old_ref).read_text())
        task = jax.device_get(state[2])
        outcomes = [dict(case_id=c['case_id'], hits=int(task.hits[i]),
            failure=FAILURES[int(task.failure_code[i])], elapsed_ms=int(task.elapsed_ms[i]))
            for i, c in enumerate(p['cases'])]
        changed = [dict(current=a, previous=b) for a, b in zip(outcomes, old['cases'])
            if a['hits'] != b['hits'] or a['failure'] != b['physical_failure']
            or abs(a['elapsed_ms'] * .001 - b['elapsed_s']) > 1e-6]
        apex_deltas = apex_replay_deltas(all_apexes, old['events'])
        report['runs'][name] = dict(actor_reference=plan['actors'][name], original_evaluation=old_ref,
            original_graph=identity, execution=execution, outcomes=outcomes,
            replay_outcome_differences=changed, matched_apex_count=len(apex_deltas),
            matched_apex_max_abs_delta_m=max(apex_deltas, default=0.),
            collision_contacts=contacts, first_failure_endpoint_poses=failures,
            selected_case_events=selected_events, selected_case_last20_controls=list(tail),
            endpoint_pose_scope='after original5ms control;failure time comes from1ms task trace',
            wall_s=time.monotonic() - started)
        print(f'[ATTR-COLLISION] {name} replay_changed={len(changed)} '
              f'apex_max_delta={max(apex_deltas,default=0.):.6g} contacts={contacts}', flush=True)
        step.clear_cache(); jax.clear_caches(); release_cpu_heap()
    report['parameters_after_sha256'] = numeric_tree_digest(actors)
    if report['parameters_after_sha256'] != before:
        raise ValueError('Frozen actor changed')
    report.update(status='PAIRED_ORIGINAL_GRAPH_COLLISION_ATTRIBUTION_COMPLETE', parameters_unchanged=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--task', choices=('gradients', 'collision', 'collision_entry'), required=True)
    parser.add_argument('--entry-actor', choices=('u30_fp32', 'u30_mixed'))
    args = parser.parse_args()
    if args.task == 'collision_entry' and args.entry_actor is None:
        parser.error('collision_entry requires one --entry-actor per fresh native process')
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, 3600., startup_reserve_bytes=2**20) as guard:
        plan = plan_and_budget(cfg, guard)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        report = source_report(cfg, args)
        report.update(schema='wmj_v1_post_pair_frozen_attribution', task=args.task,
            plan=cfg['paired_frozen_attribution']['plan'])
        write_json_unique(args.output_dir / 'implementation.json', dict(report=report,
            sources={n:(Path(__file__).parent/n).read_text() for n in
                     ('audit_paired_attribution.py', 'paired_attribution.py', 'native_entry_collision.py')}), guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            if args.task == 'collision_entry':
                from .native_entry_collision import audit
                audit(cfg, args, guard, report, plan)
            else:
                (gradients if args.task == 'gradients' else collisions)(cfg, args, guard, report, plan)
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            if args.task == 'collision_entry':
                from .native_entry_collision import write_compact_json
                write_compact_json(args.output_dir / 'result.json', report, guard.budget)
            else:
                write_json_unique(args.output_dir / 'result.json', report, guard.budget)


if __name__ == '__main__':
    main()
