"""Frozen pose-margin coefficient calibration; no optimizer or policy update."""

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import read_array_archive, write_array_archive_unique, write_json_unique
from .attribution_common import source_report
from .audit_full_prior_world import release_cpu_heap
from .audit_module_gradients import load_runtime
from .audit_paired_attribution import policies
from .contracts import authenticate, config_hash, load_config, repository_root
from .course_goals import build_goal_table
from .imagined_env import EpisodeContract
from .imagined_rewards import RewardContract
from .mixed_precision import make_runtime
from .native_entry_collision import write_compact_json
from .paired_attribution import flat_tree, target_from_checkpoint
from .pose_margin import (OBJECTIVES, PoseMarginContract, coefficient_grid,
                          frozen_objectives, select_weight)
from .resources import StorageBudget
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest


def load_plan(cfg, guard):
    ref = cfg['pose_margin_candidate']['calibration']['plan']
    plan = json.loads(authenticate(repository_root(), ref).read_text())
    forbidden = ('actor_updates', 'critic_updates', 'world_updates', 'default_reward_changes',
                 'constraint_changes', 'physics_changes', 'new_scenarios')
    if (not plan['authorized'] or any(plan[k] for k in forbidden)
            or plan['max_new_output_bytes'] != 2*2**20
            or plan['horizon_controls'] != 64 or len(plan['source_indices']) != 32
            or cfg['pose_margin_candidate']['contract'] != plan['contract']
            or plan['contract']['enabled'] or plan['contract']['weight_per_s'] != 0.
            or config_hash(cfg['imagination_implementation']['reward']['coefficients'])
            != plan['default_reward_sha256']):
        raise ValueError('Frozen default-off pose calibration contract changed')
    guard.budget = StorageBudget(guard.budget.root,
        min(guard.budget.max_bytes, plan['phase_start_used_bytes'] + plan['max_new_output_bytes']),
        plan['min_free_bytes'])
    guard.budget.check(additional_bytes=2**20)
    return plan


def check(guard):
    guard.check()
    guard.budget.check(additional_bytes=128*1024)


def audit(cfg, args, guard, report, plan):
    population = json.loads(authenticate(repository_root(), plan['training_population_plan']).read_text())
    initial, _, robot, auxiliary, options, control, refs = load_runtime(cfg, guard, population)
    if population['source_indices'] != plan['source_indices']:
        raise ValueError('Training sources changed')
    actors = policies(cfg, plan)
    critics = {name:target_from_checkpoint(ref) for name, ref in plan['learners'].items()}
    before = numeric_tree_digest((actors, critics, robot, auxiliary))
    goals = np.unique(np.asarray(initial.task.goal_xyz), axis=0)
    table = build_goal_table(goals, static_admitted=np.ones(len(goals), bool))
    keys = jax.random.split(jax.random.PRNGKey(plan['seed'] + 2), 32)
    episode = EpisodeContract.from_config(cfg)
    reward = RewardContract(**cfg['imagination_implementation']['reward']['coefficients'])
    contract = PoseMarginContract(**plan['contract'])
    if abs(contract.limit_rad - episode.roll_pitch_limit_rad) > 1e-12:
        raise ValueError('Candidate normalization does not match the unchanged hard bound')
    prior_path = authenticate(repository_root(), plan['prior_gradients'])
    prior = json.loads(prior_path.read_text())
    prior_modes = read_array_archive(prior_path.parent/'modes.npz', prior['mode_archive'])
    arrays = {}
    report.update(model_references=refs, initial_state_sha256=numeric_tree_digest(initial),
        source_indices=plan['source_indices'], source_ids=[r['source_id'] for r in population['source_rows']],
        parameters_before_sha256=before, objective_names=list(OBJECTIVES), runs={},
        coefficient_selection_scope=plan['selection']['scope'], optimizer_calls=0)

    for item in plan['gradient_runs']:
        check(guard)
        name, mode = item['name'], item['precision']
        actor, critic = actors[item['actor']], critics[item['actor']]
        with jax.enable_x64(mode == 'mixed'):
            runtime = make_runtime(mode, robot, auxiliary, options, control)
            fn = frozen_objectives(runtime, initial, table, keys, episode, reward, contract,
                exploration_std=plan['exploration_std'])
            forward = jax.jit(fn)

            def objective(parameters, target, weights):
                values, trace = fn(parameters, target)
                return jnp.sum(values.mean(axis=0)*weights), (values, trace)

            derivative = jax.jit(jax.value_and_grad(objective, argnums=0, has_aux=True))
            started = time.monotonic()
            forward_values, forward_trace = jax.device_get(forward(actor, critic))
            gradients = []; primal_deltas = []; mode_changes = []
            for index, label in enumerate(OBJECTIVES):
                check(guard)
                (_, (values, trace)), gradient = jax.device_get(derivative(actor, critic,
                    jnp.eye(len(OBJECTIVES), dtype=jnp.float32)[index]))
                vector = flat_tree(gradient)
                if not np.isfinite(vector).all() or not np.isfinite(values).all():
                    raise FloatingPointError('Nonfinite frozen calibration result')
                if np.any(trace['default_reward_delta'] != 0.):
                    raise ValueError('Default reward changed')
                if index == 0:
                    center, center_trace = values, trace
                gradients.append(vector)
                primal_deltas.append(float(np.max(np.abs(values - forward_values))))
                mode_changes.append(int(np.any(trace['modes'] != forward_trace['modes'], axis=(0, 2)).sum()))
                print(f'[POSE-GRAD] {name}/{label} norm={np.linalg.norm(vector):.7g}', flush=True)
            matrix = np.stack(gradients)
            gram = matrix @ matrix.T
            means = center.mean(axis=0)
            grid = coefficient_grid(gram, float(means[1]), float(means[4]),
                                    plan['weights_per_s'], plan['selection'])
            previous = np.asarray(prior['runs'][name]['per_source_components'])
            previous_values = np.stack((previous[:, :11].sum(axis=1), previous[:, 2],
                                        previous[:, 1], previous[:, 11]), axis=-1)
            old_shape = prior['mode_archive_shapes_before_packbits'][name+'_center']
            old_modes = np.unpackbits(prior_modes[name+'_center']).reshape(old_shape).astype(bool)
            mask = center_trace['valid'] & ~center_trace['model_truncated']
            angles = np.abs(center_trace['angles_rad'])
            active = np.any(angles > contract.onset_rad, axis=-1) & mask
            terminal = center_trace['terminated'] & mask
            result = dict(actor_reference=plan['actors'][item['actor']], critic_reference=plan['learners'][item['actor']],
                numeric_mode=mode, mean_objectives=means.tolist(), per_source_objectives=center.tolist(),
                full_actor_parameter_count=matrix.shape[1], gradient_gram_matrix=gram.tolist(),
                gradient_norms=np.linalg.norm(matrix, axis=1).tolist(), coefficient_grid=grid,
                valid_controls_per_source=mask.sum(axis=0).tolist(), charged_controls_per_source=active.sum(axis=0).tolist(),
                terminal_count=int(terminal.sum()), charged_terminal_count=int((terminal & active).sum()),
                per_source_undiscounted_unit_pose_cost=center_trace['unit_pose_penalty'].sum(axis=0).tolist(),
                abs_angles_deg_quantiles=np.quantile(np.rad2deg(angles[mask]), [0., .5, .9, .95, 1.], axis=0).tolist(),
                unit_pose_cost_per_interval_quantiles=np.quantile(-center_trace['unit_pose_penalty'][mask], [0., .5, .9, .95, 1.]).tolist(),
                default_reward_delta_max=0., forward_vs_derivative_primal_max=primal_deltas,
                forward_vs_derivative_mode_changed_counts=mode_changes,
                prior_forward_objective_max_abs_delta=np.max(np.abs(forward_values[:, :4]-previous_values), axis=0).tolist(),
                prior_forward_mode_changed_source_indices=np.flatnonzero(np.any(
                    forward_trace['modes'] != old_modes, axis=(0, 2))).tolist(),
                wall_s=time.monotonic()-started)
            for key, value in center_trace.items():
                arrays[name+'_'+key] = np.asarray(value)
            report['runs'][name] = result
            print(f'[POSE-GRID] {name} feasible={[r["weight_per_s"] for r in grid if r["eligible"]]} '
                  f'charged={int(active.sum())}/{int(mask.sum())} terminal={int((terminal & active).sum())}', flush=True)
            forward.clear_cache(); derivative.clear_cache()
            del forward, derivative, runtime, matrix, gradients
        jax.clear_caches(); release_cpu_heap()
    report['selected_weight_per_s'] = select_weight({n:r['coefficient_grid'] for n, r in report['runs'].items()})
    report['trace_archive'] = write_array_archive_unique(args.output_dir/'traces.npz', arrays, guard.budget)
    report['trace_scope'] = 'exact AD-primal frozen diagnostic windows;no full histories duplicated;seeds/checkpoints remain authenticated sources'
    report['parameters_after_sha256'] = numeric_tree_digest((actors, critics, robot, auxiliary))
    if report['parameters_after_sha256'] != before:
        raise ValueError('Frozen parameters changed')
    report.update(status='POSE_MARGIN_FROZEN_CALIBRATION_COMPLETE', parameters_unchanged=True,
        selected_weight_is_training_approval=False, default_reward_changed=False,
        closed_loop_improvement_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args(); cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, 3600., startup_reserve_bytes=2**20) as guard:
        plan = load_plan(cfg, guard)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        report = source_report(cfg, args)
        report.update(schema='wmj_v1_pose_margin_frozen_calibration',
            plan=cfg['pose_margin_candidate']['calibration']['plan'])
        write_json_unique(args.output_dir/'implementation.json', dict(report=report,
            sources={n:(Path(__file__).parent/n).read_text() for n in
                ('calibrate_pose_margin.py', 'pose_margin.py')}), guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            audit(cfg, args, guard, report, plan)
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_compact_json(args.output_dir/'result.json', report, guard.budget)


if __name__ == '__main__':
    main()
