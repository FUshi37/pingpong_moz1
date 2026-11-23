"""Frozen own-state model errors on the independent direct-support supplement."""

import argparse
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_json_unique, write_array_archive_unique
from .audit_full_prior_world import (AuditReader, make_bank_builder, read_selected_trial,
                                     release_cpu_heap, teacher_seeds)
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .fit_robot import summarize_errors
from .full_prior_seeds import select_teacher_seeds
from .full_prior_world import full_prior_world_rollout
from .native_stack import build_native_environment
from .new_pose_audit import diagnostic_starts, prefix_validity, select_windows
from .normal_checkpoint import load_components
from .physics_metrics import ball_metrics, robot_metrics
from .position_contact_data import prepare_position_contact_batch
from .release_alignment import compare_releases
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .supplement_data import SupplementReader, diagnostic_batches
from .training_seed_bank import initial_candidates
from .world_control import ControlContract


def audit(cfg, args, guard, report):
    root = repository_root() / cfg['resources']['output_root']
    reader = SupplementReader(cfg, cfg['targeted_supplement']['collection_reference'], guard.budget)
    original = AuditReader(root / 'support_replay_full_a01', guard.budget, max_bytes=0)
    parameters, options, refs, auxiliary, aux_ref = load_components(cfg, original,
        robot_checkpoint=root / 'support_robot_fit_a02/robot_support_u006144.json',
        contact_checkpoint=root / 'contact_activation_control_a01/contact_coupled_u014336.json',
        auxiliary_checkpoint=root / 'solver_aux_fit_a01/auxiliary_u004096.json')
    model_refs = dict(refs, auxiliary=aux_ref)
    frozen = numeric_tree_digest(dict(robot=parameters['robot'], auxiliary=auxiliary))
    bundle = build_native_environment(cfg, n_envs=1)
    contract = ControlContract.from_native(bundle.environment)
    builder = make_bank_builder(options, contract)
    names = list(reader.records)
    groups = reader.records[names[0]]['record']['groups']
    batches, exclusions = diagnostic_batches(groups)
    report.update(dataset_sha256=reader.identity, model_references=model_refs,
        source_collection=cfg['targeted_supplement']['collection_reference'],
        selected_sources=sum(b['useful'] for b in batches), exclusions=exclusions,
        independent_root_families=len({groups[lane]['root_seed_family']
            for b in batches for lane in b['lanes'][:b['useful']]}),
        final_test_outcomes_used=False, scopes=['task_prefix', 'finite_physics'],
        comparison='common_raw_actions;own_robot_ball_control_and_observation_recursion',
        contact_offset_strata='proposal_classes;actual_entry_offset_is_reported_separately',
        learned_contact_residual=False)

    @jax.jit
    def predict(seed, context, actions, goal, raw):
        _, trace = full_prior_world_rollout(parameters['robot'], auxiliary, seed, actions,
            context=context, control_contract=contract, task_goal_xyz=goal,
            ar1_enabled=False, **options)
        physical = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), trace['physics'])
        batch = prepare_position_contact_batch(None, raw)
        metrics, _ = ball_metrics(physical['contact'], batch)
        metrics.update(robot_metrics(physical, raw, batch))
        finite = jnp.all(jnp.isfinite(trace['actor_observation']), axis=-1)
        for value in (physical['q_after_rad'], physical['dq_after_rad_s'],
                      physical['contact']['after'].position_m, physical['contact']['after'].velocity_m_s):
            finite &= jnp.all(jnp.isfinite(value).reshape(actions.shape[0], 5, -1, value.shape[-1]), axis=(1, 3))
        return dict(metrics=metrics, finite=finite, release=physical['contact']['release'],
            entry=physical['contact']['entry'], position=physical['contact']['before'].position_m,
            velocity=physical['contact']['before'].velocity_m_s)

    pieces, attributes, reconstructions, source_candidates, event_reports = {}, {}, [], [], []
    actual_entries = []
    nonfinite = {str(h): 0 for h in (1, 8, 32, 64, 128)}
    prefix_counts = dict(nonfinite)
    for index, selection in enumerate(batches):
        guard.check()
        lanes, useful = selection['lanes'], selection['useful']
        raw, values, _ = read_selected_trial(reader, names, lanes)
        bank, context, checks = teacher_seeds(reader, reader.records[names[0]], lanes,
                                              raw, values, options, builder)
        local_groups = [groups[i] for i in lanes]
        candidate, counts = initial_candidates(local_groups[:useful],
            values['state__task_within_task_bounds'][0, :useful], raw['event_contact_entry'][:, :useful],
            trial=0, minimum_lead_s=cfg['training_seed_bank']['minimum_precontact_lead_s'])
        for row in candidate:
            row['lane'] = int(lanes[row['lane']])
        source_candidates.extend(candidate)
        reconstructions.append(dict(batch=index, lanes=lanes[:useful].tolist(), useful=useful,
                                    computational_padding=len(lanes)-useful, checks=checks, seed_counts=counts))
        initial_task = values['state__task_within_task_bounds'][0]
        _, prefix = prefix_validity(raw['ball_model_valid'], np.zeros_like(raw['ball_model_valid']),
            ~values['state__task_within_task_bounds'][1:], initial_task)
        entry_time, entry_lane = np.where(raw['event_contact_entry'][:, :useful]
                                          & raw['ball_model_valid'][:, :useful])
        surface = options['surface'].prism
        displacement = (raw['ball_qpos_before'][entry_time, entry_lane, :3]
                        - raw['racket_geom_position_before_m'][entry_time, entry_lane])
        local = np.einsum('nij,ni->nj',
            raw['racket_geom_rotation_before'][entry_time, entry_lane], displacement)
        axes = [(surface.normal_axis+1) % 3, (surface.normal_axis+2) % 3]
        # Radial fraction of the actual finite polygon, using its half spaces.
        ratio = np.max((local[:, axes] @ np.asarray(surface.edge_outward_xy).T)
                       / np.asarray(surface.edge_offset_m), axis=-1)
        actual_entries.extend(dict(lane=int(lanes[lane]), substep=int(t),
            radial_boundary_fraction=float(value), within_task_prefix=bool(prefix[t, lane]),
            proposed_class=groups[lanes[lane]]['contact_offset_class'])
            for t, lane, value in zip(entry_time, entry_lane, ratio))
        for horizon in (1, 8, 32, 64, 128):
            guard.check()
            starts = diagnostic_starts(raw['event_controlled_release'] & raw['ball_model_valid'], horizon)
            selected = {k: select_windows(v, starts * 5, horizon * 5) for k, v in raw.items()}
            primary = select_windows(prefix, starts * 5, horizon * 5)[:, :useful]
            actions = select_windows(values['action__raw_normalized'], starts, horizon)
            seed = select_teacher_seeds(bank, starts)
            result = jax.device_get(predict(seed, context, jnp.asarray(actions),
                jnp.asarray(values['state__goal_xyz_m'][starts, np.arange(len(lanes))]),
                jax.tree.map(jnp.asarray, selected)))
            nonfinite[str(horizon)] += int((~result['finite'][:, :useful].all(axis=0)).sum())
            prefix_counts[str(horizon)] += int(primary.all(axis=0).sum())
            arrays = dict(case_indices=lanes[:useful], start_controls=starts[:useful],
                          task_prefix_valid=primary, finite_physics=selected['ball_model_valid'][:, :useful],
                          model_finite=result['finite'][:, :useful])
            for metric, (value, mask) in result['metrics'].items():
                value, mask = value[..., :useful], mask[..., :useful]
                arrays[metric] = np.where(np.isfinite(value), value, 0.)
                arrays[metric + '__finite'] = np.isfinite(value)
                arrays[metric + '__valid'] = mask
                for scope, valid in (('task_prefix', primary),
                                     ('finite_physics', selected['ball_model_valid'][:, :useful])):
                    use = mask & (valid[-1] if value.ndim == 1 else valid)
                    key = f'{horizon}/{scope}/{metric}'
                    pieces.setdefault(key, []).append(value[use])
                    group_values = {label: np.asarray([row[label] for row in local_groups[:useful]])
                                    for label in ('region_id', 'split', 'generator', 'contact_offset_class')}
                    for label, vector in group_values.items():
                        attributes.setdefault(key, {}).setdefault(label, []).append(np.broadcast_to(vector, use.shape)[use])
            archive = write_array_archive_unique(args.output_dir / f'errors_b{index:02d}_h{horizon}.npz', arrays, guard.budget)
            window = dict(batch=index, horizon=horizon, lanes=lanes[:useful].tolist(), starts=starts[:useful].tolist(),
                          errors_archive=archive)
            if horizon in (8, 128):
                result_events = {}
                predicted = result['release'][:, :useful] & (result['velocity'][:, :useful, 2] > 0.)
                actual = selected['event_controlled_release'][:, :useful]
                for scope, valid in (('task_prefix', primary),
                                     ('finite_physics', selected['ball_model_valid'][:, :useful])):
                    result_events[scope] = compare_releases(actual & valid, predicted & valid,
                        selected['ball_qpos_before'][:, :useful, :3], selected['ball_qvel_before'][:, :useful, :3],
                        result['position'][:, :useful], result['velocity'][:, :useful],
                        [row['episode'] for row in local_groups[:useful]])
                window['ordinal_release_diagnostic'] = result_events
                window['event_definition'] = 'native_controlled_vs_model_upward_geometric;changed_counts_are_ambiguous'
                event_reports.append(dict(batch=index, horizon=horizon,
                    summaries={scope:{k:v for k,v in result_events[scope].items() if k != 'cases'} for scope in result_events}))
            write_json_unique(args.output_dir / f'window_b{index:02d}_h{horizon}.json', window, guard.budget)
        write_json_unique(args.output_dir / f'reconstruction_b{index:02d}.json', reconstructions[-1], guard.budget)
        print(f'[AUDIT] batch{index+1}/{len(batches)}; '
              f'{useful} unique sources; H1/8/32/64/128 complete', flush=True)
        del raw, values, bank, selected, seed, result
        release_cpu_heap()
    summary, strata = {}, {}
    for key, values in pieces.items():
        all_values = np.concatenate(values)
        summary[key] = summarize_errors([all_values])
        strata[key] = {}
        for label, groups_list in attributes[key].items():
            all_groups = np.concatenate(groups_list)
            strata[key][label] = {str(group): summarize_errors([all_values[all_groups == group]])
                                  for group in np.unique(all_groups)}
    if frozen != numeric_tree_digest(dict(robot=parameters['robot'], auxiliary=auxiliary)):
        raise ValueError('Frozen model parameters changed')
    usable_regions = sorted({r['region_id'] for r in source_candidates})
    old_plan = authenticate(repository_root(), cfg['training_seed_bank']['plan_reference'])
    old_regions = set(json.loads(old_plan.read_text())['admitted_regions'])
    report.update(status='TARGETED_FROZEN_WORLD_AUDIT_COMPLETE', metrics=summary, by_stratum=strata,
        model_nonfinite_windows=nonfinite, complete_task_prefix_windows=prefix_counts,
        reconstruction=reconstructions, event_diagnostics=event_reports,
        actual_contact_entries=actual_entries,
        actual_near_edge_entries_090_to105=sum(.9 <= r['radial_boundary_fraction'] <= 1.05
                                              for r in actual_entries),
        candidate_initial_train_sources=source_candidates, candidate_initial_train_regions=usable_regions,
        newly_covered_source_regions=sorted(set(usable_regions) - old_regions),
        previously_missing_regions_with_new_candidates=sorted(set(usable_regions) & set(cfg['targeted_supplement']['missing_source_regions'])),
        candidate_source_status='real_contact_lead_and_initial_pose_valid;not_yet_added_to_training_bank',
        frozen_before_sha256=frozen, frozen_after_sha256=frozen,
        world_qualified=False, promoted=False, training_bank_modified=False,
        limitations=['Supplement errors are diagnostics, not held-out fixed-task success',
                     'Train and validation strata reported separately; final test is not queried',
                     'No model fit, actor update or action-effect counterfactual collection'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args(); cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, 3600.) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        sources = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_v1_targeted_frozen_world_audit', status='INCOMPLETE',
            training_version='WMJ-V1', gpu_uuid=args.gpu_uuid, config_sha256=config_hash(cfg),
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            source_sha256={p.name: file_hash(p) for p in sources},
            actor_updates=0, critic_updates=0, world_updates=0)
        write_json_unique(args.output_dir / 'implementation.json',
                          dict(report=report, source={p.name:p.read_text() for p in sources}), guard.budget)
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
