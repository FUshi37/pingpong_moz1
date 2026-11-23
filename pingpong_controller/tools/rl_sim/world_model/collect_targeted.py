"""One independent1024x128 full-support collection with frozen teacher weights."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_json_unique
from .audit_action_effects import clone_batch_lanes
from .collect_teacher import compact_state, save_seed, _stack_cpu
from .collection_labels import label_task_bounds
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .frozen_initialization import StandardDropAuditor
from .generators import (correlated_action_sequences, generator_assignments,
                         propose_directed_contacts, sample_contact_lead_controls)
from .geometry import compiled_geometry
from .migration import actor_mean, load_v321_actor
from .native_observation import NativeObservationAdapter
from .native_stack import build_native_environment
from .physical_events import init_physical_events, scan_physical_events
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest, pack_numeric_state, restore_numeric_state
from .targeted_collection import acquisition_groups, missing_regions, targeted_draw, warm_fallback
from .targeted_recovery import compare_replay_arrays, restore_prepared
from .teacher_initialization import begin_ball_trial, prepare_warmup
from .teacher_trace import instrument_environment
from .training_seed_bank import incoming_admitted
from .trajectory_data import TrajectoryDataset
from .world_control import ControlContract


def collect(cfg, args, guard, report):
    from mjx_topology_execution import make_collect

    settings = cfg['targeted_supplement']; root = repository_root()
    n = settings['n_envs']; recipe = cfg['collection']['generator_execution']
    if (n, settings['controls'], settings['chunk_controls']) != (1024, 128, 64):
        raise ValueError('Supplement is limited to one1024x128 acquisition')
    # Reserve the estimated full acquisition before GPU compilation. Every
    # individual write still checks raw-array reservation and actual disk space.
    guard.budget.check(additional_bytes=512 * 2**20)
    catalog_ref = settings['catalog_reference']
    catalog = json.loads(authenticate(root, catalog_ref).read_text())
    for key in ('physics', 'control', 'domain'):
        if catalog[key + '_sha256'] != config_hash(cfg[key]):
            raise ValueError('Independent pose catalogue contract changed')
    if catalog['frozen_validation_states_used'] or catalog['status'] != 'STATIC_CATALOG_COMPLETE':
        raise ValueError('Independent complete static catalogue required')
    region_rows = [r for r in catalog['rows'] if r['candidates']]
    regions = np.array([r['region_id'] for r in region_rows], int)
    points = np.array([r['grid_node_m'] for r in region_rows])
    lo = np.asarray(cfg['domain']['racket_xyz_lower_m']); hi = np.asarray(cfg['domain']['racket_xyz_upper_m'])
    near = points[:, 1] >= settings['near_side_y_lower_m']
    boundary = np.any((points[:, :2] - lo[:2] <= settings['XY_boundary_width_m'])
                      | (hi[:2] - points[:, :2] <= settings['XY_boundary_width_m']), axis=1)
    focus = regions[near | boundary | np.isin(regions, settings['missing_source_regions'])]
    rng = np.random.default_rng(settings['seed'])
    draw = targeted_draw(rng, n, region_ids=regions, focus_region_ids=focus)
    requested_regions = regions[draw.indices]
    pose_ids = np.zeros(n, int)
    q = np.zeros((n, 7), np.float32)
    for index, region in enumerate(regions):
        lanes = np.flatnonzero(draw.indices == index)
        candidates = region_rows[index]['candidates']
        choices = np.resize(rng.permutation(len(candidates)), len(lanes))
        pose_ids[lanes] = choices
        q[lanes] = [candidates[i]['q_rad'] for i in choices]
    bundle = build_native_environment(cfg, n_envs=n)
    env = bundle.environment; control = ControlContract.from_native(env)
    if not np.allclose(control.projection.reference_rotation, catalog['reference_rotation_world'], atol=1e-7):
        raise ValueError('Native orientation convention changed')
    fresh, _ = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(settings['seed']), n))
    jax.block_until_ready(fresh)
    geometry = compiled_geometry(env, fresh)
    adapter = NativeObservationAdapter(env, fresh, full_chain=True)
    observer = instrument_environment(env, bundle.stack, support_trace=True)
    actor = load_v321_actor(root, cfg['lineage']['actor_source'])
    actor_digest = numeric_tree_digest(actor)
    context = numeric_tree_digest(fresh.model)
    context_ref, _, _ = save_seed(args.output_dir, 'context', fresh.model, context, guard.budget)
    report.update(geometry=geometry, observer=observer, context=dict(sha256=context, reference=context_ref),
        context_sha256=context, actual_env_overrides=bundle.overrides,
        native_xml=dict(path=str(bundle.stack.xml), sha256=file_hash(Path(bundle.stack.xml))),
        racket_reference_rotation_world=np.asarray(control.projection.reference_rotation).tolist(),
        joint_lower_rad=np.asarray(env.arm_lo).tolist(), joint_upper_rad=np.asarray(env.arm_hi).tolist(),
        catalog_reference=catalog_ref, sampler=draw.report,
        frozen_validation_states_used=False, geometry_execution='offline_setup_only')

    def full_initial(native, valid):
        history, _ = adapter.initialize(native)
        return dict(native=native, history=history, events=init_physical_events(n), ball_model_valid=valid)

    def transition(full, action, goal):
        after, _, _, native_done, metrics = env.step(full['native'], action)
        history, diagnostic = adapter.advance(after, full['history'])
        trace = metrics['wmj_trace']
        finite = jnp.ones(trace['contact_overlap'].shape, bool)
        for key in ('q_after', 'dq_after', 'support_q_after', 'support_dq_after',
                    'ball_qpos_after', 'ball_qvel_after'):
            finite &= jnp.all(jnp.isfinite(trace[key]), axis=-1)
        ball_valid = full['ball_model_valid'][None] & (jnp.cumsum(
            (trace['other_ball_collision'] | ~finite).astype(jnp.int32), axis=0) == 0)
        events, labels = scan_physical_events(full['events'], trace, valid=ball_valid,
                                              initialization_active=jnp.zeros_like(ball_valid))
        streams = dict(trace, **{'event_' + k: v for k, v in labels.items()},
                       ball_model_valid=ball_valid, robot_model_valid=finite)
        action_values = dict(raw_normalized=action, clipped_normalized=jnp.clip(action, -1., 1.),
            projected_normalized=after.prev_action, accepted_normalized=after.prev_action,
            native_done_diagnostic=native_done)
        diagnostic['finite'] = jnp.all(finite, axis=0)
        return dict(native=after, history=history, events=events, ball_model_valid=ball_valid[-1]), \
            compact_state(env, after, goal), action_values, streams, diagnostic

    scoped, execution = make_collect(transition, mode='project'); step = jax.jit(scoped)
    report['execution'] = execution
    goal = jnp.broadcast_to(jnp.array([0., -.5, 1.375]), (n, 3))
    if 'prepared_setup' in settings:
        template = dict(full_initial(fresh, jnp.ones(n, bool)),
                        native=fresh._replace(model=None), goal=goal)
        full, goal, groups, actions, assigned, effective_regions, initial_valid = restore_prepared(
            cfg, template, context, regions, focus, region_rows, report)
        full['native'] = full['native']._replace(model=fresh.model)
        print('[RESTORE] same admitted control0 and action schedule; no new initialization search', flush=True)
    else:
        native = jax.jit(lambda f, pose: prepare_warmup(env, f, pose))(fresh, jnp.asarray(q))
        full = full_initial(native, jnp.ones(n, bool))
        goal = jnp.broadcast_to(jnp.array([0., -.5, 1.375]), (n, 3))
        start = time.monotonic()
        for tick in range(128):
            guard.check()
            full, *_ = step(full, jnp.zeros((n, 7)), goal)
            if tick % 32 == 31:
                jax.block_until_ready(full)
                print(f'[WARM] {tick + 1}/128 controls', flush=True)
        warmed = full['native']; report['warmup_wall_s'] = time.monotonic() - start
        auditor = StandardDropAuditor(cfg)
        actual_q, actual_v = np.asarray(warmed.data.qpos), np.asarray(warmed.data.qvel)
        actual_geom = np.asarray(warmed.data.geom_xpos[:, env.racket_geom_id])
        cache, warm_rows = {}, []
        for lane in range(n):
            guard.check()
            key = hashlib.sha256(actual_q[lane].tobytes() + actual_v[lane].tobytes()).hexdigest()
            if key not in cache:
                drop = np.r_[np.clip(actual_geom[lane, :2], lo[:2], hi[:2]), 1.375]
                cache[key] = auditor.audit_warmed(actual_q[lane], actual_v[lane], drop,
                                                  control.projection.reference_rotation)
            warm_rows.append(dict(lane=lane, requested_region=int(requested_regions[lane]),
                                  pose_id=int(pose_ids[lane]), actual_state_sha256=key, **cache[key]))
        admitted = np.array([r['status'] == 'ADMITTED' for r in warm_rows])
        mapping, effective_regions, substitutions = warm_fallback(rng, admitted, requested_regions)
        report['warmup_admission'] = dict(admitted=int(admitted.sum()), unknown=int((~admitted).sum()),
            unique_actual_states=len(cache), admitted_regions=np.unique(requested_regions[admitted]).tolist(),
            missing_requested_regions=missing_regions(requested_regions, requested_regions[admitted]))
        write_json_unique(args.output_dir / 'warmup_admission.json',
            dict(summary=report['warmup_admission'], rows=warm_rows, substitutions=substitutions), guard.budget)
        full = clone_batch_lanes(full, mapping)
        warmed = full['native']
        # Preserve all controller/model values during substitution; only a new ball
        # and legitimate new-trial sensor bookkeeping follow below.
        print(f'[ADMIT] {int(admitted.sum())}/1024 initial warm proposals; '
              f'{len(np.unique(effective_regions))} actual regions', flush=True)
        assigned = generator_assignments(rng, n, cfg)
        actions, modes = correlated_action_sequences(rng, 128, n,
            amplitude_range=recipe['identification_normalized_amplitude_range'])
        mask = assigned == 'boundaries_failure'
        if mask.any():
            actions[:, mask], modes[mask] = correlated_action_sequences(rng, 128, int(mask.sum()),
                amplitude_range=recipe['boundary_normalized_amplitude_range'])
        lead = sample_contact_lead_controls(rng, cfg, n)
        preview = full
        future_p = np.zeros((n, 3), np.float32); future_r = np.zeros((n, 3, 3), np.float32)
        for tick in range(1, int(lead.max()) + 1):
            guard.check()
            preview, *_ = step(preview, jnp.asarray(actions[tick-1]), goal)
            at = lead == tick
            if at.any():
                future_p[at] = np.asarray(preview['native'].data.geom_xpos[:, env.racket_geom_id])[at]
                future_r[at] = np.asarray(preview['native'].data.geom_xmat[:, env.racket_geom_id]).reshape(n, 3, 3)[at]
        plan = propose_directed_contacts(rng, cfg, auditor.surface, future_p, future_r,
            ball_radius_m=geometry['ball']['radius_m'], contact_lead_controls=lead)
        initial_p, initial_v = plan.initial_position_m.copy(), plan.initial_velocity_m_s.copy()
        robot_only = assigned == 'robot_identification'
        initial_p[robot_only] = [0., -3., 3.]; initial_v[robot_only] = 0.
        initial_geometry = []
        for lane in range(n):
            guard.check()
            check = auditor.audit_warmed(actual_q[mapping[lane]], actual_v[mapping[lane]],
                                         initial_p[lane], control.projection.reference_rotation)
            legal = incoming_admitted(check)
            initial_geometry.append(dict(lane=lane, admitted=legal, **check))
        initial_valid = np.array([row['admitted'] for row in initial_geometry])
        # Unsafe/ambiguous initial ball placements are kept as explicit excluded
        # setup records, never converted into successful or usable ball dynamics.
        original_initial_p = initial_p.copy(); original_initial_v = initial_v.copy()
        initial_p[~initial_valid] = [0., -3., 3.]; initial_v[~initial_valid] = 0.
        write_json_unique(args.output_dir / 'initial_geometry.json', dict(rows=initial_geometry,
            invalid_count=int((~initial_valid).sum()), invalid_ball_data_excluded=True,
            requested_position_m=original_initial_p.tolist(), requested_velocity_m_s=original_initial_v.tolist()), guard.budget)
        native = jax.jit(lambda f, w, p, v: begin_ball_trial(env, f, w, p, v, ar1_enabled=False))(
            fresh, warmed, jnp.asarray(initial_p), jnp.asarray(initial_v))
        if not np.all(np.asarray(native.data.qvel[:, env.ball_vadr+3:env.ball_vadr+6]) == 0.):
            raise ValueError('Initial ball spin must remain zero')
        if numeric_tree_digest(native.rmp_state) != numeric_tree_digest(warmed.rmp_state):
            raise ValueError('New-ball placement changed complete warmed RMP memory')
        full = full_initial(native, jnp.asarray(initial_valid)); goal = jnp.asarray(plan.target_apex_m)
        source_groups = acquisition_groups(settings['seed'], mapping)
        groups = [dict(source_groups[lane],
            warm_source_lane=int(mapping[lane]),
            region_id=int(effective_regions[lane]), requested_region_id=int(requested_regions[lane]),
            generator=str(assigned[lane]), initial_geometry_admitted=bool(initial_valid[lane]),
            contact_offset_class=str(plan.offset_class[lane])) for lane in range(n)]
        write_json_unique(args.output_dir / 'trial_0000.json', dict(
            seed=settings['seed'], sampler=draw.report, requested_regions=requested_regions.tolist(),
            effective_regions=effective_regions.tolist(), original_region_probability=draw.probability.tolist(),
            substitutions=substitutions, generator=assigned.tolist(), excitation_mode=modes.tolist(),
            proposal={k: v.tolist() for k, v in asdict(plan).items()},
            actual_initial_ball_position_m=initial_p.tolist(), actual_initial_ball_velocity_m_s=initial_v.tolist(),
            warmup_controls=128, warmup_actions='zero', preview_controls=lead.tolist(),
            preview_actions_used_unchanged=(assigned != 'closed_loop').tolist(),
            frozen_validation_states_used=False, setup_or_task_success_claim=False), guard.budget)
    policy = jax.jit(lambda current, target: actor_mean(actor,
        adapter.observation(current['native'], current['history'], target)[:, :72]))
    identity = dict(config_sha256=config_hash(cfg), geometry_sha256=geometry['geometry_sha256'],
        control_sha256=config_hash(cfg['control']), contact_sha256=config_hash(cfg['physics']),
        source_commit=report['source_commit'], training_version='WMJ-V1', model_revision=0)
    dataset = TrajectoryDataset.create(args.output_dir / 'dataset', guard.budget, identity=identity,
                                      split_version=cfg['collection']['split']['id'])
    counts = Counter(); rounds = []
    for half in range(2):
        guard.check()
        seed_tree = dict(full, native=full['native']._replace(model=None), goal=goal)
        if half == 0 and 'recovered_seed_reference' in report:
            seed_ref = report['recovered_seed_reference']
        else:
            seed_ref, _, _ = save_seed(args.output_dir, f'seed_{half:04d}', seed_tree, context, guard.budget)
        state_rows = [jax.device_get(compact_state(env, full['native'], goal))]
        action_rows, substep_rows, valid_rows = [], [], []
        began = time.monotonic()
        for tick in range(64):
            guard.check()
            action = jnp.asarray(actions[half*64+tick])
            action = jnp.where(jnp.asarray(assigned == 'closed_loop')[:, None], policy(full, goal), action)
            before = full
            result = step(full, action, goal)
            full = result[0]
            values, action_values, substeps, diagnostics = jax.device_get(result[1:])
            if not np.all(diagnostics['finite']) or np.any(diagnostics['capacity_exceeded']):
                raise FloatingPointError('Nonfinite native trajectory or insufficient history capacity')
            imminent = np.any(substeps['event_contact_entry'], axis=0)
            if 'precontact_restore' not in report and imminent.any():
                tree = dict(before, native=before['native']._replace(model=None))
                arrays, seed_manifest = pack_numeric_state(tree, context_sha256=context)
                restored = restore_numeric_state(tree, arrays, seed_manifest, context_sha256=context)
                restored['native'] = restored['native']._replace(model=before['native'].model)
                repeat = step(restored, action, goal)
                jax.block_until_ready(repeat)
                exact_seed = numeric_tree_digest(restored) == numeric_tree_digest(before)
                def numeric_leaves(value):
                    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in
                            jax.tree_util.tree_flatten_with_path(jax.device_get(value))[0]}
                left, right = numeric_leaves(full), numeric_leaves(repeat[0])
                comparison = compare_replay_arrays(left, right)
                discrete_match = all(np.array_equal(left[key], right[key])
                    for key in left if left[key].dtype.kind in 'biu')
                finite_state = all(np.isfinite(x).all() for x in [*left.values(), *right.values()])
                first, replay = full['native'], repeat[0]['native']
                critical = {}
                for name, a, b, tolerance in (
                    ('qpos', first.data.qpos, replay.data.qpos, 3e-6),
                    ('qvel', first.data.qvel, replay.data.qvel, .001),
                    ('ball_velocity', first.data.qvel[:, env.ball_vadr:env.ball_vadr+3],
                        replay.data.qvel[:, env.ball_vadr:env.ball_vadr+3], 1e-5),
                    ('reference', first.arm_cmd_q, replay.arm_cmd_q, 1e-5),
                    ('accepted_action', first.prev_action, replay.prev_action, 1e-5)):
                    error = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
                    critical[name] = dict(maximum_absolute_difference=error, tolerance=tolerance,
                                          passed=error <= tolerance)
                event_match = all(np.array_equal(substeps[k], np.asarray(repeat[3][k]))
                                  for k in ('event_contact_entry', 'event_contact_release'))
                passed = exact_seed and discrete_match and finite_state and event_match and all(
                    item['passed'] for item in critical.values())
                report['precontact_restore'] = dict(serialized_seed_exact=exact_seed,
                    exact_state=comparison['exact'], exact_events=event_match,
                    discrete_match=discrete_match, finite_state=finite_state, critical=critical,
                    all_leaf_diagnostic=comparison, passed=passed,
                    control=half*64+tick, physical_entry_lanes=np.flatnonzero(imminent).tolist())
                if not passed:
                    raise ValueError('Complete-seed replay changed physical state or discrete events')
                del arrays, seed_manifest, restored, repeat, tree
            state_rows.append(values); action_rows.append(action_values); substep_rows.append(substeps)
            valid_rows.append(diagnostics['finite'])
            for name in ('contact_entry', 'contact_release', 'controlled_release', 'actual_apex'):
                counts[name] += int(substeps['event_' + name].sum())
            if tick % 32 == 31:
                print(f'[COLLECT] {half*64+tick+1}/128 controls', flush=True)
        states, recorded_actions, substeps = map(_stack_cpu, (state_rows, action_rows, substep_rows))
        bounds = label_task_bounds(cfg, states, reference_rotation=control.projection.reference_rotation,
                                  joint_lower_rad=np.asarray(env.arm_lo), joint_upper_rad=np.asarray(env.arm_hi))
        for key, value in bounds.items():
            if value.shape[:2] == (65, n):
                states['task_' + key] = value
        write_json_unique(args.output_dir / f'quality_{half:04d}.json', dict(
            initial_task_eligible=bounds['initial_task_eligible'].tolist(),
            first_task_bound_failure_control=bounds['first_task_bound_failure_control'].tolist(),
            continuous_trial_control_offset=half*64), guard.budget)
        record = dataset.append_chunk(f'r{half:04d}', states=states, actions=recorded_actions,
            substeps=substeps, valid=np.stack(valid_rows), groups=groups, seed_ref=seed_ref, context_sha256=context)
        rounds.append(dict(round=half, transitions=record['transitions'],
            valid_transitions=record['valid_transitions'], stored_bytes=record['archive']['stored_bytes'],
            wall_s=time.monotonic()-began))
        write_json_unique(args.output_dir / f'progress_{half:04d}.json', dict(rounds=rounds, events=dict(counts)), guard.budget)
        del state_rows, action_rows, substep_rows, valid_rows, states, recorded_actions, substeps, seed_tree
        print(f'[SAVE] round{half+1}/2 committed; total transitions={(half+1)*65536}', flush=True)
    if numeric_tree_digest(actor) != actor_digest:
        raise ValueError('Frozen teacher parameters changed')
    report.update(status='TARGETED_COLLECTION_COMPLETE', teacher_transitions=131072, rounds=rounds,
        events=dict(counts), actual_regions=np.unique(effective_regions).tolist(),
        missing_source_regions_still_without_warm_pose=missing_regions(settings['missing_source_regions'], effective_regions),
        invalid_initial_ball_placements=int((~initial_valid).sum()),
        data_split_unique_lanes=dict(Counter(g['split'] for g in next(iter(dataset.chunks.values()))['groups'])),
        actor_before_sha256=actor_digest, actor_after_sha256=actor_digest,
        full_support_labels_direct=True, frozen_model_error_audit='PENDING', qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args(); cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, cfg['targeted_supplement']['max_worker_wall_s']) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        sources = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_v1_independent_targeted_collection', status='INCOMPLETE',
            training_version='WMJ-V1', config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            source_sha256={p.name: file_hash(p) for p in sources}, seed=cfg['targeted_supplement']['seed'],
            actor_updates=0, critic_updates=0, world_updates=0)
        write_json_unique(args.output_dir / 'implementation.json',
                          dict(report=report, source={p.name: p.read_text() for p in sources}), guard.budget)
        write_json_unique(args.output_dir / 'config_snapshot.json', cfg, guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            collect(cfg, args, guard, report)
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_json_unique(args.output_dir / 'result.json', report, guard.budget)


if __name__ == '__main__':
    main()
