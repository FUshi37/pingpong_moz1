"""Collect warmed full-domain V324 experiments without updating any policy.

Production shape is 1024 lanes by 64 controls per round. The explicit small
diagnostic mode qualifies interfaces only. Every two rounds start a new
128-control trial; the second round preserves all physics/control/history.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_array_archive_unique, write_json_unique
from .contracts import config_hash, file_hash, load_config, repository_root
from .resources import StorageBudget, WallBudget
from .state_adapter import numeric_tree_digest, pack_numeric_state, restore_numeric_state
from .trajectory_data import TrajectoryDataset


def compact_state(env, native, goal):
    """Continuous 200 Hz streams; full memory belongs to the separate seed."""
    return dict(q_rad=native.data.qpos[:, env.arm_qadr],
        dq_rad_s=native.data.qvel[:, env.arm_vadr], q_ref_rad=native.arm_cmd_q,
        ball_qpos=native.data.qpos[:, env.ball_qadr:env.ball_qadr+7],
        ball_qvel=native.data.qvel[:, env.ball_vadr:env.ball_vadr+6],
        racket_position_m=native.data.site_xpos[:, env.racket_site_id],
        racket_rotation_world=native.data.site_xmat[:, env.racket_site_id].reshape(-1, 3, 3),
        observed_ball_position_m=native.cached_ball_obs_pos,
        ball_observed=native.fixed_contact_ball_observed,
        time_us=native.step_count*5000, ball_source_time_us=native.last_ball_obs_step*5000,
        goal_xyz_m=goal, accepted_action=native.prev_action)


def _stack_cpu(rows):
    return {name: np.stack([row[name] for row in rows]) for name in rows[0]}


def save_seed(directory, name, tree, context, budget):
    arrays, manifest = pack_numeric_state(tree, context_sha256=context)
    manifest['archive'] = write_array_archive_unique(directory/f'{name}.npz', arrays, budget)
    manifest['archive_file'] = f'{name}.npz'
    path = directory/f'{name}.json'
    digest = write_json_unique(path, manifest, budget)
    return dict(manifest=str(path.absolute()), sha256=digest), arrays, manifest


def run_collection(cfg, args, budget, wall, report):
    from .native_stack import build_native_environment
    from .native_observation import NativeObservationAdapter
    from .geometry import compiled_geometry
    from .contact_surface import surface_from_vertices
    from .generators import (load_pose_catalog, select_pose_per_region, generator_assignments,
        correlated_action_sequences, sample_contact_lead_controls, propose_directed_contacts)
    from .sampling import sample_collection_regions
    from .teacher_trace import instrument_environment
    from .teacher_initialization import prepare_warmup, begin_ball_trial
    from .physical_events import init_physical_events, scan_physical_events
    from .collection_labels import label_task_bounds
    from .migration import load_v321_actor, actor_mean
    from mjx_topology_execution import make_collect

    n = args.diagnostic_envs or cfg['collection']['n_envs']
    recipe = cfg['collection']['generator_execution']
    if recipe['trial_controls'] != 128 or recipe['warmup_controls'] != 128:
        raise ValueError('This collector records two continuous 64-control rounds per warmed trial')
    rng = np.random.default_rng(args.seed)
    catalog = load_pose_catalog(cfg)
    bundle = build_native_environment(cfg, n_envs=n)
    env = bundle.environment
    reset = jax.jit(env.reset)
    fresh, _ = reset(jax.random.split(jax.random.PRNGKey(args.seed), n))
    jax.block_until_ready(fresh)
    geometry = compiled_geometry(env, fresh)
    surface = surface_from_vertices(np.asarray(geometry['racket']['compiled_vertices_local_m']),
        normal_axis=int(env.cfg.fixed_contact_racket_surface_size_axis))
    adapter = NativeObservationAdapter(env, fresh)
    observer = instrument_environment(env, bundle.stack)
    actor = load_v321_actor(repository_root(), cfg['lineage']['actor_source'])
    report.update(geometry=geometry, observer=observer,
        admitted_regions=catalog.region_ids.tolist(), unknown_regions=catalog.unknown_region_ids.tolist(),
        actual_env_overrides=bundle.overrides,
        racket_reference_rotation_world=np.asarray(env.fixed_contact_racket_reference_rotation).tolist(),
        joint_lower_rad=np.asarray(env.arm_lo).tolist(), joint_upper_rad=np.asarray(env.arm_hi).tolist())
    context = numeric_tree_digest(fresh.model)
    context_ref, _, _ = save_seed(args.output_dir, 'context', fresh.model, context, budget)
    identity = dict(config_sha256=config_hash(cfg), geometry_sha256=geometry['geometry_sha256'],
        control_sha256=config_hash(cfg['control']), contact_sha256=config_hash(cfg['physics']),
        source_commit=report['source_commit'], training_version='WMJ-V1', model_revision=0)
    dataset = TrajectoryDataset.create(args.output_dir/'dataset', budget, identity=identity,
                                      split_version=cfg['collection']['split']['id'])
    report['context'] = dict(sha256=context, reference=context_ref)

    def initial_full(native):
        history, _ = adapter.initialize(native)
        return dict(native=native, history=history, events=init_physical_events(n),
                    ball_model_valid=jnp.ones(n, bool))

    def transition(full, action, goal):
        result = env.step(full['native'], action)
        after, _, _, native_done, metrics = result
        history, diagnostics = adapter.advance(after, full['history'])
        trace = metrics['wmj_trace']
        finite = (jnp.all(jnp.isfinite(trace['q_after']), -1)
                  & jnp.all(jnp.isfinite(trace['dq_after']), -1)
                  & jnp.all(jnp.isfinite(trace['ball_qpos_after']), -1)
                  & jnp.all(jnp.isfinite(trace['ball_qvel_after']), -1))
        ball_valid = (full['ball_model_valid'][None, :]
                      & (jnp.cumsum((trace['other_ball_collision'] | ~finite).astype(jnp.int32), axis=0) == 0))
        events, labels = scan_physical_events(full['events'], trace, valid=ball_valid,
                                              initialization_active=jnp.zeros_like(ball_valid))
        # Native fixed-point reward/termination never become WMJ returns. All
        # finite failed trajectories remain data, with separate validity tags.
        streams = {**trace, **{f'event_{key}': value for key, value in labels.items()},
                   'ball_model_valid': ball_valid, 'robot_model_valid': finite}
        actions = dict(raw_normalized=action, clipped_normalized=jnp.clip(action, -1., 1.),
                       projected_normalized=after.prev_action, accepted_normalized=after.prev_action)
        state_values = compact_state(env, after, goal)
        actions['native_done_diagnostic'] = native_done
        next_full = dict(native=after, history=history, events=events,
                         ball_model_valid=ball_valid[-1])
        diagnostics['finite'] = jnp.all(finite, axis=0)
        return next_full, state_values, actions, streams, diagnostics

    scoped, execution = make_collect(transition, mode='project')
    step = jax.jit(scoped)
    warm_start = jax.jit(lambda value, q: prepare_warmup(env, value, q))
    trial_start = jax.jit(lambda value, warm, p, v:
        begin_ball_trial(env, value, warm, p, v, ar1_enabled=False))
    policy = jax.jit(lambda full, goal: actor_mean(actor,
        adapter.observation(full['native'], full['history'], goal)[:, :72]))
    report['execution'] = execution
    report['source_snapshot'] = {p.name: dict(sha256=file_hash(p), source=p.read_text())
                                for p in Path(__file__).parent.glob('*.py')}
    write_json_unique(args.output_dir/'implementation.json', report, budget)
    coverage = np.zeros(len(catalog.region_ids), np.int64)
    counts = Counter()
    cold = True
    for trial in range((args.rounds+1)//2):
        wall.check(reserve_s=30.)
        draw = sample_collection_regions(rng, n, coverage=coverage,
            normalized_error=np.zeros_like(coverage), cost_s=np.ones_like(coverage), error_source='coverage_only')
        pose_ids, q = select_pose_per_region(rng, catalog, draw.indices)
        assigned = generator_assignments(rng, n, cfg)
        # New context is not independently randomized after warming; every
        # trial uses the exact baseline model saved above and fresh RNG keys.
        fresh, _ = reset(jax.random.split(jax.random.PRNGKey(args.seed+trial), n))
        if numeric_tree_digest(fresh.model) != context:
            raise ValueError('A new randomized context needs its own saved model before collection')
        goal = jnp.broadcast_to(jnp.array([0., -.5, 1.375]), (n, 3))
        warmed = warm_start(fresh, jnp.asarray(q))
        full = initial_full(warmed)
        warm_actions, warm_modes = correlated_action_sequences(rng, 128, n,
            amplitude_range=recipe['warmup_normalized_amplitude_range'])
        warm_actions[:recipe['warmup_initial_zero_controls']] = 0.
        started = time.monotonic()
        for tick in range(128):
            full, *_ = step(full, jnp.asarray(warm_actions[tick]), goal)
            if tick % 32 == 0:
                jax.block_until_ready(full)
                print(f'[WARM] trial={trial} tick={tick}/128', flush=True)
                wall.check(reserve_s=30.)
        jax.block_until_ready(full)
        report['warmups'].append(dict(trial=trial, wall_s=time.monotonic()-started, includes_compile=cold))
        cold = False
        warmed = full['native']
        actions, modes = correlated_action_sequences(rng, 128, n,
            amplitude_range=recipe['identification_normalized_amplitude_range'])
        boundary = assigned == 'boundaries_failure'
        if boundary.any():
            actions[:, boundary], modes[boundary] = correlated_action_sequences(rng, 128, int(boundary.sum()),
                amplitude_range=recipe['boundary_normalized_amplitude_range'])
        lead = sample_contact_lead_controls(rng, cfg, n)
        preview = full
        future_p = np.zeros((n, 3), np.float32)
        future_r = np.zeros((n, 3, 3), np.float32)
        for tick in range(1, int(lead.max())+1):
            preview, *_ = step(preview, jnp.asarray(actions[tick-1]), goal)
            mask = lead == tick
            if mask.any():
                future_p[mask] = np.asarray(preview['native'].data.geom_xpos[:, env.racket_geom_id])[mask]
                future_r[mask] = np.asarray(preview['native'].data.geom_xmat[:, env.racket_geom_id]).reshape(n, 3, 3)[mask]
        plan = propose_directed_contacts(rng, cfg, surface, future_p, future_r,
            ball_radius_m=geometry['ball']['radius_m'], contact_lead_controls=lead)
        initial_p, initial_v = plan.initial_position_m.copy(), plan.initial_velocity_m_s.copy()
        robot_only = assigned == 'robot_identification'
        initial_p[robot_only], initial_v[robot_only] = [0., -3., 3.], [0., 0., 0.]
        native = trial_start(fresh, warmed, jnp.asarray(initial_p), jnp.asarray(initial_v))
        full = initial_full(native)
        goal = jnp.asarray(plan.target_apex_m)
        if not np.all(np.asarray(native.data.qvel[:, env.ball_vadr+3:env.ball_vadr+6]) == 0):
            raise ValueError('A new trial must begin with zero ball spin')
        # Preserve warmed control leaf values exactly; this is a state-copy
        # identity check, not a numerical comparison between solver variants.
        if numeric_tree_digest(native.rmp_state) != numeric_tree_digest(warmed.rmp_state):
            raise ValueError('Trial initialization changed warmed RMP memory')
        trial_meta = dict(trial=trial, pose_ids=pose_ids.tolist(), region_ids=catalog.region_ids[draw.indices].tolist(),
            region_probability=draw.probability.tolist(), sampler=draw.report,
            generator=assigned.tolist(), excitation_mode=modes.tolist(), warm_excitation_mode=warm_modes.tolist(),
            proposal={key: value.tolist() for key, value in asdict(plan).items()},
            actual_initial_ball_position_m=initial_p.tolist(), actual_initial_ball_velocity_m_s=initial_v.tolist(),
            warmup_controls=128, preview_controls=lead.tolist(),
            preview_actions_used_unchanged=(assigned != 'closed_loop').tolist(),
            setup_or_task_success_claim=False, old_fixed_point_stable_lanes=0)
        write_json_unique(args.output_dir/f'trial_{trial:04d}.json', trial_meta, budget)
        np.add.at(coverage, draw.indices, 1)
        groups = [dict(root_seed_family=f'wmj_{args.seed}_{trial}_{lane}',
            episode=f'wmj_{args.seed}_{trial}_{lane}', session=f'wmj_{args.seed}_{trial}_{lane}',
            region_id=int(catalog.region_ids[draw.indices[lane]]), generator=str(assigned[lane])) for lane in range(n)]
        for half in range(min(2, args.rounds-2*trial)):
            round_id = trial*2+half
            wall.check(reserve_s=30.)
            full_without_model = {**full, 'native': full['native']._replace(model=None), 'goal': goal}
            seed_ref, _, _ = save_seed(args.output_dir, f'seed_{round_id:04d}',
                                      full_without_model, context, budget)
            initial = jax.device_get(compact_state(env, full['native'], goal))
            state_rows, action_rows, substep_rows, valid_rows = [initial], [], [], []
            start = time.monotonic()
            for tick in range(64):
                action = jnp.asarray(actions[half*64+tick])
                if np.any(assigned == 'closed_loop'):
                    action = jnp.where(jnp.asarray(assigned == 'closed_loop')[:, None], policy(full, goal), action)
                before = full
                result = step(full, action, goal)
                full, values, action_values, substeps, diagnostic = result
                values, action_values, substeps, diagnostic = jax.device_get((values, action_values, substeps, diagnostic))
                if not np.all(diagnostic['finite']):
                    raise FloatingPointError('Nonfinite native trajectory; preserve earlier committed chunks and stop')
                if np.any(diagnostic['capacity_exceeded']):
                    raise ValueError('History capacity exceeded')
                # Finish the previously outstanding precontact restore check
                # from an actual imminent impact, not an allocated-contact distance.
                imminent = np.any(substeps['event_contact_entry'], axis=0) & ~np.asarray(before['events'].contact_active)
                if 'precontact_replay' not in report and np.any(imminent):
                    tree = {**before, 'native': before['native']._replace(model=None)}
                    arrays, manifest = pack_numeric_state(tree, context_sha256=context)
                    restored = restore_numeric_state(tree, arrays, manifest, context_sha256=context)
                    restored['native'] = restored['native']._replace(model=before['native'].model)
                    replay = step(restored, action, goal)
                    jax.block_until_ready(replay)
                    max_q = float(jnp.max(jnp.abs(full['native'].data.qpos-replay[0]['native'].data.qpos)))
                    max_v = float(jnp.max(jnp.abs(full['native'].data.qvel-replay[0]['native'].data.qvel)))
                    event_match = all(np.array_equal(substeps[k], np.asarray(replay[3][k]))
                                      for k in ('event_contact_entry', 'event_contact_release'))
                    replay_ref, _, _ = save_seed(args.output_dir, 'precontact_replay_seed', tree, context, budget)
                    action_ref = write_array_archive_unique(args.output_dir/'precontact_replay_action.npz',
                        dict(action=np.asarray(action), goal=np.asarray(goal)), budget)
                    tight_pass = max_q <= 1e-5 and max_v <= 1e-4 and event_match
                    report['precontact_replay'] = dict(status='PASS' if tight_pass else 'DIAGNOSTIC_DIFFERENCE',
                        round=round_id, tick=tick,
                        qualifying_lanes=np.flatnonzero(imminent).tolist(), max_qpos_error=max_q,
                        max_qvel_error=max_v, discrete_events_equal=event_match,
                        seed=replay_ref, action_archive=action_ref,
                        input_leaf_integrity='exact_authenticated_restore',
                        repeated_small_difference_is_not_collection_gate=True)
                    write_json_unique(args.output_dir/'precontact_replay.json', report['precontact_replay'], budget)
                    print(f'[VERIFY] precontact replay round={round_id} tick={tick}: '
                          f'qpos={max_q:.6g}, qvel={max_v:.6g}, events_equal={event_match}', flush=True)
                    # Keep the original numerical-smoke result visible, but do
                    # not discard real teacher data for GPU reduction roundoff.
                    # These wider stop levels only triage the repeated diagnostic;
                    # they do not qualify model accuracy or change task gates.
                    if max_q > 1e-3 or max_v > 1e-2 or not event_match:
                        raise ValueError(f'Precontact replay needs diagnosis: qpos={max_q}, '
                                         f'qvel={max_v}, events_equal={event_match}')
                state_rows.append(values)
                action_rows.append(action_values)
                substep_rows.append(substeps)
                valid_rows.append(diagnostic['finite'])
                for key in ('contact_entry', 'contact_release', 'controlled_release', 'actual_apex'):
                    counts[key] += int(substeps[f'event_{key}'].sum())
                if tick % 32 == 0:
                    print(f'[COLLECT] round={round_id+1}/{args.rounds} tick={tick}/64', flush=True)
                    wall.check(reserve_s=30.)
            states, recorded_actions, substeps = map(_stack_cpu, (state_rows, action_rows, substep_rows))
            bounds = label_task_bounds(cfg, states,
                reference_rotation=np.asarray(env.fixed_contact_racket_reference_rotation),
                joint_lower_rad=np.asarray(env.arm_lo), joint_upper_rad=np.asarray(env.arm_hi))
            for key, value in bounds.items():
                if value.shape[:2] == (65, n):
                    states[f'task_{key}'] = value
            write_json_unique(args.output_dir/f'quality_{round_id:04d}.json',
                dict(initial_task_eligible=bounds['initial_task_eligible'].tolist(),
                     first_task_bound_failure_control=bounds['first_task_bound_failure_control'].tolist(),
                     continuous_trial_control_offset=half*64,
                     counts={key: int((~bounds[key]).sum()) for key in ('joint_position_valid',
                         'racket_xyz_valid', 'racket_roll_pitch_valid', 'ball_xy_valid', 'ball_z_valid')},
                     task_failure_data_retained=True, robot_self_collision_queries=0), budget)
            row = dataset.append_chunk(f'r{round_id:04d}', states=states, actions=recorded_actions,
                substeps=substeps, valid=np.stack(valid_rows), groups=groups, seed_ref=seed_ref, context_sha256=context)
            report['rounds'].append(dict(round=round_id, valid_transitions=row['valid_transitions'],
                wall_s=time.monotonic()-start, stored_bytes=row['archive']['stored_bytes'],
                substep_ball_valid_fraction=float(substeps['ball_model_valid'].mean())))
            print(f'[SAVE] round={round_id+1}; transitions={row["transitions"]}; events={dict(counts)}', flush=True)
            write_json_unique(args.output_dir/f'progress_{round_id:04d}.json',
                dict(rounds=report['rounds'], events=dict(counts), region_trial_coverage=coverage.tolist()), budget)
    report.update(status='DIAGNOSTIC_COMPLETE' if args.diagnostic_envs else 'COLLECTION_COMPLETE',
        events=dict(counts), region_trial_coverage=coverage.tolist(),
        admitted_regions_sampled=int(np.count_nonzero(coverage)),
        teacher_transitions=sum(row['transitions'] for row in dataset.chunks.values()),
        data_split_counts=dict(Counter(group['split'] for row in dataset.chunks.values() for group in row['groups'])),
        unknown_geometry_not_dropped_from_future_task_denominators=True,
        all_domain_or_model_accuracy_qualified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--diagnostic-envs', type=int, choices=(4, 16))
    parser.add_argument('--seed', type=int, default=2026110604)
    parser.add_argument('--max-wall-s', type=float, default=3600.)
    args = parser.parse_args(argv)
    cfg = load_config()
    if ((args.diagnostic_envs and args.rounds not in (2, 4))
            or (not args.diagnostic_envs and args.rounds not in cfg['collection']['budget_rounds'])):
        parser.error('Use 20/40/80 production rounds, or 2/4 explicitly diagnostic rounds')
    if not 0 < args.max_wall_s <= 12*3600:
        parser.error('Collection must remain inside the 12-hour maximum')
    if (os.environ.get('PP_MEMORY_SCOPE_ACTIVE') != 'YES'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid
            or os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE') != 'false'):
        raise ValueError('Use the guarded single-UUID launcher without preallocation')
    from .audit_gpu_replay import _gpu_sample
    resource = cfg['resources']
    if args.gpu_uuid not in resource['allowed_gpu_uuids']:
        raise ValueError('Unregistered GPU')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:')))*1024
    if available < resource['startup_min_host_available_gib']*2**30:
        raise RuntimeError('Insufficient startup host memory')
    sample = _gpu_sample(args.gpu_uuid)
    if sample['other_compute_pids'] or sample['temperature_c'] >= 78 or sample['memory_free_mib'] < 12*1024:
        raise RuntimeError(f'GPU unavailable: {sample}')
    budget = StorageBudget(repository_root()/resource['output_root'], resource['max_new_bytes'], resource['min_free_bytes'])
    budget.check(additional_bytes=128*2**20)
    if not args.output_dir.resolve().is_relative_to(budget.root.resolve()):
        raise ValueError('Output directory must remain inside the WMJ storage budget')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    wall = WallBudget(started, args.max_wall_s)
    stopped = threading.Event()
    gpu_samples = [sample]

    def watch():
        while not stopped.wait(5.):
            try:
                value = _gpu_sample(args.gpu_uuid)
                gpu_samples.append(value)
                if value['other_compute_pids'] or value['temperature_c'] >= 78 or wall.remaining_s() == 0:
                    raise RuntimeError(f'Resource/time limit: {value}')
            except Exception as exc:
                print(f'[STOP] {exc}; terminating own PID {os.getpid()}', flush=True)
                os.kill(os.getpid(), signal.SIGTERM)
                return

    monitor = threading.Thread(target=watch, daemon=True)
    monitor.start()
    report = dict(schema='wmj_teacher_collection_v1', training_version='WMJ-V1',
        config=cfg, config_sha256=config_hash(cfg), status='INCOMPLETE',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=repository_root()).strip(),
        timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), gpu_uuid=args.gpu_uuid,
        absolute_output_dir=str(args.output_dir.absolute()), n_envs=args.diagnostic_envs or 1024,
        seed=args.seed, requested_rounds=args.rounds, rounds=[], warmups=[],
        actor_updates=0, critic_updates=0, origin='mjx_teacher',
        initialization_geometry_runtime_queries=0, argv=vars(args) | {'output_dir': str(args.output_dir)})
    write_json_unique(args.output_dir/'manifest.json', report, budget)
    try:
        if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
            raise ValueError('Exactly one CUDA device is required')
        report['devices'] = [str(device) for device in jax.devices()]
        run_collection(cfg, args, budget, wall, report)
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        stopped.set()
        monitor.join(timeout=6.)
        report.update(wall_s=time.monotonic()-started, gpu_samples=gpu_samples)
        write_json_unique(args.output_dir/'result.json', report, budget)
        print(f'[SAVE] {args.output_dir}; status={report["status"]}', flush=True)


if __name__ == '__main__':
    main()
