"""Guarded small CUDA replay audit; never a trainer or a collection qualification."""

from __future__ import annotations

import argparse
from datetime import datetime
import io
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

from .artifacts import write_json_unique
from .contracts import load_config, repository_root, config_hash, file_hash
from .resources import StorageBudget
from .state_adapter import numeric_tree_digest, pack_numeric_state, restore_numeric_state
from .numerical_validation import assess_execution_difference


def _gpu_sample(uuid):
    row = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-gpu=uuid,temperature.gpu,memory.free', '--format=csv,noheader,nounits'],
        text=True, timeout=5).strip().split(',')
    if row[0].strip() != uuid:
        raise ValueError('Physical GPU identity mismatch')
    processes = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=5)
    others = [int(line.strip()) for line in processes.splitlines()
              if line.strip().isdigit() and int(line.strip()) != os.getpid()]
    return dict(temperature_c=float(row[1]), memory_free_mib=float(row[2]), other_compute_pids=others)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--max-controls', type=int, default=512)
    parser.add_argument('--accepted-execution-report', type=Path,
                        help='Reuse the recorded, user-accepted numerical smoke without recompiling a second simulator')
    args = parser.parse_args(argv)
    if not 1 <= args.max_controls <= 512:
        parser.error('The diagnostic is bounded at 512 control ticks')
    if (os.environ.get('PP_MEMORY_SCOPE_ACTIVE') != 'YES'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid
            or os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE') != 'false'):
        raise ValueError('Use a cgroup-guarded single-UUID launcher without preallocation')
    cfg = load_config()
    resource = cfg['resources']
    available_kib = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:')))
    if (args.gpu_uuid not in resource['allowed_gpu_uuids']
            or available_kib*1024 < resource['startup_min_host_available_gib']*2**30):
        raise ValueError('Unregistered GPU or insufficient startup host memory')
    budget = StorageBudget(repository_root()/resource['output_root'], resource['max_new_bytes'], resource['min_free_bytes'])
    budget.check(additional_bytes=128*2**20)
    if not args.output_dir.resolve().is_relative_to(budget.root.resolve()):
        raise ValueError('Output must remain inside the registered WMJ budget root')
    first_gpu = _gpu_sample(args.gpu_uuid)
    if first_gpu['other_compute_pids'] or first_gpu['temperature_c'] >= 78 or first_gpu['memory_free_mib'] < 12*1024:
        raise RuntimeError(f'GPU is not available for this probe: {first_gpu}')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stopped = threading.Event()
    samples = [first_gpu]

    def watchdog():
        while not stopped.wait(5.):
            try:
                sample = _gpu_sample(args.gpu_uuid)
                samples.append(sample)
                if sample['temperature_c'] >= 78 or sample['other_compute_pids']:
                    raise RuntimeError(f'GPU resource guard: {sample}')
            except Exception as exc:
                print(f'[STOP] {exc}; terminating only probe PID {os.getpid()}', flush=True)
                os.kill(os.getpid(), signal.SIGTERM)
                return

    monitor = threading.Thread(target=watchdog, daemon=True)
    monitor.start()
    started = time.monotonic()
    report = dict(schema='wmj_gpu_full_seed_replay_v1', training_version='WMJ-V1',
        config=cfg, config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
        timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        n_envs=4, max_controls=args.max_controls, phases={}, status='INCOMPLETE',
        source_snapshot={p.name: dict(sha256=file_hash(p), source=p.read_text()) for p in Path(__file__).parent.glob('*.py')},
        collection_shape_qualified=False, actor_training=False, critic_training=False,
        initialization_geometry_runtime_queries=0)
    write_json_unique(args.output_dir/'manifest.json', report, budget)
    try:
        if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
            raise ValueError('Exactly one CUDA device is required')
        report['devices'] = [str(d) for d in jax.devices()]
        from .native_stack import build_native_environment
        from .native_observation import NativeObservationAdapter
        from .geometry import compiled_geometry
        from .migration import load_v321_actor, actor_mean, expand_actor_history
        from .teacher_trace import instrument_environment
        from .audit_collision_geometry import _save_bytes
        from mjx_topology_execution import make_collect
        from train_fixed_plane_ball_obs_noise import compare_numeric_leaves

        print('[RUN] WMJ-V1 CUDA full-state/EXEC probe; 4 lanes; no learning', flush=True)
        bundle = build_native_environment(cfg, n_envs=4)
        env = bundle.environment
        state, observation = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(2026110601), 4))
        jax.block_until_ready(state)
        geometry = compiled_geometry(env, state)
        adapter = NativeObservationAdapter(env, state)
        history, _ = adapter.initialize(state)
        actor = load_v321_actor(repository_root(), cfg['lineage']['actor_source'])
        expanded = expand_actor_history(actor)
        act = jax.jit(lambda obs: actor_mean(actor, obs))
        expanded_act = jax.jit(lambda obs: actor_mean(expanded, obs))
        report['geometry'] = geometry
        report['observer'] = instrument_environment(env, bundle.stack)
        context = config_hash(dict(config=config_hash(cfg), geometry=geometry['geometry_sha256'],
                                   sampled_model=numeric_tree_digest(state.model)))
        report['context_sha256'] = context

        def step(native, history, action):
            result = env.step(native, action)
            history, diagnostic = adapter.advance(result[0], history)
            return result, history, diagnostic

        reference = None if args.accepted_execution_report else jax.jit(step)
        scoped, execution = make_collect(step, mode='project')
        optimized = jax.jit(scoped)
        report['execution'] = execution
        action = act(observation)
        if reference is not None:
            print('[VERIFY] compiling native reference step', flush=True)
            compile_start = time.monotonic()
            left = reference(state, history, action)
            jax.block_until_ready(left)
            report['reference_cold_step_s'] = time.monotonic()-compile_start
        else:
            from .contracts import authenticate
            evidence = cfg['execution']['qualification']['accepted_roundoff_evidence']
            authenticated = authenticate(repository_root(), evidence)
            if args.accepted_execution_report.resolve() != authenticated:
                raise ValueError('Only the registered execution evidence may be reused')
            prior = json.loads(authenticated.read_text())
            for key in ('physics', 'control', 'observation', 'policy'):
                if prior['config'][key] != cfg[key]:
                    raise ValueError(f'Execution evidence physical interface changed: {key}')
            if prior['observer']['observer_sha256'] != report['observer']['observer_sha256']:
                raise ValueError('Execution observer changed since the accepted smoke')
            report['reused_execution_evidence'] = evidence
        print('[VERIFY] compiling V324 EXEC step for the outstanding state-restore checks', flush=True)
        compile_start = time.monotonic()
        right = optimized(state, history, action)
        jax.block_until_ready(right)
        report['optimized_cold_step_s'] = time.monotonic()-compile_start
        initial_difference = (compare_numeric_leaves(left, right) if reference is not None
                              else prior['initial_execution_pair'])
        report['initial_execution_pair'] = initial_difference
        report['execution_numerical_assessment'] = assess_execution_difference(initial_difference)
        if report['execution_numerical_assessment']['status'] != 'PASS':
            raise ValueError('Observable execution differences exceed the numerical smoke budget')
        print('[VERIFY] execution numerical smoke accepted; floating-point differences retained', flush=True)
        observed_counts = dict(flight=0, precontact=0, in_contact=0)
        for control in range(args.max_controls):
            action = act(observation)
            result, next_history, diagnostic = optimized(state, history, action)
            jax.block_until_ready(result)
            trace = result[-1]['wmj_trace']
            distance = np.asarray(trace['contact_distance_m'][0])
            force = np.asarray(trace['contact_normal_force_n'][0])
            overlap = np.asarray(trace['contact_overlap'][0])
            vertical = np.asarray(trace['ball_qvel_before'][0, :, 2])
            masks = dict(flight=(distance > .03) & ~overlap & (np.abs(vertical) > .05),
                precontact=(distance > 0.) & (distance < .01) & (vertical < -.05) & ~overlap,
                in_contact=overlap & (force > 0.))
            if np.any(np.asarray(diagnostic['capacity_exceeded'])):
                raise ValueError('Paired observation history overflow')
            for name, mask in masks.items():
                observed_counts[name] += int(np.count_nonzero(mask))
                if name in report['phases'] or not np.any(mask):
                    continue
                full = dict(native=state._replace(model=None), history=history)
                arrays, manifest = pack_numeric_state(full, context_sha256=context)
                restored = restore_numeric_state(full, arrays, manifest, context_sha256=context)
                replay = optimized(restored['native']._replace(model=state.model), restored['history'], action)
                native_reference = reference(state, history, action) if reference is not None else None
                jax.block_until_ready((replay, native_reference))
                restore_diff = compare_numeric_leaves((result, next_history, diagnostic), replay)
                restore_assessment = assess_execution_difference(restore_diff)
                exec_diff = (compare_numeric_leaves((result, next_history, diagnostic), native_reference)
                             if native_reference is not None else None)
                exec_assessment = assess_execution_difference(exec_diff) if exec_diff is not None else None
                goal = jnp.concatenate((state.racket_anchor[:, :2], jnp.full((4, 1), 1.425)), axis=-1)
                obs969 = adapter.observation(state, history, goal)
                raw_error = float(jnp.max(jnp.abs(expanded_act(obs969)-action)))
                if restore_assessment['status'] != 'PASS' or (exec_assessment and exec_assessment['status'] != 'PASS') or raw_error > 1e-5:
                    raise ValueError(f'Phase {name} replay/migration mismatch: {restore_diff}, {exec_diff}, {raw_error}')
                stream = io.BytesIO()
                np.savez_compressed(stream, **arrays, raw_action=np.asarray(action))
                seed_path = args.output_dir/f'{name}_seed.npz'
                _save_bytes(seed_path, stream.getvalue(), budget)
                phase = dict(control_index=control, qualifying_lanes=np.flatnonzero(mask).tolist(),
                    contact_distance_m=distance.tolist(), contact_normal_force_n=force.tolist(),
                    vertical_velocity_m_s=vertical.tolist(), restore_difference=restore_diff,
                    restore_assessment=restore_assessment, restored_input_integrity='exact_leaf_hashes',
                    execution_difference=exec_diff, migration_max_raw_action_error=raw_error,
                    execution_assessment=exec_assessment,
                    seed_manifest=manifest, seed_path=seed_path.name, seed_sha256=file_hash(seed_path))
                report['phases'][name] = phase
                write_json_unique(args.output_dir/f'{name}_audit.json', phase, budget)
                print(f'[VERIFY] {name}: full-state restore and EXEC pair passed at tick {control}', flush=True)
            state, observation, history = result[0], result[1], next_history
            report['controls_executed'] = control+1
            if len(report['phases']) == 3:
                break
            if np.any(np.asarray(result[3])):
                report['stop_reason'] = 'first_native_done_before_all_phases'
                break
            if control % 64 == 0:
                print(f'[VERIFY] tick={control} phase_counts={observed_counts}', flush=True)
        report['phase_sample_counts'] = observed_counts
        report['status'] = 'PASS' if len(report['phases']) == 3 else 'INCOMPLETE'
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        stopped.set()
        monitor.join(timeout=6.)
        report['wall_s'] = time.monotonic()-started
        report['gpu_samples'] = samples
        write_json_unique(args.output_dir/'result.json', report, budget)
        print(f"[SAVE] {args.output_dir}; status={report['status']}", flush=True)


if __name__ == '__main__':
    main()
