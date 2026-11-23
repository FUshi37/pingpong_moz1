"""Fit and validate the robot execution component on immutable MJX data.

This is a real supervised module trainer, not an imagination policy trainer or
a claim that the composed world is ready. Its autoregression uses predicted
q/dq and recorded delayed commands; full feedback closure is a separate gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from .artifacts import read_array_archive, write_array_archive_unique, write_json_unique
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .fit_schedule import controls_at, parse_schedule, validate_resume_recipe
from .kinematics import extract_chain
from .model_data import RobotChunkReader, robot_chunk, robot_statistics_update, sample_robot_windows
from .model_losses import racket_outputs, robot_execution_loss
from .neural import init_adam, init_mlp, adam_update
from .robot_model import robot_rollout, scale_from_training_statistics
from .runtime_guard import RunGuard
from .state_adapter import pack_numeric_state, restore_numeric_state
from .trajectory_data import TrajectoryDataset, TrainNormalizer


def dataset_identity(dataset):
    """Bind an append-stable snapshot, including every archive and seed locator."""
    return config_hash(dict(identity=dataset.identity,
        chunks={name: record for name, record in dataset.chunks.items()}))


def load_chain(cfg, collection_report):
    path = authenticate(repository_root(), cfg['collection']['compiled_model_cache'])
    model = mujoco.MjModel.from_binary_path(str(path))
    geometry = collection_report['geometry']
    return extract_chain(model, np.asarray(geometry['reset_fk']['qpos']),
        joint_ids=geometry['reset_fk']['joint_order'],
        site_id=geometry['racket']['site_id'], geom_id=geometry['racket']['geom_id'])


def summarize_errors(values):
    values = np.concatenate(values)
    finite = np.isfinite(values)
    usable = values[finite]
    return dict(count=len(values), nonfinite=int((~finite).sum()),
        mean=float(usable.mean()) if len(usable) else None,
        p95=float(np.quantile(usable, .95)) if len(usable) else None,
        maximum=float(usable.max()) if len(usable) else None)


def validate_robot(parameters, scale, dataset, chain, guard, *, seed, batch_size=128):
    """Frozen, reproducible validation windows; final test data are inaccessible."""
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    reader = RobotChunkReader(dataset)
    horizons = (1, 8, 32, 64, 128)
    metrics = {h: {name: [] for name in ('joint_q_rmse_rad', 'joint_dq_rmse_rad_s',
        'racket_position_m', 'normal_velocity_m_s')} for h in horizons}
    regions = {h: [] for h in horizons}
    roots = {h: set() for h in horizons}

    @jax.jit
    def predict(params, scaling, batch):
        _, (q, dq) = robot_rollout(params, scaling, batch['q_before'][0],
                                    batch['dq_before'][0], batch['commands'])
        predicted_p, predicted_v = racket_outputs(chain, q[-1], dq[-1])
        target_p, target_v = racket_outputs(chain, batch['q_after'][-1], batch['dq_after'][-1])
        return dict(joint_q_rmse_rad=jnp.sqrt(jnp.mean((q[-1]-batch['q_after'][-1])**2, axis=-1)),
            joint_dq_rmse_rad_s=jnp.sqrt(jnp.mean((dq[-1]-batch['dq_after'][-1])**2, axis=-1)),
            racket_position_m=jnp.linalg.norm(predicted_p-target_p, axis=-1),
            normal_velocity_m_s=jnp.abs(predicted_v-target_v))

    names = list(dataset.chunks)
    for index in range(0, len(names), 2):
        guard.check()
        first = dataset.chunks[names[index]]
        second = dataset.chunks[names[index+1]]
        for left, right in zip(first['groups'], second['groups']):
            if any(left[key] != right[key] for key in ('root_seed_family', 'episode', 'session', 'split')):
                raise ValueError('Robot validation cannot join different episodes')
        a = reader.read(names[index])
        b = reader.read(names[index+1])
        for key in ('q_rad', 'dq_rad_s', 'time_us'):
            if not np.array_equal(a['state__'+key][-1], b['state__'+key][0]):
                raise ValueError('Robot validation chunks are discontinuous')
        a, b = robot_chunk(a), robot_chunk(b)
        joined = {key: np.concatenate((a[key], b[key])) for key in a}
        record = {**first, 'steps': first['steps']+second['steps']}
        for horizon in horizons:
            batch, refs = sample_robot_windows(rng, joined, record,
                controls=horizon, batch_size=batch_size, split='validation')
            errors = jax.device_get(predict(parameters, scale, jax.tree.map(jnp.asarray, batch)))
            for key, value in errors.items():
                metrics[horizon][key].append(value)
            regions[horizon].append(refs['region_id'])
            roots[horizon].update(refs['root_seed_family'])
        del a, b, joined
    report = {}
    for horizon in horizons:
        region_ids = np.concatenate(regions[horizon])
        regional = {}
        for region in np.unique(region_ids):
            mask = region_ids == region
            regional[str(int(region))] = {name: summarize_errors([np.concatenate(values)[mask]])
                                         for name, values in metrics[horizon].items()}
        report[str(horizon)] = dict(overall={name: summarize_errors(values)
            for name, values in metrics[horizon].items()}, by_region=regional,
            unique_root_seed_families=len(roots[horizon]), represented_regions=len(regional))
    return dict(scope='robot_module_recorded_delayed_commands_not_composed_world',
                source='validation_only', seed=seed, horizons=report,
                wall_s=time.monotonic()-started, world_model_ready=False)


def fit(cfg, args, guard, report):
    dataset = TrajectoryDataset.open(args.dataset, guard.budget)
    reader = RobotChunkReader(dataset)
    if dataset.identity['control_sha256'] != config_hash(cfg['control']):
        raise ValueError('Dataset control contract differs from WMJ-V1')
    if dataset.identity['contact_sha256'] != config_hash(cfg['physics']):
        raise ValueError('Dataset physical contract differs from WMJ-V1')
    if len(dataset.chunks) % 2:
        raise ValueError('This initial fitting entry requires complete continuous two-chunk trials')
    collection = args.dataset.parent
    collection_report = json.loads((collection/'result.json').read_text())
    data_sha = dataset_identity(dataset)
    report.update(dataset_sha256=data_sha, dataset_identity=dataset.identity,
        collection_result_sha256=file_hash(collection/'result.json'),
        dataset_transitions=sum(row['transitions'] for row in dataset.chunks.values()),
        dataset_resource_run_status='see collection integrity and scope log; not inferred from worker status')
    chain = load_chain(cfg, collection_report)
    structure = dict(widths=[35, *cfg['world_model']['robot']['hidden_sizes'], 14],
        activation='tanh_linear', normalization='train_only',
        q_scale_rad=.01, dq_scale_rad_s=.10, racket_position_scale_m=.003,
        normal_speed_scale_m_s=.05, dt_s=.001)
    signature = config_hash(dict(dataset=data_sha, structure=structure))
    report['model_signature'] = signature
    statistics_started = time.monotonic()
    if args.resume:
        metadata = json.loads(args.resume.read_text())
        if metadata['model_signature'] != signature or metadata['schema'] != 'wmj_robot_component_checkpoint_v1':
            raise ValueError('Resume dataset/model contract mismatch')
        for name in ('robot_model.py', 'model_losses.py', 'neural.py', 'kinematics.py', 'model_data.py'):
            if metadata['source_sha256'][name] != report['source_sha256'][name]:
                raise ValueError(f'Resume training semantics changed: {name}')
        stats = metadata['normalization']
    else:
        inputs, residuals = TrainNormalizer(35), TrainNormalizer(14)
        for name, record in dataset.chunks.items():
            guard.check()
            robot_statistics_update(inputs, residuals, reader.read(name), record)
            print(f'[NORMALIZE] {name}; train_substeps={inputs.count}', flush=True)
        stats = dict(inputs=inputs.to_dict(), residuals=residuals.to_dict())
    report['normalization_wall_s'] = time.monotonic()-statistics_started
    write_json_unique(args.output_dir/'normalization.json', stats, guard.budget)
    scale = scale_from_training_statistics(stats['inputs'], stats['residuals'])
    key = jax.random.PRNGKey(args.seed)
    parameters = init_mlp(key, structure['widths'], zero_output=True)
    optimizer = init_adam(parameters)
    rng = np.random.default_rng(args.seed)
    active_chunk = None
    if args.resume:
        arrays = read_array_archive(args.resume.parent/metadata['archive_file'], metadata['archive'])
        state = restore_numeric_state(dict(parameters=parameters, optimizer=optimizer, scale=scale, rng=key),
            arrays, metadata['state'], context_sha256=signature)
        parameters, optimizer, scale, key = (state[k] for k in ('parameters', 'optimizer', 'scale', 'rng'))
        rng.bit_generator.state = metadata['cpu_rng']
        active_chunk = metadata['active_chunk']
        validate_resume_recipe(metadata['fit_recipe'], report['fit_recipe'],
                               completed=int(optimizer.step))
        report['resume'] = dict(path=str(args.resume.absolute()), sha256=file_hash(args.resume))
    start_update = int(optimizer.step)
    if start_update >= args.updates:
        raise ValueError('Requested final update must exceed the restored optimizer step')

    @jax.jit(static_argnames=('autoregressive',))
    def update(params, opt, scaling, batch, *, autoregressive):
        (_, values), gradients = jax.value_and_grad(robot_execution_loss, has_aux=True)(
            params, scaling, batch, autoregressive=autoregressive, chain=chain)
        candidate, next_opt, diagnostics = adam_update(params, gradients, opt,
            learning_rate=cfg['world_model']['optimizer']['lr'], max_gradient_norm=1.)
        finite = jnp.isfinite(values['loss']) & jnp.isfinite(diagnostics['gradient_norm'])
        finite &= jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(candidate)]))
        return candidate, next_opt, {**values, **diagnostics, 'finite': finite}

    saved = {}

    def checkpoint():
        count = int(optimizer.step)
        if count in saved:
            return saved[count]
        name = f'robot_u{count:06d}'
        arrays, state = pack_numeric_state(dict(parameters=parameters, optimizer=optimizer,
            scale=scale, rng=key), context_sha256=signature)
        archive = write_array_archive_unique(args.output_dir/f'{name}.npz', arrays, guard.budget)
        payload = dict(schema='wmj_robot_component_checkpoint_v1', training_version='WMJ-V1',
            component='robot_execution', candidate_only=True, model_signature=signature,
            structure=structure, state=state, archive=archive, archive_file=f'{name}.npz',
            normalization=stats, cpu_rng=rng.bit_generator.state, active_chunk=active_chunk,
            optimizer_updates=count, fit_recipe=report['fit_recipe'],
            source_commit=report['source_commit'], source_sha256=report['source_sha256'],
            config_sha256=config_hash(cfg), dataset_sha256=data_sha,
            actor_updates=0, critic_updates=0, imagined_transitions=0,
            supervised_substep_uses=sum(controls_at(i, args.window_schedule)
                                       for i in range(count))*5*256,
            wall_s=time.monotonic()-guard.started)
        path = args.output_dir/f'{name}.json'
        digest = write_json_unique(path, payload, guard.budget)
        saved[count] = dict(path=str(path.absolute()), sha256=digest)
        print(f'[CHECKPOINT] {name}', flush=True)
        return saved[count]

    report['initial_checkpoint'] = checkpoint()
    names = list(dataset.chunks)
    data = robot_chunk(reader.read(active_chunk)) if active_chunk else None
    train_started = time.monotonic()
    try:
        for iteration in range(start_update, args.updates):
            iteration_started = time.monotonic()
            guard.check()
            if iteration % 32 == 0 or data is None:
                active_chunk = str(rng.choice(names))
                data = robot_chunk(reader.read(active_chunk))
            batch, _ = sample_robot_windows(rng, data, dataset.chunks[active_chunk],
                controls=controls_at(iteration, args.window_schedule), batch_size=256, split='train')
            started = time.monotonic()
            candidate, next_optimizer, metrics = update(parameters, optimizer, scale,
                jax.tree.map(jnp.asarray, batch), autoregressive=iteration >= args.bootstrap_updates)
            metrics = jax.device_get(metrics)
            if not bool(metrics['finite']):
                raise FloatingPointError('Nonfinite model update; retaining last finite checkpoint')
            parameters, optimizer = candidate, next_optimizer
            if iteration % 32 == 0 or iteration+1 == args.updates:
                row = dict(update=iteration+1, phase='autoregressive' if iteration >= args.bootstrap_updates else 'single_step_bootstrap',
                    control_window=controls_at(iteration, args.window_schedule),
                    **{name: float(value) for name, value in metrics.items() if name != 'finite'},
                    kernel_update_s=time.monotonic()-started,
                    full_update_s=time.monotonic()-iteration_started, chunk=active_chunk)
                report['training'].append(row)
                print(f'[FIT] u={iteration+1} phase={row["phase"]} loss={row["loss"]:.6g} '
                      f'grad={row["gradient_norm"]:.5g} update_s={row["full_update_s"]:.4f}', flush=True)
            if (iteration+1) % 256 == 0:
                checkpoint()
                write_json_unique(args.output_dir/f'training_u{iteration+1:06d}.json',
                    dict(training=report['training']), guard.budget)
        report['training_wall_s'] = time.monotonic()-train_started
        report['final_checkpoint'] = checkpoint()
        validation = validate_robot(parameters, scale, dataset, chain, guard,
                                    seed=args.seed+10001)
        write_json_unique(args.output_dir/'validation.json', validation, guard.budget)
        report['validation'] = validation
        report['status'] = 'COMPONENT_FIT_COMPLETE_NOT_WORLD_QUALIFIED'
    finally:
        report['last_finite_checkpoint'] = checkpoint()
        report['completed_updates'] = int(optimizer.step)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--updates', type=int, default=512)
    parser.add_argument('--bootstrap-updates', type=int, default=256)
    parser.add_argument('--controls', type=int, choices=(8, 16, 32, 64), default=8)
    parser.add_argument('--window-schedule', help='Ordered update:controls stages, e.g. 256:8,1024:16,2048:32,4096:64')
    parser.add_argument('--seed', type=int, default=2026110605)
    parser.add_argument('--max-wall-s', type=float, default=1800.)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(argv)
    cfg = load_config()
    if not 0 <= args.bootstrap_updates < args.updates <= cfg['world_model']['optimizer']['max_initial_updates']:
        parser.error('Require 0 <= bootstrap < total updates <= registered component update cap')
    args.window_schedule = parse_schedule(args.window_schedule, bootstrap=args.bootstrap_updates,
                                         initial=args.controls)
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, args.max_wall_s) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        paths = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_robot_fit_v1', training_version='WMJ-V1', status='INCOMPLETE',
            config=cfg, config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            absolute_output_dir=str(args.output_dir.absolute()),
            timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository_root(), text=True).strip(),
            source_sha256={p.name: file_hash(p) for p in paths},
            fit_recipe=dict(batch_windows=256, control_window=args.controls,
                window_schedule=args.window_schedule,
                bootstrap_updates=args.bootstrap_updates, learning_rate=3e-4, max_gradient_norm=1.,
                supervised_input='native_delayed_commands_and_measured_q_dq',
                sampling='uniform_acquisition_then_valid_window_32_updates_per_CPU_chunk',
                physics_dt_s=.001, heldout='validation_only', output_scope='robot_component_only'),
            arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            training=[], actor_updates=0, critic_updates=0, world_model_ready=False)
        write_json_unique(args.output_dir/'manifest.json',
            {**report, 'source_snapshot': {p.name: p.read_text() for p in paths}}, guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device is required')
            report['devices'] = [str(device) for device in jax.devices()]
            fit(cfg, args, guard, report)
        except InterruptedError as exc:
            report.update(status='INTERRUPTED', error=str(exc))
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_json_unique(args.output_dir/'result.json', report, guard.budget)
            print(f'[SAVE] {args.output_dir}; status={report["status"]}', flush=True)


if __name__ == '__main__':
    main()
