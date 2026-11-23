"""Fit the approved full-chain execution proxy using coherent native replay.

This isolated module uses recorded delayed commands, predicts all thirteen
joints, and compares FK with native pose/twist labels. It does not qualify
the composed feedback world or perform any actor/critic update.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from importlib.metadata import version
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import read_array_archive, write_array_archive_unique, write_json_unique
from .audit_support_replay import load_support_chain
from .contracts import config_hash, file_hash, load_config, repository_root
from .fit_robot import summarize_errors
from .fit_schedule import controls_at, parse_schedule, validate_resume_recipe
from .neural import init_adam, init_mlp, adam_update
from .robot_support_model import JOINT_NAMES, support_robot_rollout, support_scale_from_statistics
from .runtime_guard import RunGuard
from .state_adapter import pack_numeric_state, restore_numeric_state
from .support_data import SupportReplayReader, support_chunk, support_statistics_update, sample_support_windows
from .support_losses import racket_kinematics, support_execution_loss
from .trajectory_data import TrainNormalizer


def join_trial(reader, first_name, second_name):
    """Join one true acquisition across its storage boundary without resets."""
    first, second = reader.records[first_name], reader.records[second_name]
    if (first['record']['groups'] != second['record']['groups']
            or first['first_seed_ref'] != second['first_seed_ref']
            or (first['replay_controls_before_chunk'], second['replay_controls_before_chunk']) != (0, 64)):
        raise ValueError('Support validation needs continuous halves of one trial')
    a, b = reader.read(first_name), reader.read(second_name)
    for key in ('q_rad', 'dq_rad_s', 'support_q_rad', 'support_dq_rad_s', 'time_us'):
        if not np.array_equal(a['state__'+key][-1], b['state__'+key][0]):
            raise ValueError('Canonical support trial is discontinuous')
    a, b = support_chunk(a), support_chunk(b)
    return ({key: np.concatenate((a[key], b[key])) for key in a},
            {**first['record'], 'steps': first['record']['steps']+second['record']['steps']})


def validate(parameters, scale, reader, chain, guard, *, seed):
    """Frozen held-out 1/8/32/64/128-control predictions; test data stay sealed."""
    rng, started = np.random.default_rng(seed), time.monotonic()
    horizons = (1, 8, 32, 64, 128)
    metrics, regions, roots = ({h: {} for h in horizons},
                              {h: [] for h in horizons}, {h: set() for h in horizons})

    @jax.jit
    def errors(params, scaling, batch):
        _, (q, dq) = support_robot_rollout(params, scaling,
            batch['q_before'][0], batch['dq_before'][0], batch['commands'])
        # Native pose/twist describes the preintegration final 1 ms interval.
        pose = racket_kinematics(chain, q[-2], dq[-2])
        target_rotation = batch['racket_geom_rotation_before'][-1]
        predicted_normal, actual_normal = pose['rotation'][..., :, 0], target_rotation[..., :, 0]
        predicted_normal *= jnp.where(predicted_normal[..., 2:] >= 0., 1., -1.)
        actual_normal *= jnp.where(actual_normal[..., 2:] >= 0., 1., -1.)
        predicted_v = jnp.sum(predicted_normal*pose['velocity'], axis=-1)
        actual_v = jnp.sum(actual_normal*batch['racket_linear_velocity_before_m_s'][-1], axis=-1)
        chord = jnp.linalg.norm(pose['rotation']-target_rotation, axis=(-1, -2))
        result = dict(racket_position_m=jnp.linalg.norm(pose['position']-batch['racket_site_position_before_m'][-1], axis=-1),
            normal_velocity_m_s=jnp.abs(predicted_v-actual_v),
            racket_rotation_rad=2*jnp.arcsin(jnp.clip(chord/(2*jnp.sqrt(2.)), 0., 1.)))
        for prefix, section in (('arm', slice(0, 7)), ('support', slice(7, 13))):
            result[prefix+'_q_rmse_rad'] = jnp.sqrt(jnp.mean((q[-1, :, section]-batch['q_after'][-1, :, section])**2, axis=-1))
            result[prefix+'_dq_rmse_rad_s'] = jnp.sqrt(jnp.mean((dq[-1, :, section]-batch['dq_after'][-1, :, section])**2, axis=-1))
        return result

    names = list(reader.records)
    for index in range(0, len(names), 2):
        guard.check()
        data, record = join_trial(reader, *names[index:index+2])
        for horizon in horizons:
            batch, refs = sample_support_windows(rng, data, record,
                controls=horizon, batch_size=128, split='validation')
            values = jax.device_get(errors(parameters, scale, jax.tree.map(jnp.asarray, batch)))
            for key, value in values.items():
                metrics[horizon].setdefault(key, []).append(value)
            regions[horizon].append(refs['region_id'])
            roots[horizon].update(refs['root_seed_family'])
        print(f'[VERIFY] trial={index//2+1}/{len(names)//2} horizons={horizons}', flush=True)
    report = {}
    for h in horizons:
        region_ids = np.concatenate(regions[h])
        regional = {str(int(region)): {key: summarize_errors([np.concatenate(value)[region_ids == region]])
                    for key, value in metrics[h].items()} for region in np.unique(region_ids)}
        report[str(h)] = dict(overall={key: summarize_errors(value) for key, value in metrics[h].items()},
            by_region=regional, represented_regions=len(regional), unique_root_seed_families=len(roots[h]))
    return dict(scope='full_chain_execution_with_recorded_delayed_commands', source='validation_only',
        native_kinematic_label_time='horizon_endpoint_minus_1ms', joint_label_time='horizon_endpoint',
        seed=seed, horizons=report, wall_s=time.monotonic()-started, composed_world_qualified=False)


def fit(cfg, args, guard, report):
    reader = SupportReplayReader(args.replay, guard.budget)
    if (reader.dataset.identity['control_sha256'] != config_hash(cfg['control'])
            or reader.dataset.identity['contact_sha256'] != config_hash(cfg['physics'])):
        raise ValueError('Support replay physical/control contract mismatch')
    _, chain = load_support_chain(cfg, reader.report['geometry'])
    structure = dict(widths=[47, 128, 128, 26], activation='tanh_linear',
        input='arm_q7_dq7_command_error7_dq7_ddq7_support_q6_dq6',
        output='arm_delta_q7_delta_dq7_support_delta_q6_delta_dq6',
        joint_names=list(JOINT_NAMES), normalization='train_only',
        arm_q_scale_rad=.01, arm_dq_scale_rad_s=.10, support_q_scale_rad=.001,
        support_dq_scale_rad_s=.01, racket_position_scale_m=.003,
        racket_velocity_scale_m_s=.05, racket_rotation_scale_rad=.01,
        racket_omega_scale_rad_s=.10, dt_s=.001, labels='native_pose_twist_not_target_FK')
    signature = config_hash(dict(dataset=reader.identity, structure=structure))
    report.update(model_signature=signature, dataset_sha256=reader.identity,
        dataset_identity=reader.dataset.identity,
        replay_result_sha256=file_hash(args.replay/'result.json'), structure=structure)
    metadata = None
    started = time.monotonic()
    if args.resume:
        metadata = json.loads(args.resume.read_text())
        if metadata['schema'] != 'wmj_support_robot_checkpoint_v1' or metadata['model_signature'] != signature:
            raise ValueError('Support checkpoint dataset/architecture mismatch')
        for name in ('robot_support_model.py', 'support_losses.py', 'support_data.py', 'neural.py', 'kinematics.py'):
            if metadata['source_sha256'][name] != report['source_sha256'][name]:
                raise ValueError(f'Support resume semantics changed: {name}')
        stats = metadata['normalization']
    else:
        inputs, residuals = TrainNormalizer(47), TrainNormalizer(26)
        for name, entry in reader.records.items():
            guard.check()
            support_statistics_update(inputs, residuals, reader.read(name), entry['record'])
            print(f'[NORMALIZE] {name} train_substeps={inputs.count}', flush=True)
        stats = dict(inputs=inputs.to_dict(), residuals=residuals.to_dict())
    report['normalization_wall_s'] = time.monotonic()-started
    write_json_unique(args.output_dir/'normalization.json', stats, guard.budget)
    scale = support_scale_from_statistics(stats['inputs'], stats['residuals'])
    key = jax.random.PRNGKey(args.seed)
    parameters = init_mlp(key, structure['widths'], zero_output=True)
    optimizer, rng, active_chunk = init_adam(parameters), np.random.default_rng(args.seed), None
    if metadata:
        arrays = read_array_archive(args.resume.parent/metadata['archive_file'], metadata['archive'])
        restored = restore_numeric_state(dict(parameters=parameters, optimizer=optimizer, scale=scale, rng=key),
            arrays, metadata['state'], context_sha256=signature)
        parameters, optimizer, scale, key = (restored[k] for k in ('parameters', 'optimizer', 'scale', 'rng'))
        rng.bit_generator.state, active_chunk = metadata['cpu_rng'], metadata['active_chunk']
        validate_resume_recipe(metadata['fit_recipe'], report['fit_recipe'], completed=int(optimizer.step))
        report['resume'] = dict(path=str(args.resume.absolute()), sha256=file_hash(args.resume))
    if int(optimizer.step) >= args.updates:
        raise ValueError('Final update must exceed the restored optimizer step')

    @jax.jit(static_argnames=('autoregressive',))
    def update(params, opt, scaling, batch, *, autoregressive):
        (_, values), gradients = jax.value_and_grad(support_execution_loss, has_aux=True)(
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
        name = f'robot_support_u{count:06d}'
        arrays, state = pack_numeric_state(dict(parameters=parameters, optimizer=optimizer,
            scale=scale, rng=key), context_sha256=signature)
        archive = write_array_archive_unique(args.output_dir/f'{name}.npz', arrays, guard.budget)
        payload = dict(schema='wmj_support_robot_checkpoint_v1', training_version='WMJ-V1',
            component='robot_execution_arm_and_support', candidate_only=True, model_signature=signature,
            structure=structure, state=state, archive=archive, archive_file=f'{name}.npz',
            normalization=stats, cpu_rng=rng.bit_generator.state, active_chunk=active_chunk,
            optimizer_updates=count, fit_recipe=report['fit_recipe'], source_commit=report['source_commit'],
            source_sha256=report['source_sha256'], config_sha256=config_hash(cfg),
            dataset_sha256=reader.identity, physical_contract=reader.dataset.identity,
            actor_updates=0, critic_updates=0, imagined_transitions=0,
            supervised_substep_uses=sum(controls_at(i, args.window_schedule) for i in range(count))*5*256,
            wall_s=time.monotonic()-guard.started)
        path = args.output_dir/f'{name}.json'
        saved[count] = dict(path=str(path.absolute()), sha256=write_json_unique(path, payload, guard.budget))
        print(f'[SAVE] {name}', flush=True)
        return saved[count]

    report['initial_checkpoint'] = checkpoint()
    names, data = list(reader.records), None
    started = time.monotonic()
    try:
        for iteration in range(int(optimizer.step), args.updates):
            update_started = time.monotonic()
            guard.check()
            if iteration % 32 == 0 or data is None:
                if iteration % 32 == 0 or active_chunk is None:
                    active_chunk = str(rng.choice(names))
                data = support_chunk(reader.read(active_chunk))
            batch, _ = sample_support_windows(rng, data, reader.records[active_chunk]['record'],
                controls=controls_at(iteration, args.window_schedule), batch_size=256, split='train')
            kernel_started = time.monotonic()
            candidate, next_optimizer, metrics = update(parameters, optimizer, scale,
                jax.tree.map(jnp.asarray, batch), autoregressive=iteration >= args.bootstrap_updates)
            metrics = jax.device_get(metrics)
            if not bool(metrics['finite']):
                raise FloatingPointError('Nonfinite support update; retaining last finite checkpoint')
            parameters, optimizer = candidate, next_optimizer
            if iteration % 32 == 0 or iteration+1 == args.updates:
                row = dict(update=iteration+1, autoregressive=iteration >= args.bootstrap_updates,
                    control_window=controls_at(iteration, args.window_schedule), chunk=active_chunk,
                    **{key: float(value) for key, value in metrics.items() if key != 'finite'},
                    kernel_update_s=time.monotonic()-kernel_started, full_update_s=time.monotonic()-update_started)
                report['training'].append(row)
                print(f'[FIT] u={iteration+1} loss={row["loss"]:.6g} grad={row["gradient_norm"]:.5g} '
                      f'update_s={row["full_update_s"]:.4f}', flush=True)
            if (iteration+1) % 256 == 0:
                checkpoint()
                write_json_unique(args.output_dir/f'training_u{iteration+1:06d}.json',
                                  dict(training=report['training']), guard.budget)
        report['training_wall_s'] = time.monotonic()-started
        report['final_checkpoint'] = checkpoint()
        report['validation'] = validate(parameters, scale, reader, chain, guard, seed=args.seed+10001)
        write_json_unique(args.output_dir/'validation.json', report['validation'], guard.budget)
        report['status'] = 'COMPONENT_FIT_COMPLETE_NOT_WORLD_QUALIFIED'
    finally:
        report['last_finite_checkpoint'] = checkpoint()
        report['completed_updates'] = int(optimizer.step)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--updates', type=int, default=512)
    parser.add_argument('--bootstrap-updates', type=int, default=256)
    parser.add_argument('--controls', type=int, choices=(8, 16, 32, 64), default=8)
    parser.add_argument('--window-schedule')
    parser.add_argument('--seed', type=int, default=2026110701)
    parser.add_argument('--max-wall-s', type=float, default=1800.)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(argv)
    cfg = load_config()
    if not 0 <= args.bootstrap_updates < args.updates <= cfg['world_model']['optimizer']['max_initial_updates']:
        parser.error('Require 0 <= bootstrap < total updates <= registered cap')
    args.window_schedule = parse_schedule(args.window_schedule, bootstrap=args.bootstrap_updates, initial=args.controls)
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, args.max_wall_s) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        paths = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_support_robot_fit_v1', training_version='WMJ-V1', status='INCOMPLETE',
            config=cfg, config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            absolute_output_dir=str(args.output_dir.absolute()),
            timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository_root(), text=True).strip(),
            source_sha256={p.name: file_hash(p) for p in paths},
            fit_recipe=dict(batch_windows=256, control_window=args.controls, window_schedule=args.window_schedule,
                bootstrap_updates=args.bootstrap_updates, learning_rate=3e-4, max_gradient_norm=1.,
                sampling='uniform_acquisition_then_valid_window_32_updates_per_CPU_chunk',
                supervised_input='native_delayed_commands_and_full13_q_dq',
                physics_dt_s=.001, heldout='validation_only', output_scope='full_chain_robot_component_only'),
            arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            training=[], actor_updates=0, critic_updates=0, world_model_ready=False)
        report['dependencies'] = {name: version(name) for name in ('jax', 'jaxlib', 'mujoco', 'numpy')}
        report['training_protocol_snapshot'] = (repository_root()/cfg['training_protocol']['path']).read_text()
        report['model_initialization'] = ('restore_complete_support_checkpoint' if args.resume
                                          else 'fresh47_128_128_26_zero_output_and_fresh_Adam')
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
