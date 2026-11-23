"""Fit a finite-duration contact module on actual MJX trajectories.

Recorded robot FK conditions this component experiment. It neither qualifies
the complete world nor supplies any actor/critic training samples.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from functools import partial
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
from .contact_data import (contact_window_rows, prepare_contact_batch,
                           read_contact_trial, take_contact_windows)
from .contact_dynamics import contact_rollout
from .contact_losses import ballistic_apex, contact_module_loss
from .contact_surface import surface_from_vertices
from .contact_geometry import surface_from_compiled_model
from .contact_checkpoint import legacy_geometry_dispatch_compatible
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .fit_robot import dataset_identity, load_chain, summarize_errors
from .neural import adam_update, init_adam, init_mlp
from .runtime_guard import RunGuard
from .sampling import sample_model_windows
from .state_adapter import pack_numeric_state, restore_numeric_state
from .trajectory_data import TrajectoryDataset
from .native_contact_data import prepare_native_contact_batch, read_native_contact_trial
from .support_data import SupportReplayReader
from .soft_contact import contract_from_collection
from .hybrid_contact import hybrid_contact_loss, hybrid_contact_rollout
from .position_contact import (position_scale_from_statistics, position_contact_loss,
                               position_contact_rollout)
from .position_contact_data import (read_position_contact_trial, prepare_position_contact_batch,
                                    collect_position_statistics)
from .contact_fitting import make_contact_objective
from .contact_schedule import (contact_schedule, contact_substep_uses,
                               validate_contact_training_recipe)
from .fit_schedule import controls_at, parse_schedule
from .contact_rates import parse_rates, rate_schedule, rate_at, validate_rate_resume


def validate_contact(parameters, surface, radius, chain, dataset, guard, *, seed,
                     read_trial=read_contact_trial, prepare=prepare_contact_batch,
                     conditioning='recorded_robot_FK', rollout=contact_rollout):
    """Evaluate independent held-out events and condition-specific errors."""
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    horizons = (1, 8, 32, 64, 128)
    names = list(dataset.chunks)
    measures = {h: {} for h in horizons}
    event_counts = {h: dict(entry_missed=0, entry_extra=0, release_missed=0,
                            release_extra=0, true_entries=0, true_releases=0) for h in horizons}
    population = {h: {} for h in horizons}

    @jax.jit
    def predict(raw):
        batch = prepare(chain, raw)
        initial = jax.tree.map(lambda x: x[0], batch['before'])
        _, trace = rollout(parameters, surface, initial, batch['racket'], ball_radius_m=radius)
        before, target = trace['before'], batch['before']
        return dict(
            endpoint_position_m=jnp.linalg.norm(trace['after'].position_m[-1]-batch['after'].position_m[-1], axis=-1),
            endpoint_velocity_m_s=jnp.linalg.norm(trace['after'].velocity_m_s[-1]-batch['after'].velocity_m_s[-1], axis=-1),
            outgoing_velocity_m_s=jnp.linalg.norm(before.velocity_m_s-target.velocity_m_s, axis=-1),
            next_apex_m=jnp.linalg.norm(ballistic_apex(before.position_m, before.velocity_m_s)
                                      -ballistic_apex(target.position_m, target.velocity_m_s), axis=-1),
            outgoing_mask=batch['valid'] & batch['controlled_release'],
            entry_missed=jnp.sum(batch['entry'] & ~trace['entry']),
            entry_extra=jnp.sum(trace['entry'] & ~batch['entry']),
            release_missed=jnp.sum(batch['release'] & ~trace['release']),
            release_extra=jnp.sum(trace['release'] & ~batch['release']),
            true_entries=jnp.sum(batch['entry']), true_releases=jnp.sum(batch['release']))

    for i in range(0, len(names), 2):
        guard.check()
        data, record = read_trial(dataset, names[i], names[i+1])
        for horizon in horizons:
            guard.check()
            rows = contact_window_rows(data, record, controls=horizon, split='validation')
            if not rows:
                raise ValueError(f'No valid validation windows for horizon {horizon}')
            # Uniform independent event/acquisition, then uniform window.
            independent = {}
            for index, row in enumerate(rows):
                independent.setdefault(row['independent_event_id'], []).append(index)
            groups = list(independent.values())
            selected = [int(rng.choice(groups[int(i)])) for i in rng.integers(len(groups), size=128)]
            raw, refs = take_contact_windows(data, rows, selected, controls=horizon)
            result = jax.device_get(predict(jax.tree.map(jnp.asarray, raw)))
            regions = np.asarray([row['region_id'] for row in refs])
            for bucket in set(row['bucket'] for row in refs):
                population[horizon][bucket] = population[horizon].get(bucket, 0)+sum(row['bucket'] == bucket for row in refs)
            for key in event_counts[horizon]:
                event_counts[horizon][key] += int(result[key])
            for metric in ('endpoint_position_m', 'endpoint_velocity_m_s', 'outgoing_velocity_m_s', 'next_apex_m'):
                values = result[metric]
                if values.ndim == 2:
                    region_values = np.broadcast_to(regions, values.shape)[result['outgoing_mask']]
                    values = values[result['outgoing_mask']]
                else:
                    region_values = regions
                entry = measures[horizon].setdefault(metric, dict(values=[], regions=[]))
                entry['values'].append(values)
                entry['regions'].append(region_values)
        del data
    report = {}
    for horizon, metrics in measures.items():
        overall, regional = {}, {}
        for metric, entries in metrics.items():
            values, regions = np.concatenate(entries['values']), np.concatenate(entries['regions'])
            overall[metric] = summarize_errors([values])
            for region in np.unique(regions):
                regional.setdefault(str(int(region)), {})[metric] = summarize_errors([values[regions == region]])
        report[str(horizon)] = dict(overall=overall, by_region=regional,
            sampled_windows_by_bucket=population[horizon], event_counts=event_counts[horizon])
    return dict(scope=f'contact_module_with_{conditioning}_not_composed_world',
        horizons=report, source='validation_only', seed=seed, wall_s=time.monotonic()-started,
        edge_comparison='exact_1ms_labels; timing shifts count as missed_and_extra',
        apex_label='analytic_no_drag_apex_from_observed_controlled_release',
        world_model_ready=False)


def fit(cfg, args, guard, report):
    dataset = TrajectoryDataset.open(args.dataset, guard.budget)
    read_trial, prepare = read_contact_trial, prepare_contact_batch
    support_reader = None
    conditioning, data_sha = 'recorded_robot_FK', dataset_identity(dataset)
    if args.support_replay:
        support_reader = SupportReplayReader(args.support_replay, guard.budget)
        if support_reader.dataset.root != dataset.root:
            raise ValueError('Support replay and requested contact dataset disagree')
        read_trial = lambda unused_dataset, a, b: read_native_contact_trial(support_reader, a, b)
        prepare = prepare_native_contact_batch
        conditioning, data_sha = 'recorded_native_pose_twist', support_reader.identity
        report['support_replay'] = dict(path=str(args.support_replay.absolute()),
            result_sha256=file_hash(args.support_replay/'result.json'))
    position_mode = args.contact_inputs == 'joint_positions'
    if position_mode:
        if support_reader is None or args.contact_geometry != 'v324_mjx' or args.contact_response == 'soft_only':
            raise ValueError('39-D comparison requires native coherent replay and a learned contact response')
        read_trial = lambda unused_dataset, a, b: read_position_contact_trial(support_reader, a, b)
        prepare = prepare_position_contact_batch
        conditioning = 'recorded_native_pose_twist_and_joint_positions'
    if (dataset.identity['control_sha256'] != config_hash(cfg['control'])
            or dataset.identity['contact_sha256'] != config_hash(cfg['physics'])):
        raise ValueError('Contact data must preserve the native control and physical contract')
    names = list(dataset.chunks)
    if len(names) % 2:
        raise ValueError('Contact fitting requires complete two-chunk trials')
    collection = json.loads((args.dataset.parent/'result.json').read_text())
    chain = load_chain(cfg, collection) if support_reader is None else None
    surface = surface_from_vertices(collection['geometry']['racket']['compiled_vertices_local_m'], normal_axis=0)
    radius = float(collection['geometry']['ball']['radius_m'])
    structure = dict(widths=[39 if position_mode else 19, *cfg['world_model']['contact']['hidden_sizes'], 8],
        activation='tanh_linear', output='contact_dv_domega_entry_release', dt_s=.001,
        input_normalization='fixed_SI_scales', conditioning=conditioning)
    if args.contact_geometry == 'v324_mjx':
        import mujoco
        model = mujoco.MjModel.from_binary_path(str(authenticate(
            repository_root(), cfg['collection']['compiled_model_cache'])))
        surface = surface_from_compiled_model(model, collection['geometry']['racket']['geom_id'], normal_axis=0)
        structure['collision_geometry'] = surface.identity
        del model
    prior = None
    loss_function, rollout = contact_module_loss, contact_rollout
    if args.contact_response != 'neural':
        if args.contact_geometry != 'v324_mjx' or support_reader is None:
            raise ValueError('Soft prior comparisons require native geometry and coherent pose/twist data')
        prior, prior_context = contract_from_collection(cfg, collection)
        structure['response'] = dict(kind=args.contact_response, contract=asdict(prior), context=prior_context)
        loss_function = partial(hybrid_contact_loss, contract=prior)
        rollout = partial(hybrid_contact_rollout, contract=prior, learned=args.contact_response == 'soft_residual')
    if position_mode:
        started = time.monotonic()
        if args.resume:
            previous = json.loads(args.resume.read_text())
            if previous['dataset_sha256'] != data_sha or 'position_conditioning' not in previous['structure']:
                raise ValueError('Position normalization requires a matching39-D dataset/checkpoint')
            stats = previous['structure']['position_conditioning']['statistics']
        else:
            stats = collect_position_statistics(support_reader, guard)
        scale = position_scale_from_statistics(stats)
        report['normalization_wall_s'] = time.monotonic()-started
        write_json_unique(args.output_dir/'position_normalization.json', stats, guard.budget)
        structure['position_conditioning'] = dict(mode='joint_positions', appended_dim=20,
            fields=['RightArm_q7', 'LegWaist_q6', 'delayed_RightArm_q_command_minus_q7'],
            input_dq=False, input_ddq=False, statistics=stats, standard_deviation_floor_rad=1e-4)
        loss_function = partial(position_contact_loss, position_scale=scale, contract=prior)
        rollout = partial(position_contact_rollout, position_scale=scale, contract=prior)
    loss_function = make_contact_objective(loss_function,
                                          single_step_weight=args.single_step_aux_weight)
    signature = config_hash(dict(structure=structure, dataset=data_sha,
                                physics=cfg['physics'], geometry=collection['geometry']))
    report.update(model_signature=signature, dataset_sha256=data_sha)
    key = jax.random.PRNGKey(args.seed)
    parameters = init_mlp(key, structure['widths'], zero_output=True)
    optimizer = init_adam(parameters)
    rng = np.random.default_rng(args.seed)
    active_pair = None
    if args.resume:
        metadata = json.loads(args.resume.read_text())
        if metadata['schema'] != 'wmj_contact_component_checkpoint_v1' or metadata['model_signature'] != signature:
            raise ValueError('Resume contact dataset/model signature changed')
        validate_contact_training_recipe(
            {k: v for k, v in metadata['fit_recipe'].items() if k != 'learning_rate_schedule'},
            {k: v for k, v in report['fit_recipe'].items() if k != 'learning_rate_schedule'},
            completed=metadata['optimizer_updates'])
        validate_rate_resume(metadata['fit_recipe'], report['fit_recipe'],
                             completed=metadata['optimizer_updates'])
        if metadata['fit_recipe'] != report['fit_recipe']:
            report['resume_fit_recipe_change'] = dict(
                scope='only_future_auxiliary_loss_window_or_learning_rate_changes',
                completed_updates=metadata['optimizer_updates'],
                previous=metadata['fit_recipe'], current=report['fit_recipe'])
        core_sources = ('contact_model.py', 'contact_surface.py', 'contact_data.py',
                        'contact_dynamics.py', 'contact_losses.py', 'neural.py', 'kinematics.py', 'sampling.py')
        if support_reader is not None:
            core_sources += ('native_contact_data.py', 'support_data.py', 'replay_overlay.py')
        if 'contact_geometry.py' in metadata['source_sha256']:
            core_sources += ('contact_geometry.py',)
        if args.contact_response != 'neural':
            core_sources += ('soft_contact.py', 'hybrid_contact.py')
        if position_mode:
            core_sources += ('position_contact.py', 'position_contact_data.py',
                             'hybrid_contact.py', 'trajectory_data.py')
        if metadata['fit_recipe'].get('single_step_aux_weight', 0.) > 0:
            core_sources += ('contact_fitting.py',)
        if 'window_schedule' in metadata['fit_recipe']:
            core_sources += ('contact_schedule.py', 'fit_schedule.py')
        if 'learning_rate_schedule' in metadata['fit_recipe']:
            core_sources += ('contact_rates.py',)
        for name in core_sources:
            if metadata['source_sha256'][name] != report['source_sha256'][name]:
                if (name == 'contact_model.py' and args.contact_geometry == 'exact_prism'
                        and legacy_geometry_dispatch_compatible(metadata['source_sha256'], report['source_sha256'])):
                    report['resume_source_compatibility'] = 'audited_exact_prism_dispatch_only'
                    continue
                raise ValueError(f'Resume contact semantics changed: {name}')
        arrays = read_array_archive(args.resume.parent/metadata['archive_file'], metadata['archive'])
        restored = restore_numeric_state(dict(parameters=parameters, optimizer=optimizer), arrays,
                                          metadata['state'], context_sha256=signature)
        parameters, optimizer = restored['parameters'], restored['optimizer']
        rng.bit_generator.state = metadata['cpu_rng']
        active_pair = metadata['active_pair']
        report['resume'] = dict(path=str(args.resume), sha256=file_hash(args.resume))
    start = int(optimizer.step)
    if start >= args.updates and args.contact_response != 'soft_only':
        raise ValueError('Total updates must exceed restored optimizer step')

    @jax.jit(static_argnames=('autoregressive',))
    def update(params, opt, raw, learning_rate, *, autoregressive):
        batch = prepare(chain, raw)
        (_, losses), gradient = jax.value_and_grad(loss_function, has_aux=True)(
            params, surface, batch, ball_radius_m=radius, autoregressive=autoregressive)
        after, next_opt, diagnostics = adam_update(params, gradient, opt,
            learning_rate=(cfg['world_model']['optimizer']['lr']
                           if args.learning_rate_schedule is None else learning_rate),
            max_gradient_norm=1.)
        finite = jnp.isfinite(losses['loss']) & jnp.isfinite(diagnostics['gradient_norm'])
        finite &= jnp.all(jnp.stack([jnp.isfinite(x).all() for x in jax.tree.leaves(after)]))
        return after, next_opt, {**losses, **diagnostics, 'finite': finite}

    saved = {}
    window_schedule = contact_schedule(report['fit_recipe'])
    learning_rates = rate_schedule(report['fit_recipe'])

    def checkpoint():
        count = int(optimizer.step)
        if count in saved:
            return saved[count]
        name = f'contact_u{count:06d}'
        arrays, state = pack_numeric_state(dict(parameters=parameters, optimizer=optimizer), context_sha256=signature)
        archive = write_array_archive_unique(args.output_dir/f'{name}.npz', arrays, guard.budget)
        payload = dict(schema='wmj_contact_component_checkpoint_v1', training_version='WMJ-V1',
            component='finite_duration_contact', candidate_only=True, model_signature=signature,
            structure=structure, state=state, archive=archive, archive_file=f'{name}.npz',
            optimizer_updates=count, cpu_rng=rng.bit_generator.state, active_pair=active_pair,
            fit_recipe=report['fit_recipe'], source_sha256=report['source_sha256'],
            source_commit=report['source_commit'], physical_contract=dataset.identity,
            config_sha256=config_hash(cfg), dataset_sha256=report['dataset_sha256'],
            actor_updates=0, critic_updates=0,
            supervised_substep_uses=contact_substep_uses(report['fit_recipe'], completed=count)[0])
        if args.single_step_aux_weight > 0:
            payload['auxiliary_supervised_substep_uses'] = contact_substep_uses(
                report['fit_recipe'], completed=count)[1]
        path = args.output_dir/f'{name}.json'
        digest = write_json_unique(path, payload, guard.budget)
        saved[count] = dict(path=str(path.absolute()), sha256=digest)
        print(f'[CHECKPOINT] {name}', flush=True)
        return saved[count]

    report['initial_checkpoint'] = checkpoint()
    data = rows = None
    row_controls = None
    started = time.monotonic()
    try:
        for iteration in range(start, args.updates):
            iteration_started = time.monotonic()
            guard.check()
            controls = controls_at(iteration, window_schedule)
            if iteration % 32 == 0 or data is None:
                if iteration % 32 == 0 or active_pair is None:
                    active_pair = int(rng.integers(len(names)//2))
                data, record = read_trial(dataset, names[2*active_pair], names[2*active_pair+1])
                row_controls = None
            if row_controls != controls:
                rows = contact_window_rows(data, record, controls=controls, split='train')
                row_controls = controls
            draw = sample_model_windows(rng, rows)
            raw, _ = take_contact_windows(data, rows, draw.indices, controls=controls)
            stamp = time.monotonic()
            learning_rate = rate_at(iteration, learning_rates)
            candidate, next_optimizer, metrics = update(parameters, optimizer,
                jax.tree.map(jnp.asarray, raw), jnp.asarray(learning_rate, jnp.float32),
                autoregressive=iteration >= args.bootstrap_updates)
            metrics = jax.device_get(metrics)
            if not bool(metrics['finite']):
                raise FloatingPointError('Nonfinite contact update; retaining last finite checkpoint')
            parameters, optimizer = candidate, next_optimizer
            if iteration % 32 == 0 or iteration+1 == args.updates:
                row = dict(update=iteration+1, phase='autoregressive' if iteration >= args.bootstrap_updates else 'single_step_bootstrap',
                    **{k: float(v) for k, v in metrics.items() if k != 'finite'},
                    kernel_update_s=time.monotonic()-stamp,
                    full_update_s=time.monotonic()-iteration_started,
                    sampling=draw.report, trial_pair=active_pair)
                if args.window_schedule is not None:
                    row['controls'] = controls
                if args.learning_rate_schedule is not None:
                    row['learning_rate'] = learning_rate
                report['training'].append(row)
                print(f'[FIT] u={iteration+1} phase={row["phase"]} loss={row["loss"]:.6g} '
                      f'grad={row["gradient_norm"]:.6g} update_s={row["full_update_s"]:.4f}', flush=True)
            if (iteration+1) % 256 == 0:
                checkpoint()
                write_json_unique(args.output_dir/f'training_u{iteration+1:06d}.json',
                                  dict(training=report['training']), guard.budget)
        report['training_wall_s'] = time.monotonic()-started
        report['final_checkpoint'] = checkpoint()
        del data, rows
        validation = validate_contact(parameters, surface, radius, chain, dataset, guard, seed=args.seed+10001,
            read_trial=read_trial, prepare=prepare, conditioning=conditioning, rollout=rollout)
        write_json_unique(args.output_dir/'validation.json', validation, guard.budget)
        report.update(validation=validation, status='COMPONENT_FIT_COMPLETE_NOT_WORLD_QUALIFIED')
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
    parser.add_argument('--window-schedule',
                        help='Optional future stages, e.g.2048:8,8192:16; past updates remain fixed')
    parser.add_argument('--learning-rate-schedule',
                        help='Optional update:rate stages; consumed optimizer history is immutable')
    parser.add_argument('--seed', type=int, default=2026110606)
    parser.add_argument('--max-wall-s', type=float, default=1800.)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--single-step-aux-weight', type=float, default=0.,
                        help='Additional teacher-forced loss during recursive fitting; default off')
    parser.add_argument('--support-replay', type=Path,
                        help='Opt in to coherent replay labels with native pose/twist conditioning')
    parser.add_argument('--contact-geometry', choices=('exact_prism', 'v324_mjx'), default='exact_prism',
                        help='Native MJX geometry is opt-in; preserve the earlier Euclidean diagnostic')
    parser.add_argument('--contact-response', choices=('neural', 'soft_only', 'soft_residual'), default='neural',
                        help='Approved comparison; soft_only evaluates without optimizer updates')
    parser.add_argument('--contact-inputs', choices=('pose_twist', 'joint_positions'), default='pose_twist',
                        help='Opt-in39-D mode appends q13/position-error7 without additional dq/ddq')
    args = parser.parse_args(argv)
    cfg = load_config()
    if not np.isfinite(args.single_step_aux_weight) or args.single_step_aux_weight < 0:
        parser.error('single-step-aux-weight must be finite and nonnegative')
    if args.window_schedule is not None:
        try:
            args.window_schedule = parse_schedule(args.window_schedule,
                bootstrap=args.bootstrap_updates, initial=args.controls)
        except ValueError as exc:
            parser.error(str(exc))
    if args.learning_rate_schedule is not None:
        try:
            args.learning_rate_schedule = parse_rates(args.learning_rate_schedule,
                initial=cfg['world_model']['optimizer']['lr'])
        except ValueError as exc:
            parser.error(str(exc))
    if args.contact_inputs == 'joint_positions' and (
            not args.support_replay or args.contact_geometry != 'v324_mjx' or args.contact_response == 'soft_only'):
        parser.error('joint_positions requires coherent --support-replay, v324_mjx geometry and a learned response')
    if args.contact_response == 'soft_only':
        if (args.updates != 0 or args.bootstrap_updates != 0 or args.resume is not None
                or args.single_step_aux_weight != 0):
            parser.error('soft_only requires --updates 0 --bootstrap-updates 0 and no resume/auxiliary loss')
    elif not 0 <= args.bootstrap_updates < args.updates <= cfg['world_model']['optimizer']['max_initial_updates']:
        parser.error('Require 0 <= bootstrap < total <= registered update budget')
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, args.max_wall_s) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        paths = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_contact_fit_v1', training_version='WMJ-V1', status='INCOMPLETE',
            config=cfg, config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            absolute_output_dir=str(args.output_dir.absolute()),
            timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository_root(), text=True).strip(),
            source_sha256={p.name: file_hash(p) for p in paths},
            fit_recipe=dict(batch_windows=256, controls=args.controls, bootstrap_updates=args.bootstrap_updates,
                learning_rate=cfg['world_model']['optimizer']['lr'], gradient_norm_clip=1., sampler='P_model_128_64_38_26',
                sampling_cpu_trial_hold_updates=32,
                conditioning='recorded_native_pose_twist' if args.support_replay else 'recorded_robot_FK',
                loss_scales=dict(position_m=.003, velocity_m_s=.08, spin_rad_s=30., apex_m=.01)),
            arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            training=[], actor_updates=0, critic_updates=0, world_model_ready=False)
        if args.contact_response != 'neural':
            report['fit_recipe']['contact_response'] = args.contact_response
        if args.contact_inputs == 'joint_positions':
            report['fit_recipe'].update(contact_inputs='joint_positions',
                conditioning='recorded_native_pose_twist_and_joint_positions')
        if args.single_step_aux_weight > 0:
            report['fit_recipe']['single_step_aux_weight'] = args.single_step_aux_weight
        if args.window_schedule is not None:
            report['fit_recipe']['window_schedule'] = args.window_schedule
        if args.learning_rate_schedule is not None:
            report['fit_recipe']['learning_rate_schedule'] = args.learning_rate_schedule
        report['dependencies'] = {name: version(name) for name in ('jax', 'jaxlib', 'mujoco', 'numpy')}
        report['training_protocol_snapshot'] = (repository_root()/cfg['training_protocol']['path']).read_text()
        write_json_unique(args.output_dir/'manifest.json',
            {**report, 'source_snapshot': {p.name: p.read_text() for p in paths}}, guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA GPU is required')
            report['devices'] = [str(d) for d in jax.devices()]
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
