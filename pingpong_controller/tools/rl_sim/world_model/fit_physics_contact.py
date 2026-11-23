"""Bounded39-D contact refit on frozen execution-proxy trajectories.

Preserve contact parameters/Adam/RNG and the complete frozen robot checkpoint.
Only the recursive contact loss changes its robot conditioning; the auxiliary
single-step loss remains native. No policy, physics or feature change occurs.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import read_array_archive, write_array_archive_unique, write_json_unique
from .contact_data import contact_window_rows, take_contact_windows
from .contracts import config_hash, file_hash, load_config, repository_root
from .evaluate_physics import evaluate
from .neural import adam_update, init_adam
from .physics_components import load_physics_components
from .physics_data import read_physics_trial
from .physics_fitting import predicted_racket_motion
from .position_contact_data import prepare_position_contact_batch
from .runtime_guard import RunGuard
from .sampling import sample_model_windows
from .state_adapter import pack_numeric_state, restore_numeric_state
from .support_data import SupportReplayReader
from .model_numerics import precision_scope, validate_precision_transition
from .contact_regularization import regularized_contact_objective, validate_activation_transition


def fit(cfg, args, guard, report):
    reader = SupportReplayReader(args.support_replay, guard.budget, max_bytes=0)
    parameters, options, references = load_physics_components(cfg, reader,
        robot_checkpoint=args.robot_checkpoint, contact_checkpoint=args.contact_checkpoint)
    source = json.loads(args.contact_checkpoint.read_text())
    restored = restore_numeric_state(dict(parameters=parameters['contact'], optimizer=init_adam(parameters['contact'])),
        read_array_archive(args.contact_checkpoint.parent/source['archive_file'], source['archive']),
        source['state'], context_sha256=source['model_signature'])
    contact, optimizer = restored['parameters'], restored['optimizer']
    start = int(optimizer.step)
    if not start < args.updates <= cfg['world_model']['optimizer']['max_initial_updates']:
        raise ValueError('Requested total updates must exceed the parent and stay within the registered budget')
    recipe = dict(batch_windows=256, controls=8, learning_rate=.0003, gradient_norm_clip=1.,
        single_step_aux_weight=256., sampler='P_model_128_64_38_26', sampling_cpu_trial_hold_updates=32,
        conditioning='own_predicted_full_chain_pose_twist_and_joint_positions',
        auxiliary_conditioning='recorded_native_pose_twist_and_joint_positions',
        robot_frozen=True, frozen_robot_sha256=references['robot']['sha256'],
        seed=args.seed, validation_seed=args.seed+10001)
    precision = (source['structure'].get('composed_matmul_precision')
                 if args.matmul_precision is None else args.matmul_precision)
    if precision is not None:
        recipe['matmul_precision'] = precision
    activation_weight = (source['fit_recipe'].get('activation_penalty_weight',0.)
                         if args.activation_penalty_weight is None else args.activation_penalty_weight)
    if activation_weight:
        recipe['activation_penalty_weight'] = activation_weight
    options = dict(options, matmul_precision=precision)
    if source['schema'] == 'wmj_contact_coupled_checkpoint_v1':
        validate_activation_transition(source['fit_recipe'],recipe,start_stage=args.start_activation_stage)
        validate_precision_transition(
            {k:v for k,v in source['fit_recipe'].items() if k!='activation_penalty_weight'},
            {k:v for k,v in recipe.items() if k!='activation_penalty_weight'},start_stage=args.start_precision_stage)
        for name in ('physics_fitting.py', 'physics_data.py', 'sampling.py', 'contact_data.py'):
            if source['source_sha256'][name] != report['source_sha256'][name]:
                raise ValueError(f'Coupled fitting source changed: {name}')
        if (source['fit_recipe'].get('activation_penalty_weight',0.)
                and source['source_sha256']['contact_regularization.py'] != report['source_sha256']['contact_regularization.py']):
            raise ValueError('Active regularization fitting source changed')
        stage_start = source['coupled_stage_start_updates']
    else:
        if args.start_precision_stage or precision is not None or args.start_activation_stage or activation_weight:
            raise ValueError('Precision/activation refitting requires an existing coupled checkpoint')
        previous = source['fit_recipe']
        if (previous['controls'] != 8 or previous.get('single_step_aux_weight') != 256.
                or previous['learning_rate'] != .0003
                or 'window_schedule' in previous or 'learning_rate_schedule' in previous):
            raise ValueError('Initial paired branch requires the unchanged auxiliary H8/3e-4 parent')
        stage_start = start
    structure = dict(source['structure'], conditioning=recipe['conditioning'])
    if precision is not None:
        structure['composed_matmul_precision'] = precision
    signature = config_hash(dict(structure=structure, dataset=reader.identity,
        physics=cfg['physics'], geometry=reader.report['geometry']))
    rng = np.random.default_rng(args.seed)
    rng.bit_generator.state = source['cpu_rng']
    active_pair = source['active_pair']
    report.update(components=references, dataset_sha256=reader.identity, model_signature=signature,
        fit_recipe=recipe, starting_updates=start, coupled_stage_start_updates=stage_start,
        precision_stage_start_updates=(start if args.start_precision_stage
                                       else source.get('precision_stage_start_updates')),
        activation_stage_start_updates=(start if args.start_activation_stage
                                        else source.get('activation_stage_start_updates')))

    def update_unscoped(params, opt, raw):
        batch = prepare_position_contact_batch(None, raw)
        motion = predicted_racket_motion(parameters['robot'], raw,
            scale=options['robot_scale'], chain=options['chain'])
        (_, losses), gradient = jax.value_and_grad(regularized_contact_objective, has_aux=True)(
            params, options['surface'], batch, motion, ball_radius_m=options['ball_radius_m'],
            position_scale=options['position_scale'], contact_contract=options['contact_contract'],
            auxiliary_weight=recipe['single_step_aux_weight'],activation_penalty_weight=activation_weight)
        after, next_opt, diagnostics = adam_update(params, gradient, opt,
            learning_rate=recipe['learning_rate'], max_gradient_norm=recipe['gradient_norm_clip'])
        finite = jnp.isfinite(losses['loss']) & jnp.isfinite(diagnostics['gradient_norm'])
        finite &= jnp.all(jnp.stack([jnp.isfinite(value).all() for value in jax.tree.leaves((after, next_opt))]))
        return after, next_opt, dict(losses, **diagnostics, finite=finite)

    @jax.jit
    def update(params, opt, raw):
        with precision_scope(precision):
            return update_unscoped(params,opt,raw)

    saved = {}

    def checkpoint():
        count = int(optimizer.step)
        if count in saved:
            return saved[count]
        name = f'contact_coupled_u{count:06d}'
        arrays, state = pack_numeric_state(dict(parameters=contact, optimizer=optimizer), context_sha256=signature)
        archive = write_array_archive_unique(args.output_dir/f'{name}.npz', arrays, guard.budget)
        payload = dict(schema='wmj_contact_coupled_checkpoint_v1', training_version='WMJ-V1',
            candidate_only=True, model_signature=signature, structure=structure, state=state,
            archive=archive, archive_file=f'{name}.npz', optimizer_updates=count,
            cpu_rng=rng.bit_generator.state, active_pair=active_pair, fit_recipe=recipe,
            coupled_stage_start_updates=stage_start,
            precision_stage_start_updates=report['precision_stage_start_updates'],
            activation_stage_start_updates=report['activation_stage_start_updates'],
            frozen_robot_reference=references['robot'], parent_contact_reference=references['contact'],
            source_sha256=report['source_sha256'], source_commit=report['source_commit'],
            physical_contract=reader.dataset.identity, config_sha256=config_hash(cfg),
            dataset_sha256=reader.identity, actor_updates=0, critic_updates=0,
            supervised_substep_uses=source['supervised_substep_uses']+(count-start)*256*40,
            auxiliary_supervised_substep_uses=source.get('auxiliary_supervised_substep_uses', 0)+(count-start)*256*40)
        path = args.output_dir/f'{name}.json'
        digest = write_json_unique(path, payload, guard.budget)
        saved[count] = dict(path=str(path.absolute()), sha256=digest)
        print(f'[CHECKPOINT] {name}', flush=True)
        return saved[count]

    report['initial_checkpoint'] = checkpoint()
    names = list(reader.records)
    data = rows = None
    started = time.monotonic()
    try:
        for iteration in range(start, args.updates):
            tick = time.monotonic()
            guard.check()
            if iteration % 32 == 0 or data is None:
                if iteration % 32 == 0 or active_pair is None:
                    active_pair = int(rng.integers(len(names)//2))
                data, record = read_physics_trial(reader, names[2*active_pair], names[2*active_pair+1])
                rows = contact_window_rows(data, record, controls=8, split='train')
            draw = sample_model_windows(rng, rows)
            raw, _ = take_contact_windows(data, rows, draw.indices, controls=8)
            candidate, next_optimizer, metrics = update(contact, optimizer, jax.tree.map(jnp.asarray, raw))
            metrics = jax.device_get(metrics)
            if not bool(metrics['finite']):
                raise FloatingPointError('Nonfinite coupled contact update; retaining last finite model')
            contact, optimizer = candidate, next_optimizer
            if iteration % 32 == 0 or iteration+1 == args.updates:
                row = dict(update=iteration+1,
                    **{key:float(value) for key,value in metrics.items() if key != 'finite'},
                    full_update_s=time.monotonic()-tick, sampling=draw.report, trial_pair=active_pair)
                report['training'].append(row)
                print(f'[FIT] u={iteration+1} loss={row["loss"]:.6g} grad={row["gradient_norm"]:.6g}', flush=True)
            if (iteration+1) % 256 == 0:
                checkpoint()
                write_json_unique(args.output_dir/f'training_u{iteration+1:06d}.json',
                                  dict(training=report['training']), guard.budget)
        report['training_wall_s'] = time.monotonic()-started
        report['final_checkpoint'] = checkpoint()
        del data, rows, raw
        validation = evaluate(reader, dict(parameters, contact=contact), options, guard, seed=recipe['validation_seed'])
        write_json_unique(args.output_dir/'validation.json', validation, guard.budget)
        report.update(validation=validation, status='COUPLED_CONTACT_FIT_COMPLETE_NOT_WORLD_QUALIFIED')
    finally:
        report['last_finite_checkpoint'] = checkpoint()
        report['completed_updates'] = int(optimizer.step)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support-replay', type=Path, required=True)
    parser.add_argument('--robot-checkpoint', type=Path, required=True)
    parser.add_argument('--contact-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--updates', type=int, default=9216)
    parser.add_argument('--seed', type=int, default=2026110606)
    parser.add_argument('--max-wall-s', type=float, default=1800.)
    parser.add_argument('--matmul-precision', choices=('highest',))
    parser.add_argument('--start-precision-stage', action='store_true',
        help='Explicitly branch an existing coupled checkpoint into full-FP32 fitting')
    parser.add_argument('--activation-penalty-weight',type=float,choices=(0.,.01))
    parser.add_argument('--start-activation-stage',action='store_true',
        help='Explicitly branch a coupled checkpoint into the registered activation penalty stage')
    args = parser.parse_args(argv)
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, args.max_wall_s) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        paths = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_coupled_contact_fit_v1', training_version='WMJ-V1', status='INCOMPLETE',
            config=cfg, config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            absolute_output_dir=str(args.output_dir.absolute()),
            timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository_root(), text=True).strip(),
            source_sha256={path.name:file_hash(path) for path in paths}, invocation=sys.argv,
            arguments={key:str(value) if isinstance(value, Path) else value for key,value in vars(args).items()},
            training=[], actor_updates=0, critic_updates=0, world_model_ready=False)
        write_json_unique(args.output_dir/'manifest.json',
            {**report, 'source_snapshot':{path.name:path.read_text() for path in paths}}, guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA GPU is required')
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
