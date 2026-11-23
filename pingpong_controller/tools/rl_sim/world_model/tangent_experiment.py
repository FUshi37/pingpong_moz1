"""Authorized full-prior A/B: frozen base checks, then matched39-D refits."""

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
from .model_numerics import precision_scope
from .neural import adam_update, init_adam
from .tangent_checkpoint import SCHEMA, load_components, model_signature
from .tangent_coupling import make_predictor, tangent_coupled_objective
from .physics_data import read_physics_trial
from .runtime_guard import RunGuard
from .sampling import sample_model_windows
from .state_adapter import pack_numeric_state, restore_numeric_state
from .support_data import SupportReplayReader


def validate(reader, parameters, options, auxiliary, guard, *, enabled, learned):
    predictor = make_predictor(parameters, options, auxiliary, tangent_enabled=enabled, learned=learned)
    return evaluate(reader, parameters, options, guard, seed=2026120607, predictor=predictor)


def fit(cfg, args, guard, report, reader, parameters, options, references, auxiliary, aux_ref):
    if not args.tangent_prior:
        raise ValueError('This experiment fits only the explicitly authorized full-prior branch')
    source = json.loads(args.contact_checkpoint.read_text())
    if source['schema'] != 'wmj_contact_coupled_checkpoint_v1':
        raise ValueError('This paired experiment branches the registered original parent')
    plan = cfg['world_model']['solver_acceleration_auxiliary']['tangent_integration_comparison']
    if file_hash(args.contact_checkpoint) != plan['parent_checkpoint_sha256']:
        raise ValueError('Matched full-prior comparison requires its pinned14336 parent')
    if not 1 <= args.additional_updates <= 4096:
        raise ValueError('Each authorized paired refit is bounded to4096 additional updates')
    recipe = source['fit_recipe']
    if (recipe.get('activation_penalty_weight', 0.) != 0. or recipe.get('matmul_precision') != 'highest'
            or recipe['controls'] != 8 or recipe['single_step_aux_weight'] != 256.
            or recipe['seed'] != 2026110606 or recipe['validation_seed'] != 2026120607):
        raise ValueError('Normal comparison cannot change unrelated fitting semantics')
    restored = restore_numeric_state(dict(parameters=parameters['contact'], optimizer=init_adam(parameters['contact'])),
        read_array_archive(args.contact_checkpoint.parent/source['archive_file'], source['archive']),
        source['state'], context_sha256=source['model_signature'])
    contact, optimizer = restored['parameters'], restored['optimizer']
    start, finish = int(optimizer.step), int(optimizer.step)+args.additional_updates
    if finish > cfg['world_model']['optimizer']['max_initial_updates']:
        raise ValueError('Refit exceeds the unchanged20000-update initial-model cap')
    rng = np.random.default_rng(recipe['seed'])
    rng.bit_generator.state = source['cpu_rng']
    active_pair = source['active_pair']
    structure = dict(source['structure'], tangent_auxiliary=dict(
        enabled=args.tangent_prior, scope='normal_and_tangent_prior_not_contact_neural_features', reference=aux_ref))
    signature = model_signature(cfg, reader, structure)
    report.update(starting_updates=start, target_updates=finish, fit_recipe=recipe,
        model_signature=signature, tangent_prior=args.tangent_prior,
        frozen_auxiliary_reference=aux_ref, training=[])

    @jax.jit
    def update(params, opt, raw):
        with precision_scope('highest'):
            (_, losses), gradient = jax.value_and_grad(tangent_coupled_objective, has_aux=True)(
                params, parameters['robot'], auxiliary, raw, options=options, tangent_enabled=args.tangent_prior)
            after, next_opt, diagnostics = adam_update(params, gradient, opt,
                learning_rate=recipe['learning_rate'], max_gradient_norm=recipe['gradient_norm_clip'])
            finite = jnp.isfinite(losses['loss']) & jnp.isfinite(diagnostics['gradient_norm'])
            finite &= jnp.all(jnp.stack([jnp.isfinite(v).all() for v in jax.tree.leaves((after, next_opt))]))
            return after, next_opt, dict(losses, **diagnostics, finite=finite)

    saved = {}

    def checkpoint():
        count = int(optimizer.step)
        if count in saved:
            return saved[count]
        name = f'contact_tangent_u{count:06d}'
        arrays, state = pack_numeric_state(dict(parameters=contact, optimizer=optimizer), context_sha256=signature)
        archive = write_array_archive_unique(args.output_dir/f'{name}.npz', arrays, guard.budget)
        metadata = dict(schema=SCHEMA, training_version='WMJ-V1', candidate_only=True,
            model_signature=signature, structure=structure, state=state,
            archive=archive, archive_file=f'{name}.npz', optimizer_updates=count,
            cpu_rng=rng.bit_generator.state, active_pair=active_pair, fit_recipe=recipe,
            coupled_stage_start_updates=source['coupled_stage_start_updates'],
            tangent_stage_start_updates=start, frozen_robot_reference=references['robot'],
            frozen_auxiliary_reference=aux_ref, parent_contact_reference=references['contact'],
            source_sha256=report['source_sha256'], source_commit=report['source_commit'],
            physical_contract=reader.dataset.identity, dataset_sha256=reader.identity,
            config_sha256=config_hash(cfg), actor_updates=0, critic_updates=0,
            supervised_substep_uses=source['supervised_substep_uses']+(count-start)*256*40,
            auxiliary_supervised_substep_uses=source['auxiliary_supervised_substep_uses']+(count-start)*256*40)
        path = args.output_dir/f'{name}.json'
        digest = write_json_unique(path, metadata, guard.budget)
        saved[count] = dict(path=str(path), sha256=digest)
        print(f'[CHECKPOINT] {name}', flush=True)
        return saved[count]

    report['initial_checkpoint'] = checkpoint()
    names = list(reader.records)
    data = rows = None
    started = time.monotonic()
    try:
        for iteration in range(start, finish):
            guard.check()
            tick = time.monotonic()
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
                raise FloatingPointError('Nonfinite full-prior refit; last finite model preserved')
            contact, optimizer = candidate, next_optimizer
            if iteration % 32 == 0 or iteration+1 == finish:
                row = dict(update=iteration+1, **{k: float(v) for k, v in metrics.items() if k != 'finite'},
                    full_update_s=time.monotonic()-tick, sampling=draw.report, trial_pair=active_pair)
                report['training'].append(row)
                print(f'[FIT_TANGENT] u={iteration+1} loss={row["loss"]:.6g} grad={row["gradient_norm"]:.6g}', flush=True)
            if (iteration+1) % 256 == 0:
                checkpoint()
                write_json_unique(args.output_dir/f'training_u{iteration+1:06d}.json',
                                  dict(training=report['training']), guard.budget)
        report['training_wall_s'] = time.monotonic()-started
        report['final_checkpoint'] = checkpoint()
        del data, rows, raw
        validation = validate(reader, dict(parameters, contact=contact), options, auxiliary, guard,
                              enabled=args.tangent_prior, learned=True)
        write_json_unique(args.output_dir/'validation.json', validation, guard.budget)
        report.update(validation=validation, status='TANGENT_COMPARISON_FIT_COMPLETE_NOT_WORLD_QUALIFIED')
    finally:
        report['last_finite_checkpoint'] = checkpoint()
        report['completed_updates'] = int(optimizer.step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('base', 'fit', 'evaluate'), required=True)
    parser.add_argument('--support-replay', type=Path, required=True)
    parser.add_argument('--robot-checkpoint', type=Path, required=True)
    parser.add_argument('--contact-checkpoint', type=Path, required=True)
    parser.add_argument('--auxiliary-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--tangent-prior', action='store_true')
    parser.add_argument('--base-selection', choices=('both', 'normal', 'full'), default='both',
        help='Allow each frozen base branch in a separate resource scope')
    parser.add_argument('--additional-updates', type=int, default=4096)
    parser.add_argument('--max-wall-s', type=float, default=1800.)
    args = parser.parse_args()
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, args.max_wall_s) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        paths = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_tangent_prior_experiment_v1', training_version='WMJ-V1', status='INCOMPLETE',
            config=cfg, config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository_root(), text=True).strip(),
            source_sha256={p.name: file_hash(p) for p in paths}, invocation=sys.argv,
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            actor_updates=0, critic_updates=0, world_model_ready=False,
            scope='common_recorded_delayed_commands; own_robot_ball_recursion; no_control_feedback_qualification')
        write_json_unique(args.output_dir/'manifest.json',
            {**report, 'source_snapshot': {p.name: p.read_text() for p in paths}}, guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA GPU required')
            report['devices'] = [str(d) for d in jax.devices()]
            reader = SupportReplayReader(args.support_replay, guard.budget, max_bytes=0)
            parameters, options, references, auxiliary, aux_ref = load_components(cfg, reader,
                robot_checkpoint=args.robot_checkpoint, contact_checkpoint=args.contact_checkpoint,
                auxiliary_checkpoint=args.auxiliary_checkpoint)
            report.update(components=references, frozen_auxiliary_reference=aux_ref, dataset_sha256=reader.identity)
            if args.mode == 'fit':
                fit(cfg, args, guard, report, reader, parameters, options, references, auxiliary, aux_ref)
            elif args.mode == 'base':
                report['base_comparison'] = {}
                selections = {'both': (False, True), 'normal': (False,), 'full': (True,)}
                for enabled in selections[args.base_selection]:
                    label = 'full_auxiliary' if enabled else 'normal_auxiliary'
                    result = validate(reader, parameters, options, auxiliary, guard, enabled=enabled, learned=False)
                    report['base_comparison'][label] = result
                    write_json_unique(args.output_dir/f'base_{label}.json', result, guard.budget)
                report['status'] = 'BASE_COMPARISON_COMPLETE_NOT_WORLD_QUALIFIED'
            else:
                metadata = json.loads(args.contact_checkpoint.read_text())
                enabled = metadata['structure'].get('tangent_auxiliary', {}).get('enabled', False)
                if args.tangent_prior != enabled:
                    raise ValueError('Evaluation flag must agree with checkpoint full-prior contract')
                result = validate(reader, parameters, options, auxiliary, guard, enabled=enabled, learned=True)
                report.update(validation=result, status='TANGENT_CHECKPOINT_EVALUATED_NOT_WORLD_QUALIFIED')
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_json_unique(args.output_dir/'result.json', report, guard.budget)
            print(f'[SAVE] {args.output_dir}; status={report["status"]}', flush=True)


if __name__ == '__main__':
    main()
