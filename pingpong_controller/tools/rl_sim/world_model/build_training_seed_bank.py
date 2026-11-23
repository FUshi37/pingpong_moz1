"""Reconstruct and freeze a diverse training bank without parameter updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_json_unique
from .audit_full_prior_world import (
    AuditReader, cached_pose_sequence, make_bank_builder, read_selected_trial, release_cpu_heap, teacher_seeds,
)
from .contact_dynamics import ContactState
from .contact_model import world_spin
from .control_seed import read_control_seed
from .branch_bank import BranchSeedBank
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .course_env import course_critic_layout, init_course
from .frozen_initialization import vertical_drop_geometry
from .full_prior_seeds import initial_full_history, read_initial_camera, select_teacher_seeds
from .full_prior_world import FullPriorWorldState
from .imagined_env import init_imagined
from .imagination_checkpoint import restore_checkpoint, save_checkpoint
from .imagination_learner import slice_environment
from .native_stack import build_native_environment
from .normal_checkpoint import load_components
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .world_control import ControlContract
from .world_physics import PhysicsState


BANK_RUNTIME_SOURCES = (
    'full_prior_world.py', 'world_control.py', 'control_projection.py', 'control_rmp.py',
    'observation_engine.py', 'paired_history.py', 'kinematics.py', 'native_observation.py',
    'course_env.py', 'imagined_env.py', 'physical_events.py', 'task.py', 'course_schedule.py',
    'critic_features.py', 'branch_bank.py', 'course_sampling.py',
)


def bank_signature(cfg, *, model_references, source_rows, dataset_sha256) -> str:
    """Bind seed semantics without coupling restore to unrelated offline tools."""
    return config_hash(dict(
        schema='wmj_v1_independent_full_training_bank', stage='S01',
        models=model_references, dataset_sha256=dataset_sha256, source_rows=source_rows,
        control=cfg['control'], physics=cfg['physics'], observation=cfg['observation'],
        policy=cfg['policy'], task=cfg['task'], domain=cfg['domain'],
        curriculum=cfg['curriculum'], imagination=cfg['imagination'],
        runtime_sources={name: file_hash(Path(__file__).parent / name) for name in BANK_RUNTIME_SOURCES}))


def first_goal(cfg, raw, options, z_m) -> tuple:
    """S01 holds over the actual racket face at independently stratified heights."""
    position = np.asarray(raw['racket_geom_position_before_m'][0])
    rotation = np.asarray(raw['racket_geom_rotation_before'][0])
    lower = np.asarray(cfg['domain']['task_goal_xyz_lower_m'])
    upper = np.asarray(cfg['domain']['task_goal_xyz_upper_m'])
    z_m = np.asarray(z_m)
    if (z_m.shape != (len(position),) or not np.isfinite(z_m).all()
            or np.any(z_m < lower[2]) or np.any(z_m > upper[2])):
        raise ValueError('First-goal heights must remain inside the registered domain')
    goals = np.column_stack((np.clip(position[:, :2], lower[:2], upper[:2]), z_m)).astype(np.float32)
    surface = getattr(options['surface'], 'prism', options['surface'])
    audits = [vertical_drop_geometry(surface, p, r, goal, options['ball_radius_m'])
              for p, r, goal in zip(position, rotation, goals)]
    if not all(row['inside_face'] and row['positive_drop'] for row in audits):
        raise ValueError('Local S01 first goal lacks a valid finite-face vertical construction')
    return goals, audits


def reset_source_template(reader, record, lanes, raw, values, options):
    """Read one authentic reset without computing any future trajectory."""
    reference = record['first_seed_ref']
    control, context = read_control_seed(reference,
        context_sha256=reader.report['context_sha256'], lanes=lanes)
    camera = read_initial_camera(reference,
        context_sha256=reader.report['context_sha256'], lanes=lanes)
    qpos = jnp.asarray(raw['ball_qpos_before'][0])
    qvel = jnp.asarray(raw['ball_qvel_before'][0])
    physics = PhysicsState(jnp.asarray(raw['q_before'][0]), jnp.asarray(raw['dq_before'][0]),
        ContactState(qpos[:, :3], qvel[:, :3], world_spin(qpos[:, 3:], qvel[:, 3:]),
                     jnp.zeros(len(lanes), bool), jnp.zeros(len(lanes), jnp.float32)))
    observation = initial_full_history(camera, physics.q_rad, chain=options['chain'])
    pose = cached_pose_sequence(values['state__racket_position_m'][0],
                                values['state__racket_rotation_world'][0])
    return FullPriorWorldState(physics, control, observation, pose), context


def load_training_bank(cfg, reference, *, reader, options, control_contract,
                       model_references) -> tuple:
    """Restore a complete bank using one authentic source only as a shape template.

    The returned bank values all come from its authenticated archive, including
    every historical sample, queue, task field and RNG. No template value is
    used as a replacement for absent seed data.
    """
    payload = json.loads(authenticate(repository_root(), reference).read_text())
    metadata = payload['metadata']
    if (metadata['schema'] != 'wmj_v1_independent_full_training_bank'
            or metadata['model_references'] != model_references
            or metadata['dataset_sha256'] != reader.identity):
        raise ValueError('Training bank model or source identity changed')
    rows = metadata['source_rows']
    if not rows or any(row['split'] != 'train' for row in rows):
        raise ValueError('Training bank source partition changed')
    signature = bank_signature(cfg, model_references=model_references,
                               source_rows=rows, dataset_sha256=reader.identity)
    source = rows[0]
    raw, values, _ = read_selected_trial(reader, source['chunk_ids'], np.array([source['lane']]))
    initial, context = reset_source_template(
        reader, reader.records[source['chunk_ids'][0]], np.array([source['lane']]),
        raw, values, options)
    course = init_course(init_imagined(initial, context,
        jnp.asarray([source['first_goal_xyz_m']], jnp.float32)), jax.random.PRNGKey(0))
    environment = jax.tree.map(lambda x: jnp.broadcast_to(x, (len(rows),) + x.shape[1:]), course)
    template = dict(environment=environment, exploration_rng=jnp.zeros((len(rows), 2), jnp.uint32))
    restored = restore_checkpoint(reference, template, signature=signature)
    bank = BranchSeedBank.restore(jax.device_get(restored), metadata['branch_bank'],
                                  model_identity=config_hash(model_references), stage_id='S01')
    return bank, metadata


def verify_saved_bank(cfg, args, guard, report):
    """Verify a completed archive after a loader fix without rebuilding sources."""
    root = repository_root() / cfg['resources']['output_root']
    reference = dict(path=str(args.verify_bank.resolve()), sha256=file_hash(args.verify_bank))
    payload = json.loads(args.verify_bank.read_text())
    if payload['metadata']['source_plan'] != cfg['training_seed_bank']['plan_reference']:
        raise ValueError('Saved bank does not use the registered source plan')
    reader = AuditReader(repository_root() / cfg['training_seed_bank']['source_support_replay'],
                         guard.budget, max_bytes=0)
    parameters, options, refs, auxiliary, aux_ref = load_components(cfg, reader,
        robot_checkpoint=root / 'support_robot_fit_a02/robot_support_u006144.json',
        contact_checkpoint=root / 'contact_activation_control_a01/contact_coupled_u014336.json',
        auxiliary_checkpoint=root / 'solver_aux_fit_a01/auxiliary_u004096.json')
    model_refs = dict(refs, auxiliary=aux_ref)
    frozen = numeric_tree_digest(dict(robot=parameters['robot'], auxiliary=auxiliary))
    bank, metadata = load_training_bank(cfg, reference, reader=reader, options=options,
                                        control_contract=None, model_references=model_refs)
    tree, restored_metadata = bank.snapshot()
    original = restore_checkpoint(reference, tree, signature=payload['runtime_signature'])
    if (numeric_tree_digest(tree) != numeric_tree_digest(original)
            or metadata['branch_bank'] != restored_metadata):
        raise ValueError('Public bank restore changed exact values or metadata')
    checks = sorted(args.verify_bank.parent.glob('reconstruction_*.json'))
    reconstructed = [json.loads(path.read_text()) for path in checks]
    if sum(row['selected_rows'] for row in reconstructed) != len(bank.rows):
        raise ValueError('Complete reconstruction evidence missing')
    if frozen != numeric_tree_digest(dict(robot=parameters['robot'], auxiliary=auxiliary)):
        raise ValueError('Frozen model parameters changed')
    sources = metadata['source_rows']
    report.update(status='TRAINING_BANK_COMPLETE', training_bank=reference,
        source_plan=metadata['source_plan'], model_references=model_refs,
        unique_sources=len(sources), duplicate_source_families=len(sources)-len({r['root_seed_family'] for r in sources}),
        regions=len({r['region_id'] for r in sources}), complete_domain_coverage=False,
        missing_source_regions=cfg['training_seed_bank']['missing_source_regions'],
        reconstruction_files=[dict(path=str(p), sha256=file_hash(p)) for p in checks],
        reconstruction=reconstructed, public_loader_restore_exact=True,
        initial_history_present_counts=sorted({r['initial_history_present'] for r in sources}),
        endpoint_history_present_range=[min(r['reconstruction_endpoint_history_present'] for r in sources),
                                        max(r['reconstruction_endpoint_history_present'] for r in sources)],
        frozen_before_sha256=frozen, frozen_after_sha256=frozen,
        frozen_task_validation_sources_used=False,
        verification_scope='complete_existing_archive_restore_without_rebuilding_or_changing_any_seed')


def build(cfg, args, guard, report) -> None:
    """Read only train sources; check full640ms context before storing control0."""
    settings = cfg['training_seed_bank']
    plan_ref = settings['plan_reference']
    plan = json.loads(authenticate(repository_root(), plan_ref).read_text())
    if (plan['schema'] != 'wmj_v1_independent_training_seed_plan'
            or plan['status'] != 'PLAN_COMPLETE'
            or plan['selected_partition'] != 'train'
            or plan['frozen_task_validation_sources_used']
            or plan['physics_sha256'] != config_hash(cfg['physics'])
            or plan['control_sha256'] != config_hash(cfg['control'])):
        raise ValueError('A complete independent, physically admitted source plan is required')
    selected = plan['selected']
    if (len(selected) != settings['requested_unique_sources']
            or len({row['root_seed_family'] for row in selected}) != len(selected)
            or any(row['split'] != 'train' or row['source_control'] != 0 for row in selected)):
        raise ValueError('Training bank requires the requested unique train reset sources')
    root = repository_root() / cfg['resources']['output_root']
    reader = AuditReader(repository_root() / settings['source_support_replay'], guard.budget, max_bytes=0)
    if reader.identity != plan['dataset_sha256']:
        raise ValueError('Source replay identity changed after offline admission')
    parameters, options, refs, auxiliary, aux_ref = load_components(
        cfg, reader, robot_checkpoint=root / 'support_robot_fit_a02/robot_support_u006144.json',
        contact_checkpoint=root / 'contact_activation_control_a01/contact_coupled_u014336.json',
        auxiliary_checkpoint=root / 'solver_aux_fit_a01/auxiliary_u004096.json')
    model_refs = dict(refs, auxiliary=aux_ref)
    model_id = config_hash(model_refs)
    frozen_digest = numeric_tree_digest(dict(robot=parameters['robot'], auxiliary=auxiliary))
    bundle = build_native_environment(cfg, n_envs=1)
    control = ControlContract.from_native(bundle.environment)
    reconstruct = make_bank_builder(options, control)
    bank = BranchSeedBank(model_identity=model_id, stage_id='S01', capacity_per_category=len(selected))
    n = len(selected)
    rng = np.random.default_rng(settings['seed'] + 1)
    lower, upper = cfg['domain']['task_goal_xyz_lower_m'][2], cfg['domain']['task_goal_xyz_upper_m'][2]
    z = lower + (upper - lower) * rng.permutation((np.arange(n) + .5) / n)
    source_rows, chunks = [], []
    selected_by_trial = {}
    for index, row in enumerate(selected):
        selected_by_trial.setdefault(row['trial'], []).append((index, row))
    microbatch = 64
    layout = None
    for trial, entries in sorted(selected_by_trial.items()):
        for start in range(0, len(entries), microbatch):
            guard.check()
            began = time.monotonic()
            useful = entries[start:start + microbatch]
            padded = useful + [useful[0]] * (microbatch - len(useful))
            lanes = np.array([row['lane'] for _, row in padded])
            names = useful[0][1]['chunk_ids']
            groups = reader.records[names[0]]['record']['groups']
            for _, row in useful:
                group = groups[row['lane']]
                if group['split'] != 'train' or group['root_seed_family'] != row['root_seed_family']:
                    raise ValueError('Source group identity changed after offline admission')
            raw, values, _ = read_selected_trial(reader, names, lanes)
            complete, context, checks = teacher_seeds(
                reader, reader.records[names[0]], lanes, raw, values, options, reconstruct)
            initial = select_teacher_seeds(complete, jnp.zeros(microbatch, jnp.int32))
            goals, drop = first_goal(cfg, raw, options, np.array([z[index] for index, _ in padded]))
            base = init_imagined(initial, context, jnp.asarray(goals))
            ids = jnp.asarray([index for index, _ in padded], jnp.int32)
            course = init_course(base, jax.random.PRNGKey(settings['seed'] + 2), ids)
            if layout is None:
                layout = course_critic_layout(course)
                if layout['dimension'] != cfg['imagination_implementation']['critic']['input_dimension']:
                    raise ValueError('Training bank critic layout changed')
            present = np.asarray(initial.observation.history.paired.present).sum(axis=-1)
            endpoint_present = np.asarray(complete.observation.history.paired.present[-1]).sum(axis=-1)
            # Transfer once. Computational padding never becomes a stored source.
            course = jax.device_get(course)
            for local, (index, row) in enumerate(useful):
                key = jax.random.fold_in(jax.random.PRNGKey(settings['seed'] + 3), index)[None]
                source_id = f'{names[0]}/lane{row["lane"]}/control0'
                bank.add_teacher_precontact(
                    slice_environment(course, local, local + 1), key,
                    split='train', source_id=source_id, time_to_contact_s=row['time_to_contact_s'])
                source_rows.append(dict(
                    row, source_id=source_id, bank_row=len(source_rows), first_goal_xyz_m=goals[local].tolist(),
                    first_goal_drop_geometry=drop[local], initial_history_present=int(present[local]),
                    reconstruction_endpoint_history_present=int(endpoint_present[local])))
            summary = dict(
                trial=trial, selected_rows=len(useful), computational_lanes=microbatch,
                padding_rows=microbatch - len(useful), seed_checks=checks,
                complete_native_controls_verified=128, source_lanes=[row['lane'] for _, row in useful],
                wall_s=time.monotonic() - began)
            chunks.append(summary)
            write_json_unique(args.output_dir / f'reconstruction_{len(chunks):03d}.json', summary, guard.budget)
            del complete, raw, values, initial, context, base, course
            release_cpu_heap()
            print(f'[BANK] {len(bank.rows)}/{n} unique train sources reconstructed', flush=True)
    tree, metadata = bank.snapshot()
    if len(bank.rows) != n or sum(bank.evicted.values()):
        raise ValueError('Bank construction dropped or evicted required sources')
    signature = bank_signature(cfg, model_references=model_refs, source_rows=source_rows,
                               dataset_sha256=reader.identity)
    ref = save_checkpoint(args.output_dir / 'training_bank.json', tree, signature=signature,
        budget=guard.budget, metadata=dict(
            schema='wmj_v1_independent_full_training_bank', branch_bank=metadata,
            source_plan=plan_ref, source_rows=source_rows, model_references=model_refs,
            dataset_sha256=reader.identity, critic_layout=layout,
            actor_updates=0, critic_updates=0, world_updates=0,
            future_labels_in_rollout=False, teacher_actions_used_as_actor_labels=False))
    restored = restore_checkpoint(ref, tree, signature=signature)
    exact = numeric_tree_digest(restored) == numeric_tree_digest(tree)
    if not exact:
        raise ValueError('Complete training bank did not restore exactly')
    resumed = BranchSeedBank.restore(jax.device_get(restored), metadata,
                                    model_identity=model_id, stage_id='S01')
    resumed_tree, resumed_metadata = resumed.snapshot()
    if numeric_tree_digest(resumed_tree) != numeric_tree_digest(tree) or resumed_metadata != metadata:
        raise ValueError('Branch bank restore changed data, RNG, ordering or metadata')
    public_bank, _ = load_training_bank(
        cfg, ref, reader=reader, options=options, control_contract=control,
        model_references=model_refs)
    public_tree, public_metadata = public_bank.snapshot()
    if numeric_tree_digest(public_tree) != numeric_tree_digest(tree) or public_metadata != metadata:
        raise ValueError('Public loader changed complete training-bank values')
    if frozen_digest != numeric_tree_digest(dict(robot=parameters['robot'], auxiliary=auxiliary)):
        raise ValueError('Frozen model parameters changed during reconstruction')
    report.update(
        status='TRAINING_BANK_COMPLETE', training_bank=ref, source_plan=plan_ref,
        runtime_signature=signature, dataset_sha256=reader.identity, model_references=model_refs,
        unique_sources=n, duplicate_source_families=0, regions=len({r['region_id'] for r in source_rows}),
        missing_source_regions=plan['missing_regions'], complete_domain_coverage=False,
        reconstruction=chunks, restore_exact=True, branch_bank_restore_exact=True,
        public_loader_restore_exact=True,
        initial_history_present_counts=sorted({r['initial_history_present'] for r in source_rows}),
        endpoint_history_present_range=[min(r['reconstruction_endpoint_history_present'] for r in source_rows),
                                        max(r['reconstruction_endpoint_history_present'] for r in source_rows)],
        frozen_before_sha256=frozen_digest, frozen_after_sha256=frozen_digest,
        policy_updates=0, frozen_task_validation_sources_used=False,
        limitations=['Coverage is the observed/admitted training-source subset; missing regions remain explicit',
                     'Stored control0 states have a real reset sample and invalid history padding',
                     'Full640ms reconstruction validates history/control coherence, not learned prediction accuracy',
                     'New-pose model error and global course behavior still require independent validation'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--verify-bank', type=Path)
    args = parser.parse_args()
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, 3600.) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        sources = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(
            schema='wmj_v1_global_training_bank_build', status='INCOMPLETE',
            training_version='WMJ-V1', config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            source_sha256={p.name: file_hash(p) for p in sources},
            actor_updates=0, critic_updates=0, world_updates=0)
        write_json_unique(args.output_dir / 'implementation.json',
                          dict(report=report, source={p.name: p.read_text() for p in sources}), guard.budget)
        write_json_unique(args.output_dir / 'config_snapshot.json', cfg, guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            if args.verify_bank:
                verify_saved_bank(cfg, args, guard, report)
            else:
                build(cfg, args, guard, report)
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_json_unique(args.output_dir / 'result.json', report, guard.budget)


if __name__ == '__main__':
    main()
