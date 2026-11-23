"""Replay saved new-pose model branches for event-aligned outcome diagnostics."""

import argparse
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import read_array_archive, write_array_archive_unique, write_json_unique
from .audit_full_prior_world import AuditReader, read_selected_trial
from .build_training_seed_bank import reset_source_template
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .full_prior_world import full_prior_world_rollout
from .native_stack import build_native_environment
from .new_pose_audit import select_windows
from .normal_checkpoint import load_components
from .release_alignment import compare_releases
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest, restore_numeric_state
from .world_control import ControlContract


def audit(cfg, args, guard, report):
    root = repository_root()
    reference = cfg['new_pose_model_audit']['same_clock_evidence']
    path = authenticate(root, reference)
    source = json.loads(path.read_text())
    if source['status'] != 'FROZEN_NEW_POSE_AUDIT_COMPLETE':
        raise ValueError('A complete frozen same-clock audit is required')
    streams = source['native_streams']
    raw = read_array_archive(Path(streams['path']), streams['archive'])
    reader = AuditReader(root / cfg['training_seed_bank']['source_support_replay'], guard.budget, max_bytes=0)
    output = root / cfg['resources']['output_root']
    parameters, options, refs, auxiliary, aux_ref = load_components(cfg, reader,
        robot_checkpoint=output / 'support_robot_fit_a02/robot_support_u006144.json',
        contact_checkpoint=output / 'contact_activation_control_a01/contact_coupled_u014336.json',
        auxiliary_checkpoint=output / 'solver_aux_fit_a01/auxiliary_u004096.json')
    references = dict(refs, auxiliary=aux_ref)
    if references != source['model_references']:
        raise ValueError('Frozen world identity changed')
    frozen = numeric_tree_digest(dict(robot=parameters['robot'], auxiliary=auxiliary))
    bundle = build_native_environment(cfg, n_envs=1)
    control = ControlContract.from_native(bundle.environment)
    # Only a shape/type template is borrowed. Every numerical value below is
    # overwritten by the authenticated validation branch archive before rollout.
    metadata = json.loads(authenticate(root, cfg['training_seed_bank']['bank_reference']).read_text())['metadata']
    row = metadata['source_rows'][0]
    reset_raw, values, _ = read_selected_trial(reader, row['chunk_ids'], np.array([row['lane']]))
    world, context = reset_source_template(reader, reader.records[row['chunk_ids'][0]],
        np.array([row['lane']]), reset_raw, values, options)
    cases = json.loads(authenticate(root, cfg['frozen_evaluation_artifacts']['cases']).read_text())['stages']['S01']
    case_lookup = {case['case_id']: i for i, case in enumerate(cases)}

    @jax.jit
    def predict(seed, ctx, actions, goals):
        _, trace = full_prior_world_rollout(parameters['robot'], auxiliary, seed, actions,
            context=ctx, control_contract=control, task_goal_xyz=goals, ar1_enabled=False, **options)
        contact = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), trace['physics']['contact'])
        return dict(position=contact['before'].position_m, velocity=contact['before'].velocity_m_s,
                    entry=contact['entry'], release=contact['release'], overlap=contact['overlap'])

    results = {}
    for horizon in (8, 128):
        guard.check()
        record_path = path.parent / f'horizon_{horizon}.json'
        if file_hash(record_path) != source['horizons'][str(horizon)]['sha256']:
            raise ValueError('Saved branch manifest changed')
        record = json.loads(record_path.read_text())
        ids = np.array([case_lookup[name] for name in record['case_ids']])
        n = len(ids)
        template = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape[1:]),
                                dict(world=world, context=context))
        seeds = record['seeds']
        restored = restore_numeric_state(template,
            read_array_archive(path.parent / seeds['archive_file'], seeds['archive']), seeds['state'],
            context_sha256=config_hash(references))
        starts = np.asarray(record['starts'])
        actions = select_windows(raw['raw_action'], starts, horizon)[:, ids]
        predicted = jax.device_get(predict(restored['world'], restored['context'], jnp.asarray(actions),
            jnp.asarray([cases[i]['goals_xyz_m'][0] for i in ids], jnp.float32)))
        select = lambda name: select_windows(raw[name], starts * 5, horizon * 5)[:, ids]
        native_release = select('substep__event_controlled_release')
        model_release = predicted['release'] & (predicted['velocity'][..., 2] > 0.)
        native_p = select('substep__ball_qpos_before')[..., :3]
        native_v = select('substep__ball_qvel_before')[..., :3]
        scopes = {}
        for scope, valid in (('task_prefix', select('task_prefix_valid')),
                             ('diagnostic_physics', select('substep__ball_model_valid'))):
            scopes[scope] = compare_releases(native_release & valid, model_release & valid,
                native_p, native_v, predicted['position'], predicted['velocity'], record['case_ids'])
        finite = np.isfinite(predicted['position']).all(axis=-1) & np.isfinite(predicted['velocity']).all(axis=-1)
        arrays = {k: np.where(np.isfinite(v), v, 0) for k, v in predicted.items()}
        arrays['finite'] = finite
        artifact = write_array_archive_unique(args.output_dir / f'model_h{horizon}.npz', arrays, guard.budget)
        result = dict(horizon=horizon, case_ids=record['case_ids'], scopes=scopes,
            model_nonfinite_windows=int((~finite.all(axis=0)).sum()),
            initial_contact_active=int(np.asarray(restored['world'].physics.ball.previous_contact).sum()),
            native_total_entries=int(select('substep__event_contact_entry').sum()),
            predicted_total_entries=int(predicted['entry'].sum()),
            outcome_definition='native_controlled_release_vs_model_upward_geometric_release',
            limitation='Extra/missing geometric modes are reported; equal counts alone do not prove identical contact modes',
            prediction_archive=artifact)
        write_json_unique(args.output_dir / f'horizon_{horizon}.json', result, guard.budget)
        results[str(horizon)] = result
        print(f'[EVENT] H={horizon} primary={scopes["task_prefix"]["velocity_error_m_s"]}', flush=True)
    if frozen != numeric_tree_digest(dict(robot=parameters['robot'], auxiliary=auxiliary)):
        raise ValueError('Frozen parameters changed')
    report.update(status='SAVED_EVENT_AUDIT_COMPLETE', source=reference, model_references=references,
                  results=results, frozen_before_sha256=frozen, frozen_after_sha256=frozen, promoted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, 1800.) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        sources = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_v1_saved_new_pose_release_audit', status='INCOMPLETE',
            training_version='WMJ-V1', gpu_uuid=args.gpu_uuid, config_sha256=config_hash(cfg),
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            source_sha256={p.name: file_hash(p) for p in sources},
            actor_updates=0, critic_updates=0, world_updates=0, new_native_collection=False)
        write_json_unique(args.output_dir / 'implementation.json',
                          dict(report=report, source={p.name: p.read_text() for p in sources}), guard.budget)
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
