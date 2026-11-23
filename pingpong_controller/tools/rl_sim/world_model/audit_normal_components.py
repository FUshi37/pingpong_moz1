"""Paired H8 error decomposition using the exact existing validation windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_json_unique
from .contact_data import contact_window_rows, take_contact_windows
from .contact_losses import ballistic_apex
from .contracts import config_hash, file_hash, load_config, repository_root
from .fit_robot import summarize_errors
from .model_numerics import precision_scope
from .normal_checkpoint import load_components
from .normal_contact import normal_contact_rollout
from .normal_coupling import acceleration_terms, own_robot_motion
from .physics_data import read_physics_trial
from .position_contact_data import prepare_position_contact_batch
from .runtime_guard import RunGuard
from .support_data import SupportReplayReader


def component_errors(parameters, options, auxiliary, raw, *, enabled):
    batch = prepare_position_contact_batch(None, raw)
    motion, robot, commands = own_robot_motion(parameters['robot'], raw,
        scale=options['robot_scale'], chain=options['chain'])
    terms = (acceleration_terms(auxiliary, robot['q_before_rad'], robot['dq_before_rad_s'], commands)
             if enabled else None)
    initial = jax.tree.map(lambda value: value[0], batch['before'])
    _, trace = normal_contact_rollout(parameters['contact'], options['surface'], initial, motion, terms,
        ball_radius_m=options['ball_radius_m'], position_scale=options['position_scale'],
        contract=options['contact_contract'])
    before, actual = trace['before'], batch['before']
    dv = before.velocity_m_s-actual.velocity_m_s
    da = (ballistic_apex(before.position_m, before.velocity_m_s)
          -ballistic_apex(actual.position_m, actual.velocity_m_s))
    # Native normal is an evaluation coordinate only, never a model input.
    normal = raw['racket_geom_rotation_before'][..., :, 0]
    vn = jnp.sum(dv*normal, axis=-1)
    return dict(velocity_3d_m_s=jnp.linalg.norm(dv, axis=-1),
        velocity_xy_m_s=jnp.linalg.norm(dv[..., :2], axis=-1), velocity_z_m_s=jnp.abs(dv[..., 2]),
        velocity_normal_m_s=jnp.abs(vn),
        velocity_tangent_m_s=jnp.linalg.norm(dv-vn[..., None]*normal, axis=-1),
        apex_3d_m=jnp.linalg.norm(da, axis=-1), apex_xy_m=jnp.linalg.norm(da[..., :2], axis=-1),
        apex_z_m=jnp.abs(da[..., 2]), valid=batch['valid'] & batch['controlled_release'])


def audit(cfg, args, guard, report):
    root = repository_root()/cfg['resources']['output_root']
    reader = SupportReplayReader(root/'support_replay_full_a01', guard.budget, max_bytes=0)
    models, refs = {}, {}
    for name, path in (('control', args.control_checkpoint), ('candidate', args.candidate_checkpoint)):
        p, o, ref, a, ar = load_components(cfg, reader,
            robot_checkpoint=root/'support_robot_fit_a02/robot_support_u006144.json',
            contact_checkpoint=path, auxiliary_checkpoint=root/'solver_aux_fit_a01/auxiliary_u004096.json')
        enabled = json.loads(path.read_text())['structure']['normal_auxiliary']['enabled']
        if enabled != (name == 'candidate'):
            raise ValueError('Diagnostic requires the paired zero/corrected checkpoint roles')
        models[name] = p, o, a, enabled
        refs[name] = dict(components=ref, auxiliary=ar)

    @jax.jit
    def predict(raw):
        with precision_scope('highest'):
            return {name: component_errors(p, o, a, raw, enabled=enabled)
                    for name, (p, o, a, enabled) in models.items()}

    rng, names, identity = np.random.default_rng(2026120607), list(reader.records), hashlib.sha256()
    pieces, regions = {name: {} for name in models}, []
    for trial in range(0, len(names), 2):
        guard.check()
        data, record = read_physics_trial(reader, names[trial], names[trial+1])
        for horizon in (1, 8, 32, 64, 128):
            rows = contact_window_rows(data, record, controls=horizon, split='validation')
            independent = {}
            for i, row in enumerate(rows):
                independent.setdefault(row['independent_event_id'], []).append(i)
            groups = list(independent.values())
            selected = [int(rng.choice(groups[int(i)])) for i in rng.integers(len(groups), size=128)]
            if horizon != 8:
                continue  # Consume the exact original RNG draws for all horizons.
            raw, window_refs = take_contact_windows(data, rows, selected, controls=horizon)
            identity.update(json.dumps(window_refs, sort_keys=True).encode())
            result = jax.device_get(predict(jax.tree.map(jnp.asarray, raw)))
            region = np.asarray([row['region_id'] for row in window_refs])
            mask = result['control']['valid']
            if not np.array_equal(mask, result['candidate']['valid']):
                raise ValueError('Native release populations differ')
            regions.append(np.broadcast_to(region, mask.shape)[mask])
            for name, values in result.items():
                for metric, value in values.items():
                    if metric != 'valid':
                        pieces[name].setdefault(metric, []).append(value[mask])
        print(f'[COMPONENTS] trial={trial//2+1}/{len(names)//2}', flush=True)
    region = np.concatenate(regions)
    report.update(window_selection_sha256=identity.hexdigest(), references=refs, metrics={}, by_region={})
    for name, values in pieces.items():
        report['metrics'][name] = {k: summarize_errors(v) for k, v in values.items()}
        report['by_region'][name] = {str(int(r)): {k: summarize_errors([np.concatenate(v)[region == r]])
            for k, v in values.items()} for r in np.unique(region)}
    report['status'] = 'H8_COMPONENT_DIAGNOSTIC_COMPLETE'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-checkpoint', type=Path, required=True)
    parser.add_argument('--candidate-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, 900.) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        report = dict(schema='wmj_normal_component_audit_v1', training_version='WMJ-V1', status='INCOMPLETE',
            config_sha256=config_hash(cfg), source_sha256=file_hash(Path(__file__)), gpu_uuid=args.gpu_uuid,
            actor_updates=0, critic_updates=0, world_qualified=False,
            scope='own_robot_common_command_H8_heldout_error_decomposition',
            arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
        write_json_unique(args.output_dir/'implementation.json',
            dict(report=report, source=Path(__file__).read_text()), guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            audit(cfg, args, guard, report)
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_json_unique(args.output_dir/'result.json', report, guard.budget)


if __name__ == '__main__':
    main()
