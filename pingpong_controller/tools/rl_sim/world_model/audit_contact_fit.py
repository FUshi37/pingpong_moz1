"""Separate one-step contact fit error from published recursive validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from .artifacts import read_array_archive, write_json_unique
from .contact_dynamics import contact_step
from .contact_geometry import surface_from_compiled_model
from .contact_model import contact_features, FEATURE_SCALE
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .fit_robot import summarize_errors
from .native_contact_data import prepare_native_contact_batch, read_native_contact_trial
from .neural import init_adam, init_mlp
from .resources import StorageBudget
from .state_adapter import restore_numeric_state
from .support_data import SupportReplayReader
from .soft_contact import SoftContactContract
from .hybrid_contact import hybrid_contact_step
from .position_contact import position_scale_from_statistics, position_contact_step, position_features, POSE_KEYS
from .position_contact_data import read_position_contact_trial, prepare_position_contact_batch
from .model_numerics import precision_scope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026110701)
    parser.add_argument('--conditioning-diagnostics', action='store_true')
    args = parser.parse_args()
    if jax.default_backend() != 'cpu':
        raise ValueError('This bounded diagnostic uses CPU')
    cfg, root = load_config(), repository_root()
    r = cfg['resources']
    budget = StorageBudget(root/r['output_root'], r['max_new_bytes'], r['min_free_bytes'])
    reader = SupportReplayReader(args.replay, budget, max_bytes=0)
    model = mujoco.MjModel.from_binary_path(str(authenticate(root, cfg['collection']['compiled_model_cache'])))
    native = surface_from_compiled_model(model, reader.report['geometry']['racket']['geom_id'], normal_axis=0)
    radius = float(reader.report['geometry']['ball']['radius_m'])
    checkpoints = []
    for path in args.checkpoint:
        metadata = json.loads(path.read_text())
        structure = metadata['structure']
        accepted_conditioning = {
            'wmj_contact_component_checkpoint_v1': ('recorded_native_pose_twist',
                'recorded_native_pose_twist_and_joint_positions'),
            'wmj_contact_coupled_checkpoint_v1': ('own_predicted_full_chain_pose_twist_and_joint_positions',),
        }
        if (metadata['dataset_sha256'] != reader.identity
                or structure['conditioning'] not in accepted_conditioning.get(metadata['schema'], ())):
            raise ValueError('Require matching coherent native-pose contact checkpoint')
        expected = config_hash(dict(structure=structure, dataset=reader.identity,
            physics=cfg['physics'], geometry=reader.report['geometry']))
        if metadata['model_signature'] != expected:
            raise ValueError('Checkpoint physics/geometry signature mismatch')
        surface = native if 'collision_geometry' in structure else native.prism
        if surface is native and structure['collision_geometry'] != native.identity:
            raise ValueError('Native contact dependency or merged mesh changed')
        parameters = init_mlp(jax.random.PRNGKey(0), structure['widths'])
        arrays = read_array_archive(path.parent/metadata['archive_file'], metadata['archive'])
        restored = restore_numeric_state(dict(parameters=parameters, optimizer=init_adam(parameters)),
            arrays, metadata['state'], context_sha256=metadata['model_signature'])
        checkpoints.append((path, restored['parameters'], surface, metadata))
    names, rng = list(reader.records), np.random.default_rng(args.seed)
    metrics = [{} for _ in checkpoints]

    def make_predict(surface, response, positions, precision):
        contract = SoftContactContract(**response['contract']) if response else None
        learned = response is None or response['kind'] != 'soft_only'
        scale = position_scale_from_statistics(positions['statistics']) if positions else None
        def predict_unscoped(parameters, raw):
            if positions:
                batch = prepare_position_contact_batch(None, raw)
                after, trace = position_contact_step(parameters, surface, batch['before'], batch['racket'],
                    ball_radius_m=radius, contract=contract, position_scale=scale)
            else:
                batch = prepare_native_contact_batch(None, raw)
                after, trace = hybrid_contact_step(parameters, surface, batch['before'], batch['racket'],
                    ball_radius_m=radius, contract=contract, learned=learned)
            physical = dict(ball_position_m=batch['before'].position_m,
                ball_velocity_m_s=batch['before'].velocity_m_s,
                ball_omega_world_rad_s=batch['before'].omega_world_rad_s,
                previous_contact=batch['before'].previous_contact,
                duration_s=batch['before'].duration_s, ball_radius_m=radius,
                **{name:batch['racket'][name] for name in POSE_KEYS})
            features, geometry = contact_features(surface, **physical)
            result = dict(velocity_error_m_s=jnp.linalg.norm(after.velocity_m_s-batch['after'].velocity_m_s, axis=-1),
                spin_error_rad_s=jnp.linalg.norm(after.omega_world_rad_s-batch['after'].omega_world_rad_s, axis=-1),
                target_delta_v_m_s=jnp.linalg.norm(batch['after'].velocity_m_s-batch['before'].velocity_m_s
                                                  -jnp.array([0., 0., -.00981]), axis=-1),
                overlap_mismatch=trace['overlap'] != raw['contact_overlap'],
                near_rim=geometry['edge_signed_distance_m'] > -.01)
            if args.conditioning_diagnostics:
                value = features/jnp.asarray(FEATURE_SCALE,jnp.float32)
                if positions:
                    extra = position_features(batch['racket']['joint_q_rad'], batch['racket']['delayed_q_command_rad'])
                    value = jnp.concatenate((value,(extra-scale.mean)/scale.scale),axis=-1)
                for i in range(value.shape[-1]):
                    result[f'input_abs_{i:02d}'] = jnp.abs(value[...,i])
                for i,layer in enumerate(parameters[:-1]):
                    value = jnp.tanh(value@layer['w']+layer['b'])
                    result[f'hidden_{i}_saturated_fraction'] = jnp.mean(jnp.abs(value)>.995,axis=-1)
                    result[f'hidden_{i}_mean_derivative'] = jnp.mean(1-value**2,axis=-1)
            return result
        @jax.jit
        def predict(parameters,raw):
            with precision_scope(precision):
                return predict_unscoped(parameters,raw)
        return predict

    predictors = [make_predict(surface, metadata['structure'].get('response'),
                               metadata['structure'].get('position_conditioning'),
                               metadata['structure'].get('composed_matmul_precision'))
                  for _, _, surface, metadata in checkpoints]
    reader_function = (read_position_contact_trial if any('position_conditioning' in m['structure']
                       for _, _, _, m in checkpoints) else read_native_contact_trial)
    for i in range(0, len(names), 2):
        budget.check()
        data, record = reader_function(reader, names[i], names[i+1])
        for split in ('train', 'validation'):
            allowed = np.array([group['split'] == split for group in record['groups']])[None, :]
            valid = data['ball_model_valid'] & allowed
            for overlap in (False, True):
                times, lanes = np.nonzero(valid & (data['contact_overlap'] == overlap))
                if not len(times):
                    continue
                count = min(512, len(times))
                selected = rng.choice(len(times), size=count, replace=False)
                # Fixed shape padding contributes to no metric.
                selected = np.pad(selected, (0, 512-count), mode='edge')
                raw = {name: jnp.asarray(value[times[selected], lanes[selected]]) for name, value in data.items()}
                for index, ((_, params, _, _), predict) in enumerate(zip(checkpoints, predictors)):
                    result = jax.device_get(predict(params, raw))
                    groups = dict(all=np.ones(count, bool))
                    if overlap:
                        groups.update(near_rim=result['near_rim'][:count], interior=~result['near_rim'][:count],
                            entry=np.asarray(raw['event_contact_entry'][:count]),
                            sustained=np.asarray(raw['event_contact_duration_steps'][:count]) >= 8,
                            compression_2to7ms=(np.asarray(raw['event_contact_duration_steps'][:count])>=2)
                                & (np.asarray(raw['event_contact_duration_steps'][:count])<8))
                    for group, mask in groups.items():
                        bucket = f'{split}/{"contact" if overlap else "free"}/{group}'
                        values = metrics[index].setdefault(bucket, {})
                        selected_metrics = ['velocity_error_m_s','spin_error_rad_s','target_delta_v_m_s','overlap_mismatch']
                        if args.conditioning_diagnostics:
                            selected_metrics += [key for key in result if key.startswith(('input_abs_','hidden_'))]
                        for name in selected_metrics:
                            values.setdefault(name, []).append(result[name][:count][mask].astype(float))
        print(f'[AUDIT] trial={i//2+1}/{len(names)//2}', flush=True)
        del data
    reports = []
    for (path, _, _, metadata), groups in zip(checkpoints, metrics):
        reports.append(dict(checkpoint=str(path.absolute()), sha256=file_hash(path),
            optimizer_updates=metadata['optimizer_updates'],
            checkpoint_conditioning=metadata['structure']['conditioning'],
            recorded_native_pose_is_diagnostic_teacher_forcing=True,
            matmul_precision=metadata['structure'].get('composed_matmul_precision'),
            by_population={group: {key: summarize_errors(values) for key, values in measures.items()}
                           for group, measures in groups.items()}))
    payload = dict(schema='wmj_contact_one_step_audit_v1', training_version='WMJ-V1',
        source_sha256=file_hash(Path(__file__)), replay_sha256=reader.identity, seed=args.seed,
        sampling='without_replacement_up_to512_native_overlap_and512_free_per_trial_per_split',
        scope='teacher_forcing_diagnostic_not_recursive_or_world_qualification',
        conditioning_diagnostics=args.conditioning_diagnostics,
        final_test_used=False, checkpoints=reports)
    digest = write_json_unique(args.output, payload, budget)
    print(json.dumps(dict(path=str(args.output.absolute()), sha256=digest)))


if __name__ == '__main__':
    main()
