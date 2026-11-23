"""Compare full-chain measured-state FK directly to native replay pose/twist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from .artifacts import read_array_archive, write_json_unique
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .fit_robot import summarize_errors
from .kinematics import extract_chain
from .resources import StorageBudget
from .robot_support_model import JOINT_NAMES
from .support_data import SupportReplayReader, support_chunk
from .support_losses import racket_kinematics


def load_support_chain(cfg, geometry):
    """Select all thirteen ancestor hinges without altering the native model."""
    model = mujoco.MjModel.from_binary_path(str(authenticate(repository_root(), cfg['collection']['compiled_model_cache'])))
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
    if any(value < 0 for value in ids):
        raise ValueError('Full-chain joint names are absent from the native model')
    chain = extract_chain(model, np.asarray(geometry['reset_fk']['qpos']), joint_ids=ids,
        site_id=geometry['racket']['site_id'], geom_id=geometry['racket']['geom_id'])
    return model, chain


def audit(cfg, reader):
    model, chain = load_support_chain(cfg, reader.report['geometry'])
    predict = jax.jit(lambda q, dq: racket_kinematics(chain, q, dq))
    all_errors, rows = {}, []
    for name in reader.records:
        data = support_chunk(reader.read(name))
        q, dq = data['q_before'][::8, ::16], data['dq_before'][::8, ::16]
        predicted = jax.device_get(predict(jnp.asarray(q), jnp.asarray(dq)))
        errors = {}
        for field, label, unit in (
            ('position', 'racket_site_position_before_m', 'site_position_m'),
            ('geom_position', 'racket_geom_position_before_m', 'geom_position_m'),
            ('velocity', 'racket_linear_velocity_before_m_s', 'linear_velocity_m_s'),
            ('omega', 'racket_angular_velocity_before_rad_s', 'angular_velocity_rad_s')):
            errors[unit] = np.linalg.norm(predicted[field]-data[label][::8, ::16], axis=-1).ravel()
        # Chordal rotation distance is stable for microradian differences.
        chord = np.linalg.norm(predicted['rotation']-data['racket_geom_rotation_before'][::8, ::16], axis=(-1, -2))
        errors['rotation_rad'] = (2*np.arcsin(np.clip(chord/(2*np.sqrt(2)), 0., 1.))).ravel()
        rows.append(dict(chunk=name, errors={key: summarize_errors([value]) for key, value in errors.items()}))
        for key, value in errors.items():
            all_errors.setdefault(key, []).append(value)

    # Independently check the new native twist observer at exact original seeds,
    # using CPU MuJoCo Jacobians and every moving base/waist/arm coordinate.
    oracle_errors = dict(position_m=[], rotation_rad=[], linear_velocity_m_s=[], angular_velocity_rad_s=[])
    native = mujoco.MjData(model)
    geom_id = reader.report['geometry']['racket']['geom_id']
    for index, name in enumerate(reader.records):
        if index % 2:
            continue
        reference = reader.records[name]['first_seed_ref']
        seed_path = Path(reference['manifest'])
        if file_hash(seed_path) != reference['sha256']:
            raise ValueError('Full seed manifest changed')
        seed = json.loads(seed_path.read_text())
        leaves = {row['path']: f'leaf_{i:06d}' for i, row in enumerate(seed['leaves'])}
        q_key, dq_key = leaves["['native'].data.qpos"], leaves["['native'].data.qvel"]
        state = read_array_archive(seed_path.parent/seed['archive_file'], seed['archive'], names=[q_key, dq_key])
        arrays = reader.read(name)
        for lane in range(0, reader.records[name]['record']['lanes'], 16):
            native.qpos[:], native.qvel[:] = state[q_key][lane], state[dq_key][lane]
            mujoco.mj_forward(model, native)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacGeom(model, native, jp, jr, geom_id)
            for name_, truth, label in (
                ('position_m', native.geom_xpos[geom_id], 'racket_geom_position_before_m'),
                ('linear_velocity_m_s', jp@native.qvel, 'racket_linear_velocity_before_m_s'),
                ('angular_velocity_rad_s', jr@native.qvel, 'racket_angular_velocity_before_rad_s')):
                oracle_errors[name_].append(float(np.linalg.norm(truth-arrays['substep__'+label][0, 0, lane])))
            difference = np.linalg.norm(native.geom_xmat[geom_id].reshape(3, 3)-arrays['substep__racket_geom_rotation_before'][0, 0, lane])
            oracle_errors['rotation_rad'].append(float(2*np.arcsin(np.clip(difference/(2*np.sqrt(2)), 0., 1.))))
    return dict(scope='measured_state_FK_and_native_observer_not_model_prediction',
        sample_rule='all_available_chunks: substeps_every8_lanes_every16; CPU_oracle_first_substep_every16_lanes',
        joint_names=list(JOINT_NAMES), overall={key: summarize_errors(value) for key, value in all_errors.items()},
        chunks=rows, independent_cpu_oracle={key: summarize_errors([np.asarray(value)]) for key, value in oracle_errors.items()},
        fixed_base_numerical_residual_included=True, full_world_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != 'cpu':
        raise ValueError('Use CPU for this independent FK audit')
    cfg = load_config()
    r = cfg['resources']
    budget = StorageBudget(repository_root()/r['output_root'], r['max_new_bytes'], r['min_free_bytes'])
    reader = SupportReplayReader(args.replay, budget, diagnostic=True)
    result = dict(schema='wmj_support_FK_audit_v1', training_version='WMJ-V1',
        config_sha256=config_hash(cfg), replay_sha256=reader.identity,
        source_sha256=file_hash(Path(__file__)), **audit(cfg, reader))
    digest = write_json_unique(args.output, result, budget)
    print(json.dumps(dict(path=str(args.output.absolute()), sha256=digest,
                         overall=result['overall'], independent_cpu_oracle=result['independent_cpu_oracle'])))


if __name__ == '__main__':
    main()
