"""Compare seven-joint FK's frozen support context against actual teacher poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from .artifacts import write_json_unique
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .fit_robot import load_chain
from .kinematics import forward_kinematics
from .resources import StorageBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config()
    if jax.default_backend() != 'cpu':
        raise ValueError('This small FK-context audit uses CPU only')
    report = json.loads((args.collection/'result.json').read_text())
    model = mujoco.MjModel.from_binary_path(str(authenticate(repository_root(), cfg['collection']['compiled_model_cache'])))
    native = mujoco.MjData(model)
    chain = load_chain(cfg, report)
    fk = jax.jit(jax.vmap(lambda q: forward_kinematics(chain, q)))
    joint_ids = report['geometry']['reset_fk']['joint_order']
    qadr = model.jnt_qposadr[np.asarray(joint_ids)]
    site = report['geometry']['racket']['site_id']
    results = []
    for index in (0, 1, 4, 5, 18, 19):
        chunk_path = args.collection/'dataset'/f'chunk_r{index:04d}.npz'
        manifest = json.loads(chunk_path.with_suffix('.json').read_text())
        digest = file_hash(chunk_path)
        if digest != manifest['archive']['sha256']:
            raise ValueError('Teacher chunk changed')
        seed_path = args.collection/f'seed_{index:04d}.json'
        seed = json.loads(seed_path.read_text())
        if file_hash(seed_path) != manifest['seed_ref']['sha256']:
            raise ValueError('Seed manifest changed')
        archive_path = args.collection/seed['archive_file']
        if file_hash(archive_path) != seed['archive']['sha256']:
            raise ValueError('Seed archive changed')
        leaf = next(i for i, row in enumerate(seed['leaves']) if row['path'] == "['native'].data.qpos")
        with np.load(archive_path, allow_pickle=False) as archive:
            full_q = archive[f'leaf_{leaf:06d}']
        with np.load(chunk_path, allow_pickle=False) as archive:
            q = archive['state__q_rad']
            actual = archive['state__racket_position_m']
        measured_q = q[::8, ::16]
        predicted = np.asarray(fk(jnp.asarray(measured_q.reshape(-1, 7))).site_position)
        fixed_error = np.linalg.norm(predicted-actual[::8, ::16].reshape(-1, 3), axis=-1)
        per_seed_error, complete_seed_error = [], []
        for lane in range(0, len(full_q), 16):
            native.qpos[:] = full_q[lane]
            mujoco.mj_forward(model, native)
            complete_seed_error.append(np.linalg.norm(native.site_xpos[site]-actual[0, lane]))
            for tick in range(0, len(q), 8):
                native.qpos[:] = full_q[lane]
                native.qpos[qadr] = q[tick, lane]
                mujoco.mj_forward(model, native)
                per_seed_error.append(np.linalg.norm(native.site_xpos[site]-actual[tick, lane]))

        def summary(value):
            return dict(count=len(value), median_m=float(np.median(value)),
                        p95_m=float(np.quantile(value, .95)), maximum_m=float(np.max(value)))

        results.append(dict(chunk=index, archive_sha256=digest,
            global_fixed_support=summary(fixed_error),
            per_seed_fixed_support=summary(per_seed_error),
            full_seed_qpos=summary(complete_seed_error)))
    result = dict(schema='wmj_fk_context_audit_v1', training_version='WMJ-V1',
        config_sha256=config_hash(cfg), collection_result_sha256=file_hash(args.collection/'result.json'),
        source_sha256=file_hash(Path(__file__)), sample_rule='ticks every8, lanes every16, chunks0/1/4/5/18/19',
        results=results, finding='fixed_non_arm_ancestor_context_is_not_exact_along_native_motion',
        implication='component FK errors do not yet certify absolute native racket pose; contact fit is unqualified',
        physics_modified=False, full_world_completed=False)
    resource = cfg['resources']
    budget = StorageBudget(repository_root()/resource['output_root'], resource['max_new_bytes'], resource['min_free_bytes'])
    digest = write_json_unique(args.output, result, budget)
    print(json.dumps(dict(path=str(args.output.absolute()), sha256=digest, results=results)))


if __name__ == '__main__':
    main()
