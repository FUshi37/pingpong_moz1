"""Compare actual V324 sphere-mesh kernel, Euclidean prism and saved labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco.mjx._src import collision_convex, mesh
from mujoco.mjx._src.collision_types import ConvexInfo, GeomInfo

from .artifacts import write_json_unique
from .contact_surface import surface_from_vertices, surface_features
from .contracts import authenticate, file_hash, load_config, repository_root
from .fit_robot import summarize_errors
from .resources import StorageBudget
from .support_data import SupportReplayReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != 'cpu':
        raise ValueError('Use CPU for this contact-kernel audit')
    cfg = load_config()
    r = cfg['resources']
    budget = StorageBudget(repository_root()/r['output_root'], r['max_new_bytes'], r['min_free_bytes'])
    reader = SupportReplayReader(args.replay, budget, max_bytes=0)
    geometry = reader.report['geometry']
    model = mujoco.MjModel.from_binary_path(str(authenticate(repository_root(), cfg['collection']['compiled_model_cache'])))
    native = mujoco.MjData(model)
    racket_id, ball_id = geometry['racket']['geom_id'], model.geom('ball').id
    surface = surface_from_vertices(geometry['racket']['compiled_vertices_local_m'], normal_axis=0)
    radius = float(geometry['ball']['radius_m'])
    convex = mesh.convex(model, geometry['racket']['mesh_id'])

    def one(ball, position, rotation):
        sphere = GeomInfo(ball, jnp.eye(3), jnp.array([radius, 0., 0.]))
        racket = ConvexInfo(position, rotation, jnp.asarray(model.geom_size[racket_id]), **vars(convex))
        distance, point, normal = collision_convex._sphere_convex(sphere, racket)
        euclidean = surface_features(surface, ball, position, rotation, radius)['sphere_gap_m']
        return distance, euclidean

    predict = jax.jit(jax.vmap(one))
    fields = ('ball_qpos_before', 'racket_geom_position_before_m', 'racket_geom_rotation_before',
              'contact_distance_m', 'contact_present', 'contact_overlap', 'ball_model_valid')
    rows, native_errors, euclidean_errors, cpu_errors, examples = [], [], [], [], []
    for name, record in reader.records.items():
        arrays = reader.read(name, names={'substep__'+field for field in fields})
        lanes = np.flatnonzero([g['split'] == 'train' for g in record['record']['groups']])[::4]
        data = {field: arrays['substep__'+field].reshape((-1,)+arrays['substep__'+field].shape[2:])[::4, lanes]
                for field in fields}
        selected = data['ball_model_valid'] & data['contact_present'] & (np.abs(data['contact_distance_m']) < 1.)
        if not selected.any():
            continue
        p, x, rot = (data[key][selected] for key in
                     ('ball_qpos_before', 'racket_geom_position_before_m', 'racket_geom_rotation_before'))
        p = p[:, :3]
        actual = data['contact_distance_m'][selected]
        distances, gaps = [], []
        for start in range(0, len(p), 512):
            count = min(512, len(p)-start)
            pad = 512-count
            def padded(value):
                return np.concatenate((value[start:start+count], np.repeat(value[start:start+1], pad, axis=0)))
            distance, gap = jax.device_get(predict(jnp.asarray(padded(p)), jnp.asarray(padded(x)), jnp.asarray(padded(rot))))
            distances.append(distance[:count])
            gaps.append(gap[:count])
        distances, gaps = np.concatenate(distances), np.concatenate(gaps)
        kernel_error, euclidean_error = np.abs(distances-actual), np.abs(gaps-actual)
        native_errors.append(kernel_error)
        euclidean_errors.append(euclidean_error)
        for index in np.argsort(euclidean_error)[-min(8, len(p)):]:
            native.geom_xpos[ball_id], native.geom_xmat[ball_id] = p[index], np.eye(3).ravel()
            native.geom_xpos[racket_id], native.geom_xmat[racket_id] = x[index], rot[index].ravel()
            exact = mujoco.mj_geomDistance(model, native, ball_id, racket_id, 1., np.zeros(6))
            cpu_errors.append(abs(float(gaps[index])-exact))
            local = rot[index].T@(p[index]-x[index])
            examples.append(dict(chunk=name, saved_distance_m=float(actual[index]),
                kernel_distance_m=float(distances[index]), euclidean_gap_m=float(gaps[index]),
                cpu_distance_m=exact, ball_center_geom_local_m=local.tolist()))
        rows.append(dict(chunk=name, count=len(p), kernel_error_m=summarize_errors([kernel_error]),
            euclidean_error_m=summarize_errors([euclidean_error]),
            kernel_overlap_mismatches=int(np.count_nonzero((distances <= 0.) != (actual <= 0.))),
            euclidean_overlap_mismatches=int(np.count_nonzero((gaps <= 0.) != (actual <= 0.)))))
    result = dict(schema='wmj_contact_kernel_audit_v1', training_version='WMJ-V1',
        source_sha256=file_hash(Path(__file__)), replay_sha256=reader.identity,
        collision_kernel_source=str(Path(collision_convex.__file__)),
        collision_kernel_sha256=file_hash(Path(collision_convex.__file__)),
        mesh_builder_sha256=file_hash(Path(mesh.__file__)),
        sample_rule='training_only_every4th_substep_and_every4th_training_lane; native_finite_pair_distance',
        kernel_error_m=summarize_errors(native_errors), euclidean_error_m=summarize_errors(euclidean_errors),
        cpu_vs_euclidean_outlier_error_m=summarize_errors([np.asarray(cpu_errors)]),
        chunks=rows, largest_euclidean_error_examples=examples,
        physical_configuration_changed=False, surrogate_changed=False)
    digest = write_json_unique(args.output, result, budget)
    print(json.dumps(dict(path=str(args.output.absolute()), sha256=digest,
        kernel_error_m=result['kernel_error_m'], euclidean_error_m=result['euclidean_error_m'],
        cpu_vs_euclidean_outlier_error_m=result['cpu_vs_euclidean_outlier_error_m'],
        kernel_overlap_mismatches=sum(row['kernel_overlap_mismatches'] for row in rows),
        euclidean_overlap_mismatches=sum(row['euclidean_overlap_mismatches'] for row in rows))))


if __name__ == '__main__':
    main()
