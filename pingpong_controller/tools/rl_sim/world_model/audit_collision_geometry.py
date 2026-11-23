"""Reproduce a convex-mesh alarm from an authenticated native reset artifact.

This diagnostic does not change collision masks or certify real-robot geometry.
It exports the instantiated model and original compiled triangle surfaces so
that an alarm can be inspected without repeated MJX reset compilation.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import io
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import mujoco
import numpy as np


def mesh_world(model, data, geom_id: int):
    """Return original compiled triangle vertices in the geom's world frame."""
    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError('Expected a mesh geom')
    mesh = int(model.geom_dataid[geom_id])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    local = np.asarray(model.mesh_vert[start:start+count], np.float64)
    world = local @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
    start, count = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
    faces = np.array(model.mesh_face[start:start+count], dtype=np.int32)
    return world, faces


def _ancestry(model, geom_id):
    result = []
    body = int(model.geom_bodyid[geom_id])
    while body:
        result.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body))
        body = int(model.body_parentid[body])
    return result


def _save_bytes(path, value, budget):
    if not path.resolve().is_relative_to(budget.root.resolve()):
        raise ValueError('Artifact must remain in the registered output root')
    budget.check(additional_bytes=len(value)+8192)
    with path.open('xb') as stream:
        stream.write(value)


def mesh_components(vertices, faces):
    """Count original vertex-connected components without filling mesh holes."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = np.concatenate((faces[:, :2], faces[:, 1:], faces[:, [2, 0]]))
    graph = coo_matrix((np.ones(len(edges), dtype=np.uint8), edges.T),
                      shape=(len(vertices), len(vertices))).tocsr()
    _, labels = connected_components(graph, directed=False)
    used = np.unique(faces)
    return [dict(vertices=int(np.count_nonzero(labels[used] == label)),
                 faces=int(np.count_nonzero(labels[faces[:, 0]] == label)))
            for label in np.unique(labels[used])]


def inspect_pair(model, data, geoms):
    """Locate convex witnesses on the actual triangles without claiming overlap."""
    import trimesh
    witness = np.zeros(6, dtype=np.float64)
    signed_distance = float(mujoco.mj_geomDistance(model, data, *geoms, .03, witness))
    meshes, rows = {}, []
    for index, geom in enumerate(geoms):
        vertices, faces = mesh_world(model, data, geom)
        triangles = vertices[faces]
        point = witness.reshape(2, 3)[index]
        nearest = trimesh.triangles.closest_point(triangles, np.tile(point, (len(faces), 1)))
        distances = np.linalg.norm(nearest-point, axis=1)
        closest_id = int(np.argmin(distances))
        original = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        processed = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=True)
        rows.append(dict(geom_id=int(geom), body_ancestry=_ancestry(model, geom),
            mesh_id=int(model.geom_dataid[geom]), vertices=len(vertices), triangles=len(faces),
            original_watertight=bool(original.is_watertight),
            processed_watertight=bool(processed.is_watertight),
            original_components=mesh_components(vertices, faces),
            geom_world_position_m=data.geom_xpos[geom].tolist(),
            world_aabb_m=[vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
            convex_witness_m=point.tolist(),
            witness_to_original_surface_distance_m=float(distances[closest_id]),
            closest_original_surface_point_m=nearest[closest_id].tolist(),
            closest_triangle_id=closest_id))
        meshes[f'vertices_{geom}'], meshes[f'faces_{geom}'] = vertices, faces
    return dict(geoms=list(geoms), convex_signed_distance_m=signed_distance,
                scope='convex_alarm_diagnostic_not_actual_surface_or_hardware_collision',
                surfaces=rows), meshes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reset-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    import json
    from .contracts import load_config, repository_root, config_hash, file_hash
    from .artifacts import write_json_unique
    from .resources import StorageBudget
    from .native_stack import build_native_environment

    started = time.monotonic()
    cfg = load_config()
    resources = cfg['resources']
    budget = StorageBudget(repository_root()/resources['output_root'],
                           resources['max_new_bytes'], resources['min_free_bytes'])
    budget.check(additional_bytes=256*2**20)
    if not args.output_dir.resolve().is_relative_to(budget.root.resolve()):
        raise ValueError('Output must remain inside the registered WMJ budget root')
    reset_report = json.loads(args.reset_report.read_text())
    if reset_report['config_sha256'] != config_hash(cfg):
        raise ValueError('Reset report configuration does not match this run')
    for name, source in reset_report['source_snapshot'].items():
        if name in ('native_stack.py', 'contracts.py', 'geometry.py'):
            if file_hash(Path(__file__).parent/name) != source['sha256']:
                raise ValueError(f'Native construction source changed since reset: {name}')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print('[RUN] exact recorded reset on newly instantiated native compiled model', flush=True)
    bundle = build_native_environment(cfg, n_envs=1)
    env, model = bundle.environment, bundle.environment.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = reset_report['geometry']['reset_fk']['qpos']
    mujoco.mj_forward(model, data)
    expected = np.array(reset_report['geometry']['reset_fk']['racket_site_world_m'])
    error = float(np.max(np.abs(data.site_xpos[env.racket_site_id]-expected)))
    if error > 1e-12:
        raise ValueError(f'Compiled native reset site mismatch: {error}')
    geoms = tuple(reset_report['nominal_self_collision']['nearest']['geoms'])
    diagnostic, meshes = inspect_pair(model, data, geoms)
    model_bytes = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, model_bytes)
    model_path = args.output_dir/'native_model.mjb'
    _save_bytes(model_path, model_bytes.tobytes(), budget)
    arrays = io.BytesIO()
    np.savez_compressed(arrays, qpos=data.qpos, qvel=data.qvel, **meshes)
    arrays_path = args.output_dir/'reset_and_surfaces.npz'
    _save_bytes(arrays_path, arrays.getvalue(), budget)
    payload = dict(schema='wmj_convex_alarm_diagnostic_v1', training_version='WMJ-V1',
        timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        config_sha256=config_hash(cfg), reset_report=str(args.reset_report.resolve()),
        reset_report_sha256=file_hash(args.reset_report),
        code_sha256=file_hash(Path(__file__)), mujoco_version=mujoco.__version__,
        reset_site_max_error_m=error,
        model=dict(path=str(model_path.resolve()), sha256=file_hash(model_path)),
        arrays=dict(path=str(arrays_path.resolve()), sha256=file_hash(arrays_path)),
        pair=diagnostic, wall_s=time.monotonic()-started,
        ready_for_collection=False, contact_and_execution_changed=False)
    write_json_unique(args.output_dir/'report.json', payload, budget)
    print(json.dumps(diagnostic, ensure_ascii=False), flush=True)
    print(f'[SAVE] {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
