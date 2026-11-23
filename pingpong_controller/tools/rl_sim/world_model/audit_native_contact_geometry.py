"""Check contact geometry/gating against real replayed solver intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_json_unique
from .contact_surface import surface_from_vertices, surface_features
from .contracts import file_hash, load_config, repository_root
from .fit_robot import summarize_errors
from .resources import StorageBudget
from .support_data import SupportReplayReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != 'cpu':
        raise ValueError('This label audit uses CPU only')
    cfg = load_config()
    r = cfg['resources']
    budget = StorageBudget(repository_root()/r['output_root'], r['max_new_bytes'], r['min_free_bytes'])
    reader = SupportReplayReader(args.replay, budget, max_bytes=0)
    geometry = reader.report['geometry']
    surface = surface_from_vertices(geometry['racket']['compiled_vertices_local_m'], normal_axis=0)
    radius = float(geometry['ball']['radius_m'])
    predict = jax.jit(lambda p, x, rot: surface_features(surface, p, x, rot, radius)['sphere_gap_m'])
    fields = ('ball_qpos_before', 'ball_qvel_before', 'ball_qvel_after',
              'racket_geom_position_before_m', 'racket_geom_rotation_before',
              'contact_distance_m', 'contact_overlap', 'contact_present', 'ball_model_valid')
    rows, gap_errors, far_field_deltas = [], [], []
    for name, record in reader.records.items():
        arrays = reader.read(name, names={'substep__'+field for field in fields})
        train_lanes = np.flatnonzero([g['split'] == 'train' for g in record['record']['groups']])[::4]
        data = {field: arrays['substep__'+field].reshape((-1,)+arrays['substep__'+field].shape[2:])[::4, train_lanes]
                for field in fields}
        gap = np.asarray(predict(jnp.asarray(data['ball_qpos_before'][..., :3]),
            jnp.asarray(data['racket_geom_position_before_m']), jnp.asarray(data['racket_geom_rotation_before'])))
        overlap, valid = gap <= 0., data['ball_model_valid']
        finite_native = data['contact_present'] & (np.abs(data['contact_distance_m']) < 1.) & valid
        actual = data['contact_overlap']
        delta = data['ball_qvel_after'][..., :3]-data['ball_qvel_before'][..., :3]-np.array([0., 0., -.00981], np.float32)
        speed_change = np.linalg.norm(delta, axis=-1)
        outside = ~overlap & valid
        gap_error = np.abs(gap-data['contact_distance_m'])[finite_native]
        gap_errors.append(gap_error)
        far_field_deltas.append(speed_change[outside])
        rows.append(dict(chunk=name, train_samples=int(valid.sum()), native_overlaps=int((actual & valid).sum()),
            native_missed_by_surface=int((actual & ~overlap & valid).sum()),
            surface_extra_overlaps=int((~actual & overlap & valid).sum()),
            finite_native_distances=int(finite_native.sum()),
            gap_error_m=summarize_errors([gap_error]),
            outside_surface_delta_v_above_1mm_s=int((outside & (speed_change > .001)).sum()),
            outside_surface_max_delta_v_m_s=float(speed_change[outside].max()) if outside.any() else None))
    result = dict(schema='wmj_native_contact_geometry_audit_v1', training_version='WMJ-V1',
        replay_sha256=reader.identity, source_sha256=file_hash(Path(__file__)),
        sample_rule='train_only: every4th_physics_substep_and_every4th_training_lane',
        gap_error_m=summarize_errors(gap_errors), far_field_delta_v_m_s=summarize_errors(far_field_deltas),
        counts={key: sum(row[key] for row in rows) for key in ('train_samples', 'native_overlaps',
            'native_missed_by_surface', 'surface_extra_overlaps', 'outside_surface_delta_v_above_1mm_s')},
        chunks=rows, model_accuracy_qualified=False)
    digest = write_json_unique(args.output, result, budget)
    print(json.dumps(dict(path=str(args.output.absolute()), sha256=digest, counts=result['counts'],
                         gap_error_m=result['gap_error_m'], far_field_delta_v_m_s=result['far_field_delta_v_m_s'])))


if __name__ == '__main__':
    main()
