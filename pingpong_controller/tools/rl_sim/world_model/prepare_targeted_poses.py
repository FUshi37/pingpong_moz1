"""Offline independent flat-pose catalogue for one supplemental native batch."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .artifacts import write_json_unique
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .domain_atlas import pure_vertical_speed_capacity
from .flat_initialization import solve_flat_pose
from .frozen_initialization import StandardDropAuditor
from .resources import StorageBudget, WallBudget
from .targeted_collection import independent_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(); settings = cfg['targeted_supplement']; root = repository_root()
    resources = cfg['resources']
    budget = StorageBudget(root / resources['output_root'], resources['max_new_bytes'], resources['min_free_bytes'])
    budget.check(additional_bytes=32 * 2**20)
    started = time.monotonic(); wall = WallBudget(started, 3600.)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json_unique(args.output_dir / 'config_snapshot.json', cfg, budget)
    source_paths = sorted(Path(__file__).parent.glob('*.py'))
    write_json_unique(args.output_dir / 'implementation.json',
                      dict(source={p.name: p.read_text() for p in source_paths}), budget)
    auditor = StandardDropAuditor(cfg)
    support = json.loads((root / cfg['training_seed_bank']['source_support_replay'] / 'result.json').read_text())
    reference_path = Path(support['source_collection']) / 'result.json'
    if file_hash(reference_path) != support['source_result_sha256']:
        raise ValueError('Native reference-orientation source changed')
    reference = np.asarray(json.loads(reference_path.read_text())['racket_reference_rotation_world'])
    lower = np.asarray(cfg['domain']['racket_xyz_lower_m']); upper = np.asarray(cfg['domain']['racket_xyz_upper_m'])
    results = []
    for row in auditor.atlas['rows']:
        wall.check(reserve_s=10.)
        targets = independent_targets(row['requested_racket_xyz_m'], lower, upper,
                                      seed=settings['seed'], region_id=row['id'])
        attempts, candidates = [], []
        for target in targets:
            nearest = np.argsort(np.linalg.norm(auditor.catalog.position_m - target, axis=1))
            starts = []
            for index in nearest:
                q = auditor.catalog.q_rad[index]
                if not starts or all(np.linalg.norm(q - previous) > .15 for previous in starts):
                    starts.append(q)
                if len(starts) == 4:
                    break
            for start in starts:
                wall.check(reserve_s=10.)
                ik = solve_flat_pose(auditor.model, auditor.baseline, auditor.jids, auditor.site,
                                    reference, target, start, max_nfev=120)
                attempt = dict(target_site_m=target.tolist(), ik=ik)
                attempts.append(attempt)
                if ik['status'] != 'IK_FEASIBLE':
                    continue
                qpos = auditor.baseline.copy(); qpos[auditor.qadr] = ik['q_rad']
                goal = np.r_[target[:2], 1.375]
                check = auditor.audit_warmed(qpos, np.zeros(auditor.model.nv), goal, reference)
                attempt['geometry'] = check
                if check['status'] != 'ADMITTED':
                    continue
                capacity = pure_vertical_speed_capacity(auditor.model, auditor.data, auditor.jids,
                                                       auditor.site, auditor.velocity_limits)
                attempt['velocity_capacity'] = capacity
                if capacity['status'] != 'PASS':
                    continue
                candidates.append(dict(region_id=row['id'], q_rad=ik['q_rad'], target_site_m=target.tolist(),
                                       static_admission=check, velocity_capacity=capacity))
                break
        result = dict(region_id=row['id'], grid_node_m=row['requested_racket_xyz_m'],
            target_site_candidates_m=targets.tolist(), candidates=candidates, attempts=attempts,
            status='STATIC_CANDIDATES' if candidates else 'UNKNOWN', proven_unreachable=False)
        sha = write_json_unique(args.output_dir / f'region_{row["id"]:03d}.json', result, budget)
        results.append(dict(result, evidence_sha256=sha))
        if len(results) % 16 == 0:
            print(f'[CATALOG] {len(results)}/189; admitted_regions={sum(bool(x["candidates"]) for x in results)}', flush=True)
    admitted = [r['region_id'] for r in results if r['candidates']]
    missing = settings['missing_source_regions']
    report = dict(schema='wmj_v1_independent_targeted_pose_catalog', status='STATIC_CATALOG_COMPLETE',
        training_version='WMJ-V1', seed=settings['seed'], rows=results, admitted_regions=admitted,
        unknown_regions=[r['region_id'] for r in results if not r['candidates']],
        missing_source_regions=missing, missing_regions_with_static_candidates=sorted(set(missing) & set(admitted)),
        atlas_reference=cfg['collection']['initialization_atlas'],
        model_reference=cfg['collection']['compiled_model_cache'],
        reference_rotation_world=reference.tolist(), reference_source=dict(path=str(reference_path), sha256=file_hash(reference_path)),
        physics_sha256=config_hash(cfg['physics']), control_sha256=config_hash(cfg['control']),
        domain_sha256=config_hash(cfg['domain']), config_sha256=config_hash(cfg),
        candidate_count=sum(len(row['candidates']) for row in results),
        frozen_validation_states_used=False, policy_or_model_outcomes_used=False,
        geometry_additional_clearance_m=0., runtime_geometry_queries=0,
        native_warmup_admission='PENDING', actor_updates=0, critic_updates=0, world_updates=0,
        source_sha256={p.name: file_hash(p) for p in source_paths}, wall_s=time.monotonic() - started)
    write_json_unique(args.output_dir / 'catalog.json', report, budget)
    print(f'[CATALOG] {len(admitted)}/189 regions; missing sources recovered statically '
          f'{len(report["missing_regions_with_static_candidates"])}/14', flush=True)


if __name__ == '__main__':
    main()
