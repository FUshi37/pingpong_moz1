"""CPU-only offline admission and coverage report for existing training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .artifacts import write_json_unique
from .audit_full_prior_world import AuditReader
from .contracts import config_hash, file_hash, load_config, repository_root
from .frozen_initialization import StandardDropAuditor
from .resources import StorageBudget, WallBudget
from .training_seed_bank import (
    balanced_selection, incoming_admitted, initial_candidates, read_initial_pose,
)


def main() -> None:
    """Retain every exclusion, without creating or updating any learned model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config()
    settings = cfg['training_seed_bank']
    resource = cfg['resources']
    root = repository_root()
    budget = StorageBudget(root / resource['output_root'], resource['max_new_bytes'], resource['min_free_bytes'])
    budget.check(additional_bytes=32 * 2**20)
    if not args.output_dir.resolve().is_relative_to(budget.root.resolve()):
        raise ValueError('Training seed evidence must remain inside the WMJ budget')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json_unique(args.output_dir / 'config_snapshot.json', cfg, budget)
    write_json_unique(args.output_dir / 'implementation.json', dict(
        source={p.name: p.read_text() for p in Path(__file__).parent.glob('*.py')}), budget)
    started = time.monotonic()
    wall = WallBudget(started, settings['max_preparation_wall_s'])
    reader = AuditReader(root / settings['source_support_replay'], budget, max_bytes=0)
    if reader.identity != settings['source_dataset_sha256']:
        raise ValueError('The approved source dataset changed')
    source_report = json.loads((Path(reader.report['source_collection']) / 'result.json').read_text())
    reference = np.asarray(source_report['racket_reference_rotation_world'])
    auditor = StandardDropAuditor(cfg)
    ball_qadr = int(auditor.model.jnt_qposadr[auditor.model.body_jntadr[auditor.ball_body]])
    names = list(reader.records)
    eligible, trial_reports = [], []
    all_regions = set()
    for trial in range(len(names) // 2):
        wall.check(reserve_s=10.)
        pair = names[2 * trial:2 * trial + 2]
        record = reader.records[pair[0]]
        groups = record['record']['groups']
        all_regions.update(g['region_id'] for g in groups if g['split'] == 'train')
        chunks = [reader.read(name, names={
            'state__task_within_task_bounds', 'substep__event_contact_entry'}) for name in pair]
        entries = np.concatenate([chunk['substep__event_contact_entry'].reshape(-1, len(groups))
                                  for chunk in chunks])
        candidates, counts = initial_candidates(
            groups, chunks[0]['state__task_within_task_bounds'][0], entries,
            trial=trial, minimum_lead_s=settings['minimum_precontact_lead_s'])
        qpos, qvel = read_initial_pose(record['first_seed_ref'], context_sha256=reader.report['context_sha256'])
        audits = []
        for candidate in candidates:
            wall.check(reserve_s=10.)
            lane = candidate['lane']
            if np.isfinite(qpos[lane]).all() and np.isfinite(qvel[lane]).all():
                result = auditor.audit_warmed(
                    qpos[lane], qvel[lane], qpos[lane, ball_qadr:ball_qadr + 3], reference)
                admitted = incoming_admitted(result)
                result['drop_geometry_required'] = False
                result['scope'] = 'recorded_incoming_ball;vertical_drop_test_not_an_admission_gate'
            else:
                admitted = False
                result = dict(status='UNKNOWN', reason='nonfinite_initial_native_state')
            row = dict(candidate, geometry_admitted=admitted, geometry=result)
            audits.append(row)
            if admitted:
                eligible.append(dict(candidate, chunk_ids=pair,
                                     initial_seed_reference=record['first_seed_ref']))
        path = args.output_dir / f'trial_{trial:02d}.json'
        payload = dict(trial=trial, chunk_ids=pair, counts=counts, rows=audits,
                       source_groups=groups, source_context_sha256=reader.report['context_sha256'])
        sha = write_json_unique(path, payload, budget)
        trial_reports.append(dict(trial=trial, counts=counts,
                                  geometry_admitted=sum(r['geometry_admitted'] for r in audits),
                                  reference=dict(path=str(path.absolute()), sha256=sha)))
        print(f'[ADMISSION] trial{trial}: {counts["eligible"]} basic candidates, '
              f'{trial_reports[-1]["geometry_admitted"]} geometry admitted', flush=True)
    if eligible:
        selected, selection = balanced_selection(
            eligible, count=min(settings['requested_unique_sources'], len(eligible)), seed=settings['seed'])
    else:
        selected, selection = [], dict(regions=0, available=0, selected=0, duplicated_families=0)
    report = dict(
        schema='wmj_v1_independent_training_seed_plan',
        status='PLAN_COMPLETE' if len(selected) == settings['requested_unique_sources'] else 'SOURCE_SHORTAGE',
        config_sha256=config_hash(cfg), dataset_sha256=reader.identity,
        source_support_replay=settings['source_support_replay'], source_context_sha256=reader.report['context_sha256'],
        source_collection=reader.report['source_collection'], source_result_sha256=reader.report['source_result_sha256'],
        selected=selected, selection=selection, trials=trial_reports,
        training_regions=sorted(all_regions), admitted_regions=sorted({r['region_id'] for r in eligible}),
        missing_regions=sorted(all_regions - {r['region_id'] for r in eligible}),
        complete_candidate_domain_coverage=False,
        physics_sha256=config_hash(cfg['physics']), control_sha256=config_hash(cfg['control']),
        geometry_scope='offline_only_no_training_tick_cost', policy_scores_used=False,
        selected_partition='train', frozen_task_validation_sources_used=False,
        source_clock='control0;real_reset_sample_plus_invalid_history_padding',
        actor_updates=0, critic_updates=0, world_updates=0,
        source_sha256={p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')},
        wall_s=time.monotonic() - started)
    write_json_unique(args.output_dir / 'plan.json', report, budget)
    print(f'[PLAN] {len(selected)} unique sources, {selection["regions"]} regions; '
          f'missing={report["missing_regions"]}', flush=True)


if __name__ == '__main__':
    main()
