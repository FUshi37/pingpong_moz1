"""Guarded offline search for flatter, natively warmed frozen-case seeds."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_json_unique
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .flat_initialization import (
    merge_complete_native, pose_preference, recover_nonexact_case, search_case,
)
from .frozen_cases import GROUPS, validate_case_manifest
from .frozen_initialization import StandardDropAuditor
from .frozen_seed import load_warmed_seed, save_warmed_seed
from .native_stack import build_native_environment
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .teacher_initialization import prepare_warmup
from .world_control import ControlContract


def angle_statistics(rows: list, mask: np.ndarray) -> dict:
    """Describe maximum absolute roll/pitch on an explicitly fixed population."""
    values = np.array([max(map(abs, row['relative_roll_pitch_deg']))
                       for row, included in zip(rows, mask) if included])
    return dict(count=len(values), mean_deg=float(values.mean()),
                p95_deg=float(np.percentile(values, 95)), maximum_deg=float(values.max()))


def search(cfg: dict, args, guard, report: dict) -> None:
    """Keep the cases and constraints fixed; evaluate no policy actions."""
    from mjx_topology_execution import make_collect

    artifacts = cfg['frozen_evaluation_artifacts']
    parent = json.loads(authenticate(repository_root(), artifacts['static_initialization']).read_text())
    manifest = json.loads(authenticate(repository_root(), artifacts['cases']).read_text())
    validate_case_manifest(manifest, cfg)
    if parent['cases_reference']['sha256'] != artifacts['cases']['sha256']:
        raise ValueError('Parent initialization and frozen cases differ')
    cases = manifest['stages']['S01']
    if len(cases) != 128 or [r['case_id'] for r in parent['rows']] != [c['case_id'] for c in cases]:
        raise ValueError('All 128 fixed cases must remain in their original order')
    settings = cfg['offline_initialization_search']
    index_rows = None
    if args.recover_nonexact:
        index_ref = settings['nonexact_candidate_index']
        index = json.loads(authenticate(repository_root(), index_ref).read_text())
        if (index['cases_sha256'] != manifest['cases_sha256']
                or index['physics_sha256'] != config_hash(cfg['physics'])
                or index['control_sha256'] != config_hash(cfg['control'])):
            raise ValueError('Nonexact candidate index context changed')
        index_rows = index['rows']
        if [r['case_id'] for r in index_rows] != [c['case_id'] for c in cases]:
            raise ValueError('Nonexact candidate case identities changed')
        report['nonexact_candidate_index'] = index_ref
    bundle = build_native_environment(cfg, n_envs=128)
    env = bundle.environment
    control = ControlContract.from_native(env)
    reference = np.asarray(control.projection.reference_rotation)
    context = dict(
        schema='wmj_fixed_warmed_population', cases_sha256=manifest['cases_sha256'],
        physics_sha256=config_hash(cfg['physics']), control_sha256=config_hash(cfg['control']),
        domain_sha256=config_hash(cfg['domain']), ids=list(range(128)),
        reset_seed=settings['reset_seed'], warmup_controls=settings['warmup_controls'],
        native_source_sha256=bundle.stack.spec.native_env_sha256,
        initialization_sha256=artifacts['static_initialization']['sha256'],
        warmup_ar1_enabled=False)
    fresh, _ = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(settings['reset_seed']), 128))
    jax.block_until_ready(fresh)
    winner, original_rows = load_warmed_seed(artifacts['warmed_seed'], fresh, context)
    parent_state = winner
    winner_rows = copy.deepcopy(original_rows)
    parent_known = np.array([r['status'] == 'ADMITTED' for r in original_rows])
    selections = [dict(source='parent_warmed_fallback', candidate_id=None,
                       q_rad=row.get('q_rad'), round=None) for row in parent['rows']]
    report.update(
        parent_references=artifacts, parent_admitted=int(parent_known.sum()),
        cases_sha256=manifest['cases_sha256'], native_overrides=bundle.overrides,
        native_xml=dict(path=str(bundle.stack.xml), sha256=file_hash(Path(bundle.stack.xml))),
        reference_rotation=reference.tolist(), parent_context=context,
        original_orientation=angle_statistics(original_rows, parent_known))
    auditor = StandardDropAuditor(cfg)
    candidates = []
    started = time.monotonic()
    for index, case in enumerate(cases):
        guard.check()
        if index_rows is None:
            row = search_case(auditor, case, reference, cfg, guard.check)
        elif parent_known[index]:
            row = dict(case_id=case['case_id'], group=case['group'], candidates=[], attempts=[],
                       reason='parent_admitted_state_frozen_without_research')
        else:
            source = json.loads(authenticate(repository_root(), index_rows[index]['reference']).read_text())
            if source['case_id'] != case['case_id'] or source['goal_xyz_m'] != case['goals_xyz_m'][0]:
                raise ValueError('Candidate identity or goal changed')
            row = recover_nonexact_case(auditor, source, reference, cfg, guard.check)
        candidates.append(row)
        write_json_unique(args.output_dir / (case['case_id'] + '_candidates.json'), row, guard.budget)
        print(f'[SEARCH] {index + 1}/128 {case["case_id"]} '
              f'{len(row["candidates"])} candidates/{len(row["attempts"])} attempts', flush=True)
    report['static_search_wall_s'] = time.monotonic() - started
    report['static_candidates'] = sum(len(r['candidates']) for r in candidates)
    prepare = jax.jit(lambda state, q: prepare_warmup(env, state, q))
    scoped, execution = make_collect(lambda state, action: env.step(state, action)[0], mode='project')
    step = jax.jit(scoped)
    report['execution'] = execution
    zero = jnp.zeros((128, 7), jnp.float32)
    rounds = []
    for round_index in range(settings['static_candidates_per_case']):
        guard.check()
        started = time.monotonic()
        available = np.array([len(r['candidates']) > round_index for r in candidates])
        if not available.any():
            break
        q = jnp.asarray([
            r['candidates'][round_index]['q_rad'] if valid else auditor.catalog.q_rad[0]
            for r, valid in zip(candidates, available)], jnp.float32)
        warmed = prepare(fresh, q)
        for _ in range(settings['warmup_controls']):
            guard.check()
            warmed = step(warmed, zero)
        jax.block_until_ready(warmed)
        qpos, qvel = np.asarray(warmed.data.qpos), np.asarray(warmed.data.qvel)
        selected = np.zeros(128, bool)
        rows = []
        for index, (case, valid) in enumerate(zip(cases, available)):
            guard.check()
            if not valid:
                result = dict(status='UNKNOWN', reason='no_candidate_in_this_round')
            elif not (np.isfinite(qpos[index]).all() and np.isfinite(qvel[index]).all()):
                result = dict(status='UNKNOWN', reason='nonfinite_native_state')
            else:
                result = auditor.audit_warmed(qpos[index], qvel[index], case['goals_xyz_m'][0], reference)
            row = dict(case_id=case['case_id'], group=case['group'], **result)
            rows.append(row)
            goal = case['goals_xyz_m'][0]
            kw = dict(resolution_deg=settings['orientation_preference_resolution_deg'])
            if pose_preference(row, goal, **kw) < pose_preference(winner_rows[index], goal, **kw):
                selected[index] = True
                winner_rows[index] = row
                candidate = candidates[index]['candidates'][round_index]
                selections[index] = dict(
                    source='offline_flat_native_warmup', candidate_id=candidate['candidate_id'],
                    q_rad=candidate['q_rad'], round=round_index)
        winner = merge_complete_native(winner, warmed, selected)
        jax.block_until_ready(winner)
        known = np.array([r['status'] == 'ADMITTED' for r in winner_rows])
        if np.any(parent_known & ~known):
            raise ValueError('Parent admitted population was not preserved')
        # Save each complete best population so a resource stop loses no evidence.
        checkpoint = save_warmed_seed(
            args.output_dir / f'round_{round_index:02d}_seed.json', winner, winner_rows,
            dict(context, schema='wmj_offline_search_intermediate', round=round_index), guard.budget)
        summary = dict(round=round_index, candidate_lanes=int(available.sum()),
                       candidate_admitted=sum(r['status'] == 'ADMITTED' for r in rows),
                       selected_lanes=int(selected.sum()), total_admitted=int(known.sum()),
                       admission=rows, selections=copy.deepcopy(selections), checkpoint=checkpoint,
                       wall_s=time.monotonic() - started)
        digest = write_json_unique(args.output_dir / f'round_{round_index:02d}.json', summary, guard.budget)
        rounds.append(dict(round=round_index, sha256=digest, total_admitted=int(known.sum())))
        print(f'[WARMED] round {round_index + 1}: {int(selected.sum())} selected, '
              f'{int(known.sum())}/128 admitted', flush=True)
    known = np.array([r['status'] == 'ADMITTED' for r in winner_rows])
    static_rows = []
    for case, selection, admitted in zip(cases, selections, known):
        static_rows.append(dict(
            case_id=case['case_id'], group=case['group'], goal_xyz_m=case['goals_xyz_m'][0],
            status='STATIC_INITIALIZATION_ADMITTED' if admitted else 'UNKNOWN',
            selection=selection, q_rad=selection['q_rad'],
            scope='selected_after_native_warmup', warmup_admission='ADMITTED' if admitted else 'UNKNOWN',
            proven_unreachable=False))
    groups = {group: dict(requested=32, admitted=sum(
        r['group'] == group and r['status'] == 'ADMITTED' for r in winner_rows)) for group in GROUPS}
    initialization = dict(
        schema='wmj_v1_frozen_initialization', status='OFFLINE_WARMED_SEARCH_COMPLETE',
        cases_reference=artifacts['cases'], atlas_reference=cfg['collection']['initialization_atlas'],
        model_reference=cfg['collection']['compiled_model_cache'],
        config_sha256=config_hash(cfg), physics_sha256=config_hash(cfg['physics']),
        control_sha256=config_hash(cfg['control']), rows=static_rows, groups=groups,
        parent_references=artifacts, runtime_geometry_queries=0, policy_outputs_used=False)
    init_path = args.output_dir / 'initialization.json'
    init_hash = write_json_unique(init_path, initialization, guard.budget)
    new_context = dict(context, initialization_sha256=init_hash)
    seed_ref = save_warmed_seed(args.output_dir / 'warmed_seed.json', winner, winner_rows, new_context, guard.budget)
    restored, restored_rows = load_warmed_seed(seed_ref, fresh, new_context)
    exact = numeric_tree_digest(winner) == numeric_tree_digest(restored) and restored_rows == winner_rows
    orientation = angle_statistics(winner_rows, parent_known)
    checks = dict(
        fixed_goals_unchanged=True, parent_admitted_cases_preserved=bool(np.all(known[parent_known])),
        admitted_count_improves_over61=int(known.sum()) > int(parent_known.sum()),
        orientation_on_parent_admitted_population_improves=(
            orientation['mean_deg'] < report['original_orientation']['mean_deg']
            and orientation['p95_deg'] < report['original_orientation']['p95_deg']),
        complete_native_state_roundtrip=bool(exact))
    if args.recover_nonexact:
        del checks['orientation_on_parent_admitted_population_improves']
        checks['parent_admitted_states_exactly_preserved'] = (
            numeric_tree_digest(merge_complete_native(parent_state, winner, parent_known))
            == numeric_tree_digest(parent_state))
    report.update(
        status='COMPLETE', rounds=rounds, final_admitted=int(known.sum()), unknown=int((~known).sum()),
        groups=groups, orientation_on_parent_population=orientation,
        orientation_all_admitted=angle_statistics(winner_rows, known), checks=checks,
        pass_=all(checks.values()), selections=selections,
        static_initialization=dict(path=str(init_path.absolute()), sha256=init_hash), warmed_seed=seed_ref,
        runtime_geometry_queries=0, policy_outputs_used=False, policy_training_authorized=False)
    print(f'[RESULT] {int(known.sum())}/128 admitted; checks={checks}', flush=True)


def main() -> None:
    """Run a single bounded, source-snapshotted search with no optimizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--recover-nonexact', action='store_true')
    args = parser.parse_args()
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir,
                  cfg['offline_initialization_search']['max_worker_wall_s']) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        sources = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(
            schema='wmj_v1_flat_warmed_initialization_search', status='INCOMPLETE',
            search_mode='nonexact_recovery' if args.recover_nonexact else 'flat_ik_search',
            config_sha256=config_hash(cfg), training_version='WMJ-V1', gpu_uuid=args.gpu_uuid,
            command=[sys.executable, '-m', __spec__.name, *sys.argv[1:]],
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            source_sha256={p.name: file_hash(p) for p in sources},
            actor_updates=0, critic_updates=0, world_updates=0)
        write_json_unique(args.output_dir / 'implementation.json',
                          dict(report=report, source={p.name: p.read_text() for p in sources}), guard.budget)
        write_json_unique(args.output_dir / 'config_snapshot.json', cfg, guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            search(cfg, args, guard, report)
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_json_unique(args.output_dir / 'result.json', report, guard.budget)


if __name__ == '__main__':
    main()
