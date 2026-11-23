"""Offline horizontal-pose search and complete native-state lane selection."""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from scipy.optimize import least_squares

from .domain_atlas import _rpy_and_rate, pure_vertical_speed_capacity


def solve_flat_pose(model, fixed_qpos, joint_ids, site_id, reference_rotation,
                    target_xyz_m, initial_q, *, max_nfev=120,
                    position_scale_m=.001, orientation_scale_rad=np.pi / 180,
                    joint_prior_weight=.01, roll_pitch_limit_deg=2.) -> dict:
    """Prefer zero relative roll/pitch without weakening any admission limit.

    The seven joint coordinates follow the provided native joint IDs. Position
    is in world metres; the reference rotation is the native projection frame.
    Failure of this bounded local search is UNKNOWN, not unreachability.
    """
    data = mujoco.MjData(model)
    joints = np.asarray(joint_ids, int)
    qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    lower, upper = model.jnt_range[joints].T
    initial = np.clip(np.asarray(initial_q), lower + 1e-8, upper - 1e-8)
    target = np.asarray(target_xyz_m)
    cache = {}

    def evaluate(q):
        if 'q' in cache and np.array_equal(cache['q'], q):
            return cache['result']
        data.qpos[:] = fixed_qpos
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        jp = np.zeros((3, model.nv))
        jr = np.zeros_like(jp)
        mujoco.mj_jacSite(model, data, jp, jr, site_id)
        rpy, rpy_jac = _rpy_and_rate(
            data.site_xmat[site_id].reshape(3, 3), reference_rotation,
            jr[:, vadr])
        residual = np.concatenate((
            (data.site_xpos[site_id] - target) / position_scale_m,
            rpy / orientation_scale_rad, joint_prior_weight * (q - initial)))
        jacobian = np.concatenate((
            jp[:, vadr] / position_scale_m,
            rpy_jac / orientation_scale_rad,
            joint_prior_weight * np.eye(len(joints))))
        result = residual, jacobian, data.site_xpos[site_id].copy(), rpy
        cache.update(q=q.copy(), result=result)
        return result

    solution = least_squares(
        lambda q: evaluate(q)[0], initial, jac=lambda q: evaluate(q)[1],
        bounds=(lower, upper), max_nfev=max_nfev,
        xtol=1e-9, ftol=1e-9, gtol=1e-9)
    _, _, position, rpy = evaluate(solution.x)
    error = float(np.linalg.norm(position - target))
    finite = np.isfinite(solution.x).all() and np.isfinite(position).all()
    feasible = (finite and error <= .0005
                and np.all(np.abs(rpy) <= np.deg2rad(roll_pitch_limit_deg) + 1e-5))
    return dict(
        status='IK_FEASIBLE' if feasible else 'UNKNOWN',
        q_rad=solution.x.tolist(), position_m=position.tolist(),
        position_error_m=error, relative_roll_pitch_deg=np.rad2deg(rpy).tolist(),
        nfev=int(solution.nfev), optimizer_status=int(solution.status),
        position_tolerance_m=.0005, proven_unreachable=False)


def pose_preference(row: dict, goal_xyz_m, *, resolution_deg=.1) -> tuple:
    """Admitted poses first, then flatness band, centring and fine flatness."""
    if row['status'] != 'ADMITTED':
        return (1, np.inf, np.inf, np.inf, np.inf)
    angle = float(np.max(np.abs(row['relative_roll_pitch_deg'])))
    position = np.asarray(row['racket_site_m'])
    offset = float(np.linalg.norm(position[:2] - np.asarray(goal_xyz_m)[:2]))
    return (0, int(np.floor(angle / resolution_deg)), offset, angle,
            abs(float(position[2]) - 1.075))


def merge_complete_native(old, new, selected):
    """Replace entire leading-axis lanes; shared queue phases must agree.

    No reconstructed controller or history values are permitted. A scalar or
    non-batched model leaf must be exactly identical before lane mixing.
    """
    selected = np.asarray(selected, bool)
    if selected.ndim != 1:
        raise ValueError('Native lane mask must be one-dimensional')
    old_leaves, old_tree = jax.tree.flatten(old)
    new_leaves, new_tree = jax.tree.flatten(new)
    if old_tree != new_tree:
        raise ValueError('Native state structures differ')
    merged = []
    for a, b in zip(old_leaves, new_leaves):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('Native leaf shape or dtype differs')
        if a.ndim and a.shape[0] == len(selected):
            mask = jnp.asarray(selected).reshape((-1,) + (1,) * (a.ndim - 1))
            merged.append(jnp.where(mask, b, a))
        else:
            if not np.array_equal(np.asarray(a), np.asarray(b)):
                raise ValueError('Native shared model value or queue phase differs')
            merged.append(a)
    return jax.tree.unflatten(old_tree, merged)


def candidate_targets(goal, catalog, cfg) -> list:
    """Fixed deterministic position proposals; never inspect actor outcomes."""
    search = cfg['offline_initialization_search']
    goal = np.asarray(goal)
    domain = cfg['domain']
    lower = np.asarray(domain['racket_xyz_lower_m'])
    upper = np.asarray(domain['racket_xyz_upper_m'])
    direction = .5 * (lower[:2] + upper[:2]) - goal[:2]
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-12 else np.zeros(2)
    xy_candidates = [goal[:2] + offset * direction
                     for offset in search['inward_xy_offsets_m']]
    if search['include_nearest_atlas_xy']:
        nearest = np.argmin(np.linalg.norm(catalog.position_m[:, :2] - goal[:2], axis=1))
        xy_candidates.append(catalog.position_m[nearest, :2])
    targets = []
    for xy in xy_candidates:
        for z in search['site_height_candidates_m']:
            target = np.r_[np.clip(xy, lower[:2], upper[:2]), z]
            if not any(np.linalg.norm(target - x) < 1e-7 for x in targets):
                targets.append(target)
    return targets


def search_case(auditor, case: dict, reference_rotation, cfg: dict,
                check=lambda: None) -> dict:
    """Generate diverse static candidates for later native warmed admission."""
    search = cfg['offline_initialization_search']
    goal = np.asarray(case['goals_xyz_m'][0])
    attempts, feasible = [], []
    for target in candidate_targets(goal, auditor.catalog, cfg):
        order = np.argsort(np.linalg.norm(auditor.catalog.position_m - target, axis=1))
        starts = []
        for index in order:
            q = auditor.catalog.q_rad[index]
            if not starts or all(np.linalg.norm(q - previous) > .15 for previous in starts):
                starts.append(q)
            if len(starts) == search['ik_starts_per_target']:
                break
        for start in starts:
            check()
            result = solve_flat_pose(
                auditor.model, auditor.baseline, auditor.jids, auditor.site,
                reference_rotation, target, start,
                max_nfev=search['ik_max_function_evaluations'],
                position_scale_m=search['ik_position_scale_m'],
                orientation_scale_rad=search['ik_orientation_scale_rad'],
                joint_prior_weight=search['ik_joint_prior_weight'],
                roll_pitch_limit_deg=cfg['domain']['roll_pitch_limit_deg'])
            attempt = dict(target_site_m=target.tolist(), initial_q_rad=start.tolist(),
                           ik=result, candidate_id=len(attempts))
            attempts.append(attempt)
            if result['status'] != 'IK_FEASIBLE':
                continue
            if any(np.linalg.norm(np.asarray(result['q_rad']) - x['q_rad']) < 1e-3
                   for x in feasible):
                attempt['excluded_reason'] = 'duplicate_joint_pose'
                continue
            feasible.append(dict(
                candidate_id=attempt['candidate_id'], q_rad=result['q_rad'],
                target_site_m=target.tolist(), status='ADMITTED',
                racket_site_m=result['position_m'],
                relative_roll_pitch_deg=result['relative_roll_pitch_deg']))
    # ADMITTED here is only a sorting sentinel. Every selected candidate below
    # must separately pass the full original-mesh and drop-geometry audits.
    ordered = sorted(feasible, key=lambda x: pose_preference(
        x, goal, resolution_deg=search['orientation_preference_resolution_deg']))
    candidates = []
    # Try different target positions first. A failed geometry check never marks
    # that target as covered, so alternate joint solutions can still be tried.
    for distinct_only in (True, False):
        for row in ordered:
            check()
            attempt = attempts[row['candidate_id']]
            if len(candidates) >= search['static_candidates_per_case']:
                break
            if 'static_admission' in attempt:
                continue
            if distinct_only and any(
                    np.linalg.norm(np.asarray(row['target_site_m']) - x['target_site_m']) < .005
                    for x in candidates):
                continue
            qpos = auditor.baseline.copy()
            qpos[auditor.qadr] = row['q_rad']
            audit = auditor.audit_warmed(qpos, np.zeros(auditor.model.nv), goal, reference_rotation)
            audit['execution_scope'] = 'offline_static_candidate_before_native_warmup'
            attempt['static_admission'] = audit
            if audit['status'] != 'ADMITTED':
                continue
            capacity = pure_vertical_speed_capacity(
                auditor.model, auditor.data, auditor.jids, auditor.site, auditor.velocity_limits)
            attempt['velocity_capacity'] = capacity
            if capacity['status'] == 'PASS':
                candidates.append(dict(row, **audit))
    for row in feasible:
        attempt = attempts[row['candidate_id']]
        if 'static_admission' not in attempt:
            attempt['excluded_reason'] = 'lower_priority_geometry_not_evaluated_in_bounded_search'
    return dict(case_id=case['case_id'], group=case['group'], goal_xyz_m=goal.tolist(),
                candidates=candidates, attempts=attempts,
                static_admitted_candidates=len(candidates), policy_outputs_used=False,
                proven_unreachable=False)


def recover_nonexact_case(auditor, source: dict, reference_rotation, cfg: dict,
                          check=lambda: None) -> dict:
    """Audit actual poses when IK missed a proposed site, not a task constraint.

    Preserve the solver's UNKNOWN result. A separate actual finite-face and
    geometry audit can still admit that joint pose as an initialization.
    """
    goal = np.asarray(source['goal_xyz_m'])
    attempts, candidates = [], []
    for original in source['attempts']:
        result = original['ik']
        if result['status'] != 'UNKNOWN':
            continue
        check()
        attempt = copy.deepcopy(original)
        attempts.append(attempt)
        if (not np.isfinite(result['q_rad']).all()
                or max(map(abs, result['relative_roll_pitch_deg']))
                > cfg['domain']['roll_pitch_limit_deg'] + np.rad2deg(1e-5)):
            attempt['excluded_reason'] = 'actual_pose_nonfinite_or_orientation_outside_limit'
            continue
        qpos = auditor.baseline.copy()
        qpos[auditor.qadr] = result['q_rad']
        audit = auditor.audit_warmed(qpos, np.zeros(auditor.model.nv), goal, reference_rotation)
        audit['execution_scope'] = 'offline_nonexact_reference_pose_before_native_warmup'
        attempt['static_admission'] = audit
        if audit['status'] != 'ADMITTED':
            continue
        capacity = pure_vertical_speed_capacity(
            auditor.model, auditor.data, auditor.jids, auditor.site, auditor.velocity_limits)
        attempt['velocity_capacity'] = capacity
        if capacity['status'] == 'PASS':
            candidates.append(dict(
                candidate_id=attempt['candidate_id'], q_rad=result['q_rad'],
                ik_reference_status=result['status'], target_site_m=attempt['target_site_m'], **audit))
    settings = cfg['offline_initialization_search']
    candidates.sort(key=lambda row: pose_preference(
        row, goal, resolution_deg=settings['orientation_preference_resolution_deg']))
    return dict(
        case_id=source['case_id'], group=source['group'], goal_xyz_m=goal.tolist(),
        candidates=candidates[:settings['static_candidates_per_case']], attempts=attempts,
        static_admitted_candidates=len(candidates), policy_outputs_used=False,
        scope='existing_nonexact_IK_solutions_only_no_new_solver_or_limits', proven_unreachable=False)
