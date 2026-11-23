"""Policy-independent compiled geometry, bounded IK and velocity diagnostics.

An optimizer failure is UNKNOWN, never a proof of unreachability. A geometric
pass certifies the sampled pose only, not a dynamically feasible multi-hit
transition. Unverified mesh convex hull overlap is not actual self-collision.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
from scipy.optimize import least_squares, linprog


def collision_pairs(model, *, moving_body_ids, excluded_body_ids):
    """Inspect retained robot geometry even when training collision masks are off."""
    moving, excluded = set(moving_body_ids), {0, *excluded_body_ids}
    unique, geoms = set(), []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ''
        if body in excluded or name.startswith('mjx_ball_contact_'):
            continue
        key = (body, int(model.geom_type[geom]), int(model.geom_dataid[geom]),
            *model.geom_size[geom], *model.geom_pos[geom], *model.geom_quat[geom])
        if key not in unique:
            unique.add(key)
            geoms.append(geom)
    pairs = []
    for index, first in enumerate(geoms):
        a = int(model.geom_bodyid[first])
        for second in geoms[index+1:]:
            b = int(model.geom_bodyid[second])
            if not ({a, b} & moving):
                continue
            if (a == b or model.body_parentid[a] == b or model.body_parentid[b] == a
                    or model.body_weldid[a] == model.body_weldid[b]):
                continue
            pairs.append((first, second))
    return pairs


def check_self_collision(model, data, pairs) -> dict:
    """Distinguish primitive overlap from unresolved mesh-hull broad-phase alarms."""
    collisions, unsupported, mesh_alarms, zero_distances = [], [], [], []
    minimum, nearest = .03, None
    for a, b in pairs:
        try:
            distance = float(mujoco.mj_geomDistance(model, data, a, b, .03, None))
        except (ValueError, RuntimeError) as exc:
            unsupported.append(dict(geoms=[a, b], error=str(exc)))
            continue
        if not np.isfinite(distance):
            unsupported.append(dict(geoms=[a, b], error='nonfinite_signed_distance'))
            continue
        row = dict(geoms=[int(a), int(b)], bodies=[
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
            for g in (a, b)], distance_m=distance)
        if distance < minimum:
            minimum, nearest = distance, row
        if distance < -1e-6:
            if any(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH for g in (a, b)):
                mesh_alarms.append(dict(row, reason='unverified_convex_hull_can_fill_mesh_voids'))
            else:
                collisions.append(row)
        elif abs(distance) <= 1e-6:
            zero_distances.append(dict(row, reason='touching_or_unresolved_distance_query'))
    status = ('FAIL' if collisions else
              'UNKNOWN' if unsupported or mesh_alarms or zero_distances or not pairs else 'PASS')
    return dict(status=status, tested_pairs=len(pairs), minimum_distance_m=minimum,
        distance_upper_censor_m=.03, penetration_tolerance_m=1e-6,
        nearest=nearest, interpenetrations=collisions, unsupported=unsupported,
        unresolved_mesh_hull_overlaps=mesh_alarms,
        unresolved_zero_distance=zero_distances,
        geometry_scope='compiled_robot_convex_geometry; adjacent_and_welded_bodies_excluded')


def _rpy_and_rate(rotation, reference, angular_jacobian):
    relative = reference.T @ rotation
    pitch = np.arcsin(np.clip(relative[0, 2], -1., 1.))
    roll = np.arctan2(-relative[1, 2], relative[2, 2])
    matrix = np.array([[1., 0., np.sin(pitch)],
        [0., np.cos(roll), -np.sin(roll)*np.cos(pitch)],
        [0., np.sin(roll), np.cos(roll)*np.cos(pitch)]])
    rate = np.linalg.lstsq(matrix, reference.T @ angular_jacobian, rcond=1e-6)[0]
    return np.array([roll, pitch]), rate[:2]


def solve_ik(model, fixed_qpos, joint_ids, site_id, reference_rotation, target_xyz_m,
             initial_q, lower_q, upper_q, *, max_nfev=100, roll_pitch_limit_deg=2.) -> dict:
    """Bounded position IK with the actual relative-roll/pitch inequality."""
    data = mujoco.MjData(model)
    joint_ids = np.asarray(joint_ids, int)
    qadr, vadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    limit = np.deg2rad(roll_pitch_limit_deg)
    cache = {}

    def evaluate(q):
        if 'q' in cache and np.array_equal(cache['q'], q):
            return cache['result']
        data.qpos[:] = fixed_qpos
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site_id)
        rpy, rpy_jacobian = _rpy_and_rate(data.site_xmat[site_id].reshape(3, 3),
            reference_rotation, jr[:, vadr])
        error = np.concatenate(((data.site_xpos[site_id]-target_xyz_m)/.001,
                                (rpy-np.clip(rpy, -limit, limit))/.01))
        jacobian = np.concatenate((jp[:, vadr]/.001,
            rpy_jacobian*(np.abs(rpy) > limit)[:, None]/.01))
        result = error, jacobian, data.site_xpos[site_id].copy(), rpy
        cache.update(q=q.copy(), result=result)
        return result

    solution = least_squares(lambda q: evaluate(q)[0],
        np.clip(initial_q, lower_q+1e-8, upper_q-1e-8), jac=lambda q: evaluate(q)[1],
        bounds=(lower_q, upper_q), max_nfev=max_nfev, xtol=1e-9, ftol=1e-9, gtol=1e-9)
    _, _, position, rpy = evaluate(solution.x)
    error = float(np.linalg.norm(position-target_xyz_m))
    finite = np.isfinite(solution.x).all() and np.isfinite(position).all()
    feasible = finite and error <= .0005 and np.all(np.abs(rpy) <= limit+1e-5)
    return dict(status='IK_FEASIBLE' if feasible else 'UNKNOWN',
        reason='sampled_pose_satisfies_bounds' if feasible else 'no_valid_solution_found_from_this_start',
        q_rad=solution.x.tolist(), position_m=position.tolist(), position_error_m=error,
        relative_roll_pitch_deg=np.rad2deg(rpy).tolist(), nfev=int(solution.nfev),
        optimizer_status=int(solution.status), position_tolerance_m=.0005)


def pure_vertical_speed_capacity(model, data, joint_ids, site_id, velocity_limits) -> dict:
    """Kinematic upper bound at one pose; not a claimed RMP/PD dynamic response."""
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, site_id)
    columns = model.jnt_dofadr[np.asarray(joint_ids, int)]
    task = np.concatenate((jp[:, columns], jr[:, columns]))
    vertical = np.array([0., 0., 1., 0., 0., 0.])
    size = len(joint_ids)
    objective = np.zeros(size+1)
    objective[-1] = -1.
    result = linprog(objective, A_eq=np.column_stack((task, -vertical)), b_eq=np.zeros(6),
        bounds=[(-float(v), float(v)) for v in velocity_limits]+[(0., None)], method='highs')
    return dict(status='PASS' if result.success else 'UNKNOWN',
        maximum_vertical_speed_m_s=float(result.x[-1]) if result.success else None,
        qvel_rad_s=result.x[:-1].tolist() if result.success else None,
        scope='instantaneous_joint_velocity_only; acceleration_projection_RMP_and_contact_not_certified')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nominal-only', action='store_true')
    args = parser.parse_args(argv)
    from .contracts import load_config, repository_root, config_hash, file_hash
    from .artifacts import write_json_unique
    from .resources import StorageBudget
    from .native_stack import build_native_environment
    from .geometry import compiled_geometry
    from .initialization_geometry import InitializationGeometryAuditor
    import jax

    started = time.monotonic()
    cfg = load_config()
    resource = cfg['resources']
    budget = StorageBudget(repository_root()/resource['output_root'], resource['max_new_bytes'], resource['min_free_bytes'])
    budget.check(additional_bytes=8*2**20)
    source_snapshot = {p.name: dict(sha256=file_hash(p), source=p.read_text())
                       for p in Path(__file__).parent.glob('*.py')}
    print('[RUN] WMJ-V1 compiled geometry/IK audit; actor_learning=off', flush=True)
    bundle = build_native_environment(cfg, n_envs=1)
    env = bundle.environment
    state, _ = env.reset(jax.random.split(jax.random.PRNGKey(2026110504), 1))
    geometry = compiled_geometry(env, state)
    model, data = env.mj_model, mujoco.MjData(env.mj_model)
    baseline = np.asarray(state.data.qpos[0], np.float64)
    data.qpos[:] = baseline
    mujoco.mj_forward(model, data)
    moving = [*model.jnt_bodyid[np.asarray(env.arm_jids)], env.racket_body_id]
    pairs = collision_pairs(model, moving_body_ids=moving, excluded_body_ids=[env.ball_body_id])
    geometry_auditor = InitializationGeometryAuditor(model, pairs,
        tolerance_m=cfg['domain']['self_collision_audit']['surface_tolerance_m'])
    nominal = geometry_auditor.audit(data)
    print(f"[VERIFY] nominal self-collision={nominal['status']} pairs={len(pairs)} "
          f"refined_pairs={len(nominal['original_geometry_refinements'])}", flush=True)
    report = dict(schema='wmj_domain_atlas_v1', training_version='WMJ-V1',
        timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), config_sha256=config_hash(cfg),
        config=cfg, source_snapshot=source_snapshot,
        backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
        geometry=geometry, nominal_self_collision=nominal, rows=[], ready_for_collection=False,
        candidate_target_domain=cfg['domain'], transition_admission='UNKNOWN_pending_dynamic_audit',
        execution_scope='offline_initialization_only',
        runtime_geometry_queries=0, cached_seed_bank_required=True,
        atlas_status='NOT_RUN')
    if not args.nominal_only:
        rng = np.random.default_rng(2026110504)
        axes = [np.linspace(a, b, count) for a, b, count in zip(
            cfg['domain']['racket_xyz_lower_m'], cfg['domain']['racket_xyz_upper_m'], (9, 7, 3))]
        grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
        qadr = np.asarray(env.arm_qadr)
        low, high = np.asarray(env.arm_lo, np.float64), np.asarray(env.arm_hi, np.float64)
        for index, point in enumerate(grid):
            starts = [baseline[qadr], (low+high)/2, *rng.uniform(low, high, (2, 7))]
            attempts = []
            for initial in starts:
                attempt = solve_ik(model, baseline, env.arm_jids, env.racket_site_id,
                    np.asarray(env.fixed_contact_racket_reference_rotation), point, initial, low, high)
                if attempt['status'] == 'IK_FEASIBLE':
                    data.qpos[:] = baseline
                    data.qpos[qadr] = attempt['q_rad']
                    mujoco.mj_forward(model, data)
                    audit_started = time.monotonic()
                    attempt['self_collision'] = geometry_auditor.audit(data)
                    attempt['offline_geometry_audit_wall_s'] = time.monotonic()-audit_started
                    attempt['velocity_capacity'] = pure_vertical_speed_capacity(
                        model, data, env.arm_jids, env.racket_site_id, np.asarray(env.arm_vel_limit_rad_s))
                attempts.append(attempt)
            valid = any(a.get('self_collision', {}).get('status') == 'PASS' for a in attempts)
            report['rows'].append(dict(id=index, requested_racket_xyz_m=point.tolist(),
                status='STATIC_POSE_FEASIBLE' if valid else 'UNKNOWN', attempts=attempts))
            if index % 16 == 0:
                print(f'[VERIFY] IK points={index+1}/{len(grid)}', flush=True)
            if (index+1) % 32 == 0:
                report['atlas_status'] = 'INCOMPLETE'
                report['wall_s'] = time.monotonic()-started
                partial = args.output.with_name(f'{args.output.stem}.partial_{index+1:03d}.json')
                write_json_unique(partial, report, budget)
        report['atlas_status'] = 'SAMPLED_STATIC_GEOMETRY_ONLY'
        report['static_feasible_points'] = sum(r['status'] == 'STATIC_POSE_FEASIBLE' for r in report['rows'])
        report['unknown_points'] = len(report['rows'])-report['static_feasible_points']
    report['wall_s'] = time.monotonic()-started
    write_json_unique(args.output, report, budget)
    print(f'[SAVE] {args.output}', flush=True)
    if ((args.nominal_only and nominal['status'] != 'PASS')
            or (not args.nominal_only and not report['static_feasible_points'])):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
