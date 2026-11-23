"""One native acquisition with read-only solver rows; no model fitting."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import read_array_archive, write_json_unique
from .contracts import config_hash, file_hash, load_config
from .fit_robot import summarize_errors
from .replay_support import _restore
from .runtime_guard import RunGuard
from .soft_contact import contract_from_collection, softness
from .trajectory_data import TrajectoryDataset


def solver_rows(env, previous, current, prior):
    """Select the native ball/racket row without changing the physical carry."""
    from .teacher_trace import contact_pair_masks
    pair, _ = contact_pair_masks(current.contact.geom, current.contact.dist,
        ball_geom_id=env.ball_geom_id, racket_geom_id=env.racket_contact_geom_id)
    slot = jnp.argmax(pair, axis=-1)
    lane = jnp.arange(pair.shape[0])
    address = jnp.asarray(current.contact.efc_address)[slot]
    jacobian = current.efc_J[lane, address]
    dist = current.contact.dist[lane, slot]
    d = current.efc_D[lane, address]
    aref = current.efc_aref[lane, address]
    force = current.efc_force[lane, address]
    velocity = jnp.sum(jacobian*previous.qvel, axis=-1)
    acceleration = jnp.sum(jacobian*current.qacc, axis=-1)
    k, b, imp = softness(prior, dist)
    expected_aref = -b*velocity-k*imp*dist
    expected_r = prior.reference_inverse_mass_kg_inv*(1-imp)/imp
    interval = slice(env.ball_vadr, env.ball_vadr+3)
    ball_acceleration = jnp.sum(jacobian[:, interval]*current.qacc[:, interval], axis=-1)
    gravity = jnp.sum(jacobian[:, interval]*jnp.array([0., 0., -9.81]), axis=-1)
    robot_acceleration = acceleration-ball_acceleration
    measured_acceleration = (current.qvel-previous.qvel)/.001
    measured_robot_acceleration = (jnp.sum(jacobian*measured_acceleration, axis=-1)
        -jnp.sum(jacobian[:, interval]*measured_acceleration[:, interval], axis=-1))
    zero_force = jnp.maximum((aref-gravity)/(1/prior.mass_kg+expected_r), 0.)
    solver_force = jnp.maximum((aref-gravity-robot_acceleration)/(1/prior.mass_kg+expected_r), 0.)
    measured_force = jnp.maximum((aref-gravity-measured_robot_acceleration)/(1/prior.mass_kg+expected_r), 0.)
    return dict(active=jnp.any(pair, axis=-1) & (dist < 0.), dist_m=dist,
        force_n=force, normal_velocity_m_s=velocity,
        aref_m_s2=aref, expected_aref_m_s2=expected_aref,
        aref_error_m_s2=aref-expected_aref,
        native_regularizer=1/jnp.maximum(d, 1e-20), expected_regularizer=expected_r,
        regularizer_error=1/jnp.maximum(d, 1e-20)-expected_r,
        normal_constraint_residual_m_s2=acceleration-aref+force/jnp.maximum(d, 1e-20),
        robot_normal_acceleration_m_s2=robot_acceleration,
        measured_robot_normal_acceleration_m_s2=measured_robot_acceleration,
        zero_robot_acceleration_velocity_error_m_s=(zero_force-force)*.001/prior.mass_kg,
        solver_robot_acceleration_velocity_error_m_s=(solver_force-force)*.001/prior.mass_kg,
        measured_robot_acceleration_velocity_error_m_s=(measured_force-force)*.001/prior.mass_kg,
        ball_normal_acceleration_m_s2=ball_acceleration,
        ball_force_balance_m_s2=ball_acceleration-gravity-force/prior.mass_kg,
        force_with_solver_robot_acceleration_n=solver_force,
        solver_niter=current.solver_niter)


def audit(cfg, args, guard, report):
    from .native_stack import build_native_environment
    from .native_observation import NativeObservationAdapter
    from .physical_events import init_physical_events
    from . import teacher_trace
    from mjx_topology_execution import make_collect

    source = json.loads((args.collection/'result.json').read_text())
    dataset = TrajectoryDataset.open(args.collection/'dataset', guard.budget)
    names = list(dataset.chunks)[:2]
    first = dataset.chunks[names[0]]
    n = first['lanes']
    prior, context = contract_from_collection(cfg, source)
    bundle = build_native_environment(cfg, n_envs=n)
    env = bundle.environment
    fresh, _ = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(source['seed']), n))
    jax.block_until_ready(fresh)
    model = _restore(source['context']['reference'], fresh.model, source['context']['sha256'])
    history, _ = NativeObservationAdapter(env, fresh).initialize(fresh)
    template = dict(native=fresh._replace(model=None), history=history,
        events=init_physical_events(n), ball_model_valid=jnp.ones(n, bool),
        goal=jnp.zeros((n, 3), jnp.float32))
    restored = _restore(first['seed_ref'], template, first['context_sha256'])
    native = restored['native']._replace(model=model)
    original_loop = teacher_trace.recorded_loop

    def diagnostic_loop(substep, initial, count, sample):
        def with_rows(index, previous, current):
            return {**sample(index, previous, current),
                    'solver_diagnostic': solver_rows(env, previous[0], current[0], prior)}
        return original_loop(substep, initial, count, with_rows)

    teacher_trace.recorded_loop = diagnostic_loop
    observer = teacher_trace.instrument_environment(env, bundle.stack, support_trace=True)

    def transition(state, action):
        after, _, _, _, metrics = env.step(state, action)
        trace = metrics['wmj_trace']
        return after, trace['solver_diagnostic'], trace['other_ball_collision']

    scoped, execution = make_collect(transition, mode='project')
    step = jax.jit(scoped)
    train = np.array([group['split'] == 'train' for group in first['groups']])
    valid = np.ones(n, bool)
    parts = {}
    report.update(context=context, observer=observer, execution=execution,
        source_result_sha256=file_hash(args.collection/'result.json'),
        source_chunks=names, seed_reference=first['seed_ref'], source_context_sha256=source['context']['sha256'])
    try:
        for name in names:
            record = dataset.chunks[name]
            action = read_array_archive(dataset.root/f'chunk_{name}.npz', record['archive'],
                                        names=['action__raw_normalized'])['action__raw_normalized']
            for tick in range(record['steps']):
                if tick % 16 == 0:
                    guard.check()
                    print(f'[AUDIT] {name} tick={tick}/64', flush=True)
                native, diagnostic, other = step(native, jnp.asarray(action[tick]))
                diagnostic, other = jax.device_get((diagnostic, other))
                allowed = valid[None] & (np.cumsum(other, axis=0) == 0)
                valid = allowed[-1]
                mask = diagnostic['active'] & allowed & train[None]
                for key, value in diagnostic.items():
                    if key != 'active':
                        parts.setdefault(key, []).append(value[mask])
    finally:
        teacher_trace.recorded_loop = original_loop
    values = {key: np.concatenate(value) for key, value in parts.items()}
    report['overall_absolute'] = {key: summarize_errors([np.abs(value)]) for key, value in values.items()}
    pushing = values['force_n'] > 1e-5
    report['positive_force_absolute'] = {key: summarize_errors([np.abs(value[pushing])]) for key, value in values.items()}
    report['solver_niter_note'] = 'Native field only; a zero does not establish zero solver iterations'
    worst = np.argsort(np.abs(values['normal_constraint_residual_m_s2']))[-20:][::-1]
    report['largest_constraint_residual_samples'] = [
        {key: value[index].tolist() for key, value in values.items()} for index in worst]
    report['status'] = 'DIAGNOSTIC_COMPLETE'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--max-wall-s', type=float, default=900.)
    args = parser.parse_args()
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, args.max_wall_s) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        report = dict(schema='wmj_native_contact_solver_audit_v1', training_version='WMJ-V1',
            status='INCOMPLETE', timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            config_sha256=config_hash(cfg), source_sha256=file_hash(Path(__file__)),
            scope='training_only_first128_control_acquisition_native_solver_diagnostic',
            physical_modifications=False, actor_updates=0, critic_updates=0, world_model_ready=False,
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        write_json_unique(args.output_dir/'implementation.json',
            dict(report=report, source=Path(__file__).read_text()), guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            audit(cfg, args, guard, report)
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_json_unique(args.output_dir/'result.json', report, guard.budget)
            print(f'[SAVE] {args.output_dir}; status={report["status"]}', flush=True)


if __name__ == '__main__':
    main()
