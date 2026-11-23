"""One-step native-MJX versus world projection from the same full1024-lane seed."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_json_unique
from .audit_support_replay import load_support_chain
from .contracts import config_hash, file_hash, load_config
from .control_projection import project_reference
from .fit_robot import summarize_errors
from .runtime_guard import RunGuard
from .support_data import SupportReplayReader
from .world_control import ControlContract
from .kinematics import forward_kinematics
from .control_kinematics import extract_control_chain, forward_control_kinematics
from cartesian_pose_action_projection_jax import (
    matrix_to_euler_xyz_jax, relative_xyz_euler_rate_jacobian_jax,
)


def projection_inputs(env, native, chain, q, *, kinematics=forward_kinematics):
    """Expose same-clock native/FK inputs solely for implementation diagnosis."""
    reference = env.batched_forward(native.model, native.data.replace(
        qpos=native.data.qpos.at[:, env.arm_qadr].set(native.arm_cmd_q),
        qvel=native.data.qvel.at[:, env.arm_vadr].set(0.)))
    own = jax.vmap(lambda x: kinematics(chain, x))(
        q.at[:, :7].set(native.arm_cmd_q))
    position = reference.site_xpos[:, env.racket_site_id]
    axis = reference.xaxis[:, env.arm_jids]
    anchor = reference.xanchor[:, env.arm_jids]
    native_values = (position, reference.site_xmat[:, env.racket_site_id].reshape((-1, 3, 3)),
        jnp.swapaxes(jnp.cross(axis, position[:, None]-anchor), 1, 2), jnp.swapaxes(axis, 1, 2))
    own_values = (own.site_position, own.site_rotation, own.site_jacobian[..., :7], own.angular_jacobian[..., :7])

    def transform(values):
        p, r, jp, jr = values
        rpy = matrix_to_euler_xyz_jax(jnp.einsum('ji,njk->nik', env.fixed_contact_racket_reference_rotation, r))
        return dict(position=jnp.einsum('ij,nj->ni', env.cartesian_world_to_policy_rotation,
                        p-env.cartesian_policy_origin_world[None]), rpy=rpy,
            translation_jacobian=jnp.einsum('ij,njk->nik', env.cartesian_world_to_policy_rotation, jp),
            rpy_jacobian=relative_xyz_euler_rate_jacobian_jax(rpy, env.fixed_contact_racket_reference_rotation, jr))
    return dict(native=transform(native_values), world=transform(own_values))


def highest_matrix_fk(chain, q):
    """Isolate FK dot precision without changing native/projector precision."""
    with jax.default_matmul_precision('highest'):
        return forward_kinematics(chain, q)


def audit(cfg, reader, guard, *, include_quaternion=False):
    from .native_stack import build_native_environment
    from .native_observation import NativeObservationAdapter
    from .physical_events import init_physical_events
    from .replay_support import _restore
    from mjx_topology_execution import make_collect

    names = list(reader.records)
    record = reader.records[names[0]]
    source = json.loads((Path(reader.report['source_collection'])/'result.json').read_text())
    n = record['record']['lanes']
    bundle = build_native_environment(cfg, n_envs=n)
    env = bundle.environment
    fresh, _ = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(source['seed']), n))
    jax.block_until_ready(fresh)
    guard.check()
    context = source['context']['sha256']
    model = _restore(source['context']['reference'], fresh.model, context)
    adapter = NativeObservationAdapter(env, fresh)
    history, _ = adapter.initialize(fresh)
    template = dict(native=fresh._replace(model=None), history=history,
        events=init_physical_events(n), ball_model_valid=jnp.ones(n, bool),
        goal=jnp.zeros((n, 3), jnp.float32))
    full = _restore(record['first_seed_ref'], template, context)
    native = full['native']._replace(model=model)
    raw = reader.read(names[0], names={'action__raw_normalized', 'action__accepted_normalized', 'state__q_ref_rad'})
    action = jnp.asarray(raw['action__raw_normalized'][0])
    velocity = jnp.clip(jnp.clip(action, -1., 1.)*env.arm_vel_limit_rad_s[None]
        *float(env.cfg.action_velocity_scale)*native.action_scale_mult[:, None],
        -env.arm_vel_limit_rad_s[None], env.arm_vel_limit_rad_s[None])
    scoped, execution = make_collect(lambda state,v:env._project_cartesian_pose_reference(
        state, state.data.qpos[:, env.arm_qadr], v), mode='project')
    native_result = jax.device_get(jax.jit(scoped)(native, velocity))
    guard.check()
    mj_model, chain = load_support_chain(cfg, reader.report['geometry'])
    q = native.data.qpos[:, mj_model.jnt_qposadr[np.asarray(chain.joint_ids)]]
    contract = replace(ControlContract.from_native(env).projection, fk_matmul_precision=None)
    world = jax.device_get(jax.jit(lambda q,reference,v:project_reference(
        chain, contract, q, reference, v))(q, native.arm_cmd_q, velocity))
    inputs = jax.device_get(jax.jit(lambda s,q: projection_inputs(env,s,chain,q))(native,q))
    if include_quaternion:
        quaternion_chain = extract_control_chain(mj_model, reader.report['geometry']['reset_fk']['qpos'],
            joint_ids=chain.joint_ids, site_id=reader.report['geometry']['racket']['site_id'],
            geom_id=reader.report['geometry']['racket']['geom_id'])
        quaternion_world = jax.device_get(jax.jit(lambda q,reference,v:project_reference(
            quaternion_chain, contract, q, reference, v, kinematics=forward_control_kinematics))(
                q, native.arm_cmd_q, velocity))
        quaternion_inputs = jax.device_get(jax.jit(lambda s,q: projection_inputs(env,s,quaternion_chain,q,
            kinematics=forward_control_kinematics))(native,q))
    highest_world = jax.device_get(jax.jit(lambda q,reference,v:project_reference(
        chain, contract, q, reference, v, kinematics=highest_matrix_fk))(q,native.arm_cmd_q,velocity))
    highest_inputs = jax.device_get(jax.jit(lambda s,q: projection_inputs(env,s,chain,q,
        kinematics=highest_matrix_fk))(native,q))
    guard.check()
    denominator = np.asarray(env.arm_vel_limit_rad_s[None]*native.action_scale_mult[:, None])
    predictions = dict(native_scoped=dict(reference=np.asarray(native_result[0]),
        action=np.clip(np.asarray(native_result[1])/denominator, -1., 1.),
        fallback=np.asarray(native_result[6])),
        world_scoped=dict(reference=np.asarray(world.reference_q),
            action=np.clip(np.asarray(world.accepted_qvel)/denominator, -1., 1.),
            fallback=np.asarray(world.fallback_to_hold)),
        highest_matrix_world=dict(reference=np.asarray(highest_world.reference_q),
            action=np.clip(np.asarray(highest_world.accepted_qvel)/denominator, -1., 1.),
            fallback=np.asarray(highest_world.fallback_to_hold)))
    if include_quaternion:
        predictions['quaternion_world'] = dict(reference=np.asarray(quaternion_world.reference_q),
            action=np.clip(np.asarray(quaternion_world.accepted_qvel)/denominator, -1., 1.),
            fallback=np.asarray(quaternion_world.fallback_to_hold))
    labels = dict(reference=raw['state__q_ref_rad'][1], action=raw['action__accepted_normalized'][0])
    validation = np.asarray([g['split'] == 'validation' for g in record['record']['groups']])
    reports = {}
    for scope, keep in (('all_lanes_identity_diagnostic', np.ones(n, bool)), ('validation', validation)):
        comparisons = {}
        for mode, values in predictions.items():
            comparisons[mode+'_vs_recorded'] = {key:summarize_errors([np.abs(values[key][keep]-labels[key][keep]).ravel()])
                                                for key in ('reference', 'action')}
        comparisons['world_vs_native'] = {key:summarize_errors([np.abs(
            predictions['world_scoped'][key][keep]-predictions['native_scoped'][key][keep]).ravel()])
            for key in ('reference', 'action')}
        comparisons['world_native_fallback_disagreements'] = int(np.sum(keep &
            (predictions['world_scoped']['fallback'] != predictions['native_scoped']['fallback'])))
        if include_quaternion:
            comparisons['quaternion_native_fallback_disagreements'] = int(np.sum(keep &
                (predictions['quaternion_world']['fallback'] != predictions['native_scoped']['fallback'])))
        comparisons['highest_matrix_native_fallback_disagreements'] = int(np.sum(keep &
            (predictions['highest_matrix_world']['fallback'] != predictions['native_scoped']['fallback'])))
        reports[scope] = comparisons
    mismatch = predictions['world_scoped']['fallback'] != predictions['native_scoped']['fallback']
    input_errors = {key:summarize_errors([np.abs(inputs['native'][key]-inputs['world'][key]).ravel()])
                    for key in inputs['native']}
    quaternion_input_errors = None
    if include_quaternion:
        quaternion_input_errors = {key:summarize_errors([np.abs(quaternion_inputs['native'][key]
            -quaternion_inputs['world'][key]).ravel()]) for key in quaternion_inputs['native']}
    highest_input_errors = {key:summarize_errors([np.abs(highest_inputs['native'][key]
        -highest_inputs['world'][key]).ravel()]) for key in highest_inputs['native']}
    cases = [dict(lane=int(i), split=record['record']['groups'][i]['split'],
        native_fallback=bool(native_result[6][i]), world_fallback=bool(world.fallback_to_hold[i]),
        native_linear_violation=float(native_result[10][i]), world_linear_violation=float(world.linear_max_violation[i]),
        native_current_safe=bool(native_result[7][i]), world_current_safe=bool(world.current_safe[i]),
        input_maximum_error={key:float(np.max(np.abs(inputs['native'][key][i]-inputs['world'][key][i])))
                            for key in inputs['native']}) for i in np.flatnonzero(mismatch)]
    return dict(n_envs=n, execution=execution, seed_reference=record['first_seed_ref'],
                comparisons=reports, input_errors=input_errors, fallback_cases=cases,
                quaternion_input_errors=quaternion_input_errors,
                highest_input_errors=highest_input_errors,
                default_matmul_precision=jax.config.jax_default_matmul_precision, use_for_model_fitting=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support-replay', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--include-quaternion', action='store_true')
    parser.add_argument('--max-wall-s', type=float, default=900.)
    args = parser.parse_args(argv)
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, args.max_wall_s) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        paths = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_full_seed_projection_diagnostic_v1', training_version='WMJ-V1',
            status='INCOMPLETE', config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), invocation=sys.argv,
            source_sha256={p.name:file_hash(p) for p in paths},
            actor_updates=0, critic_updates=0, world_model_ready=False)
        write_json_unique(args.output_dir/'manifest.json',
            {**report, 'source_snapshot':{p.name:p.read_text() for p in paths}}, guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA GPU is required')
            reader = SupportReplayReader(args.support_replay, guard.budget, max_bytes=0)
            report.update(diagnostic=audit(cfg, reader, guard, include_quaternion=args.include_quaternion), dataset_sha256=reader.identity,
                          status='DIAGNOSTIC_COMPLETE_NOT_WORLD_QUALIFIED')
        except InterruptedError as exc:
            report.update(status='INTERRUPTED', error=str(exc))
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            digest = write_json_unique(args.output_dir/'result.json', report, guard.budget)
            print(dict(path=str(args.output_dir), sha256=digest, status=report['status']), flush=True)


if __name__ == '__main__':
    main()
