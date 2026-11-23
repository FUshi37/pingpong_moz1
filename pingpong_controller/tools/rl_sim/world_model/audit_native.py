"""Bounded CPU audit of the actual WMJ/V324 constructor, camera and full seed."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_json_unique
from .contracts import config_hash, load_config, repository_root
from .geometry import compiled_geometry, audit_differentiable_fk
from .native_stack import build_native_environment
from .observation_engine import advance_world_observation, actor_observation
from .native_observation import camera_from_native, NativeObservationAdapter
from .resources import StorageBudget
from .state_adapter import pack_numeric_state, restore_numeric_state


def compare_camera(environment, state, *, noisy: bool) -> dict:
    """Compare 200 native camera ticks, sparse history and the unchanged prefix."""
    from fixed_plane_v92 import rotation_matrix_to_6d
    adapter = NativeObservationAdapter(environment, state)
    history, _ = adapter.initialize(state)
    imagined_observation = adapter.world_observation_state(state, history)
    advance = jax.jit(lambda current, q, truth: advance_world_observation(
        current, q, truth, ar1_enabled=noisy, pose_from_q=adapter.pose_from_q))
    bpos = state.data.qpos[:, environment.ball_qadr:environment.ball_qadr+3]
    bvel = state.data.qvel[:, environment.ball_vadr:environment.ball_vadr+3]
    prefix_error = 0.
    camera_error = 0.
    history_error = 0.
    for index in range(1, 201):
        # Known changing truth distinguishes held samples from fresh input.
        truth = bpos + jnp.array([.0001, -.0002, .0003]) * index
        state = state._replace(step_count=state.step_count + 1)
        state, native_obs, _ = environment._apply_observation_pipeline(state, truth, bvel)
        imagined_observation, _ = advance(imagined_observation,
            state.data.qpos[:, environment.arm_qadr], truth)
        camera = imagined_observation.camera
        history, diagnostics = adapter.advance(state, history)
        if np.any(np.asarray(diagnostics['capacity_exceeded'])):
            raise ValueError('Native 90Hz samples exceeded the registered history capacity')
        expected = camera_from_native(state)
        for left, right in zip(jax.tree.leaves(camera), jax.tree.leaves(expected)):
            if np.asarray(left).dtype.kind == 'f':
                np.testing.assert_allclose(left, right, rtol=0., atol=1e-6)
                camera_error = max(camera_error, float(np.max(np.abs(np.asarray(left-right)))))
            else:
                np.testing.assert_array_equal(left, right)
        for left, right in zip(jax.tree.leaves(imagined_observation.history),
                               jax.tree.leaves(history)):
            if np.asarray(left).dtype.kind == 'f':
                np.testing.assert_allclose(left, right, rtol=0., atol=1e-6)
                history_error = max(history_error, float(np.max(np.abs(np.asarray(left-right)))))
            else:
                np.testing.assert_array_equal(left, right)
        q = state.data.qpos[:, environment.arm_qadr]
        dq = state.data.qvel[:, environment.arm_vadr]
        pose = jnp.concatenate((state.data.site_xpos[:, environment.racket_site_id],
            rotation_matrix_to_6d(state.data.site_xmat[:, environment.racket_site_id]
                                 .reshape(-1, 3, 3))), axis=-1)
        goal = jnp.concatenate((state.racket_anchor[:, :2],
                               jnp.full((environment.n_envs, 1), 1.425)), axis=-1)
        obs73 = actor_observation(q, dq, state.arm_cmd_q, state.prev_action, pose, camera, goal)
        np.testing.assert_allclose(obs73[:, :72], native_obs, rtol=0., atol=1e-6)
        extended = adapter.observation(state, history, goal)
        np.testing.assert_allclose(extended[:, :73], obs73, rtol=0., atol=1e-6)
        prefix_error = max(prefix_error, float(np.max(np.abs(np.asarray(obs73[:, :72]-native_obs)))))
    rows = np.asarray(extended[:, 73:]).reshape(environment.n_envs, 64, 14)
    if not np.all((rows[:, :, 13].sum(axis=1) >= 57) & (rows[:, :, 13].sum(axis=1) <= 59)):
        raise ValueError('Expected roughly 58 fresh ball/racket pairs per 640ms')
    return dict(status='PASS', native_ticks=200, ar1_enabled=noisy,
                complete_camera_state_numerically_equal=True, float_atol=1e-6,
                rng_clock_and_masks_exact=True, camera_max_error=camera_error,
                history_max_error=history_error, actor_prefix_max_error=prefix_error,
                world_observation_roll_forward_matches_native=True,
                actor_shape=list(extended.shape), paired_history_samples=rows[:, :, 13].sum(axis=1).tolist(),
                sparse_racket_history=True, joint_action_history_in_actor=False,
                probe_scope='native_camera_pipeline_with_synthetic_ball_input_and_stationary_robot')


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--physical-step', action='store_true')
    parser.add_argument('--observe-substeps', action='store_true')
    args = parser.parse_args(argv)
    if args.observe_substeps and not args.physical_step:
        parser.error('--observe-substeps requires --physical-step')
    started = time.monotonic()
    cfg = load_config()
    resource = cfg['resources']
    budget = StorageBudget(repository_root()/resource['output_root'],
                           resource['max_new_bytes'], resource['min_free_bytes'])
    budget.check(additional_bytes=8*1024**2)
    print('[RUN] WMJ-V1 native CPU audit; n=2; S01/S05; training=off', flush=True)
    bundle = build_native_environment(cfg, n_envs=2)
    environment = bundle.environment
    keys = jax.random.split(jax.random.PRNGKey(2026110501), 2)
    state, obs = environment.reset(keys)
    geometry = compiled_geometry(environment, state)
    geometry['differentiable_fk'] = audit_differentiable_fk(environment, state)
    camera_quiet = compare_camera(environment, state, noisy=False)
    print('[VERIFY] S01 native camera/prefix parity passed', flush=True)
    noisy = build_native_environment(cfg, n_envs=2, stage_index=4, stack=bundle.stack).environment
    noisy_state, _ = noisy.reset(keys)
    camera_noisy = compare_camera(noisy, noisy_state, noisy=True)
    print('[VERIFY] S05 native AR1 camera/prefix parity passed', flush=True)
    # Model is episode-constant context; retain ALL data/control/history/RNG
    # leaves. Reattach that exact context after a verified numerical restore.
    context = config_hash(dict(config=config_hash(cfg), geometry=geometry['geometry_sha256']))
    stripped = state._replace(model=None)
    adapter = NativeObservationAdapter(environment, state)
    history, _ = adapter.initialize(state)
    complete = dict(native=stripped, observation_history=history)
    arrays, manifest = pack_numeric_state(complete, context_sha256=context)
    restored_bundle = restore_numeric_state(complete, arrays, manifest, context_sha256=context)
    restored = restored_bundle['native']._replace(model=state.model)
    restored_history = restored_bundle['observation_history']
    restored_obs = environment._make_obs(restored, restored.ball_obs_valid_pos,
        restored.ball_obs_valid_vel, restored.ball_obs_age_seconds)
    np.testing.assert_array_equal(restored_obs, obs)
    goal = jnp.concatenate((state.racket_anchor[:, :2], jnp.full((environment.n_envs, 1), 1.425)), axis=-1)
    np.testing.assert_array_equal(adapter.observation(state, history, goal),
                                 adapter.observation(restored, restored_history, goal))
    step_report = dict(status='NOT_RUN')
    observer_report = dict(status='NOT_RUN')
    if args.physical_step:
        print('[VERIFY] paired full-state common-action 5x1ms step starting', flush=True)
        action = jnp.array([[.001, -.002, .001, 0., 0., 0., 0.]] * 2)
        left = environment.step(state, action)
        right = environment.step(restored, action)
        jax.block_until_ready((left, right))
        from train_fixed_plane_ball_obs_noise import compare_numeric_leaves
        differences = compare_numeric_leaves(left, right)
        if differences['changed_leaves']:
            raise ValueError(f'Full-state replay differs: {differences}')
        left_history, _ = adapter.advance(left[0], history)
        right_history, _ = adapter.advance(right[0], restored_history)
        history_diff = compare_numeric_leaves(left_history, right_history)
        if history_diff['changed_leaves']:
            raise ValueError(f'Paired observation history replay differs: {history_diff}')
        step_report = dict(status='PASS', controls=1, physics_substeps=5,
                           phases=['native_reset_state'], differences=differences,
                           paired_history_differences=history_diff)
        if args.observe_substeps:
            from .teacher_trace import instrument_environment
            identity = instrument_environment(environment, bundle.stack)
            recorded = environment.step(state, action)
            jax.block_until_ready(recorded)
            metadata = dict(recorded[-1])
            trace = metadata.pop('wmj_trace')
            observed_pair = compare_numeric_leaves(left, (*recorded[:-1], metadata))
            if observed_pair['changed_leaves']:
                raise ValueError(f'Observation changed native physics: {observed_pair}')
            np.testing.assert_array_equal(trace['q_before'][0], state.data.qpos[:, environment.arm_qadr])
            np.testing.assert_array_equal(trace['q_after'][-1], recorded[0].data.qpos[:, environment.arm_qadr])
            np.testing.assert_array_equal(trace['q_after'][:-1], trace['q_before'][1:])
            observer_report = dict(status='PASS', identity=identity,
                differences=observed_pair, phases=['native_reset_state'],
                trace_shapes={k: list(v.shape) for k, v in trace.items()})
    report = dict(schema='wmj_native_cpu_audit_v1', training_version='WMJ-V1',
        timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        config_hash=config_hash(cfg), native_source=str(bundle.stack.snapshot),
        actor_source_sha256=cfg['lineage']['actor_source']['sha256'],
        overrides=bundle.overrides, geometry=geometry,
        camera=[camera_quiet, camera_noisy],
        seed_roundtrip=dict(status='PASS', leaf_count=manifest['leaf_count'],
                            uncompressed_bytes=sum(v.nbytes for v in arrays.values())),
        common_action_replay=step_report, substep_observer=observer_report,
        wall_s=time.monotonic()-started,
        backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
        ready_for_training=False,
        remaining=['flight_precontact_in_contact_seed_replay', 'domain_atlas',
                   'CUDA_pairing', 'world_model_and_imagination_training'])
    write_json_unique(args.output, report, budget)
    print(f'[SAVE] native CPU audit: {args.output}', flush=True)


if __name__ == '__main__':
    main()
