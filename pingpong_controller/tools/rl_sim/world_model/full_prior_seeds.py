"""Reconstruct complete full-chain teacher seeds from a coherent replay.

Only this offline seed builder reads recorded future trajectories. Each seed
contains information available by its own timestamp; imagined transitions
receive a selected seed and cannot consult the rest of this bank.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import read_array_archive
from .contracts import file_hash
from .control_rmp import apply_output_delay, rmp_step_lanes
from .full_prior_world import FullPriorWorldState, measured_racket_pose
from .observation_engine import (
    CameraState, ObservationHistory, WorldObservationState,
    advance_observation_history, advance_world_observation,
)
from .paired_history import init_feedback, init_history


CAMERA_FIELDS = ('rng', 'step_count', 'last_ball_obs_step', 'cached_ball_obs_pos',
    'actor_ball_position_history', 'actor_ball_age_history',
    'ball_position_noise_epsilon_m', 'fixed_contact_ball_observed')


def read_initial_camera(reference, *, context_sha256, lanes):
    """Authenticate reset camera memory without loading a full native model."""
    path = Path(reference['manifest'])
    if file_hash(path) != reference['sha256']:
        raise ValueError('Camera seed manifest changed')
    metadata = json.loads(path.read_text())
    if metadata['schema'] != 'wmj_numeric_seed_v1' or metadata['context_sha256'] != context_sha256:
        raise ValueError('Camera seed context changed')
    lookup = {row['path']:(f'leaf_{i:06d}', row) for i,row in enumerate(metadata['leaves'])}
    required = ["['native']."+name for name in CAMERA_FIELDS]
    arrays = read_array_archive(path.parent/metadata['archive_file'], metadata['archive'],
                                names={lookup[key][0] for key in required})
    selected = np.asarray(lanes)
    values = []
    for key in required:
        name, expected = lookup[key]
        value = arrays[name]
        if (list(value.shape) != expected['shape'] or value.dtype.str != expected['dtype']
                or hashlib.sha256(value.tobytes()).hexdigest() != expected['sha256']):
            raise ValueError('Camera seed leaf changed: '+key)
        if (selected.ndim != 1 or selected.dtype.kind not in 'iu' or not len(selected)
                or np.any(selected < 0) or np.any(selected >= len(value))):
            raise ValueError('Invalid camera seed lanes')
        values.append(jnp.asarray(value[selected]))
    camera = CameraState(*values)
    if np.any(np.asarray(camera.step_count) != 0) or np.any(np.asarray(camera.last_sample_step) != 0):
        raise ValueError('Full history reconstruction must begin at an actual reset')
    return camera


def initial_full_history(camera, q_rad, *, chain):
    """Create the one real reset row, retaining exact realized camera noise."""
    if np.any(np.asarray(camera.step_count) != 0):
        raise ValueError('Cannot invent earlier full-chain history at a mid-episode seed')
    n = q_rad.shape[0]
    empty = ObservationHistory(init_feedback(n, n_joints=13), init_history(n))
    history, _ = advance_observation_history(empty, q_rad, camera.held_position_m,
        camera.last_sample_step*5000, camera.step_count*5000, camera.observed,
        pose_from_q=lambda q:measured_racket_pose(chain, q))
    return WorldObservationState(camera, history)


def reconstruct_teacher_bank(initial, physical_endpoints, cached_poses, references,
                              accepted_actions, *, context, control_contract, chain,
                              ar1_enabled):
    """Warm controller/camera/history along recorded states without learned repair.

    References here are actual native published targets. The exact common-
    reference RMP path was separately audited; returned commands and camera
    samples must also be compared to this replay before using its seeds.
    """
    c = control_contract

    def tick(state, values):
        physics, cached_pose, reference, accepted = values
        rmp, commands = rmp_step_lanes(state.control.rmp, reference, context.rmp,
            q_min=c.projection.q_lower, q_max=c.projection.q_upper,
            velocity_limit=c.projection.velocity_limit,
            acceleration_limit=c.acceleration_limit_rad_s2,
            jerk_limit=c.jerk_limit_rad_s3, params=c.rmp_params)
        delayed, buffers = {}, {}
        for key in ('q', 'dq', 'ddq'):
            delayed[key], buffers[key] = apply_output_delay(state.control.output_delay[key],
                commands[key], context.output_delay_steps)
        control = state.control._replace(reference_q_rad=reference,
            previous_accepted_action=accepted, rmp=rmp, output_delay=buffers)
        observation, _ = advance_world_observation(state.observation, physics.q_rad,
            physics.ball.position_m, pose_from_q=lambda q:measured_racket_pose(chain, q),
            ar1_enabled=ar1_enabled)
        after = FullPriorWorldState(physics, control, observation, cached_pose)
        return after, (after, delayed)

    _, (after, commands) = jax.lax.scan(tick, initial,
        (physical_endpoints, cached_poses, references, accepted_actions))
    bank = jax.tree.map(lambda first, rest:jnp.concatenate((first[None], rest)), initial, after)
    return bank, commands


def select_teacher_seeds(bank, start_controls):
    """Gather one time per lane and cut all pre-branch gradients."""
    times = jnp.asarray(start_controls, jnp.int32)
    lanes = jnp.arange(times.shape[0])
    return jax.tree.map(lambda x:jax.lax.stop_gradient(x[times, lanes]), bank)
