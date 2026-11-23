"""No-update course integration with explicit preflight and commit boundaries."""

from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .branch_bank import prepare_training_window
from .actor_critic import average_gradients
from .contracts import config_hash, file_hash
from .course_goals import validate_start_goals
from .course_sampling import COURSE_COUNTS, CATEGORIES
from .course_rollout import course_actor_objective, validate_course_window
from .imagination_checkpoint import runtime_signature
from .imagination_learner import (actor_gradient_window, commit_policy_windows,
    critic_training_arrays, slice_environment)


class CompactCourseWindow(NamedTuple):
    """Complete endpoint/captures plus valid critic rows; discard used features."""
    environment: Any
    exploration_rng: Any
    transitions: Any
    critic_inputs: Any
    critic_targets: Any
    captures: Any


def compact_course_window(window, target, *, lambda_value=.99):
    """Compute detached returns before releasing large next-state feature arrays."""
    validate_course_window(window)
    if np.any(np.asarray(window.transitions.valid)):
        inputs, targets = critic_training_arrays([window],target,lambda_value=lambda_value)
    else:
        inputs = np.empty((0,window.critic_inputs.shape[-1]),np.float32)
        targets = np.empty((0,),np.float32)
    transitions = window.transitions._replace(next_critic_features=
        jnp.zeros(window.transitions.valid.shape+(0,)))
    return CompactCourseWindow(*jax.device_get((window.environment,window.exploration_rng,
        transitions,inputs,targets,window.captures)))


def compact_critic_arrays(windows):
    if not windows or not sum(len(w.critic_targets) for w in windows):
        raise ValueError('No valid imagined critic rows')
    return (np.concatenate([w.critic_inputs for w in windows]),
            np.concatenate([w.critic_targets for w in windows]))


def validate_course_config(cfg):
    """Keep approved executable course constants bound to their JSON contract."""
    critic = cfg['imagination_implementation']['critic']
    if (critic['input_dimension'] != 2602
            or critic['schema'] != 'wmj_v1_critic_course_current_force'
            or critic['hidden_sizes'] != [256, 256]
            or critic['current_sequence_feature_scales'] != dict(
                completed_goals=4., pulse_used=1., remaining_steps=20., force_world_n=.02)):
        raise ValueError('Unregistered course critic layout or physical scales')
    expected = dict(goals_per_group=4, groups_physically_continuous=True, pulse_per_group=1,
        trigger='uniform_one_of_three_internal_goal_switch_apexes',
        onset='next_1ms_boundary_after_detected_apex', duration_1ms_steps=20,
        early_failure_or_time_limit='record_incomplete_without_forced_pulse')
    for stage in cfg['curriculum']['stages']:
        if stage['id'] in ('S04', 'S05') and stage.get('training_sequence') != expected:
            raise ValueError('Unregistered continuous sequence/pulse contract')
    actual = cfg['samplers']['P_start']['course_counts']
    if actual != {stage:dict(zip(CATEGORIES, counts)) for stage,counts in COURSE_COUNTS.items()}:
        raise ValueError('Unregistered course-dependent branch counts')


def make_course_gradient(robot, auxiliary, table, **options):
    """Compile a fixed-table microbatch; no parameter or optimizer mutation."""
    def objective(actor, target, environment, keys, shape_template):
        return course_actor_objective(actor, target, robot, auxiliary,
            environment, keys, table, horizon=shape_template.shape[0], **options)
    return jax.jit(jax.value_and_grad(objective, argnums=0, has_aux=True))


def course_gradient_window(session, table, gradient_fn, *, stage_id,
                           microbatch=256, horizon=64, compact=False, lambda_value=.99):
    """Use the registered logical batch; reject shortages before any commit."""
    state = session.pool.environment
    validate_start_goals(table, state.task.goal_xyz, stage_id=stage_id)
    template = jnp.zeros((horizon, state.episode_steps.shape[0], 3))
    if compact:
        n = state.episode_steps.shape[0]
        if microbatch < 1 or n % microbatch:
            raise ValueError('Microbatches must exactly partition logical environments')
        gradients, windows, losses = [], [], []
        for start in range(0,n,microbatch):
            end = start+microbatch
            (loss,window),gradient = gradient_fn(session.actor,session.target_critic,
                slice_environment(state,start,end),session.pool.exploration_rng[start:end],
                template[:,start:end])
            jax.block_until_ready((loss,window,gradient))
            if any(not np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradient)):
                raise FloatingPointError('Nonfinite actor gradient; no optimizer update applied')
            gradients.append(gradient); losses.append(float(loss))
            windows.append(compact_course_window(window,session.target_critic,lambda_value=lambda_value))
            del window
        return average_gradients(gradients,[microbatch]*len(gradients)),windows,dict(
            loss=float(np.mean(losses)),logical_envs=n,microbatch=microbatch,
            microbatches=len(windows),actor_updates=0,critic_updates=0,
            includes_detached_critic_return_preparation=True)
    gradient, windows, metrics = actor_gradient_window(
        session, template, gradient_fn, microbatch=microbatch)
    for window in windows:
        validate_course_window(window)
    return gradient, windows, metrics


def commit_course_windows(session, windows):
    for window in windows:
        validate_course_window(window)
    return commit_policy_windows(session, windows)


def prepare_course_session(session, fresh_persistent, bank, *, minimum_precontact_lead_s):
    """All resampling/exploration/camera randomness descends from saved RNG."""
    next_key, draw_key, exploration_key = jax.random.split(session.sampling_rng, 3)
    rng = np.random.default_rng(np.asarray(draw_key, dtype=np.uint32))
    keys = jax.random.split(exploration_key, 1024)
    pool, report = prepare_training_window(session.pool, fresh_persistent, bank, rng, keys,
        minimum_precontact_lead_s=minimum_precontact_lead_s)
    return session._replace(pool=pool, sampling_rng=next_key), report


def course_runtime_signature(cfg, **references):
    """Bind course/sampler/control implementations and exact current layout."""
    parent = runtime_signature(cfg, **references)
    sources = sorted(Path(__file__).parent.glob('*.py'))
    return config_hash(dict(core=parent, curriculum=cfg['curriculum'],
        samplers=cfg['samplers'], sources={p.name:file_hash(p) for p in sources}))
