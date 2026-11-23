"""Diagnostic-only reward returns and directional checks under corrected control."""
from __future__ import annotations

import json
import jax
import jax.numpy as jnp
import numpy as np

from .actor_critic import ImaginedBatch, actor_return, frozen_value
from .calibrate_velocity_reward import OBJECTIVES
from .contracts import authenticate, repository_root
from .course_env import course_critic_features
from .course_rollout import course_policy_window
from .imagined_rewards import release_reward
from .module_gradient_tools import distribution
from .paired_precision import random32_clone
from .velocity_reward import diagnostic_course_step, velocity_costs


observed_step = diagnostic_course_step
BASE = np.array([1., 0., 0., 0., 0.])
REPLACEMENT = np.array([1., -1., -1., 1., 1.])


def objective_returns(transitions, next_values):
    """Return five (N,) objectives, with the same failure and bootstrap masks."""
    tr = transitions
    base = sum(v for k, v in tr.reward_terms.items() if k not in ('velocity_xy', 'velocity_z'))
    batch = ImaginedBatch(base, tr.discount, tr.trace_continue, tr.valid)
    zeros = jnp.zeros_like(next_values)
    return jnp.stack([actor_return(batch, next_values)] + [
        actor_return(batch._replace(reward=tr.reward_terms[name]), zeros)
        for name in ('apex_xy', 'apex_z', 'velocity_xy', 'velocity_z')], axis=-1)


def objective_function(runtime, initial, keys, table, episode, reward, std):
    """One trajectory scores both rewards without modifying state or task logic."""
    window_fn = random32_clone(course_policy_window, dict(course_step=observed_step(runtime),
        course_critic_features=course_critic_features))
    options = dict(stage_id='S01', control_contract=runtime.control,
                   episode_contract=episode, reward_contract=reward, **runtime.options)
    def function(actor, critic):
        window = window_fn(actor, runtime.robot, runtime.auxiliary, initial, keys, table,
                           horizon=64, exploration_std=std, **options)
        tr = window.transitions
        values = frozen_value(critic, tr.next_critic_features)
        returns = objective_returns(tr, values)
        discrete = jnp.stack((tr.valid, tr.terminated, tr.model_truncated,
            tr.task_events['valid_release'] > 0, tr.task_events['valid_apex'] > 0,
            *[tr.task_events['bound_'+k] for k in ('ball', 'racket', 'orientation', 'joints')]), axis=-1)
        # Clip flags are meaningful only before the first physical/model failure.
        clips = jnp.concatenate((window.raw_action < -1., window.raw_action > 1.), axis=-1)
        modes = jnp.concatenate((discrete, clips & tr.valid[..., None]), axis=-1)
        batch = ImaginedBatch(tr.reward, tr.discount, tr.trace_continue, tr.valid)
        terms = {k:actor_return(batch._replace(reward=v), jnp.zeros_like(values))
                 for k, v in tr.reward_terms.items()}
        terms['target_bootstrap'] = actor_return(batch._replace(reward=jnp.zeros_like(values)), values)
        return returns, dict(modes=modes, valid=tr.valid, raw=window.raw_action,
            accepted=window.accepted_action, terms=terms,
            releases=tr.task_events['valid_release'].sum(axis=0),
            terminated=window.environment.terminated, model_truncated=window.environment.model_truncated)
    return function


def directional_summary(center, repeat, tangent, plus, minus, center_modes,
                        plus_modes, minus_modes, epsilon, weights, criteria):
    """Do not discard tiny, reversed, or changed-mode cases from the denominator."""
    center, repeat, tangent, plus, minus = map(lambda x:np.asarray(x, float),
                                              (center, repeat, tangent, plus, minus))
    delta = (plus-minus) @ weights
    automatic = tangent @ weights
    finite_difference = delta/(2*epsilon)
    noise = np.abs((center-repeat) @ weights)
    changed = np.any((plus_modes != center_modes) | (minus_modes != center_modes), axis=(0, 2))
    finite = np.isfinite(automatic+finite_difference+noise)
    resolved = (~changed & finite & (np.abs(delta) > np.maximum(
        criteria['return_difference_floor'], criteria['repeat_multiplier']*noise)))
    relative = np.abs(finite_difference-automatic)/np.maximum(
        np.maximum(np.abs(finite_difference), np.abs(automatic)), 1e-8)
    agree = automatic*finite_difference > 0.
    count = int(resolved.sum())
    rate = float(agree[resolved].mean()) if count else None
    close = float((relative[resolved] <= criteria['relative_derivative_tolerance']).mean()) if count else None
    return dict(source_count=len(center), resolved_indices=np.flatnonzero(resolved).tolist(),
        unresolved_indices=np.flatnonzero(~resolved).tolist(), mode_changed_indices=np.flatnonzero(changed).tolist(),
        resolved_count=count, direction_agreement_fraction=rate,
        relative_derivative_pass_fraction=close, stable_relative_error=distribution(relative[resolved]),
        automatic=automatic.tolist(), finite_difference=finite_difference.tolist(),
        central_return_difference=delta.tolist(), repeat_return_difference=noise.tolist(),
        relative_error=relative.tolist(), diagnostic_pass=bool(count >= criteria['minimum_resolved_sources']
            and rate >= criteria['minimum_pass_fraction'] and close >= criteria['minimum_pass_fraction']))


def native_reward_calibration(plan, reward):
    """Score archived a44 release states; native labels never enter a learner."""
    result = json.loads(authenticate(repository_root(), plan['native_actions']).read_text())
    if result['status'] != 'FROZEN_CORRECTED_CANDIDATE_AUDIT_COMPLETE':
        raise ValueError('Incomplete native action evidence')
    goals = {i:np.asarray(row['first_goal_xyz_m']) for i, row in zip(plan['source_indices'], plan['source_rows'])}
    rows = []
    for group in result['action_effects']['groups']:
        for lane, source in enumerate(group['source_indices']):
            row = dict(source_index=source, goal_xyz_m=goals[source].tolist(), outcomes={})
            for model, branches in [('native', group['native']),
                                   ('candidate_u12288', group['world']['candidate_u12288'])]:
                scored = {}
                for branch in ('base', 'plus', 'minus', 'repeat'):
                    event = branches[branch][lane]
                    if not event['release_present']:
                        scored[branch] = dict(valid=False)
                        continue
                    p, v, g = [jnp.asarray(x)[None] for x in
                        (event['release_position_m'], event['outgoing_velocity_m_s'], goals[source])]
                    old = release_reward(p, v, g, jnp.array([True]), contract=reward)
                    new, info = velocity_costs(p, v, g, jnp.array([True]), contract=reward)
                    scored[branch] = dict(valid=bool(info['valid'][0]), release_position_m=event['release_position_m'],
                        velocity_m_s=event['outgoing_velocity_m_s'], target_velocity_m_s=np.asarray(info['target_velocity_m_s'][0]).tolist(),
                        scales_m_s=np.asarray(info['scale_xyz_m_s'][0]).tolist(),
                        old_xy=float(old['apex_xy'][0]), old_z=float(old['apex_z'][0]),
                        velocity_xy=float(new['velocity_xy'][0]), velocity_z=float(new['velocity_z'][0]))
                row['outcomes'][model] = scored
            rows.append(row)
    return dict(total_sources=len(rows), native_base_releases=sum(r['outcomes']['native']['base']['valid'] for r in rows),
        rows=rows, source117=next(r for r in rows if r['source_index']==117),
        scope='frozen released-state reward calibration; no new native simulations or native learner targets')
