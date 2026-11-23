"""Frozen post-trial attribution helpers; never used by an optimizer or native step."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .actor_critic import ImaginedBatch, actor_return, frozen_value
from .artifacts import read_array_archive
from .contracts import authenticate, repository_root
from .course_env import course_critic_features
from .course_rollout import course_policy_window
from .paired_precision import random32_clone


TERM_NAMES = ('hit', 'apex_xy', 'apex_z', 'precision', 'steady_horizontal',
              'survival', 'failure', 'action_smooth', 'projection', 'approach')
OBJECTIVE_NAMES = TERM_NAMES + ('target_bootstrap', 'diagnostic_orientation_margin_deg')


def return_components(transitions: Any, next_values: Any) -> Any:
    """Return (N,11) discounted components with unchanged masks and bootstrap."""
    tr = transitions
    values = next_values.astype(tr.reward.dtype)
    zeros = jnp.zeros_like(tr.reward)
    batch = ImaginedBatch(tr.reward, tr.discount, tr.trace_continue, tr.valid)
    direct = [actor_return(batch._replace(reward=tr.reward_terms[k]), zeros) for k in TERM_NAMES]
    bootstrap = actor_return(batch._replace(reward=zeros), values)
    return jnp.stack(direct + [bootstrap], axis=-1)


def orientation_margin_deg(pose: Any, reference_rotation: Any, limit_rad: float) -> Any:
    """Signed (N,) margin using the exact existing roll/pitch convention."""
    column0, column1 = pose[:, 3:6], pose[:, 6:9]
    rotation = jnp.stack((column0, column1, jnp.cross(column0, column1)), axis=-1)
    relative = jnp.einsum('ji,njk->nik', reference_rotation, rotation)
    roll = jnp.arctan2(-relative[:, 1, 2], relative[:, 2, 2])
    pitch = jnp.arcsin(jnp.clip(relative[:, 0, 2], -1., 1.))
    return (limit_rad - jnp.maximum(jnp.abs(roll), jnp.abs(pitch))) * (180. / jnp.pi)


def flat_tree(tree: Any) -> np.ndarray:
    return np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in jax.tree.leaves(tree)])


def gradient_statistics(gradients: list, actual_update: Any, *, reward_count: int) -> dict:
    """Measure contribution and cancellation; gradient norms are not additive shares."""
    matrix = np.stack([flat_tree(g) for g in gradients])
    if not np.isfinite(matrix).all():
        raise FloatingPointError('Nonfinite frozen gradient component')
    norms = np.linalg.norm(matrix, axis=1)
    total = matrix[:reward_count].sum(axis=0)
    norm = float(np.linalg.norm(total))
    cosine = [float(x @ total / (n * norm)) if n > 1e-12 and norm > 1e-12 else None
              for x, n in zip(matrix, norms)]
    pairwise = [[float(x @ y / (a * b)) if a > 1e-12 and b > 1e-12 else None
                 for y, b in zip(matrix, norms)] for x, a in zip(matrix, norms)]
    return dict(parameter_count=matrix.shape[1], component_norms=norms.tolist(),
        total_reward_gradient_norm=norm, cosine_to_total_reward=cosine,
        pairwise_cosines=pairwise, dot_actual_update=(matrix @ flat_tree(actual_update)).tolist(),
        orientation_is_diagnostic_only=True, sign='gradient_of_positive_return_not_training_loss')


def target_from_checkpoint(reference: dict) -> tuple:
    """Read only authenticated FP32 EMA critic leaves, retaining their original order."""
    path = authenticate(repository_root(), reference)
    manifest = json.loads(path.read_text())
    specs = {f'leaf_{i:06d}': s for i, s in enumerate(manifest['state']['leaves'])
             if "['learner'].target_critic" in s['path']}
    arrays = read_array_archive(path.parent / manifest['archive_file'], manifest['archive'], names=specs)
    result = []
    for layer, width in enumerate(((2602, 256), (256, 256), (256, 1))):
        values = {}
        for key in ('w', 'b'):
            name = next(k for k, s in specs.items()
                        if s['path'] == f"['learner'].target_critic[{layer}]['{key}']")
            value, spec = arrays[name], specs[name]
            shape = width if key == 'w' else (width[1],)
            if (value.shape != shape or value.dtype != np.float32
                    or hashlib.sha256(value.tobytes()).hexdigest() != spec['sha256']):
                raise ValueError('Frozen target critic shape/dtype/hash changed')
            values[key] = jnp.asarray(value)
        result.append(values)
    return tuple(result)


def ball_other_contacts(pairs: Any, distances: Any, *, ball_id: int, racket_id: int) -> list:
    """Indices of active ball contacts excluding the registered racket surface."""
    pairs, distances = np.asarray(pairs), np.asarray(distances)
    mask = ((pairs >= 0).all(axis=-1) & (pairs == ball_id).any(axis=-1)
            & ~(pairs == racket_id).any(axis=-1) & (distances <= 0.))
    return np.flatnonzero(mask).tolist()


def component_function(runtime: Any, initial: Any, table: Any, keys: Any,
                       episode: Any, reward: Any, *, exploration_std: float) -> Callable:
    """Expose existing reward plus one separate margin diagnostic, without changing reward."""
    def step(*args, **kwargs):
        after, tr = runtime.course_step(*args, **kwargs)
        margin = orientation_margin_deg(after.world.cached_racket_pose,
            runtime.control.projection.reference_rotation, episode.roll_pitch_limit_rad)
        return after, tr._replace(next_critic_features=tr.next_critic_features.astype(jnp.float32),
            task_events=dict(tr.task_events, diagnostic_orientation_margin_deg=margin))

    window_fn = random32_clone(course_policy_window, dict(course_step=step,
        course_critic_features=lambda state: course_critic_features(state).astype(jnp.float32)))
    initial = runtime.prepare_course(initial)
    options = dict(stage_id='S01', control_contract=runtime.control,
        episode_contract=episode, reward_contract=reward, **runtime.options)

    def components(actor, critic):
        window = window_fn(actor, runtime.robot, runtime.auxiliary, initial, keys, table,
            horizon=64, exploration_std=exploration_std, **options)
        tr = window.transitions
        values = frozen_value(critic, tr.next_critic_features).astype(tr.reward.dtype)
        returns = return_components(tr, values)
        batch = ImaginedBatch(tr.reward, tr.discount, tr.trace_continue, tr.valid)
        original = actor_return(batch, values)
        margin = jnp.min(jnp.where(tr.valid,
            tr.task_events['diagnostic_orientation_margin_deg'], jnp.inf), axis=0)
        objectives = jnp.concatenate((returns, margin[:, None]), axis=-1)
        modes = jnp.stack((tr.valid, tr.terminated, tr.model_truncated,
            tr.task_events['valid_release'] > 0, tr.task_events['valid_apex'] > 0,
            tr.task_events['bound_ball'], tr.task_events['bound_racket'],
            tr.task_events['bound_orientation'], tr.task_events['bound_joints']), axis=-1)
        final = window.environment
        auxiliary = dict(modes=modes, valid=tr.valid, reward=tr.reward,
            original_return=original, component_sum_error=returns.sum(axis=-1) - original,
            terminated=final.terminated, model_truncated=final.model_truncated,
            hit_count=final.task.hit_count, apex_count=final.task.apex_count,
            final_margin_deg=orientation_margin_deg(final.world.cached_racket_pose,
                runtime.control.projection.reference_rotation, episode.roll_pitch_limit_rad),
            clipped_action_fraction=jnp.sum((jnp.abs(window.raw_action) >= 1.) & tr.valid[..., None])
                / jnp.maximum(jnp.sum(tr.valid) * 7, 1))
        return objectives, auxiliary
    return components
