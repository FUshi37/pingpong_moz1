"""Default-off pose-margin reward candidate and frozen calibration mathematics."""

from dataclasses import dataclass, replace
import math
from typing import Any

import jax.numpy as jnp
import numpy as np

from .actor_critic import ImaginedBatch, actor_return, frozen_value
from .course_env import course_critic_features
from .course_rollout import course_policy_window
from .paired_precision import random32_clone


OBJECTIVES = ('base_return', 'height_return', 'xy_return',
              'minimum_margin_deg', 'unit_pose_return')


@dataclass(frozen=True)
class PoseMarginContract:
    enabled: bool = False
    weight_per_s: float = 0.
    onset_rad: float = math.radians(1.5)
    limit_rad: float = math.radians(2.)

    def __post_init__(self):
        values = (self.weight_per_s, self.onset_rad, self.limit_rad)
        if (not isinstance(self.enabled, bool) or not all(math.isfinite(v) for v in values)
                or self.weight_per_s < 0. or not 0. < self.onset_rad < self.limit_rad):
            raise ValueError('Expected finite nonnegative weight and positive pose-margin interval')


def roll_pitch_rad(pose: Any, reference_rotation: Any) -> Any:
    """Use the existing endpoint-bound convention; output shape (...,2), radians."""
    a, b = pose[..., 3:6], pose[..., 6:9]
    rotation = jnp.stack((a, b, jnp.cross(a, b)), axis=-1)
    relative = jnp.einsum('ji,...jk->...ik', reference_rotation, rotation)
    return jnp.stack((jnp.arctan2(-relative[..., 1, 2], relative[..., 2, 2]),
        jnp.arcsin(jnp.clip(relative[..., 0, 2], -1., 1.))), axis=-1)


def pose_cost(angles_rad: Any, contract: PoseMarginContract) -> Any:
    """Unitless C1 squared hinges, summed over roll/pitch, unsaturated at failure."""
    if angles_rad.shape[-1] != 2:
        raise ValueError('Expected roll,pitch in the final dimension')
    excess = jnp.maximum((jnp.abs(angles_rad) - contract.onset_rad)
                         / (contract.limit_rad - contract.onset_rad), 0.)
    return jnp.sum(excess**2, axis=-1)


def penalty(angles_rad: Any, valid: Any, model_truncated: Any, dt_s: float,
            contract: PoseMarginContract) -> Any:
    """Charge a trusted active interval, including its physical-failure endpoint."""
    if dt_s <= 0.:
        raise ValueError('Expected positive actual interval duration')
    if not contract.enabled or contract.weight_per_s == 0.:
        return jnp.zeros_like(angles_rad[..., 0])
    return jnp.where(jnp.asarray(valid) & ~jnp.asarray(model_truncated),
                     -contract.weight_per_s * dt_s * pose_cost(angles_rad, contract), 0.)


def augment_reward(base: Any, angles_rad: Any, valid: Any, model_truncated: Any,
                   dt_s: float, contract: PoseMarginContract) -> Any:
    """Explicit candidate-only composition; default path returns the original value."""
    if not contract.enabled or contract.weight_per_s == 0.:
        return base
    return base + penalty(angles_rad, valid, model_truncated, dt_s, contract)


def coefficient_grid(gram: Any, mean_height: float, mean_unit_pose: float,
                     weights: list, criteria: dict) -> list:
    """Evaluate full-gradient directions from the five-objective Gram matrix."""
    gram = np.asarray(gram, np.float64)
    if gram.shape != (5, 5) or not np.isfinite(gram).all():
        raise ValueError('Expected finite five-objective gradient Gram matrix')
    base_norm, height_norm, margin_norm, pose_norm = np.sqrt(np.maximum(np.diag(gram), 0.))[[0, 1, 3, 4]]
    baseline_height = gram[1, 0] / max(base_norm, 1e-30)
    rows = []
    for weight in weights:
        norm = math.sqrt(max(gram[0, 0] + 2*weight*gram[0, 4] + weight**2*gram[4, 4], 0.))
        height = (gram[1, 0] + weight*gram[1, 4]) / max(norm, 1e-30)
        margin = (gram[3, 0] + weight*gram[3, 4]) / max(norm, 1e-30)
        retention = height / baseline_height if baseline_height > 1e-12 else None
        norm_ratio = weight * pose_norm / height_norm if height_norm > 1e-12 else None
        value_ratio = abs(weight * mean_unit_pose) / max(abs(mean_height), 1e-12)
        checks = dict(positive_weight=weight > 0., positive_margin_direction=margin > 1e-12,
            retains_height=retention is not None and retention >= criteria['height_direction_retention_min'],
            bounded_pose_gradient=norm_ratio is not None and norm_ratio <= criteria['pose_to_height_gradient_norm_max'],
            bounded_mean_pose_cost=value_ratio <= criteria['mean_pose_to_abs_height_return_max'])
        checks = {name:bool(value) for name, value in checks.items()}
        rows.append(dict(weight_per_s=weight, eligible=all(checks.values()), checks=checks,
            total_gradient_norm=norm, height_directional_derivative=height,
            height_direction_retention=retention, minimum_margin_directional_derivative_deg=margin,
            minimum_margin_cosine=margin / margin_norm if margin_norm > 1e-12 else None,
            pose_to_height_gradient_norm=norm_ratio, mean_pose_return=weight * mean_unit_pose,
            mean_pose_to_abs_height_return=value_ratio))
    return rows


def select_weight(run_grids: dict) -> float | None:
    """Select only a positive coefficient that passes every registered run."""
    if not run_grids:
        return None
    grids = list(run_grids.values())
    weights = [r['weight_per_s'] for r in grids[0]]
    if any([r['weight_per_s'] for r in grid] != weights for grid in grids):
        raise ValueError('Coefficient grids must be identical across runs')
    accepted = [w for i, w in enumerate(weights) if w > 0. and all(g[i]['eligible'] for g in grids)]
    return min(accepted) if accepted else None


def frozen_objectives(runtime, initial, table, keys, episode, reward, contract,
                      *, exploration_std):
    """Replay the registered fixed-policy window; attach diagnostics after each step."""
    unit = replace(contract, enabled=True, weight_per_s=1.)

    def step(*args, **kwargs):
        after, tr = runtime.course_step(*args, **kwargs)
        angles = roll_pitch_rad(after.world.cached_racket_pose,
                                runtime.control.projection.reference_rotation)
        return after, tr._replace(next_critic_features=tr.next_critic_features.astype(jnp.float32),
            task_events=dict(tr.task_events, diagnostic_pose_angles_rad=angles))

    window_fn = random32_clone(course_policy_window, dict(course_step=step,
        course_critic_features=lambda s:course_critic_features(s).astype(jnp.float32)))
    initial = runtime.prepare_course(initial)
    options = dict(stage_id='S01', control_contract=runtime.control,
        episode_contract=episode, reward_contract=reward, **runtime.options)

    def function(actor, critic):
        window = window_fn(actor, runtime.robot, runtime.auxiliary, initial, keys, table,
            horizon=64, exploration_std=exploration_std, **options)
        tr = window.transitions
        batch = ImaginedBatch(tr.reward, tr.discount, tr.trace_continue, tr.valid)
        values = frozen_value(critic, tr.next_critic_features).astype(tr.reward.dtype)
        angles = tr.task_events['diagnostic_pose_angles_rad']
        unit_penalty = penalty(angles, tr.valid, tr.model_truncated, .005, unit).astype(tr.reward.dtype)
        margin = (episode.roll_pitch_limit_rad - jnp.max(jnp.abs(angles), axis=-1)) * (180./jnp.pi)
        trusted = tr.valid & ~tr.model_truncated
        zeros = jnp.zeros_like(values)
        outputs = jnp.stack((actor_return(batch, values),
            actor_return(batch._replace(reward=tr.reward_terms['apex_z']), zeros),
            actor_return(batch._replace(reward=tr.reward_terms['apex_xy']), zeros),
            jnp.min(jnp.where(trusted, margin, jnp.inf), axis=0),
            actor_return(batch._replace(reward=unit_penalty), zeros)), axis=-1)
        modes = jnp.stack((tr.valid, tr.terminated, tr.model_truncated,
            tr.task_events['valid_release'] > 0, tr.task_events['valid_apex'] > 0,
            tr.task_events['bound_ball'], tr.task_events['bound_racket'],
            tr.task_events['bound_orientation'], tr.task_events['bound_joints']), axis=-1)
        trace = dict(angles_rad=angles, unit_pose_penalty=unit_penalty, margin_deg=margin,
            base_reward=tr.reward, height_reward=tr.reward_terms['apex_z'],
            valid=tr.valid, model_truncated=tr.model_truncated, terminated=tr.terminated,
            discount=tr.discount, trace_continue=tr.trace_continue, modes=modes,
            default_reward_delta=augment_reward(tr.reward, angles, tr.valid,
                tr.model_truncated, .005, contract) - tr.reward)
        return outputs, trace
    return function
