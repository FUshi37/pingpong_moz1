"""Default-off inverse-flight outgoing-velocity reward and calibration helpers."""

from dataclasses import dataclass
import math
from types import FunctionType
from typing import Any

import jax.numpy as jnp

from .ball_flight import velocity_for_apex
from .imagined_rewards import release_reward


@dataclass(frozen=True)
class VelocityRewardContract:
    enabled: bool = False
    xy_weight: float = 1.
    z_weight: float = 1.

    def __post_init__(self):
        if (not isinstance(self.enabled, bool) or any(not math.isfinite(x) or x < 0.
                for x in (self.xy_weight, self.z_weight))):
            raise ValueError('Velocity reward requires explicit enable and finite nonnegative weights')


def velocity_costs(position_m: Any, velocity_m_s: Any, goal_xyz_m: Any,
                   accepted_release: Any, *, contract: Any) -> tuple:
    """Infer release speed from goal apex; retain gradients through release position.

    The same continuous, no-drag ballistic model as the existing apex reward
    is inverted. A scale of s metres is converted to s/T metres per second,
    where T is the desired ascent time. The vertical conversion matches the
    apex-height derivative at the target speed, not at the observed speed.
    Invalid inverses are marked; this helper does not silently score them.
    """
    finite = (jnp.all(jnp.isfinite(position_m), axis=-1)
              & jnp.all(jnp.isfinite(velocity_m_s), axis=-1)
              & jnp.all(jnp.isfinite(goal_xyz_m), axis=-1))
    possible = finite & (goal_xyz_m[..., 2] > position_m[..., 2])
    valid = possible & accepted_release & (velocity_m_s[..., 2] > 0.)
    p = jnp.where(possible[..., None], position_m, jnp.zeros_like(position_m))
    g = jnp.where(possible[..., None], goal_xyz_m,
                  jnp.broadcast_to(jnp.array([0., 0., .2]), goal_xyz_m.shape))
    v = jnp.where(finite[..., None], velocity_m_s, jnp.zeros_like(velocity_m_s))
    target, _ = velocity_for_apex(p, g)
    time_s = target[..., 2] / 9.81
    scales = jnp.asarray([contract.apex_xy_scale_m, contract.apex_xy_scale_m,
                          contract.apex_z_scale_m]) / jnp.maximum(time_s[..., None], 1e-6)
    error = (v-target)/scales
    costs = dict(velocity_xy=-contract.apex_xy_weight*jnp.log1p(jnp.sum(error[..., :2]**2, axis=-1)),
                 velocity_z=-contract.apex_z_weight*jnp.log1p(error[..., 2]**2))
    costs = {key:jnp.where(valid, value, 0.) for key, value in costs.items()}
    info = dict(valid=valid, inverse_possible=possible, target_velocity_m_s=target,
                time_s=time_s, scale_xyz_m_s=scales)
    return costs, info


def replacement_release_reward(position_m, velocity_m_s, goal_xyz_m, accepted_release,
                               *, contract, candidate: VelocityRewardContract):
    """Replace exactly two costs; invalid finite inverses retain their old costs."""
    old = release_reward(position_m, velocity_m_s, goal_xyz_m, accepted_release, contract=contract)
    if not candidate.enabled:
        return old
    costs, info = velocity_costs(position_m, velocity_m_s, goal_xyz_m, accepted_release, contract=contract)
    terms = {key:value for key,value in old.items() if key not in ('apex_xy', 'apex_z', 'total')}
    terms.update(velocity_xy=jnp.where(info['inverse_possible'], candidate.xy_weight*costs['velocity_xy'], old['apex_xy']),
                 velocity_z=jnp.where(info['inverse_possible'], candidate.z_weight*costs['velocity_z'], old['apex_z']))
    return dict(terms, total=sum(terms.values()))


def clone_globals(function, **overrides):
    """Copy explicit dependencies even for an already-private compiled source copy."""
    result = FunctionType(function.__code__, dict(function.__globals__, **overrides),
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def candidate_course_step(runtime, candidate: VelocityRewardContract):
    """No shared globals, control paths, task counters or parameter state change."""
    if not candidate.enabled:
        return runtime.course_step
    def reward(*args, **kwargs):
        return replacement_release_reward(*args, candidate=candidate, **kwargs)
    task = clone_globals(runtime.course_step.__globals__['task_substep'], release_reward=reward)
    return clone_globals(runtime.course_step, task_substep=task)


def diagnostic_course_step(runtime):
    """Expose both cost families in one frozen rollout; never used for learning.

    Extra diagnostic terms alter this private transition's total. Calibration
    explicitly reconstructs the old total excluding these two terms, so no
    reward, optimizer or default path is changed by observation of the costs.
    """
    def observe(*args, **kwargs):
        old = release_reward(*args, **kwargs)
        costs, info = velocity_costs(*args, **kwargs)
        terms = {key:value for key,value in old.items() if key != 'total'}
        terms.update(velocity_xy=jnp.where(info['inverse_possible'],costs['velocity_xy'],old['apex_xy']),
                     velocity_z=jnp.where(info['inverse_possible'],costs['velocity_z'],old['apex_z']))
        return dict(terms,total=sum(terms.values()))
    task = clone_globals(runtime.course_step.__globals__['task_substep'], release_reward=observe)
    return clone_globals(runtime.course_step, task_substep=task)
