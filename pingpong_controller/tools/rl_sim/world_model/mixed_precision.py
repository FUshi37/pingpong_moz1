"""Explicit, default-off WMJ numerical candidate; native FP32 control is retained.

Physical q/dq and ball state, feature normalization, FK/contact arithmetic and
residual accumulation use FP64. Learned MLP operations remain FP32. Camera and
paired-history values are FP32 at their existing interface. A separate reference
also promotes world MLP/control arithmetic, but never changes the actor or native
simulator. No globals are patched and no checkpoint is rewritten.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from . import neural
from .module_gradient_tools import cast_floats, private_clone
from .robot_support_model import support_features


def normalized_prediction(parameters: Any, scale: Any, features: Any,
                          *, network: Callable = neural.mlp) -> Any:
    """Preserve feature differences before the original MLP's FP32 boundary."""
    x = jnp.asarray(features, jnp.float64)
    scale = cast_floats(scale, jnp.float64)
    z = (x-scale.input_mean)/scale.input_scale
    with jax.default_matmul_precision('highest'):
        output = network(parameters, z)
    return output.astype(jnp.float64)*scale.residual_scale+scale.residual_mean


def robot_increment_step(parameters: Any, scale: Any, q_rad: Any, dq_rad_s: Any,
                         commands: dict, *, dt_s: float = .001,
                         network: Callable = neural.mlp) -> tuple:
    """Accumulate thirteen-joint increments without first rounding absolute q."""
    q, dq = jnp.asarray(q_rad, jnp.float64), jnp.asarray(dq_rad_s, jnp.float64)
    features = support_features(q, dq, cast_floats(commands, jnp.float64))
    residual = normalized_prediction(parameters, scale, features, network=network)
    delta_q = jnp.concatenate((residual[..., :7], residual[..., 14:20]), axis=-1)
    delta_dq = jnp.concatenate((residual[..., 7:14], residual[..., 20:26]), axis=-1)
    return q+dq*dt_s+delta_q, dq+delta_dq


def control_boundary(function: Callable, chain: Any, contract: Any,
                     *, reference: bool) -> Callable:
    """Enforce the original controller dtype, independently of physical dtype."""
    dtype = jnp.float64 if reference else jnp.float32
    def step(state, raw, q, context, **unused):
        with jax.enable_x64(reference):
            return function(cast_floats(state, dtype), jnp.asarray(raw, dtype),
                jnp.asarray(q, dtype), cast_floats(context, dtype),
                chain=chain, contract=contract)
    return step


def stable_course_counters(after: Any, before: Any) -> Any:
    """Keep bounded task counts int32 when x64 promotes reduction accumulators."""
    return after._replace(
        completed_goals=after.completed_goals.astype(before.completed_goals.dtype),
        successful_goals=after.successful_goals.astype(before.successful_goals.dtype))


@dataclass(frozen=True)
class PrecisionRuntime:
    mode: str
    robot: Any
    auxiliary: Any
    options: dict
    control: Any
    controller: Callable
    course_step: Callable
    world_step: Callable
    prepare_course: Callable
    prepare_world: Callable
    prepare_context: Callable
    prepare_events: Callable


def make_runtime(mode: str, robot: Any, auxiliary: Any, options: dict,
                 control: Any) -> PrecisionRuntime:
    """Build an isolated runtime; the default FP32 mode returns original functions."""
    from . import course_env, course_control, course_schedule, full_prior_world
    from . import soft_contact, hybrid_contact, tangent_contact, observation_engine
    if mode not in ('fp32', 'mixed', 'fp64_reference'):
        raise ValueError('Unregistered precision mode')
    identity = lambda value: value
    if mode == 'fp32':
        return PrecisionRuntime(mode, robot, auxiliary, options, control, course_control.control_step,
            course_env.course_step, full_prior_world.full_prior_world_step,
            identity, identity, identity, identity)
    if not jax.config.x64_enabled:
        raise ValueError('Candidate construction requires an explicit independent x64 scope')
    reference = mode == 'fp64_reference'
    fp32_chain = cast_floats(options['chain'], jnp.float32)
    physics_options = cast_floats(options, jnp.float64)
    chain = physics_options['chain']
    candidate_control = cast_floats(control, jnp.float64 if reference else jnp.float32)
    network = private_clone(neural.mlp, float64=True) if reference else neural.mlp
    frozen_robot = cast_floats(robot, jnp.float64) if reference else robot
    frozen_auxiliary = dict(auxiliary, scale=cast_floats(auxiliary['scale'], jnp.float64))
    if reference:
        frozen_auxiliary['parameters'] = cast_floats(auxiliary['parameters'], jnp.float64)

    def robot_step(parameters, scale, q, dq, commands, *, dt_s=.001):
        return robot_increment_step(parameters, scale, q, dq, commands,
                                    dt_s=dt_s, network=network)

    def auxiliary_terms(aux, q, dq, commands):
        features = support_features(q, dq, cast_floats(commands, jnp.float64))
        return normalized_prediction(aux['parameters'], aux['scale'], features, network=network)

    # Source-pinned function copies alter only explicitly registered dtype casts.
    soft = private_clone(soft_contact.soft_contact_increment, float64=True)
    prior = private_clone(tangent_contact.tangent_prior_increment, float64=True)
    hybrid = private_clone(hybrid_contact.hybrid_contact_step, float64=True,
                           overrides={'soft_contact_increment': soft})
    contact = private_clone(tangent_contact.tangent_contact_step, overrides={
        'soft_contact_increment': soft, 'hybrid_contact_step': hybrid,
        'tangent_prior_increment': prior})
    physics = private_clone(full_prior_world.full_prior_physics_step, overrides={
        'support_robot_step': robot_step, 'acceleration_terms': auxiliary_terms,
        'tangent_contact_step': contact})
    controller = control_boundary(course_control.control_step,
        chain if reference else fp32_chain, candidate_control, reference=reference)

    def pose(unused_chain, q):
        # The observation-facing site cache remains FP32; physical contact FK is64.
        return full_prior_world.measured_racket_pose(chain, q).astype(jnp.float32)

    def observe(state, q, ball, **kwargs):
        with jax.enable_x64(False):
            return observation_engine.advance_world_observation(state,
                q.astype(jnp.float32), ball.astype(jnp.float32),
                pose_from_q=lambda x: full_prior_world.measured_racket_pose(fp32_chain, x),
                ar1_enabled=kwargs['ar1_enabled'])

    def schedule(*args, **kwargs):
        with jax.enable_x64(False):
            return course_schedule.finish_substep(*args, **kwargs)

    replacements = dict(full_prior_physics_step=physics, control_step=controller,
                        measured_racket_pose=pose, advance_world_observation=observe)
    course = private_clone(course_env.course_step,
                           overrides=dict(replacements, finish_substep=schedule))
    world = private_clone(full_prior_world.full_prior_world_step, overrides=replacements)

    def course_s01(*args, **kwargs):
        if kwargs['stage_id'] != 'S01':
            raise ValueError('Numerical candidate is validated only as an S01 diagnostic')
        after, transition = course(*args, **kwargs)
        return stable_course_counters(after, args[2]), transition

    def world_zero_noise(*args, **kwargs):
        if kwargs['ar1_enabled']:
            raise ValueError('Numerical candidate AR1 path is not validated')
        return world(*args, **kwargs)

    def prepare_world(state):
        return state._replace(physics=cast_floats(state.physics, jnp.float64),
            control=cast_floats(state.control, jnp.float64 if reference else jnp.float32),
            observation=cast_floats(state.observation, jnp.float32),
            cached_racket_pose=cast_floats(state.cached_racket_pose, jnp.float32))

    context = lambda value: cast_floats(value, jnp.float64 if reference else jnp.float32)
    events = lambda value: cast_floats(value, jnp.float64)
    def prepare_course(state):
        return state._replace(world=prepare_world(state.world),
                              context=context(state.context), events=events(state.events))
    return PrecisionRuntime(mode, frozen_robot, frozen_auxiliary, physics_options,
        candidate_control, controller, course_s01, world_zero_noise, prepare_course, prepare_world,
        context, events)
