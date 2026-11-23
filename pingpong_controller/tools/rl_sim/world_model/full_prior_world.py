"""Opt-in own-feedback world with frozen execution/auxiliary and no residual.

The public step accepts an action and complete world memory. Recorded future
commands, joints, ball states and observations are deliberately absent from
that interface. No actor update, task reset or native simulation occurs here.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax

from .model_numerics import precision_scope
from .native_observation import racket_pose_from_chain
from .normal_coupling import acceleration_terms
from .observation_engine import (
    WorldObservationState, actor_observation_with_history, advance_world_observation,
)
from .robot_support_model import support_robot_step
from .support_losses import racket_kinematics
from .tangent_contact import tangent_contact_step
from .world_control import ControlState, control_step
from .world_physics import PhysicsState, _check_state


class FullPriorWorldState(NamedTuple):
    """One coherent physical/control/camera/history state at a 200 Hz boundary."""

    physics: PhysicsState
    control: ControlState
    observation: WorldObservationState
    cached_racket_pose: Any


def measured_racket_pose(chain, q_rad):
    """Return same-time site XYZ/rotation6D, with private FK at full precision."""
    if q_rad.ndim != 2 or q_rad.shape[-1] != 13 or len(chain.joint_ids) != 13:
        raise ValueError('Full-prior world requires RightArm7 then LegWaist6')
    with precision_scope('highest'):
        return jax.vmap(lambda q:racket_pose_from_chain(chain, q))(q_rad)


def full_prior_physics_step(robot_parameters, auxiliary, state: PhysicsState,
                            command: dict, *, robot_scale, chain, surface,
                            ball_radius_m, position_scale, contact_contract,
                            matmul_precision='highest'):
    """Advance both robot and ball from the same 1 ms before-state.

    The 39-D residual has no parameter input. Its frozen checkpoint can still
    supply the authenticated geometry/normalization contract to the loader.
    Solver terms are predictions from current own joints and delayed commands.
    """
    _check_state(state)
    if (set(command) != {'q', 'dq', 'ddq'}
            or any(value.shape != (state.q_rad.shape[0], 7) for value in command.values())
            or len(chain.joint_ids) != 13 or contact_contract is None):
        raise ValueError('Full prior requires full joints, current commands and soft contact')
    with precision_scope(matmul_precision):
        pose = racket_kinematics(chain, state.q_rad, state.dq_rad_s)
        terms = acceleration_terms(auxiliary, state.q_rad, state.dq_rad_s, command)
        racket = dict(geom_position_m=pose['geom_position'], geom_rotation_world=pose['rotation'],
            racket_linear_velocity_m_s=pose['velocity'], racket_angular_velocity_rad_s=pose['omega'],
            joint_q_rad=state.q_rad, delayed_q_command_rad=command['q'])
        ball, contact = tangent_contact_step(None, surface, state.ball, racket, terms,
            ball_radius_m=ball_radius_m, position_scale=position_scale,
            contract=contact_contract, learned=False)
        q, dq = support_robot_step(robot_parameters, robot_scale,
                                   state.q_rad, state.dq_rad_s, command)
    return PhysicsState(q, dq, ball), dict(q_before_rad=state.q_rad,
        dq_before_rad_s=state.dq_rad_s, q_after_rad=q, dq_after_rad_s=dq,
        racket_before=pose, solver_terms=terms, contact=contact)


def world_actor_observation(state: FullPriorWorldState, task_goal_xyz):
    """Keep the legacy cached-site prefix separate from timestamp-paired history."""
    physics, control, observation = state.physics, state.control, state.observation
    return actor_observation_with_history(physics.q_rad[:, :7], physics.dq_rad_s[:, :7],
        control.reference_q_rad, control.previous_accepted_action, state.cached_racket_pose,
        observation.camera, task_goal_xyz, observation.history.paired,
        observation.camera.step_count*5000)


def full_prior_world_step(robot_parameters, auxiliary, state: FullPriorWorldState,
                          raw_action, *, context, control_contract,
                          task_goal_xyz, ar1_enabled: bool, **physics_options):
    """Apply one action, five physical ticks, then the known camera/history step."""
    chain = physics_options['chain']
    # Do not wrap native RMP in the learned components' precision scope.
    control, control_trace = control_step(state.control, raw_action, state.physics.q_rad,
        context, chain=chain, contract=control_contract)

    def tick(current, command):
        return full_prior_physics_step(robot_parameters, auxiliary, current, command,
                                       **physics_options)

    physics, trace = jax.lax.scan(tick, state.physics, control_trace['delayed_commands'])
    # V324 calls batched_forward after all five ticks, refreshing the site.
    cached_pose = measured_racket_pose(chain, physics.q_rad)
    observation, diagnostics = advance_world_observation(state.observation, physics.q_rad,
        physics.ball.position_m, pose_from_q=lambda q:measured_racket_pose(chain, q),
        ar1_enabled=ar1_enabled)
    after = FullPriorWorldState(physics, control, observation, cached_pose)
    return after, dict(physics=trace, control=control_trace,
        cached_racket_pose=cached_pose, observation_diagnostics=diagnostics,
        actor_observation=world_actor_observation(after, task_goal_xyz))


def full_prior_world_rollout(robot_parameters, auxiliary, initial: FullPriorWorldState,
                             raw_actions, **options):
    """Recurse raw actions with complete memory, never future teacher corrections."""
    if (raw_actions.ndim != 3 or raw_actions.shape[0] < 1
            or raw_actions.shape[1:] != (initial.physics.q_rad.shape[0], 7)):
        raise ValueError('Expected time-major raw actions (T,N,7)')

    def tick(state, action):
        return full_prior_world_step(robot_parameters, auxiliary, state, action, **options)

    return jax.lax.scan(tick, initial, raw_actions)
