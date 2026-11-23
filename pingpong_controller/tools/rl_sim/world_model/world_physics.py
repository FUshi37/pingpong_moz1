"""Own-state robot/FK/ball recursion using the approved47-D and39-D models.

This physical layer consumes delayed seven-joint commands. A complete policy
world must generate them through its own projection/RMP/delay state; supplying
recorded commands qualifies only a common-command component experiment.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax

from .contact_dynamics import ContactState
from .position_contact import position_contact_step
from .robot_support_model import support_robot_step
from .support_losses import racket_kinematics
from .model_numerics import precision_scope


class PhysicsState(NamedTuple):
    """RightArm-0..6 then LegWaist-0..5, plus complete physical ball state."""

    q_rad: Any
    dq_rad_s: Any
    ball: ContactState


def _check_state(state: PhysicsState) -> None:
    if (state.q_rad.ndim != 2 or state.q_rad.shape[-1] != 13
            or state.dq_rad_s.shape != state.q_rad.shape):
        raise ValueError('Physics requires all thirteen current q/dq values')
    n = state.q_rad.shape[0]
    if any(value.shape != (n, 3) for value in state.ball[:3]):
        raise ValueError('Ball position/velocity/world spin must match the robot batch')
    if any(value.shape != (n,) for value in state.ball[3:]):
        raise ValueError('Ball contact mode and age must match the robot batch')


def _physics_substep(robot_parameters, contact_parameters, state: PhysicsState,
                    command: dict, *, robot_scale, position_scale, chain, surface,
                    ball_radius_m: float, contact_contract):
    """Advance1 ms from one joint/ball before-state without future corrections.

    The existing execution proxy retains command dq/ddq. Only q13 and delayed
    q_command-q7 augment the original ball/racket features in contact39.
    Both models advance from the same interval start, matching native labels.
    """
    _check_state(state)
    n = state.q_rad.shape[0]
    if (set(command) != {'q', 'dq', 'ddq'}
            or any(value.shape != (n, 7) for value in command.values())):
        raise ValueError('Expected only the current delayed arm q/dq/ddq commands')
    if len(chain.joint_ids) != 13:
        raise ValueError('Full-chain FK must include arm and support joints')
    pose = racket_kinematics(chain, state.q_rad, state.dq_rad_s)
    racket = dict(geom_position_m=pose['geom_position'], geom_rotation_world=pose['rotation'],
        racket_linear_velocity_m_s=pose['velocity'], racket_angular_velocity_rad_s=pose['omega'],
        joint_q_rad=state.q_rad, delayed_q_command_rad=command['q'])
    ball, contact = position_contact_step(contact_parameters, surface, state.ball, racket,
        ball_radius_m=ball_radius_m, position_scale=position_scale, contract=contact_contract)
    q, dq = support_robot_step(robot_parameters, robot_scale,
        state.q_rad, state.dq_rad_s, command)
    after = PhysicsState(q, dq, ball)
    return after, dict(q_before_rad=state.q_rad, dq_before_rad_s=state.dq_rad_s,
        q_after_rad=q, dq_after_rad_s=dq, racket_before=pose, contact=contact)


def physics_substep(robot_parameters, contact_parameters, state: PhysicsState,
                    command: dict, *, matmul_precision=None, **options):
    """Apply the checkpoint's learned-component precision locally, excluding RMP."""
    with precision_scope(matmul_precision):
        return _physics_substep(robot_parameters, contact_parameters, state, command, **options)


def physics_rollout(robot_parameters, contact_parameters, initial: PhysicsState,
                    commands: dict, **options):
    """Recurse over time-major delayed commands; all physical states are owned."""
    _check_state(initial)
    if (set(commands) != {'q', 'dq', 'ddq'} or commands['q'].ndim != 3
            or commands['q'].shape[0] < 1
            or any(value.shape != commands['q'].shape for value in commands.values())
            or commands['q'].shape[1:] != (initial.q_rad.shape[0], 7)):
        raise ValueError('Delayed command traces must have shape(T,N,7)')

    def tick(state, command):
        return physics_substep(robot_parameters, contact_parameters, state, command, **options)

    return jax.lax.scan(tick, initial, commands)


def physics_control_step(robot_parameters, contact_parameters, state: PhysicsState,
                         commands: dict, **options):
    """Keep the200 Hz external period as exactly five physical1 ms substeps."""
    if 'q' not in commands or commands['q'].ndim < 1 or commands['q'].shape[0] != 5:
        raise ValueError('One control step requires exactly five delayed-command substeps')
    return physics_rollout(robot_parameters, contact_parameters, state, commands, **options)
