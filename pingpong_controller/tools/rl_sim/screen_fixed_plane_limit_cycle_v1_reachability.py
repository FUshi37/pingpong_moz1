#!/usr/bin/env python3
"""Screen launch and steady-orbit reachability before limit-cycle PPO.

This fixed-seed, no-learning diagnostic separates two physically different
questions. The launch probe starts with a stationary ball placed 1 mm above
the racket and tests the low 1.10 m launch target. The steady probe starts on
the descending half of the 1.45 m orbit and tests whether the registered
bounded-QVEL -> recovered-RMP -> XML-PD stack can return the ball to that
orbit. Both probes use causal Cartesian requests converted to normalized
QVEL at the exact policy boundary. The diagnostic controller is never used
by training, reward, reset, checkpoint, or deployment code.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from fixed_contact_plane_training import (
    predict_descending_plane_crossing,
    racket_surface_frame,
    so3_log_error,
)
from mjx_juggle_env import MjxJuggleEnv
from fixed_plane_limit_cycle_v1 import (
    limit_cycle_dynamic_racket_vz_target_m_s,
    limit_cycle_workspace_bounded_stroke_reference,
)
from train_juggle_mjx_curriculum import (
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V1_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V2_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V3_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V4_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V5_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V6_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V7_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V8_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V9_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V10_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V11_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V12_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V13_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V14_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V15_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V16_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V17_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V18_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V19_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V20_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V21_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V22_PROFILE,
    build_curriculum,
)


SURFACE_Z_M = 1.025
LAUNCH_APEX_Z_M = 1.10
STEADY_APEX_Z_M = 1.45
GRAVITY_M_S2 = 9.8
EFFECTIVE_RESTITUTION = 0.665
# The longest registered diagnostic strike lead is 0.26 s.  The reset must
# start earlier than that so the plant can actually settle at the 23 mm
# below-plane pre-stroke pose before the upward pulse begins.  Using 0.26 s
# here made the longest-lead lanes enter ``strike`` on the first simulator
# tick and turned the intended pre-stroke into an immediate upward sweep.
# 0.39 s is the already validated nominal falling-reset preparation time used
# by the open-loop fixed-contact witness under this RMP/PD plant.
STEADY_RESET_TIME_TO_CONTACT_S = 0.39
STEADY_PRESTROKE_DEPTH_M = 0.023
STEADY_MAX_STRIKE_LEAD_S = 0.26


def _validate_steady_probe_timing() -> None:
    """Reject a reachability probe that cannot execute its own pre-stroke."""

    preparation_s = (
        STEADY_RESET_TIME_TO_CONTACT_S - STEADY_MAX_STRIKE_LEAD_S
    )
    # At 0.20 m/s, 115 ms is the optimistic minimum time for 23 mm travel.
    # Keep a small margin for the recovered-RMP/PD delay and discretization.
    minimum_preparation_s = STEADY_PRESTROKE_DEPTH_M / 0.20 + 0.010
    if preparation_s < minimum_preparation_s:
        raise ValueError(
            "steady reachability reset leaves no executable pre-stroke window"
        )


def _steady_quality_probe_candidates(
    profile_version: int = 13,
) -> list[tuple[float, ...]]:
    """Cover the analytic steady impact speed and low-VXY witness regime."""

    # The 1.025 -> 1.45 m orbit requires about 0.57 m/s racket surface Vz
    # under the registered effective restitution. With real velocity feedback
    # active, the old high-compensation grid produced almost zero realized
    # contact Vz. These families also include the 3/3 low-bandwidth XY braking
    # regime already validated by the non-learning plant witness.
    # Attempts 17--18 showed that a constant horizontal feed-forward can
    # cancel the configuration-dependent coupling, but applying it for the
    # full 190 ms vertical stroke accumulates too much contact-position
    # error. Attempt 19 showed that high measured-velocity damping destroys
    # contact instead. Search an independent short terminal feed-forward
    # window: it preserves the vertical run-up and limits horizontal travel.
    demonstrated_vxy_m_s = tuple(
        (vx, vy)
        for vx in (-0.55, -0.40, -0.25, -0.12)
        for vy in (-0.55, -0.40, -0.25, -0.12)
    )
    legacy_vertical_witnesses = (
        # The 1.0/0.75 witness sits close to the upper apex boundary on GPU.
        # Bracket its effective feed-forward without weakening any pass gate,
        # so reachability does not depend on a single numerical-edge lane.
        (0.9, 0.70),
        (0.9, 0.75),
        (0.9, 0.80),
        (1.0, 0.65),
        (1.0, 0.75),
        (1.2, 0.75),
        (1.4, 0.75),
    )
    if int(profile_version) >= 14:
        # V14-only append: retain the complete earlier V14 order so every
        # existing candidate keeps its lane/seed identity, then add the two
        # robust-neighborhood witnesses. Older profile screens remain exact.
        vertical_witnesses = (
            *legacy_vertical_witnesses[:-1],
            (1.25, 0.75),
            legacy_vertical_witnesses[-1],
            (1.0, 0.70),
            (1.0, 0.80),
        )
    else:
        vertical_witnesses = legacy_vertical_witnesses
    return [
        (
            speed,
            lead,
            position_compensation,
            stroke_vx,
            stroke_vy,
            3.0,
            3.0,
            4.0,
            2.5,
            3.0e-4,
            vxy_lead,
        )
        for speed, position_compensation in vertical_witnesses
        for lead in (0.19,)
        for stroke_vx, stroke_vy in demonstrated_vxy_m_s
        for vxy_lead in (0.06, 0.10, 0.14, 0.19)
    ]


def _select_probe_stages(curriculum, profile_version: int):
    """Use acquisition for launch and V4 steady semantics for quality probes."""

    launch_stage = curriculum[0]
    steady_stage = curriculum[-1] if int(profile_version) >= 4 else launch_stage
    return launch_stage, steady_stage


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _surface_state(env: MjxJuggleEnv, state):
    data = state.data
    rotation = data.geom_xmat[:, env.racket_geom_id].reshape((-1, 3, 3))
    if env.cfg.fixed_contact_surface_frame_mode == "mesh_axes_v1":
        rotation = racket_surface_frame(
            rotation,
            normal_axis=int(env.cfg.fixed_contact_racket_surface_size_axis),
        )
    normal = rotation[:, :, 2]
    normal = normal * jnp.where(normal[:, 2:3] >= 0.0, 1.0, -1.0)
    half_thickness = state.model.geom_size[
        :,
        env.racket_geom_id,
        int(env.cfg.fixed_contact_racket_surface_size_axis),
    ]
    position = (
        data.geom_xpos[:, env.racket_geom_id]
        + normal * half_thickness[:, None]
    )
    return position, rotation


def _measured_racket_site_velocity(state) -> jax.Array:
    """Return the velocity cache produced by the preceding environment step.

    ``MjxJuggleEnv.step`` advances ``prev_racket_pos`` to the newly returned
    site position and stores the finite-difference velocity separately in
    ``prev_racket_vel``.  Re-differencing a returned state therefore reads
    zero and silently disables the damping terms in this diagnostic.
    """

    return state.prev_racket_vel


def _qvel_action_from_surface_twist(
    env: MjxJuggleEnv,
    state,
    *,
    surface_position: jax.Array,
    target_xy: jax.Array,
    target_rotation: jax.Array,
    desired_vz: jax.Array,
    desired_vxy: jax.Array,
    xy_kp: float | jax.Array = 8.0,
    xy_kd: float | jax.Array = 1.5,
    rotation_kp: float | jax.Array = 6.0,
    rotation_kd: float | jax.Array = 1.5,
    dls_regularization: float | jax.Array = 2.0e-3,
) -> jax.Array:
    data = state.data
    site_position = data.site_xpos[:, env.racket_site_id]
    site_velocity = _measured_racket_site_velocity(state)
    angular_velocity = data.cvel[:, env.racket_body_id, :3]
    current_rotation = data.site_xmat[:, env.racket_site_id].reshape(
        (-1, 3, 3)
    )
    xy_kp = jnp.broadcast_to(jnp.asarray(xy_kp), (env.n_envs,))[:, None]
    xy_kd = jnp.broadcast_to(jnp.asarray(xy_kd), (env.n_envs,))[:, None]
    rotation_kp = jnp.broadcast_to(
        jnp.asarray(rotation_kp), (env.n_envs,)
    )[:, None]
    rotation_kd = jnp.broadcast_to(
        jnp.asarray(rotation_kd), (env.n_envs,)
    )[:, None]
    dls_regularization = jnp.broadcast_to(
        jnp.asarray(dls_regularization), (env.n_envs,)
    )
    desired_linear = jnp.concatenate(
        [
            desired_vxy
            + xy_kp * (target_xy - surface_position[:, :2])
            - xy_kd * site_velocity[:, :2],
            desired_vz[:, None],
        ],
        axis=-1,
    )
    orientation_error_body = so3_log_error(
        current_rotation,
        target_rotation,
    )
    orientation_error_world = jnp.einsum(
        "nij,nj->ni", current_rotation, orientation_error_body
    )
    desired_angular = (
        rotation_kp * orientation_error_world
        - rotation_kd * angular_velocity
    )
    task_twist = jnp.concatenate([desired_linear, desired_angular], axis=-1)

    joint_axes = data.xaxis[:, env.arm_jids, :]
    joint_anchors = data.xanchor[:, env.arm_jids, :]
    surface_offsets = surface_position[:, None, :] - joint_anchors
    linear_jacobian = jnp.swapaxes(
        jnp.cross(joint_axes, surface_offsets), 1, 2
    )
    angular_jacobian = jnp.swapaxes(joint_axes, 1, 2)
    jacobian = jnp.concatenate([linear_jacobian, angular_jacobian], axis=1)
    transpose = jnp.swapaxes(jacobian, 1, 2)
    regularized = (
        jacobian @ transpose
        + dls_regularization[:, None, None]
        * jnp.eye(6, dtype=jnp.float32)[None, :, :]
    )
    desired_qvel = jnp.einsum(
        "nij,nj->ni",
        transpose,
        jnp.linalg.solve(regularized, task_twist[..., None]).squeeze(-1),
    )
    return jnp.clip(
        desired_qvel
        / (
            env.arm_vel_limit_rad_s[None, :]
            * float(env.cfg.action_velocity_scale)
            * state.action_scale_mult[:, None]
        ),
        -1.0,
        1.0,
    )


def _run_probe(
    *,
    env: MjxJuggleEnv,
    state,
    obs: jax.Array,
    steps: int,
    make_desired_vz: Callable[[Any, jax.Array, jax.Array, jax.Array], jax.Array],
    make_desired_vxy: Callable[
        [Any, jax.Array, jax.Array, jax.Array], jax.Array
    ] | None = None,
    xy_kp: float | jax.Array = 8.0,
    xy_kd: float | jax.Array = 1.5,
    rotation_kp: float | jax.Array = 6.0,
    rotation_kd: float | jax.Array = 1.5,
    dls_regularization: float | jax.Array = 2.0e-3,
) -> dict[str, np.ndarray]:
    initial_surface_position, initial_surface_rotation = _surface_state(
        env, state
    )
    initial_ball_position = np.asarray(
        state.data.xpos[:, env.ball_body_id]
    )
    initial_target_offset = np.asarray(state.reset_target_offset)
    target_xy = initial_surface_position[:, :2]
    target_rotation = jnp.broadcast_to(
        env.fixed_contact_racket_reference_rotation,
        (env.n_envs, 3, 3),
    )

    def controlled_step(current_state, tick):
        surface_position, _ = _surface_state(env, current_state)
        desired_vz = make_desired_vz(
            current_state,
            tick,
            surface_position,
            target_xy,
        )
        desired_vxy = (
            make_desired_vxy(
                current_state,
                tick,
                surface_position,
                target_xy,
            )
            if make_desired_vxy is not None
            else jnp.zeros((env.n_envs, 2), dtype=jnp.float32)
        )
        action = _qvel_action_from_surface_twist(
            env,
            current_state,
            surface_position=surface_position,
            target_xy=target_xy,
            target_rotation=target_rotation,
            desired_vz=desired_vz,
            desired_vxy=desired_vxy,
            xy_kp=xy_kp,
            xy_kd=xy_kd,
            rotation_kp=rotation_kp,
            rotation_kd=rotation_kd,
            dls_regularization=dls_regularization,
        )
        next_state, _next_obs, _reward, done, metrics = env.step(
            current_state, action
        )
        next_surface_position, _ = _surface_state(env, next_state)
        realized_surface_velocity = (
            next_surface_position - surface_position
        ) / env.dt
        ball_position = next_state.data.xpos[:, env.ball_body_id]
        ball_velocity = next_state.data.qvel[
            :, env.ball_vadr : env.ball_vadr + 3
        ]
        new_hit = metrics["new_hit"] > 0.5
        contact_edge = metrics["logical_hit_edge"] > 0.5
        predicted_post_hit_apex = (
            ball_position[:, 2]
            + jnp.maximum(ball_velocity[:, 2], 0.0) ** 2
            / (2.0 * GRAVITY_M_S2)
        )
        return next_state, (
            ball_position[:, 2],
            next_surface_position[:, 2],
            realized_surface_velocity[:, 2],
            metrics["arm_qvel_limit_utilization_max"],
            metrics["arm_qacc_limit_utilization_max"],
            metrics["arm_qvel_limit_exceed_fraction"],
            metrics["arm_qacc_limit_exceed_fraction"],
            new_hit,
            contact_edge,
            jnp.where(new_hit, predicted_post_hit_apex, -jnp.inf),
            jnp.where(new_hit, ball_velocity[:, 2], -jnp.inf),
            done,
            metrics["cartesian_pose_projection_fallback_to_hold"],
            jnp.where(new_hit, metrics["fixed_contact_q_z_sum"], -jnp.inf),
            jnp.where(new_hit, metrics["fixed_contact_q_pose_sum"], -jnp.inf),
            jnp.where(new_hit, metrics["fixed_contact_q_vxy_sum"], -jnp.inf),
            jnp.where(new_hit, metrics["fixed_contact_q_omega_sum"], -jnp.inf),
            jnp.where(new_hit, metrics["fixed_contact_q_offset_sum"], -jnp.inf),
            jnp.where(new_hit, metrics["hit_racket_vxy_sum"], -jnp.inf),
            jnp.where(
                new_hit,
                metrics["hit_racket_full_angular_speed_rad_s"],
                -jnp.inf,
            ),
            jnp.where(
                new_hit, metrics["hit_contact_center_dist_sum"], -jnp.inf
            ),
            jnp.where(new_hit, metrics["hit_ball_z_sum"], -jnp.inf),
            jnp.where(new_hit, metrics["hit_racket_z_sum"], -jnp.inf),
            jnp.where(contact_edge, ball_position[:, 2], -jnp.inf),
            jnp.where(contact_edge, ball_velocity[:, 2], -jnp.inf),
            jnp.where(contact_edge, next_surface_position[:, 2], -jnp.inf),
            jnp.where(
                contact_edge, realized_surface_velocity[:, 0], -jnp.inf
            ),
            jnp.where(
                contact_edge, realized_surface_velocity[:, 1], -jnp.inf
            ),
            jnp.where(contact_edge, realized_surface_velocity[:, 2], -jnp.inf),
        )

    controlled_step_jit = jax.jit(controlled_step)
    max_ball_z = state.data.xpos[:, env.ball_body_id, 2]
    min_surface_z = initial_surface_position[:, 2]
    max_surface_z = initial_surface_position[:, 2]
    max_surface_vz = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    max_qvel_ratio = jnp.zeros((env.n_envs,), dtype=jnp.float32)
    max_qacc_ratio = jnp.zeros((env.n_envs,), dtype=jnp.float32)
    qvel_exceed_sum = jnp.zeros((env.n_envs,), dtype=jnp.float32)
    qacc_exceed_sum = jnp.zeros((env.n_envs,), dtype=jnp.float32)
    confirmed_hit_count = jnp.zeros((env.n_envs,), dtype=jnp.int32)
    contact_edge_count = jnp.zeros((env.n_envs,), dtype=jnp.int32)
    post_hit_apex = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    post_hit_vz = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    projection_fallback_count = jnp.zeros(
        (env.n_envs,), dtype=jnp.float32
    )
    event_q_z = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    event_q_pose = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    event_q_vxy = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    event_q_omega = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    event_q_offset = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    event_racket_vxy = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    event_racket_omega = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    event_contact_offset = jnp.full(
        (env.n_envs,), -jnp.inf, dtype=jnp.float32
    )
    event_ball_z = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    event_racket_z = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    edge_ball_z = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    edge_ball_vz = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    edge_surface_z = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    edge_surface_vx = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    edge_surface_vy = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    edge_surface_vz = jnp.full((env.n_envs,), -jnp.inf, dtype=jnp.float32)
    edge_captured = jnp.zeros((env.n_envs,), dtype=bool)
    event_captured = jnp.zeros((env.n_envs,), dtype=bool)
    ever_done = jnp.zeros((env.n_envs,), dtype=bool)
    evaluated_step_count = jnp.zeros((env.n_envs,), dtype=jnp.float32)
    for tick in range(int(steps)):
        active = ~(event_captured | ever_done)
        state, values = controlled_step_jit(
            state, jnp.asarray(tick, dtype=jnp.int32)
        )
        (
            ball_z,
            surface_z,
            surface_vz,
            qvel_ratio,
            qacc_ratio,
            qvel_exceed,
            qacc_exceed,
            new_hit,
            contact_edge,
            event_apex,
            event_vz,
            done,
            projection_fallback,
            step_q_z,
            step_q_pose,
            step_q_vxy,
            step_q_omega,
            step_q_offset,
            step_racket_vxy,
            step_racket_omega,
            step_contact_offset,
            step_ball_z,
            step_racket_z,
            step_edge_ball_z,
            step_edge_ball_vz,
            step_edge_surface_z,
            step_edge_surface_vx,
            step_edge_surface_vy,
            step_edge_surface_vz,
        ) = values
        first_event = active & new_hit
        first_edge = active & contact_edge & (~edge_captured)
        max_ball_z = jnp.where(
            active, jnp.maximum(max_ball_z, ball_z), max_ball_z
        )
        min_surface_z = jnp.where(
            active, jnp.minimum(min_surface_z, surface_z), min_surface_z
        )
        max_surface_z = jnp.where(
            active, jnp.maximum(max_surface_z, surface_z), max_surface_z
        )
        max_surface_vz = jnp.where(
            active, jnp.maximum(max_surface_vz, surface_vz), max_surface_vz
        )
        max_qvel_ratio = jnp.where(
            active, jnp.maximum(max_qvel_ratio, qvel_ratio), max_qvel_ratio
        )
        max_qacc_ratio = jnp.where(
            active, jnp.maximum(max_qacc_ratio, qacc_ratio), max_qacc_ratio
        )
        qvel_exceed_sum = qvel_exceed_sum + jnp.where(
            active, qvel_exceed, 0.0
        )
        qacc_exceed_sum = qacc_exceed_sum + jnp.where(
            active, qacc_exceed, 0.0
        )
        evaluated_step_count = evaluated_step_count + active.astype(
            jnp.float32
        )
        confirmed_hit_count = confirmed_hit_count + first_event.astype(
            jnp.int32
        )
        contact_edge_count = contact_edge_count + (
            active & contact_edge
        ).astype(jnp.int32)
        post_hit_apex = jnp.where(first_event, event_apex, post_hit_apex)
        post_hit_vz = jnp.where(first_event, event_vz, post_hit_vz)
        projection_fallback_count = (
            projection_fallback_count
            + jnp.where(active, projection_fallback, 0.0)
        )
        event_q_z = jnp.where(first_event, step_q_z, event_q_z)
        event_q_pose = jnp.where(first_event, step_q_pose, event_q_pose)
        event_q_vxy = jnp.where(first_event, step_q_vxy, event_q_vxy)
        event_q_omega = jnp.where(first_event, step_q_omega, event_q_omega)
        event_q_offset = jnp.where(first_event, step_q_offset, event_q_offset)
        event_racket_vxy = jnp.where(
            first_event, step_racket_vxy, event_racket_vxy
        )
        event_racket_omega = jnp.where(
            first_event, step_racket_omega, event_racket_omega
        )
        event_contact_offset = jnp.where(
            first_event, step_contact_offset, event_contact_offset
        )
        event_ball_z = jnp.where(first_event, step_ball_z, event_ball_z)
        event_racket_z = jnp.where(
            first_event, step_racket_z, event_racket_z
        )
        edge_ball_z = jnp.where(first_edge, step_edge_ball_z, edge_ball_z)
        edge_ball_vz = jnp.where(first_edge, step_edge_ball_vz, edge_ball_vz)
        edge_surface_z = jnp.where(
            first_edge, step_edge_surface_z, edge_surface_z
        )
        edge_surface_vx = jnp.where(
            first_edge, step_edge_surface_vx, edge_surface_vx
        )
        edge_surface_vy = jnp.where(
            first_edge, step_edge_surface_vy, edge_surface_vy
        )
        edge_surface_vz = jnp.where(
            first_edge, step_edge_surface_vz, edge_surface_vz
        )
        edge_captured = edge_captured | first_edge
        event_captured = event_captured | first_event
        ever_done = ever_done | (active & done)

    return {
        "initial_obs": np.asarray(obs),
        "initial_surface_position": np.asarray(initial_surface_position),
        "initial_surface_rotation": np.asarray(initial_surface_rotation),
        "initial_ball_position": initial_ball_position,
        "initial_target_offset": initial_target_offset,
        "max_ball_z": np.asarray(max_ball_z),
        "min_surface_z": np.asarray(min_surface_z),
        "max_surface_z": np.asarray(max_surface_z),
        "max_surface_vz": np.asarray(max_surface_vz),
        "max_qvel_ratio": np.asarray(max_qvel_ratio),
        "max_qacc_ratio": np.asarray(max_qacc_ratio),
        "qvel_exceed_fraction": np.asarray(
            qvel_exceed_sum / jnp.maximum(evaluated_step_count, 1.0)
        ),
        "qacc_exceed_fraction": np.asarray(
            qacc_exceed_sum / jnp.maximum(evaluated_step_count, 1.0)
        ),
        "confirmed_hit_count": np.asarray(confirmed_hit_count),
        "contact_edge_count": np.asarray(contact_edge_count),
        "post_hit_apex": np.asarray(post_hit_apex),
        "post_hit_vz": np.asarray(post_hit_vz),
        "projection_fallback_steps": np.asarray(projection_fallback_count),
        "event_q_z": np.asarray(event_q_z),
        "event_q_pose": np.asarray(event_q_pose),
        "event_q_vxy": np.asarray(event_q_vxy),
        "event_q_omega": np.asarray(event_q_omega),
        "event_q_offset": np.asarray(event_q_offset),
        "event_racket_vxy": np.asarray(event_racket_vxy),
        "event_racket_omega": np.asarray(event_racket_omega),
        "event_contact_offset": np.asarray(event_contact_offset),
        "event_ball_z": np.asarray(event_ball_z),
        "event_racket_z": np.asarray(event_racket_z),
        "edge_ball_z": np.asarray(edge_ball_z),
        "edge_ball_vz": np.asarray(edge_ball_vz),
        "edge_surface_z": np.asarray(edge_surface_z),
        "edge_surface_vx": np.asarray(edge_surface_vx),
        "edge_surface_vy": np.asarray(edge_surface_vy),
        "edge_surface_vz": np.asarray(edge_surface_vz),
        "ever_done": np.asarray(ever_done),
    }


def _lane_common(result: dict[str, np.ndarray], index: int) -> dict[str, object]:
    def finite_or_none(value: np.generic | float) -> float | None:
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None

    return {
        "confirmed_hit_count": int(result["confirmed_hit_count"][index]),
        "contact_edge_count": int(result["contact_edge_count"][index]),
        "post_hit_apex_z_m": finite_or_none(
            result["post_hit_apex"][index]
        ),
        "post_hit_ball_vz_m_s": finite_or_none(
            result["post_hit_vz"][index]
        ),
        "max_ball_z_m": float(result["max_ball_z"][index]),
        "max_surface_vz_m_s": float(result["max_surface_vz"][index]),
        "min_surface_z_m": float(result["min_surface_z"][index]),
        "max_surface_z_m": float(result["max_surface_z"][index]),
        "max_joint_qvel_limit_ratio": float(result["max_qvel_ratio"][index]),
        "max_joint_qacc_limit_ratio_diagnostic": float(
            result["max_qacc_ratio"][index]
        ),
        "joint_qvel_exceed_fraction": float(
            result["qvel_exceed_fraction"][index]
        ),
        "joint_qacc_exceed_fraction": float(
            result["qacc_exceed_fraction"][index]
        ),
        "projection_fallback_steps": float(
            result["projection_fallback_steps"][index]
        ),
        "terminated_or_truncated": bool(result["ever_done"][index]),
        "target_x_m": float(result["initial_target_offset"][index, 0]),
        "target_y_m": float(result["initial_target_offset"][index, 1]),
        "initial_surface_x_m": float(
            result["initial_surface_position"][index, 0]
        ),
        "initial_surface_y_m": float(
            result["initial_surface_position"][index, 1]
        ),
        "initial_ball_x_m": float(result["initial_ball_position"][index, 0]),
        "initial_ball_y_m": float(result["initial_ball_position"][index, 1]),
        "contact_q_z": finite_or_none(result["event_q_z"][index]),
        "contact_q_pose": finite_or_none(result["event_q_pose"][index]),
        "contact_q_vxy": finite_or_none(result["event_q_vxy"][index]),
        "contact_q_omega": finite_or_none(result["event_q_omega"][index]),
        "contact_q_offset": finite_or_none(result["event_q_offset"][index]),
        "hit_racket_vxy_m_s": finite_or_none(
            result["event_racket_vxy"][index]
        ),
        "hit_racket_omega_rad_s": finite_or_none(
            result["event_racket_omega"][index]
        ),
        "hit_contact_offset_m": finite_or_none(
            result["event_contact_offset"][index]
        ),
        "hit_ball_center_z_m": finite_or_none(result["event_ball_z"][index]),
        "hit_racket_z_m": finite_or_none(result["event_racket_z"][index]),
        "edge_ball_center_z_m": finite_or_none(result["edge_ball_z"][index]),
        "edge_ball_vz_m_s": finite_or_none(result["edge_ball_vz"][index]),
        "edge_surface_z_m": finite_or_none(result["edge_surface_z"][index]),
        "edge_surface_vx_m_s": finite_or_none(
            result["edge_surface_vx"][index]
        ),
        "edge_surface_vy_m_s": finite_or_none(
            result["edge_surface_vy"][index]
        ),
        "edge_surface_vz_m_s": finite_or_none(
            result["edge_surface_vz"][index]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument(
        "--profile-version",
        type=int,
        choices=(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22,
        ),
        default=1,
    )
    parser.add_argument(
        "--quality-grid-x-half-range-m", type=float
    )
    parser.add_argument(
        "--quality-grid-y-half-range-m", type=float
    )
    parser.add_argument(
        "--quality-grid-x-center-m", type=float
    )
    parser.add_argument(
        "--quality-grid-y-center-m", type=float
    )
    parser.add_argument("--steps", type=int, default=180)
    args = parser.parse_args()
    _validate_steady_probe_timing()
    if args.steps < 120:
        parser.error("--steps must be at least 120 (0.6 s at 200 Hz)")
    uses_directional_grid = int(args.profile_version) >= 5
    quality_grid_x_half_range_m = (
        float(args.quality_grid_x_half_range_m)
        if args.quality_grid_x_half_range_m is not None
        else (0.003125 if uses_directional_grid else 0.05)
    )
    quality_grid_y_half_range_m = (
        float(args.quality_grid_y_half_range_m)
        if args.quality_grid_y_half_range_m is not None
        else (0.01 if uses_directional_grid else 0.04)
    )
    quality_grid_x_center_m = (
        float(args.quality_grid_x_center_m)
        if args.quality_grid_x_center_m is not None
        else (0.003125 if uses_directional_grid else 0.0)
    )
    quality_grid_y_center_m = (
        float(args.quality_grid_y_center_m)
        if args.quality_grid_y_center_m is not None
        else (-0.01 if uses_directional_grid else 0.0)
    )
    if not 0.0 < quality_grid_x_half_range_m <= 0.05:
        parser.error(
            "--quality-grid-x-half-range-m must be in (0, 0.05]"
        )
    if not 0.0 < quality_grid_y_half_range_m <= 0.04:
        parser.error(
            "--quality-grid-y-half-range-m must be in (0, 0.04]"
        )
    quality_grid_x_m = tuple(
        quality_grid_x_center_m + offset
        for offset in (
            -quality_grid_x_half_range_m,
            0.0,
            quality_grid_x_half_range_m,
        )
    )
    quality_grid_y_m = tuple(
        quality_grid_y_center_m + offset
        for offset in (
            -quality_grid_y_half_range_m,
            0.0,
            quality_grid_y_half_range_m,
        )
    )
    if quality_grid_x_m[0] < -0.05 or quality_grid_x_m[-1] > 0.05:
        parser.error("quality-grid X cells must remain within [-0.05, 0.05]")
    if quality_grid_y_m[0] < -0.04 or quality_grid_y_m[-1] > 0.04:
        parser.error("quality-grid Y cells must remain within [-0.04, 0.04]")
    if int(args.profile_version) in (15, 16, 17, 18, 19, 20, 21, 22):
        if any(
            value is not None
            for value in (
                args.quality_grid_x_half_range_m,
                args.quality_grid_y_half_range_m,
                args.quality_grid_x_center_m,
                args.quality_grid_y_center_m,
            )
        ):
            parser.error(
                f"V{int(args.profile_version)} fixes reachability to the "
                "center cell and rejects "
                "quality-grid overrides"
            )
        quality_grid_x_m = (0.0,)
        quality_grid_y_m = (0.0,)
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(args.physical_gpu_index),
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    physical_gpu_uuid = gpu_query.stdout.strip()
    if physical_gpu_uuid != str(args.expected_gpu_uuid):
        parser.error(
            "physical GPU UUID mismatch: expected "
            f"{args.expected_gpu_uuid}, got {physical_gpu_uuid}"
        )
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices not in {
        str(args.physical_gpu_index),
        physical_gpu_uuid,
    }:
        parser.error(
            "CUDA_VISIBLE_DEVICES must select the audited physical GPU"
        )

    speeds = np.asarray([1.2, 1.7, 2.2, 2.7], dtype=np.float32)
    durations = np.asarray([0.035, 0.055, 0.075, 0.095], dtype=np.float32)
    speed_grid, duration_grid = np.meshgrid(speeds, durations, indexing="ij")
    launch_speed = jnp.asarray(speed_grid.reshape(-1))
    launch_duration = jnp.asarray(duration_grid.reshape(-1))
    n_envs = int(launch_speed.shape[0])

    profile = {
        1: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V1_PROFILE,
        2: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V2_PROFILE,
        3: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V3_PROFILE,
        4: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V4_PROFILE,
        5: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V5_PROFILE,
        6: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V6_PROFILE,
        7: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V7_PROFILE,
        8: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V8_PROFILE,
        9: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V9_PROFILE,
        10: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V10_PROFILE,
        11: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V11_PROFILE,
        12: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V12_PROFILE,
        13: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V13_PROFILE,
        14: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V14_PROFILE,
        15: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V15_PROFILE,
        16: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V16_PROFILE,
        17: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V17_PROFILE,
        18: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V18_PROFILE,
        19: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V19_PROFILE,
        20: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V20_PROFILE,
        21: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V21_PROFILE,
        22: FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V22_PROFILE,
    }[int(args.profile_version)]
    curriculum = build_curriculum(
        curriculum_profile=profile,
        measured_ball_mass_range_kg=(0.0039, 0.0041),
        asymmetric_critic=True,
        critic_command_history_steps=12,
        right_arm_pd_profile="recovered_rmp_rmpmd_v2",
    )
    stage, steady_source_stage = _select_probe_stages(
        curriculum, int(args.profile_version)
    )
    keys = jnp.stack(
        [
            jnp.full((n_envs,), 20262103, dtype=jnp.uint32),
            jnp.arange(1, n_envs + 1, dtype=jnp.uint32),
        ],
        axis=-1,
    )

    launch_env = MjxJuggleEnv(args.xml, n_envs=n_envs, cfg=stage.cfg)
    launch_state, launch_obs = launch_env.reset(keys)
    launch_surface, launch_rotation = _surface_state(launch_env, launch_state)
    launch_normal = launch_rotation[:, :, 2]
    launch_ball_position = launch_state.data.xpos[:, launch_env.ball_body_id]
    ball_radius = launch_state.model.geom_size[:, launch_env.ball_geom_id, 0]
    reset_normal_gap = (
        jnp.sum(
            (launch_ball_position - launch_surface) * launch_normal,
            axis=-1,
        )
        - ball_radius
    )

    def launch_desired_vz(current_state, tick, surface_position, _target_xy):
        site_velocity = _measured_racket_site_velocity(current_state)
        elapsed = tick.astype(jnp.float32) * launch_env.dt
        return jnp.where(
            elapsed < launch_duration,
            launch_speed,
            jnp.clip(
                10.0 * (SURFACE_Z_M - surface_position[:, 2])
                - 1.5 * site_velocity[:, 2],
                -0.9,
                0.4,
            ),
        )

    launch_result = _run_probe(
        env=launch_env,
        state=launch_state,
        obs=launch_obs,
        steps=int(args.steps),
        make_desired_vz=launch_desired_vz,
    )
    launch_pass = (
        (launch_result["confirmed_hit_count"] >= 1)
        & (launch_result["post_hit_apex"] >= 1.085)
        & (launch_result["post_hit_apex"] <= 1.125)
        & (launch_result["min_surface_z"] >= 1.002 - 0.002)
        & (launch_result["max_surface_z"] <= 1.120 + 0.002)
        & (launch_result["max_qvel_ratio"] <= 1.01)
        & (launch_result["qvel_exceed_fraction"] <= 1.0e-6)
        & (launch_result["qacc_exceed_fraction"] <= 0.02)
    )

    quality_grid_enabled = int(args.profile_version) >= 4
    if quality_grid_enabled:
        candidates = _steady_quality_probe_candidates(args.profile_version)
        steady_speed_base = np.asarray(
            [candidate[0] for candidate in candidates], dtype=np.float32
        )
        steady_lead_base = np.asarray(
            [candidate[1] for candidate in candidates], dtype=np.float32
        )
        steady_position_compensation_base = np.asarray(
            [candidate[2] for candidate in candidates], dtype=np.float32
        )
        steady_stroke_vx_base = np.asarray(
            [candidate[3] for candidate in candidates], dtype=np.float32
        )
        steady_stroke_vy_base = np.asarray(
            [candidate[4] for candidate in candidates], dtype=np.float32
        )
        steady_xy_kp_base = np.asarray(
            [candidate[5] for candidate in candidates], dtype=np.float32
        )
        steady_xy_kd_base = np.asarray(
            [candidate[6] for candidate in candidates], dtype=np.float32
        )
        steady_rotation_kp_base = np.asarray(
            [candidate[7] for candidate in candidates], dtype=np.float32
        )
        steady_rotation_kd_base = np.asarray(
            [candidate[8] for candidate in candidates], dtype=np.float32
        )
        steady_dls_regularization_base = np.asarray(
            [candidate[9] for candidate in candidates], dtype=np.float32
        )
        steady_vxy_lead_base = np.asarray(
            [candidate[10] for candidate in candidates], dtype=np.float32
        )
        # Equal-bin reset assignment cycles over the registered XY cells.
        # Repeating each controller candidate once per cell gives every cell
        # an identical speed/lead/controller witness population. V15--V17 use
        # one independently reachable center cell; older profiles retain nine.
        grid_cell_count = len(quality_grid_x_m) * len(quality_grid_y_m)
        steady_speed_np = np.repeat(steady_speed_base, grid_cell_count)
        steady_lead_np = np.repeat(steady_lead_base, grid_cell_count)
        steady_position_compensation_np = np.repeat(
            steady_position_compensation_base, grid_cell_count
        )
        steady_stroke_vx_np = np.repeat(
            steady_stroke_vx_base, grid_cell_count
        )
        steady_stroke_vy_np = np.repeat(
            steady_stroke_vy_base, grid_cell_count
        )
        steady_advance_np = np.zeros_like(steady_speed_np)
        steady_xy_kp_np = np.repeat(steady_xy_kp_base, grid_cell_count)
        steady_xy_kd_np = np.repeat(steady_xy_kd_base, grid_cell_count)
        steady_rotation_kp_np = np.repeat(
            steady_rotation_kp_base, grid_cell_count
        )
        steady_rotation_kd_np = np.repeat(
            steady_rotation_kd_base, grid_cell_count
        )
        steady_dls_regularization_np = np.repeat(
            steady_dls_regularization_base, grid_cell_count
        )
        steady_vxy_lead_np = np.repeat(
            steady_vxy_lead_base, grid_cell_count
        )
    else:
        steady_speeds_np = np.asarray(
            [0.9, 1.1, 1.3, 1.5], dtype=np.float32
        )
        steady_leads_np = np.asarray(
            [0.16, 0.18, 0.20, 0.22], dtype=np.float32
        )
        steady_speed_grid, steady_lead_grid = np.meshgrid(
            steady_speeds_np, steady_leads_np, indexing="ij"
        )
        steady_speed_base = steady_speed_grid.reshape(-1)
        steady_lead_base = steady_lead_grid.reshape(-1)
        steady_speed_np = steady_speed_base
        steady_lead_np = steady_lead_base
        steady_position_compensation_np = np.zeros_like(steady_speed_np)
        steady_stroke_vx_np = np.zeros_like(steady_speed_np)
        steady_stroke_vy_np = np.zeros_like(steady_speed_np)
        steady_advance_np = np.zeros_like(steady_speed_np)
        steady_xy_kp_np = np.full_like(steady_speed_np, 8.0)
        steady_xy_kd_np = np.full_like(steady_speed_np, 1.5)
        steady_rotation_kp_np = np.full_like(steady_speed_np, 6.0)
        steady_rotation_kd_np = np.full_like(steady_speed_np, 1.5)
        steady_dls_regularization_np = np.full_like(
            steady_speed_np, 2.0e-3
        )
        steady_vxy_lead_np = steady_lead_np.copy()
    steady_speed = jnp.asarray(steady_speed_np)
    steady_lead = jnp.asarray(steady_lead_np)
    steady_position_compensation = jnp.asarray(
        steady_position_compensation_np
    )
    steady_stroke_vx = jnp.asarray(steady_stroke_vx_np)
    steady_stroke_vy = jnp.asarray(steady_stroke_vy_np)
    steady_advance = jnp.asarray(steady_advance_np)
    steady_xy_kp = jnp.asarray(steady_xy_kp_np)
    steady_xy_kd = jnp.asarray(steady_xy_kd_np)
    steady_rotation_kp = jnp.asarray(steady_rotation_kp_np)
    steady_rotation_kd = jnp.asarray(steady_rotation_kd_np)
    steady_dls_regularization = jnp.asarray(
        steady_dls_regularization_np
    )
    steady_vxy_lead = jnp.asarray(steady_vxy_lead_np)
    steady_n_envs = int(steady_speed.shape[0])
    contact_center_z_m = SURFACE_Z_M + float(np.asarray(ball_radius)[0])
    steady_apex_height_m = STEADY_APEX_Z_M - contact_center_z_m
    steady_cfg = replace(
        steady_source_stage.cfg,
        ball_reset_mode="falling_contact",
        racket_launch_racket_follow_episode_target_xy=False,
        racket_launch_racket_follow_fixed_contact_surface_z=False,
        falling_reset_time_to_contact_range_s=(
            STEADY_RESET_TIME_TO_CONTACT_S,
            STEADY_RESET_TIME_TO_CONTACT_S,
        ),
        falling_reset_apex_height_range_m=(
            steady_apex_height_m,
            steady_apex_height_m,
        ),
        falling_reset_vxy_max=0.0,
        falling_reset_contact_xy_jitter=0.0,
        falling_reset_racket_follow_episode_target_xy=True,
        falling_reset_racket_follow_fixed_contact_surface_z=True,
        terminate_after_confirmed_hits=1,
        episode_target_x_range_m=(
            (quality_grid_x_m[0], quality_grid_x_m[-1])
            if quality_grid_enabled
            else stage.cfg.episode_target_x_range_m
        ),
        episode_target_y_range_m=(
            (quality_grid_y_m[0], quality_grid_y_m[-1])
            if quality_grid_enabled
            else stage.cfg.episode_target_y_range_m
        ),
        fixed_contact_eval_grid_x_m=(
            quality_grid_x_m
            if quality_grid_enabled
            else stage.cfg.fixed_contact_eval_grid_x_m
        ),
        fixed_contact_eval_grid_y_m=(
            quality_grid_y_m
            if quality_grid_enabled
            else stage.cfg.fixed_contact_eval_grid_y_m
        ),
        fixed_contact_xy_bin_sampling_enabled=quality_grid_enabled,
        fixed_contact_xy_equal_bin_evaluation=quality_grid_enabled,
        fixed_contact_xy_equal_bin_seed_permutation=False,
        fixed_contact_xy_center_probability=0.0,
        fixed_contact_xy_bin_jitter_m=0.0,
    )
    steady_env = MjxJuggleEnv(
        args.xml, n_envs=steady_n_envs, cfg=steady_cfg
    )
    steady_keys = jnp.stack(
        [
            jnp.full((steady_n_envs,), 20262874, dtype=jnp.uint32),
            jnp.arange(1, steady_n_envs + 1, dtype=jnp.uint32),
        ],
        axis=-1,
    )
    steady_state, steady_obs = steady_env.reset(steady_keys)
    steady_state = steady_state._replace(
        hit_count=jnp.full(
            (steady_n_envs,), 2, dtype=steady_state.hit_count.dtype
        )
    )

    def steady_desired_vz(current_state, _tick, surface_position, _target_xy):
        site_velocity = _measured_racket_site_velocity(current_state)
        ball_position = current_state.data.xpos[:, steady_env.ball_body_id]
        ball_velocity = current_state.data.qvel[
            :, steady_env.ball_vadr : steady_env.ball_vadr + 3
        ]
        _crossing_xy, time_to_contact, valid = predict_descending_plane_crossing(
            ball_position,
            ball_velocity,
            contact_center_z_m,
            gravity_m_s2=GRAVITY_M_S2,
            max_time_s=1.0,
        )
        strike = valid & (time_to_contact <= steady_lead)
        if not quality_grid_enabled:
            desired_z = jnp.where(
                strike, SURFACE_Z_M, SURFACE_Z_M - 0.009
            )
            feedforward = jnp.where(strike, steady_speed, 0.0)
            return jnp.clip(
                feedforward
                + 12.0 * (desired_z - surface_position[:, 2])
                - 1.5 * site_velocity[:, 2],
                -1.0,
                3.2,
            )
        # Give the delayed RMP/PD plant 23 mm of monotone run-up, then search
        # a causal constant-velocity pulse whose measured surface crosses the
        # registered plane while moving upward.  Unlike the high-terminal-
        # velocity Hermite diagnostic this trajectory never contains a
        # preparatory downstroke after the pulse begins.
        prestroke_z = SURFACE_Z_M - STEADY_PRESTROKE_DEPTH_M
        strike_target_z = (
            SURFACE_Z_M
            - steady_position_compensation * steady_speed / 12.0
        )
        desired_z = jnp.where(strike, strike_target_z, prestroke_z)
        trajectory_vz = jnp.where(strike, steady_speed, 0.0)
        return jnp.clip(
            trajectory_vz
            + 12.0 * (desired_z - surface_position[:, 2])
            - 1.5 * site_velocity[:, 2],
            -1.0,
            3.2,
        )

    def steady_desired_vxy(
        current_state, _tick, surface_position, _target_xy
    ):
        del surface_position, _target_xy
        ball_position = current_state.data.xpos[:, steady_env.ball_body_id]
        ball_velocity = current_state.data.qvel[
            :, steady_env.ball_vadr : steady_env.ball_vadr + 3
        ]
        _crossing_xy, time_to_contact, valid = predict_descending_plane_crossing(
            ball_position,
            ball_velocity,
            contact_center_z_m,
            gravity_m_s2=GRAVITY_M_S2,
            max_time_s=1.0,
        )
        strike = valid & (time_to_contact <= steady_vxy_lead)
        return jnp.where(
            strike[:, None],
            jnp.stack((steady_stroke_vx, steady_stroke_vy), axis=-1),
            0.0,
        )

    steady_result = _run_probe(
        env=steady_env,
        state=steady_state,
        obs=steady_obs,
        steps=int(args.steps),
        make_desired_vz=steady_desired_vz,
        make_desired_vxy=steady_desired_vxy,
        xy_kp=steady_xy_kp,
        xy_kd=steady_xy_kd,
        rotation_kp=steady_rotation_kp,
        rotation_kd=steady_rotation_kd,
        dls_regularization=steady_dls_regularization,
    )
    steady_pass = (
        (steady_result["confirmed_hit_count"] >= 1)
        & (steady_result["post_hit_apex"] >= 1.425)
        & (steady_result["post_hit_apex"] <= 1.475)
        & (steady_result["min_surface_z"] >= 1.002 - 0.002)
        & (steady_result["max_surface_z"] <= 1.120 + 0.002)
        & (steady_result["max_qvel_ratio"] <= 1.01)
        & (steady_result["qvel_exceed_fraction"] <= 1.0e-6)
        & (steady_result["qacc_exceed_fraction"] <= 0.02)
    )
    if quality_grid_enabled:
        steady_pass = steady_pass & (
            (steady_result["event_q_z"] >= 0.45)
            & (steady_result["event_q_pose"] >= 0.45)
            & (steady_result["event_q_vxy"] >= 0.45)
            & (steady_result["event_q_omega"] >= 0.45)
            & (steady_result["event_q_offset"] >= 0.45)
        )

    reset_gap_np = np.asarray(reset_normal_gap)
    reset_pass = bool(
        np.all(np.isfinite(launch_result["initial_obs"]))
        and np.all(np.isfinite(steady_result["initial_obs"]))
        and np.max(np.abs(reset_gap_np - 0.001)) <= 0.002
        and np.max(np.abs(np.asarray(launch_surface[:, 2]) - SURFACE_Z_M))
        <= 0.010
    )
    launch_winners = np.flatnonzero(launch_pass)
    steady_winners = np.flatnonzero(steady_pass)
    launch_reachable = bool(launch_winners.size > 0)
    steady_quality_grid = []
    steady_quality_grid_reachable = True
    minimum_quality_witnesses = 2 if int(args.profile_version) >= 14 else 1
    if quality_grid_enabled:
        target_offsets = np.asarray(steady_result["initial_target_offset"])
        for target_x in quality_grid_x_m:
            for target_y in quality_grid_y_m:
                cell_mask = np.isclose(target_offsets[:, 0], target_x) & np.isclose(
                    target_offsets[:, 1], target_y
                )
                cell_winners = np.flatnonzero(cell_mask & steady_pass)
                cell_reachable = bool(
                    cell_winners.size >= minimum_quality_witnesses
                )
                steady_quality_grid_reachable = (
                    steady_quality_grid_reachable and cell_reachable
                )
                steady_quality_grid.append(
                    {
                        "target_x_m": target_x,
                        "target_y_m": target_y,
                        "candidate_count": int(np.count_nonzero(cell_mask)),
                        "winning_lane_indices": cell_winners.tolist(),
                        "quality_reachable": cell_reachable,
                    }
                )
    steady_reachable = bool(
        steady_winners.size > 0 and steady_quality_grid_reachable
    )
    dynamic_method_contract: dict[str, Any] | None = None
    dynamic_method_contract_passed = True
    if int(args.profile_version) >= 14:
        transition_hits = int(
            getattr(
                steady_source_stage.cfg,
                "fixed_plane_limit_cycle_transition_confirmed_hits",
            )
        )
        transition_apex_z = np.linspace(
            LAUNCH_APEX_Z_M,
            STEADY_APEX_Z_M,
            transition_hits + 1,
            dtype=np.float32,
        )
        nominal_apex_z = np.concatenate(
            [transition_apex_z, transition_apex_z[-1:]]
        )
        nominal_incoming_vz = np.concatenate(
            [
                np.zeros((1,), dtype=np.float32),
                -np.sqrt(
                    2.0
                    * GRAVITY_M_S2
                    * np.maximum(
                        nominal_apex_z[:-1] - contact_center_z_m,
                        0.0,
                    )
                ).astype(np.float32),
            ]
        )
        raw_vz, executable_vz, clipped_vz = (
            limit_cycle_dynamic_racket_vz_target_m_s(
                incoming_ball_vz_m_s=jnp.asarray(nominal_incoming_vz),
                desired_apex_z_m=jnp.asarray(nominal_apex_z),
                contact_ball_center_z_m=contact_center_z_m,
                gravity_m_s2=GRAVITY_M_S2,
                effective_restitution=EFFECTIVE_RESTITUTION,
                min_racket_vz_m_s=0.0,
                max_racket_vz_m_s=0.90,
            )
        )
        raw_vz_np = np.asarray(raw_vz)
        executable_vz_np = np.asarray(executable_vz)
        clipped_vz_np = np.asarray(clipped_vz)
        prepare_s = float(
            steady_source_stage.cfg.fixed_plane_limit_cycle_stroke_prepare_s
        )
        prestroke_depth_m = float(
            steady_source_stage.cfg.fixed_plane_limit_cycle_prestroke_depth_m
        )
        sample_ttc = np.linspace(prepare_s, 0.0, 401, dtype=np.float32)
        tiled_speed = np.repeat(executable_vz_np, sample_ttc.size)
        tiled_ttc = np.tile(sample_ttc, executable_vz_np.size)
        stroke_z, stroke_vz, stroke_active = (
            limit_cycle_workspace_bounded_stroke_reference(
                time_to_contact_s=jnp.asarray(tiled_ttc),
                target_contact_vz_m_s=jnp.asarray(tiled_speed),
                crossing_valid=jnp.ones(tiled_ttc.shape, dtype=jnp.bool_),
                pending_hit=jnp.zeros(tiled_ttc.shape, dtype=jnp.bool_),
                hit_count=jnp.ones(tiled_ttc.shape, dtype=jnp.int32),
                contact_surface_z_m=SURFACE_Z_M,
                prestroke_depth_m=prestroke_depth_m,
                prepare_s=prepare_s,
                min_hit_count=1,
            )
        )
        stroke_z_np = np.asarray(stroke_z).reshape((-1, sample_ttc.size))
        stroke_vz_np = np.asarray(stroke_vz).reshape((-1, sample_ttc.size))
        stroke_active_np = np.asarray(stroke_active)
        dynamic_method_contract_passed = bool(
            np.all(np.isfinite(raw_vz_np))
            and np.all(raw_vz_np >= 0.0)
            and np.all(raw_vz_np <= 0.90)
            and not np.any(clipped_vz_np)
            and np.all(stroke_active_np)
            and np.min(stroke_z_np) >= 1.010 - 1.0e-6
            and np.max(stroke_z_np) <= SURFACE_Z_M + 1.0e-6
            and np.allclose(
                stroke_z_np[:, 0], SURFACE_Z_M, atol=1.0e-6, rtol=0.0
            )
            and np.allclose(
                stroke_vz_np[:, 0], 0.0, atol=1.0e-6, rtol=0.0
            )
            and np.allclose(
                stroke_z_np[:, -1], SURFACE_Z_M, atol=1.0e-6, rtol=0.0
            )
            and np.allclose(
                stroke_vz_np[:, -1],
                executable_vz_np,
                atol=1.0e-5,
                rtol=0.0,
            )
        )
        dynamic_method_contract = {
            "passed": dynamic_method_contract_passed,
            "min_contact_vz_m_s": 0.0,
            "max_contact_vz_m_s": 0.90,
            "prestroke_depth_m": prestroke_depth_m,
            "stroke_prepare_s": prepare_s,
            "minimum_sampled_surface_z_m": float(np.min(stroke_z_np)),
            "maximum_sampled_surface_z_m": float(np.max(stroke_z_np)),
            "phases": [
                {
                    "name": name,
                    "incoming_ball_vz_m_s": float(nominal_incoming_vz[index]),
                    "desired_apex_z_m": float(nominal_apex_z[index]),
                    "raw_racket_vz_target_m_s": float(raw_vz_np[index]),
                    "executable_racket_vz_target_m_s": float(
                        executable_vz_np[index]
                    ),
                    "clipped": bool(clipped_vz_np[index]),
                }
                for index, name in enumerate(
                    (
                        "launch",
                        *(
                            f"transition_{phase_index}"
                            for phase_index in range(1, transition_hits + 1)
                        ),
                        "steady",
                    )
                )
            ],
        }
    passed = bool(
        reset_pass
        and launch_reachable
        and steady_reachable
        and dynamic_method_contract_passed
    )

    launch_lanes = []
    for index in range(n_envs):
        lane = {
            "index": index,
            "requested_surface_vz_m_s": float(np.asarray(launch_speed)[index]),
            "pulse_duration_s": float(np.asarray(launch_duration)[index]),
            **_lane_common(launch_result, index),
            "passed": bool(launch_pass[index]),
        }
        launch_lanes.append(lane)
    steady_lanes = []
    for index in range(steady_n_envs):
        lane = {
            "index": index,
            "requested_surface_vz_m_s": float(np.asarray(steady_speed)[index]),
            "strike_lead_s": float(
                np.asarray(steady_lead + steady_advance)[index]
            ),
            "stroke_duration_s": None,
            "execution_advance_s": 0.0,
            "position_equilibrium_compensation": float(
                np.asarray(steady_position_compensation)[index]
            ),
            "requested_surface_vx_m_s": float(
                np.asarray(steady_stroke_vx)[index]
            ),
            "requested_surface_vy_m_s": float(
                np.asarray(steady_stroke_vy)[index]
            ),
            "stroke_trajectory": "low_hold_constant_velocity",
            "xy_kp_s_inv": float(np.asarray(steady_xy_kp)[index]),
            "xy_kd": float(np.asarray(steady_xy_kd)[index]),
            "rotation_kp_s_inv": float(
                np.asarray(steady_rotation_kp)[index]
            ),
            "rotation_kd": float(np.asarray(steady_rotation_kd)[index]),
            "dls_regularization": float(
                np.asarray(steady_dls_regularization)[index]
            ),
            "horizontal_feedforward_lead_s": float(
                np.asarray(steady_vxy_lead)[index]
            ),
            **_lane_common(steady_result, index),
            "passed": bool(steady_pass[index]),
        }
        steady_lanes.append(lane)

    desired_steady_outgoing_vz = math.sqrt(
        2.0 * GRAVITY_M_S2 * steady_apex_height_m
    )
    required_steady_racket_vz = (
        desired_steady_outgoing_vz
        - EFFECTIVE_RESTITUTION * desired_steady_outgoing_vz
    ) / (1.0 + EFFECTIVE_RESTITUTION)
    launch_height_m = LAUNCH_APEX_Z_M - contact_center_z_m
    desired_launch_outgoing_vz = math.sqrt(
        2.0 * GRAVITY_M_S2 * launch_height_m
    )
    required_resting_launch_racket_vz = (
        desired_launch_outgoing_vz / (1.0 + EFFECTIVE_RESTITUTION)
    )
    script_dir = Path(__file__).resolve().parent
    code_paths = {
        "reachability_screen": Path(__file__).resolve(),
        "limit_cycle_contract": script_dir / "fixed_plane_limit_cycle_v1.py",
        "mjx_environment": script_dir / "mjx_juggle_env.py",
        "curriculum_trainer": script_dir / "train_juggle_mjx_curriculum.py",
    }
    report = {
        "schema": (
            f"fixed_plane_limit_cycle_reachability_v{args.profile_version}"
        ),
        "passed": passed,
        "rmp_pd_reachable": bool(launch_reachable and steady_reachable),
        "launch_reachable": launch_reachable,
        "steady_reachable": steady_reachable,
        "reset_geometry_passed": reset_pass,
        "profile": profile,
        "actor_observation_dim": int(launch_env.obs_dim),
        "control_dt_s": float(launch_env.dt),
        "racket_launch_hold_time_s": float(
            launch_env.cfg.racket_launch_hold_time_s
        ),
        "contact_surface_z_m": SURFACE_Z_M,
        "contact_ball_center_z_m": contact_center_z_m,
        "launch_apex_z_m": LAUNCH_APEX_Z_M,
        "transition_apex_z_m": 0.5 * (
            LAUNCH_APEX_Z_M + STEADY_APEX_Z_M
        ),
        "transition_confirmed_hits": int(
            getattr(
                steady_source_stage.cfg,
                "fixed_plane_limit_cycle_transition_confirmed_hits",
            )
        ),
        "apex_schedule_z_m": np.linspace(
            LAUNCH_APEX_Z_M,
            STEADY_APEX_Z_M,
            int(
                getattr(
                    steady_source_stage.cfg,
                    "fixed_plane_limit_cycle_transition_confirmed_hits",
                )
            )
            + 1,
        ).tolist(),
        "steady_apex_z_m": STEADY_APEX_Z_M,
        "steady_reset_time_to_contact_s": STEADY_RESET_TIME_TO_CONTACT_S,
        "steady_prestroke_depth_m": STEADY_PRESTROKE_DEPTH_M,
        "steady_max_strike_lead_s": STEADY_MAX_STRIKE_LEAD_S,
        "desired_launch_outgoing_ball_vz_m_s": desired_launch_outgoing_vz,
        "required_resting_ball_launch_racket_vz_m_s": (
            required_resting_launch_racket_vz
        ),
        "desired_steady_outgoing_ball_vz_m_s": desired_steady_outgoing_vz,
        "steady_incoming_ball_vz_m_s": -desired_steady_outgoing_vz,
        "required_steady_racket_vz_m_s": required_steady_racket_vz,
        "reset_gap_error_max_m": float(
            np.max(np.abs(reset_gap_np - 0.001))
        ),
        "launch_winning_lane_indices": launch_winners.tolist(),
        "steady_winning_lane_indices": steady_winners.tolist(),
        "steady_quality_grid_reachable": bool(
            steady_quality_grid_reachable
        ),
        "steady_quality_grid_min_witnesses": int(
            minimum_quality_witnesses
        ),
        "steady_quality_grid": steady_quality_grid,
        "steady_quality_grid_x_m": quality_grid_x_m,
        "steady_quality_grid_y_m": quality_grid_y_m,
        "steady_quality_component_floor": (
            0.45 if quality_grid_enabled else None
        ),
        "dynamic_method_contract": dynamic_method_contract,
        "diagnostic_controller_is_training_method": False,
        "xml": str(args.xml.resolve()),
        "xml_sha256": _sha256(args.xml),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "physical_gpu_index": int(args.physical_gpu_index),
        "physical_gpu_uuid": physical_gpu_uuid,
        "cuda_visible_devices": visible_devices,
        "code_sha256": {
            name: _sha256(path) for name, path in code_paths.items()
        },
        "launch_lanes": launch_lanes,
        "steady_lanes": steady_lanes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
