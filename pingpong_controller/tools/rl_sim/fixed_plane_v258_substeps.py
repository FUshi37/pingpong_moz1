"""V258 Gaussian rebound-force limiting around the unchanged MJX substep."""

from typing import Callable

import jax
import jax.numpy as jnp

from fixed_plane_v256_substeps import initialize_metadata as initialize_workspace_metadata
from fixed_plane_v256_workspace_force import (
    RACKET_LOW_M, RACKET_HIGH_M, workspace_outside,
)
from fixed_plane_v258_gaussian_force import limit_rebound_force, sample_velocity_scaled_forces

META_FIELDS = 12


def initialize_metadata(state) -> tuple:
    """Retain the existing persistent contact fields; add transient diagnostics."""
    n = state.step_count.shape[0]
    return (*initialize_workspace_metadata(state), jnp.zeros(n, jnp.int32),
            jnp.zeros(n), jnp.zeros(n), jnp.zeros(n), jnp.zeros(n, jnp.int32))


def wrap_substep(env, state, native_substep: Callable) -> Callable:
    """Bound the combined disturbance on the rising1ms release tick only."""
    def substep(index, carry):
        native = carry[:-META_FIELDS]
        (previous, cooldown, ball_exit, racket_exit, squares, events, exit_tick,
         limited_count, force_peak, angle_peak, speed_peak, near_cap_count) = carry[-META_FIELDS:]
        data = env.batched_forward(state.model, native[0])
        pre_ball_exit = workspace_outside(data.xpos[:, env.ball_body_id],
            env.cfg.v256_ball_workspace_lower_m, env.cfg.v256_ball_workspace_upper_m)
        pre_racket_exit = workspace_outside(data.site_xpos[:, env.racket_site_id],
                                           RACKET_LOW_M, RACKET_HIGH_M)
        stopped = ball_exit | racket_exit | pre_ball_exit | pre_racket_exit
        contact, other = env._ball_contact_flags(data)
        keys = jax.vmap(lambda key: jax.random.fold_in(key,
            jnp.uint32(25801) + jnp.asarray(index, jnp.uint32)))(state.rng)
        original = data.xfrc_applied
        mass = state.model.body_mass[:, env.ball_body_id]
        interval = slice(env.ball_vadr, env.ball_vadr + 3)
        free_velocity = data.qvel[:, interval] + env.timestep * (
            state.model.opt.gravity + (original[:, env.ball_body_id, :3]
                + data.qfrc_applied[:, interval]) / mass[:, None])
        force, rebound, cooldown = sample_velocity_scaled_forces(keys, contact=contact,
            previous_contact=previous, other_contact=other,
            vz_m_s=data.qvel[:, env.ball_vadr + 2], cooldown_ticks=cooldown,
            flight_std_n=env.cfg.v256_flight_force_std_n,
            rebound_std_n=env.cfg.v256_rebound_force_std_n,
            free_velocity_m_s=free_velocity,
            reference_speed_m_s=env.cfg.v258_rebound_reference_speed_m_s)
        force = jnp.where(stopped[:, None], 0., force)
        rebound &= ~stopped
        bounded, scale = limit_rebound_force(force, free_velocity, mass_kg=mass,
            dt_s=env.timestep, force_limit_n=env.cfg.v258_rebound_force_limit_n,
            angle_limit_deg=env.cfg.v258_rebound_angle_limit_deg,
            speed_relative_limit=env.cfg.v258_rebound_speed_relative_limit)
        force = jnp.where(rebound[:, None], bounded, force)
        predicted = free_velocity + force * (env.timestep / mass[:, None])
        angle = jnp.rad2deg(jnp.arctan2(jnp.linalg.norm(jnp.cross(
            free_velocity, predicted), axis=-1), jnp.sum(free_velocity * predicted, axis=-1)))
        speed = jnp.abs(jnp.linalg.norm(predicted, axis=-1) /
                       jnp.maximum(jnp.linalg.norm(free_velocity, axis=-1), 1e-12) - 1.)
        forced = data.replace(xfrc_applied=original.at[:, env.ball_body_id, :3].add(force))
        result = native_substep(index, (forced, *native[1:]))
        candidate = env.batched_forward(state.model, result[0].replace(xfrc_applied=original))
        from fixed_plane_v225_near_side import select_intervention_rows
        candidate = select_intervention_rows(data, candidate, ~stopped)
        ball_now = pre_ball_exit | workspace_outside(candidate.xpos[:, env.ball_body_id],
            env.cfg.v256_ball_workspace_lower_m, env.cfg.v256_ball_workspace_upper_m)
        racket_now = pre_racket_exit | workspace_outside(candidate.site_xpos[:, env.racket_site_id],
                                                       RACKET_LOW_M, RACKET_HIGH_M)
        exit_tick = jnp.where((exit_tick < 0) & (ball_now | racket_now), index + 1, exit_tick)
        return (candidate, *result[1:], contact, cooldown, ball_exit | ball_now,
            racket_exit | racket_now, squares + force**2,
            events + rebound.astype(jnp.int32), exit_tick,
            limited_count + (rebound & (scale < 1.)).astype(jnp.int32),
            jnp.maximum(force_peak, jnp.where(rebound, jnp.linalg.norm(force, axis=-1), 0.)),
            jnp.maximum(angle_peak, jnp.where(rebound, angle, 0.)),
            jnp.maximum(speed_peak, jnp.where(rebound, speed, 0.)),
            near_cap_count + (rebound & (angle >= .999 *
                env.cfg.v258_rebound_angle_limit_deg)).astype(jnp.int32))
    return substep
