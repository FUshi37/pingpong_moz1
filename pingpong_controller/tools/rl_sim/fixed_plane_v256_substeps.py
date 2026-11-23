"""V256-only wrapper around the unchanged native MJX physics substep."""

from typing import Callable

import jax
import jax.numpy as jnp

from fixed_plane_v256_workspace_force import (
    RACKET_LOW_M, RACKET_HIGH_M, workspace_outside, sample_forces,
)

META_FIELDS = 7


def initialize_metadata(state) -> tuple:
    """Initialize transient diagnostics plus two persistent Markov fields."""
    n = state.step_count.shape[0]
    return (state.v256_previous_contact, state.v256_rebound_cooldown_ticks,
            jnp.zeros(n, bool), jnp.zeros(n, bool), jnp.zeros((n, 3)),
            jnp.zeros(n, jnp.int32), jnp.full(n, -1, jnp.int32))


def wrap_substep(env, state, native_substep: Callable) -> Callable:
    """Apply world forces for1ms, then latch/freeze the first true workspace exit."""
    def substep(index, carry):
        native = carry[:-META_FIELDS]
        previous, cooldown, ball_exit, racket_exit, squares, events, exit_tick = carry[-META_FIELDS:]
        data = env.batched_forward(state.model, native[0])
        pre_ball_exit = workspace_outside(data.xpos[:, env.ball_body_id],
            env.cfg.v256_ball_workspace_lower_m, env.cfg.v256_ball_workspace_upper_m)
        pre_racket_exit = workspace_outside(data.site_xpos[:, env.racket_site_id],
                                             RACKET_LOW_M, RACKET_HIGH_M)
        stopped = ball_exit | racket_exit | pre_ball_exit | pre_racket_exit
        contact, other = env._ball_contact_flags(data)
        keys = jax.vmap(lambda key: jax.random.fold_in(key,
            jnp.uint32(25601) + jnp.asarray(index, jnp.uint32)))(state.rng)
        force, rebound, cooldown = sample_forces(keys, contact=contact,
            previous_contact=previous, other_contact=other,
            vz_m_s=data.qvel[:, env.ball_vadr + 2], cooldown_ticks=cooldown,
            flight_std_n=env.cfg.v256_flight_force_std_n,
            rebound_std_n=env.cfg.v256_rebound_force_std_n)
        force = jnp.where(stopped[:, None], 0., force)
        rebound &= ~stopped
        original_force = data.xfrc_applied
        forced = data.replace(xfrc_applied=original_force.at[:, env.ball_body_id, :3].add(force))
        result = native_substep(index, (forced, *native[1:]))
        candidate = env.batched_forward(state.model, result[0].replace(xfrc_applied=original_force))
        # Native loop still evaluates statically; exited lanes retain their
        # first-exit state and cannot re-enter during the remaining policy tick.
        from fixed_plane_v225_near_side import select_intervention_rows
        candidate = select_intervention_rows(data, candidate, ~stopped)
        ball_now = pre_ball_exit | workspace_outside(candidate.xpos[:, env.ball_body_id],
            env.cfg.v256_ball_workspace_lower_m, env.cfg.v256_ball_workspace_upper_m)
        racket_now = pre_racket_exit | workspace_outside(candidate.site_xpos[:, env.racket_site_id],
                                                        RACKET_LOW_M, RACKET_HIGH_M)
        exit_tick = jnp.where((exit_tick < 0) & (ball_now | racket_now), index + 1, exit_tick)
        return (candidate, *result[1:], contact, cooldown, ball_exit | ball_now,
                racket_exit | racket_now, squares + force**2,
                events + rebound.astype(jnp.int32), exit_tick)
    return substep
