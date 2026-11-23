"""Causal four-goal groups and20 ms external-force scheduling at actual apexes."""

from typing import Any, NamedTuple
import jax
import jax.numpy as jnp


class ScheduleState(NamedTuple):
    goals_completed: Any
    pulse_used: Any
    remaining_steps: Any
    force_world_n: Any
    group_id: Any
    pulse_rng: Any
    goal_rng: Any
    direction_key: Any
    direction_sign: Any


def init_schedule(key,lane_ids,episode_ids=None):
    """Assign persistent logical identities before microbatch partitioning.

    Adjacent logical lanes have opposite proposals for the same episode/group.
    Actual delivery can become unbalanced after failure; record delivery, never
    invent a matching pulse in a failed lane. RNG/proposals are not critic input.
    """
    n=lane_ids.shape[0]
    episodes=jnp.zeros(n,jnp.uint32) if episode_ids is None else episode_ids.astype(jnp.uint32)
    fold=jax.vmap(lambda i,e:jax.random.fold_in(jax.random.fold_in(key,i),e))
    keys=fold(lane_ids.astype(jnp.uint32),episodes)
    directions=fold((lane_ids//2).astype(jnp.uint32)+jnp.uint32(0x40000000),episodes)
    pulse=jax.vmap(lambda k:jax.random.fold_in(k,1))(keys)
    goals=jax.vmap(lambda k:jax.random.fold_in(k,2))(keys)
    zero=jnp.zeros(n,jnp.int32)
    return ScheduleState(zero,jnp.zeros(n,bool),zero,jnp.zeros((n,3)),zero,pulse,goals,
        directions,jnp.where(lane_ids%2==0,1.,-1.))


def pulse_proposal(state):
    """Uniform disk horizontal delta-v and bounded vertical delta-v, in m/s."""
    keys=jax.vmap(jax.random.fold_in)(state.direction_key,state.group_id)
    values=jax.vmap(lambda k:jax.random.uniform(k,(3,)))(keys)
    theta=values[:,0]*(2*jnp.pi)
    radius=.05*jnp.sqrt(values[:,1])
    dv=jnp.stack((radius*jnp.cos(theta),radius*jnp.sin(theta),.1*(2*values[:,2]-1)),axis=-1)
    return dv*state.direction_sign[:,None]


def finish_substep(state,goal_switched,valid,*,enabled,mass_kg):
    """Consume the current1 ms force tick, then arm any new apex-triggered pulse.

    The crossing is observed at the end of a1 ms step. Onset is the following
    boundary, at most1 ms after the interpolated actual apex, for exactly20
    full steps. A failure/time limit may interrupt it; never extend an episode.
    """
    if not enabled:
        return state,jnp.zeros_like(valid)
    switched=goal_switched & valid
    count=state.goals_completed+switched.astype(jnp.int32)
    boundary=switched & (count==4)
    remaining=jnp.maximum(state.remaining_steps-1,0)
    force=jnp.where((remaining>0)[:,None],state.force_world_n,0.)
    keys=jax.vmap(lambda k:jax.random.split(k,2))(state.pulse_rng)
    uniform=jax.vmap(lambda k:jax.random.uniform(k))(keys[:,1])
    # Conditional hazards1/3,1/2,1 produce a uniform internal switch without
    # storing a future trigger index in the visible state.
    hazard=1./jnp.maximum(4-count,1)
    start=(enabled & switched & ~boundary & ~state.pulse_used
           & (remaining==0) & (uniform<hazard))
    candidate=state._replace(goals_completed=jnp.where(boundary,0,count),
        pulse_used=jnp.where(boundary,False,state.pulse_used|start),
        remaining_steps=jnp.where(start,20,remaining),
        force_world_n=jnp.where(start[:,None],pulse_proposal(state)*(mass_kg/.02),force),
        group_id=state.group_id+boundary.astype(jnp.int32),
        pulse_rng=jnp.where(switched[:,None],keys[:,0],state.pulse_rng))
    after=jax.tree.map(lambda a,b:jnp.where(valid.reshape(valid.shape+(1,)*(a.ndim-1)),a,b),
                       candidate,state)
    return after,start


def schedule_features(state):
    """Six approved current-state values; no pending pulse or RNG disclosure."""
    return jnp.concatenate((state.goals_completed[:,None]/4.,state.pulse_used[:,None].astype(jnp.float32),
        state.remaining_steps[:,None]/20.,state.force_world_n/.02),axis=-1)
