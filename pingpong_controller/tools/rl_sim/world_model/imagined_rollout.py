"""Reparameterized actor rollouts through a frozen structured world model."""

from __future__ import annotations

from typing import Any, NamedTuple
import jax
import jax.numpy as jnp

from .actor_critic import ImaginedBatch, actor_return, frozen_value
from .critic_features import critic_features
from .full_prior_world import world_actor_observation
from .imagined_env import imagined_step
from .migration import actor_mean


class PolicyWindow(NamedTuple):
    environment: Any
    exploration_rng: Any
    transitions: Any
    critic_inputs: Any
    raw_mean: Any
    raw_action: Any
    accepted_action: Any


def policy_window(actor,robot,auxiliary,environment,exploration_rng,goal_proposals,*,
                  exploration_std=.03,rematerialize=True,**step_options):
    """No teacher observations/commands/rewards can enter this transition API."""
    n=environment.episode_steps.shape[0]
    if goal_proposals.ndim!=3 or goal_proposals.shape[1:]!=(n,3) or exploration_rng.shape!=(n,2):
        raise ValueError('Expected(T,N,3) task proposals and independent exploration keys')
    if exploration_std<0.:
        raise ValueError('Exploration scale must be nonnegative')
    def tick(carry,goals):
        state,keys=carry
        splits=jax.vmap(lambda k:jax.random.split(k,2))(keys)
        noise=jax.vmap(lambda k:jax.random.normal(k,(7,)))(splits[:,1])
        observation=world_actor_observation(state.world,state.task.goal_xyz)
        mean=actor_mean(actor,observation)
        raw=mean+exploration_std*noise
        inputs=critic_features(state.world,state.task,state.events,state.context,state.episode_steps)
        after,transition=imagined_step(robot,auxiliary,state,raw,goals,**step_options)
        return (after,splits[:,0]),(transition,inputs,mean,raw,after.world.control.previous_accepted_action)
    # Recompute within a control step during reverse mode instead of retaining
    # every 1 kHz network/RMP/FK intermediate for the whole gradient window.
    tick=jax.checkpoint(tick) if rematerialize else tick
    (state,keys),(transitions,inputs,mean,raw,accepted)=jax.lax.scan(tick,
        (environment,exploration_rng),goal_proposals)
    return PolicyWindow(state,keys,transitions,inputs,mean,raw,accepted)


def window_batch(window):
    t=window.transitions
    return ImaginedBatch(t.reward,t.discount,t.trace_continue,t.valid)


def actor_window_objective(actor,target_critic,robot,auxiliary,environment,
                           exploration_rng,goal_proposals,**options):
    window=policy_window(actor,robot,auxiliary,environment,exploration_rng,goal_proposals,**options)
    values=frozen_value(target_critic,window.transitions.next_critic_features)
    returns=actor_return(window_batch(window),values)
    return -jnp.mean(returns),window
