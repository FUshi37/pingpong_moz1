"""Critic-only transactions over immutable, parent-generated lambda labels."""

from __future__ import annotations

import jax
import numpy as np

from .critic_fit import update_session
from .state_adapter import numeric_tree_digest


def invariant_digest(state: dict) -> str:
    """Everything except value learning, its sampling key and local counter."""
    learner = state['learner']
    mutable = {'critic', 'target_critic', 'critic_optimizer', 'sampling_rng', 'critic_updates'}
    preserved = {name:getattr(learner, name) for name in learner._fields if name not in mutable}
    other = {k:v for k,v in state.items() if k not in ('learner', 'fixed_fit_transactions')}
    return numeric_tree_digest((preserved, other))


def update_fixed_state(state: dict, inputs: np.ndarray, update_fn, *,
                       initial_updates: int, maximum_updates: int = 512,
                       authorized: bool = False) -> tuple:
    """Read saved labels, never recompute them from the evolving EMA network."""
    if not authorized:
        raise PermissionError('Fixed-data Critic updates require completed admission')
    learner = state['learner']
    if int(learner.critic_updates)-initial_updates != 4*int(state['fixed_fit_transactions']):
        raise ValueError('Fixed-data transaction count differs from actual optimizer updates')
    labels = np.asarray(state['fixed_lambda_targets'])
    after, metrics = update_session(learner, inputs, labels, update_fn,
        authorized=True, origin='imagination', initial_updates=initial_updates,
        maximum_updates=maximum_updates)
    result = dict(state, learner=after,
                  fixed_fit_transactions=state['fixed_fit_transactions']+1)
    return result, metrics


def next_draw(learner, count: int, *, batch: int = 8192, microbatch: int = 2048) -> np.ndarray:
    """Inspect the next exact sampler draw without advancing or fitting."""
    _, key = jax.random.split(learner.sampling_rng)
    return np.asarray(jax.random.randint(key, (batch//microbatch, microbatch), 0, count,
                                        dtype=np.int32))
