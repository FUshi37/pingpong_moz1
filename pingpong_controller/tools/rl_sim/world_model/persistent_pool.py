"""Persistent versus short-branch state ownership; no implicit goal reset."""

from __future__ import annotations

from typing import Any, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

from .imagined_env import reset_lanes


class PersistentPool(NamedTuple):
    environment: Any
    exploration_rng: Any
    model_revision: Any
    imagined_transition_words: Any
    windows: Any


def init_pool(environment,exploration_rng,*,model_revision=0):
    n=environment.world.physics.q_rad.shape[0]
    if exploration_rng.shape!=(n,2):
        raise ValueError('Each logical environment needs its own exploration RNG')
    return PersistentPool(environment,exploration_rng,jnp.asarray(model_revision,jnp.int32),
        jnp.zeros(2,jnp.uint32),jnp.asarray(0,jnp.int32))


def add_count(words,amount):
    """Two uint32 words preserve transition counts beyond2**32 without x64."""
    low=words[1]+jnp.asarray(amount,jnp.uint32)
    return jnp.stack((words[0]+(low<words[1]).astype(jnp.uint32),low))


def count_as_int(words):
    a=np.asarray(words,dtype=np.uint32)
    return (int(a[0])<<32)+int(a[1])


def prepare_window(pool,fresh_seeds,fresh_exploration_rng,*,persistent_envs=768):
    """Replace all short branches and ended persistent lanes before a window."""
    n=pool.environment.episode_steps.shape[0]
    if not 0<=persistent_envs<=n or fresh_seeds.episode_steps.shape!=(n,):
        raise ValueError('Pool split/seed batch mismatch')
    ended=(pool.environment.terminated|pool.environment.truncated|pool.environment.model_truncated)
    reset=ended | (jnp.arange(n)>=persistent_envs)
    environment=reset_lanes(pool.environment,fresh_seeds,reset)
    rng=jnp.where(reset[:,None],fresh_exploration_rng,pool.exploration_rng)
    return pool._replace(environment=jax.tree.map(jax.lax.stop_gradient,environment),
                         exploration_rng=rng),reset


def commit_window(pool,environment,exploration_rng,valid):
    """Detach gradient windows while keeping their complete forward history."""
    if valid.ndim!=2 or valid.shape[1]!=pool.environment.episode_steps.shape[0]:
        raise ValueError('Expected time-major logical transition validity')
    return pool._replace(environment=jax.tree.map(jax.lax.stop_gradient,environment),
        exploration_rng=exploration_rng,
        imagined_transition_words=add_count(pool.imagined_transition_words,valid.astype(jnp.uint32).sum()),
        windows=pool.windows+1)


def refresh_model(pool,new_seeds,new_exploration_rng,*,new_revision):
    """A model refresh rebuilds every state; old drift never crosses revisions."""
    if new_revision<=int(np.asarray(pool.model_revision)):
        raise ValueError('A model refresh must have a newer checkpoint revision')
    fresh=init_pool(new_seeds,new_exploration_rng,model_revision=new_revision)
    return fresh._replace(imagined_transition_words=pool.imagined_transition_words,
                          windows=pool.windows)
