"""Opt-in Critic-only transactions, exact parent learning state and seed resets."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .actor_critic import critic_loss, target_ema
from .artifacts import read_array_archive
from .contracts import authenticate, repository_root
from .neural import adam_update
from .persistent_pool import prepare_window
from .pilot_learning import validate_learning_rows
from .state_adapter import numeric_tree_digest


def restore_learning(reference: dict, template: Any) -> Any:
    """Restore every non-pool leaf from u30; never import old physical states."""
    path = authenticate(repository_root(), reference)
    payload = json.loads(path.read_text())
    if payload['schema'] != 'wmj_v1_imagination_checkpoint':
        raise ValueError('Expected an authenticated complete learner')
    selected = {f'leaf_{i:06d}': row for i, row in enumerate(payload['state']['leaves'])
                if not row['path'].startswith("['learner'].pool")}
    arrays = read_array_archive(path.parent/payload['archive_file'], payload['archive'],
                                names=selected)
    leaves = {}
    for name, row in selected.items():
        a = arrays[name]
        if (list(a.shape) != row['shape'] or a.dtype.str != row['dtype']
                or hashlib.sha256(a.tobytes()).hexdigest() != row['sha256']):
            raise ValueError('Parent learning leaf failed authentication')
        leaves[row['path']] = a
    tree, structure = jax.tree_util.tree_flatten_with_path(dict(learner=template))
    result, used = [], set()
    for key, value in tree:
        name = jax.tree_util.keystr(key)
        if not name.startswith("['learner'].pool"):
            a = leaves[name]
            if a.shape != value.shape or a.dtype != np.asarray(value).dtype:
                raise ValueError('Parent learning leaf layout differs: '+name)
            value = jnp.asarray(a); used.add(name)
        result.append(value)
    if used != set(leaves):
        raise ValueError('Missing or extra parent learning state')
    restored = jax.tree_util.tree_unflatten(structure, result)['learner']
    if (int(restored.actor_updates) != 30 or int(restored.actor_optimizer.step) != 30
            or int(restored.critic_updates) != 120 or int(restored.critic_optimizer.step) != 120):
        raise ValueError('Expected the exact u30/120-step learning state')
    return restored


def make_update(recipe: dict) -> Callable:
    """Compile four unchanged Adam steps followed by one EMA; no Actor path."""
    batch, micro = recipe['critic_batch'], recipe['critic_microbatch']
    if batch % micro or recipe['critic_updates_per_actor'] != 4:
        raise ValueError('Critic batch must retain its registered microbatch partition')
    gradient = jax.value_and_grad(critic_loss)

    @jax.jit
    def update(parameters, optimizer, target, key, inputs, targets, count):
        def one(carry, _):
            params, opt, rng = carry
            rng, draw = jax.random.split(rng)
            ids = jax.random.randint(draw, (batch//micro, micro), 0, count,
                                     dtype=jnp.int32)
            zeros = jax.tree.map(jnp.zeros_like, params)

            def accumulate(total, indices):
                value, grad = gradient(params, inputs[indices], targets[indices],
                                       jnp.ones((micro,), bool))
                return jax.tree.map(lambda a, b: a+b, total, grad), value

            sums, losses = jax.lax.scan(accumulate, zeros, ids)
            grad = jax.tree.map(lambda v: v/(batch//micro), sums)
            params, opt, info = adam_update(params, grad, opt,
                learning_rate=recipe['critic_lr'], max_gradient_norm=1.)
            return (params, opt, rng), dict(loss=losses.mean(), **info)

        (parameters, optimizer, key), info = jax.lax.scan(one,
            (parameters, optimizer, key), None, length=4)
        return parameters, optimizer, target_ema(target, parameters,
            recipe['target_critic_ema']), key, info
    return update


def update_session(session: Any, inputs: np.ndarray, targets: np.ndarray,
                   update_fn: Callable, *, authorized: bool = False,
                   origin: str = 'imagination', initial_updates: int = 0,
                   maximum_updates: int = 512, padded_rows: int | None = None) -> tuple:
    """Commit exactly four Critic steps, retaining Actor/Adam/counters unchanged."""
    if not authorized:
        raise PermissionError('Critic-only fitting requires completed admission')
    validate_learning_rows(inputs, targets, origin=origin)
    if int(session.critic_updates)-initial_updates+4 > maximum_updates:
        raise ValueError('Critic-only update budget exhausted')
    count = len(inputs); padded_rows = count if padded_rows is None else padded_rows
    if count > padded_rows:
        raise ValueError('Valid rows exceed registered rollout capacity')
    x = np.zeros((padded_rows, inputs.shape[1]), np.float32)
    y = np.zeros(padded_rows, np.float32)
    x[:count] = inputs; y[:count] = targets
    params, optimizer, target, key, info = update_fn(session.critic,
        session.critic_optimizer, session.target_critic, session.sampling_rng,
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(count, jnp.int32))
    jax.block_until_ready((params, optimizer, target, info))
    if any(not np.isfinite(np.asarray(v)).all() for v in
           jax.tree.leaves((params, optimizer, target, info))):
        raise FloatingPointError('Nonfinite Critic transaction; not committed')
    return session._replace(critic=params, critic_optimizer=optimizer,
        target_critic=target, sampling_rng=key, critic_updates=session.critic_updates+4), \
        {k: np.asarray(v).tolist() for k, v in info.items()}


def prepare_fit_state(state: dict, *, persistent_envs: int = 96) -> tuple:
    """Reset only ended persistent lanes and short branches from training roots."""
    learner = state['learner']; bank = state['reset_bank']
    n = learner.pool.environment.episode_steps.shape[0]
    next_key, draw, noise = jax.random.split(learner.sampling_rng, 3)
    ids = jax.random.randint(draw, (n,), 0, bank.episode_steps.shape[0], dtype=jnp.int32)
    fresh = jax.tree.map(lambda x: x[ids], bank)
    pool, reset = prepare_window(learner.pool, fresh, jax.random.split(noise, n),
                                 persistent_envs=persistent_envs)
    assignment = jnp.where(reset, ids, state['current_source_indices'])
    return dict(state, learner=learner._replace(pool=pool, sampling_rng=next_key),
                current_source_indices=assignment), dict(
                    reset=np.asarray(reset).tolist(), source_indices=np.asarray(assignment).tolist())


def unchanged_actor_digest(session: Any) -> str:
    return numeric_tree_digest((session.actor, session.actor_optimizer,
                                session.actor_updates, session.goal_rng))
