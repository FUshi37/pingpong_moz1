"""Explicit bounded optimizer transactions; all regression rows are imagined."""

import jax
import numpy as np

from .actor_critic import apply_accumulated_update, target_ema
from .imagination_learner import critic_gradient_batch


def validate_learning_rows(inputs, targets, *, origin='imagination'):
    if origin != 'imagination':
        raise ValueError('Native/teacher transitions cannot update the pilot')
    if len(inputs) != len(targets) or not len(inputs) or np.ndim(inputs) != 2:
        raise ValueError('Invalid imagined critic batch')
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise FloatingPointError('Nonfinite imagined regression rows')


def optimize_pilot(session, actor_gradient, inputs, targets, recipe, *, authorized=False):
    """Return one complete1Actor/4Critic/1EMA commit; leave the caller unchanged."""
    if not authorized:
        raise PermissionError('The bounded pilot has not passed its admission checks')
    validate_learning_rows(inputs, targets)
    if recipe['critic_updates_per_actor'] != 4:
        raise ValueError('Expected exactly four critic optimizer steps')
    if any(not np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(actor_gradient)):
        raise FloatingPointError('Nonfinite actor gradient')
    actor, actor_optimizer, actor_metrics = apply_accumulated_update(session.actor,
        session.actor_optimizer, [actor_gradient], [1], learning_rate=recipe['actor_lr'], authorized=True)
    critic, optimizer, key = session.critic, session.critic_optimizer, session.sampling_rng
    reports = []
    for _ in range(4):
        key, draw = jax.random.split(key)
        ids = np.asarray(jax.random.randint(draw, (recipe['critic_batch'],), 0, len(inputs)))
        gradient, loss = critic_gradient_batch(critic, inputs[ids], targets[ids],
            microbatch=recipe['critic_microbatch'])
        critic, optimizer, metrics = apply_accumulated_update(critic, optimizer, [gradient],
            [len(ids)], learning_rate=recipe['critic_lr'], authorized=True)
        jax.block_until_ready((critic, optimizer))
        reports.append(dict(loss=loss, gradient_norm=float(metrics['gradient_norm']),
            distinct_source_rows=len(np.unique(ids)), samples=len(ids)))
    result = session._replace(actor=actor, actor_optimizer=actor_optimizer, critic=critic,
        critic_optimizer=optimizer, target_critic=target_ema(session.target_critic, critic,
            recipe['target_critic_ema']), sampling_rng=key,
        actor_updates=session.actor_updates + 1, critic_updates=session.critic_updates + 4)
    if any(not np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(
            (result.actor, result.critic, result.target_critic, result.actor_optimizer, result.critic_optimizer))):
        raise FloatingPointError('Nonfinite optimizer output; transaction not committed')
    return result, dict(actor_gradient_norm=float(actor_metrics['gradient_norm']),
        critic=reports, imagined_rows=len(inputs), teacher_rows=0)
