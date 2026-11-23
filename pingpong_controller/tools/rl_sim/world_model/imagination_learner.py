"""Transactional learning primitives; this implementation phase is read-only."""

from __future__ import annotations

from typing import Any, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

from .actor_critic import (average_gradients,critic_loss,frozen_value,lambda_returns,
                           apply_accumulated_update,target_ema)
from .imagined_rollout import actor_window_objective,window_batch
from .neural import init_adam
from .persistent_pool import commit_window


class LearnerState(NamedTuple):
    actor: Any
    critic: Any
    target_critic: Any
    actor_optimizer: Any
    critic_optimizer: Any
    pool: Any
    sampling_rng: Any
    goal_rng: Any
    actor_updates: Any
    critic_updates: Any


def init_learner(actor,critic,pool,*,seed):
    key1,key2=jax.random.split(jax.random.PRNGKey(seed))
    return LearnerState(actor,critic,critic,init_adam(actor),init_adam(critic),pool,
        key1,key2,jnp.asarray(0,jnp.int32),jnp.asarray(0,jnp.int32))


def slice_environment(environment,start,end):
    return jax.tree.map(lambda x:x[start:end],environment)


def make_actor_gradient(robot,auxiliary,**options):
    """One compiled microbatch, fixed models and target weights throughout."""
    def objective(actor,target,environment,keys,goals):
        return actor_window_objective(actor,target,robot,auxiliary,environment,keys,goals,**options)
    return jax.jit(jax.value_and_grad(objective,argnums=0,has_aux=True))


def actor_gradient_window(session,goal_proposals,gradient_fn,*,microbatch,host_windows=True):
    """Accumulate gradients without applying Adam or replacing policy weights."""
    n=session.pool.environment.episode_steps.shape[0]
    if microbatch<1 or n%microbatch or goal_proposals.shape[1]!=n:
        raise ValueError('Microbatches must exactly partition logical environments')
    gradients=[];windows=[];losses=[];counts=[]
    for start in range(0,n,microbatch):
        end=start+microbatch
        (loss,window),gradient=gradient_fn(session.actor,session.target_critic,
            slice_environment(session.pool.environment,start,end),
            session.pool.exploration_rng[start:end],goal_proposals[:,start:end])
        jax.block_until_ready((loss,gradient,window))
        if any(not np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradient)):
            raise FloatingPointError('Nonfinite actor gradient; no optimizer update applied')
        gradients.append(gradient);counts.append(microbatch);losses.append(float(loss))
        windows.append(jax.device_get(window) if host_windows else window)
    gradient=average_gradients(gradients,counts)
    return gradient,windows,dict(loss=float(np.average(losses,weights=counts)),
        logical_envs=n,microbatch=microbatch,microbatches=len(windows),actor_updates=0,critic_updates=0)


def commit_policy_windows(session,windows):
    """Commit only physical/observation/task/RNG progression at a window edge."""
    environment=jax.tree.map(lambda *x:jnp.concatenate(x,axis=0),*(w.environment for w in windows))
    keys=jnp.concatenate([w.exploration_rng for w in windows],axis=0)
    valid=jnp.concatenate([w.transitions.valid for w in windows],axis=1)
    return session._replace(pool=commit_window(session.pool,environment,keys,valid))


def critic_training_arrays(windows,target,*,lambda_value=.99):
    """Model-generated states and detached targets only; exclude all padding."""
    inputs=[];targets=[]
    calculate=jax.jit(lambda w:frozen_value(target,w))
    for window in windows:
        batch=window_batch(window)
        values=calculate(jnp.asarray(window.transitions.next_critic_features))
        returns=np.asarray(lambda_returns(batch,values,lambda_value=lambda_value))
        valid=np.asarray(batch.valid,dtype=bool)
        inputs.append(np.asarray(window.critic_inputs)[valid]);targets.append(returns[valid])
    if not sum(len(x) for x in inputs):
        raise ValueError('No valid imagined transitions for critic diagnostics')
    return np.concatenate(inputs),np.concatenate(targets)


def critic_gradient_batch(parameters,inputs,targets,*,microbatch=2048):
    """Frozen-input critic regression with correct uneven-microbatch weighting."""
    if len(inputs)!=len(targets) or not len(inputs) or microbatch<1:
        raise ValueError('Invalid critic diagnostic batch')
    gradient_fn=jax.jit(jax.value_and_grad(lambda p,x,y:critic_loss(p,x,y,jnp.ones(y.shape,bool))))
    gradients=[];losses=[];counts=[]
    for start in range(0,len(inputs),microbatch):
        x=jnp.asarray(inputs[start:start+microbatch]);y=jnp.asarray(targets[start:start+microbatch])
        loss,gradient=gradient_fn(parameters,x,y)
        jax.block_until_ready((loss,gradient))
        if any(not np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradient)):
            raise FloatingPointError('Nonfinite critic gradient; no optimizer update applied')
        gradients.append(gradient);losses.append(float(loss));counts.append(len(x))
    return average_gradients(gradients,counts),float(np.average(losses,weights=counts))


def apply_training_transaction(session,actor_gradient,windows,recipe,*,authorized=False):
    """One actor update, four critic updates, then one target EMA.

    No caller in the no-update validation CLI authorizes this operation. A
    returned state is the transaction commit; the input remains immutable if
    interrupted. Checkpoints are saved only before/after complete transactions.
    """
    if not authorized:
        raise PermissionError('Actor/critic optimizer updates have not been authorized')
    if recipe['critic_updates_per_actor']!=4:
        raise ValueError('Expected the registered four critic updates')
    inputs,targets=critic_training_arrays(windows,session.target_critic,lambda_value=recipe['lambda'])
    actor,actor_optimizer,_=apply_accumulated_update(session.actor,session.actor_optimizer,
        [actor_gradient],[1],learning_rate=recipe['actor_lr'],authorized=True)
    critic=session.critic;optimizer=session.critic_optimizer;key=session.sampling_rng
    for _ in range(4):
        key,draw=jax.random.split(key)
        indices=np.asarray(jax.random.randint(draw,(recipe['critic_batch'],),0,len(inputs)))
        gradient,_=critic_gradient_batch(critic,inputs[indices],targets[indices],microbatch=recipe['critic_microbatch'])
        critic,optimizer,_=apply_accumulated_update(critic,optimizer,[gradient],[len(indices)],
            learning_rate=recipe['critic_lr'],authorized=True)
    committed=commit_policy_windows(session,windows)
    return committed._replace(actor=actor,actor_optimizer=actor_optimizer,critic=critic,
        critic_optimizer=optimizer,target_critic=target_ema(session.target_critic,critic,recipe['target_critic_ema']),
        sampling_rng=key,actor_updates=session.actor_updates+1,critic_updates=session.critic_updates+4)
