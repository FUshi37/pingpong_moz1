"""Pure imagined actor/critic objectives with explicit termination semantics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp

from .neural import adam_update, init_adam, init_mlp, mlp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ImaginedBatch:
    """Time-major rewards and discounts; no native returns are accepted."""

    reward: Any
    discount: Any
    trace_continue: Any
    valid: Any
    origin: str = 'imagination'

    def tree_flatten(self):
        return (self.reward,self.discount,self.trace_continue,self.valid),self.origin

    @classmethod
    def tree_unflatten(cls,origin,children):
        return cls(*children,origin=origin)

    def _replace(self,**values):
        return replace(self,**values)


def validate_batch(batch):
    if batch.origin!='imagination':
        raise ValueError('Actor and critic accept only model-generated transitions/returns')
    if (batch.reward.ndim!=2 or batch.reward.shape[0]<1
            or any(x.shape!=batch.reward.shape for x in
                   (batch.discount,batch.trace_continue,batch.valid))):
        raise ValueError('Expected identical (T,N) imagined transition masks')


def lambda_returns(batch, next_values, *, lambda_value=.99):
    """Bootstrap a time limit from its final state, never a post-reset state.

    discount=0 at physical failure and model truncation; time limits retain
    gamma but cut the trace. Missing padding cuts the trace as well. Targets
    are detached in their entirety, including next-value and reward paths.
    """
    validate_batch(batch)
    if next_values.shape!=batch.reward.shape or not 0.<=lambda_value<=1.:
        raise ValueError('Invalid bootstrap layout or lambda')
    next_valid=jnp.concatenate((batch.valid[1:],jnp.zeros_like(batch.valid[:1])))
    trace=batch.trace_continue & next_valid
    def tick(carry,values):
        reward,discount,value,valid,continued=values
        future=jnp.where(continued,(1.-lambda_value)*value+lambda_value*carry,value)
        target=jnp.where(valid,reward+discount*future,0.)
        return target,target
    _,targets=jax.lax.scan(tick,jnp.zeros_like(next_values[0]),
        (batch.reward,batch.discount,next_values,batch.valid,trace),reverse=True)
    return jax.lax.stop_gradient(targets)


def actor_return(batch,next_values):
    """Discounted pathwise return; preserve derivatives through bootstrap state."""
    validate_batch(batch)
    if next_values.shape!=batch.reward.shape:
        raise ValueError('Expected next-state values (T,N)')
    next_valid=jnp.concatenate((batch.valid[1:],jnp.zeros_like(batch.valid[:1])))
    boundary=~batch.trace_continue | ~next_valid
    def tick(weight,values):
        reward,discount,value,valid,end=values
        term=weight*jnp.where(valid,reward+jnp.where(end,discount*value,0.),0.)
        return weight*jnp.where(valid & ~end,discount,0.),term
    _,terms=jax.lax.scan(tick,jnp.ones_like(next_values[0]),
        (batch.reward,batch.discount,next_values,batch.valid,boundary))
    return terms.sum(axis=0)


def frozen_value(parameters,inputs):
    """Freeze critic weights while retaining its input-state derivative."""
    return mlp(jax.tree.map(jax.lax.stop_gradient,parameters),inputs)[...,0]


def init_critic(key,input_dim):
    return init_mlp(key,(input_dim,256,256,1),zero_output=True)


def critic_loss(parameters,inputs,targets,valid):
    """Value regression is detached from the imagined state and target graph."""
    inputs,targets,valid=jax.tree.map(jax.lax.stop_gradient,(inputs,targets,valid))
    prediction=mlp(parameters,inputs)[...,0]
    error=jnp.where(valid,prediction-targets,0.)
    return .5*jnp.sum(error**2)/jnp.maximum(valid.sum(),1)


def average_gradients(gradients,counts):
    """Weight microbatch sums correctly before a single global norm clip."""
    if not gradients or len(gradients)!=len(counts) or any(n<=0 for n in counts):
        raise ValueError('Positive logical sample counts required')
    total=sum(counts)
    return jax.tree.map(lambda *leaves:sum(x*n for x,n in zip(leaves,counts))/total,*gradients)


def target_ema(target,online,rate=.01):
    if not 0.<rate<=1.:
        raise ValueError('Invalid target EMA rate')
    return jax.tree.map(lambda t,v:jax.lax.stop_gradient((1.-rate)*t+rate*v),target,online)


def apply_accumulated_update(parameters,optimizer,gradients,counts,*,learning_rate,authorized=False):
    """Actual optimization remains gated off during no-update verification."""
    if not authorized:
        raise PermissionError('This phase authorizes gradient diagnostics, not optimizer updates')
    return adam_update(parameters,average_gradients(gradients,counts),optimizer,
                       learning_rate=learning_rate,max_gradient_norm=1.)
