"""Normalized robot execution losses; composed event/outcome losses follow separately."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .robot_model import robot_step,robot_rollout
from .kinematics import forward_kinematics


def racket_outputs(chain,q,dq,normal_axis=0):
    """Return pose and normal speed from the same predicted joint state."""
    shape=q.shape[:-1]
    fk=jax.vmap(lambda value:forward_kinematics(chain,value))(q.reshape(-1,7))
    velocity=jnp.einsum('nij,nj->ni',fk.geom_jacobian,dq.reshape(-1,7))
    normal=fk.geom_rotation[:,:,normal_axis]
    normal=normal*jnp.where(normal[:,2]>=0.,1.,-1.)[:,None]
    return fk.site_position.reshape(shape+(3,)),jnp.sum(normal*velocity,axis=-1).reshape(shape)


def robot_execution_loss(parameters,scale,batch,*,autoregressive,chain=None,normal_axis=0):
    """Recurse on predicted q/dq after the real initial state.

    A recorded delayed-command sequence isolates the execution module. The
    complete world later closes its own measured-feedback RMP/control chain.
    Bootstrap single-step fitting is explicit and never called a world rollout.
    """
    if autoregressive:
        _,(q,dq)=robot_rollout(parameters,scale,batch['q_before'][0],batch['dq_before'][0],batch['commands'])
    else:
        q,dq=robot_step(parameters,scale,batch['q_before'],batch['dq_before'],batch['commands'])
    mask=batch['valid'].astype(jnp.float32)
    denominator=jnp.maximum(mask.sum(),1.)
    q_error=jnp.mean(jnp.square((q-batch['q_after'])/.01),axis=-1)
    dq_error=jnp.mean(jnp.square((dq-batch['dq_after'])/.10),axis=-1)
    metrics=dict(q_loss=jnp.sum(q_error*mask)/denominator,
                 dq_loss=jnp.sum(dq_error*mask)/denominator)
    if chain is not None:
        # Endpoint FK is sufficient for this module loss; the model still
        # predicts every 1 ms q/dq rather than skipping physical substeps.
        predicted_p,predicted_v=racket_outputs(chain,q[4::5],dq[4::5],normal_axis)
        actual_p,actual_v=racket_outputs(chain,batch['q_after'][4::5],batch['dq_after'][4::5],normal_axis)
        end_mask=mask[4::5]
        end_denominator=jnp.maximum(end_mask.sum(),1.)
        metrics['racket_position_loss']=jnp.sum(jnp.mean(jnp.square((predicted_p-actual_p)/.003),axis=-1)*end_mask)/end_denominator
        metrics['normal_velocity_loss']=jnp.sum(jnp.square((predicted_v-actual_v)/.05)*end_mask)/end_denominator
    loss=sum(metrics.values())
    metrics['loss']=loss
    return loss,metrics
