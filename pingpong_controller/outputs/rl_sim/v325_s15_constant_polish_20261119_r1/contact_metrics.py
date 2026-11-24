"""Actual force-on-ball normal, distinct from the racket-face normal."""
import jax.numpy as jnp


def contact_normal_errors(frame,geom,dist,normal_force,incoming,outgoing,ball_id,racket_id):
    delta=outgoing-incoming
    length=jnp.linalg.norm(delta,axis=-1)
    target=delta/jnp.maximum(length[:,None],1.e-9)
    ball_first=(geom[...,0]==ball_id)&(geom[...,1]==racket_id)
    ball_second=(geom[...,1]==ball_id)&(geom[...,0]==racket_id)
    valid=(ball_first|ball_second)&(dist<0.)&(normal_force>1.e-6)&(length[:,None]>1.e-6)
    # MuJoCo contact.frame points from geom1 toward geom2.
    normal=jnp.where(ball_first[...,None],-frame[...,0,:],frame[...,0,:])
    error=jnp.rad2deg(jnp.arccos(jnp.clip(jnp.sum(normal*target[:,None,:],axis=-1),-1.,1.)))
    count=jnp.sum(valid,axis=-1).astype(jnp.float32)
    return dict(available_contacts=count,error_deg_sum=jnp.sum(jnp.where(valid,error,0.),axis=-1),
                error_deg_sq_sum=jnp.sum(jnp.where(valid,error**2,0.),axis=-1))
