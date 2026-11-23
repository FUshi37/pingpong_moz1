"""Direct cost on recorded physical outgoing ball XY speed, default off."""
import math
import jax.numpy as jnp


def physical_ball_xy_penalty(*, event, vxy_m_s, weight: float):
    """Return (N,) loss on every confirmed contact, normalized by .025m/s."""
    if not math.isfinite(float(weight)) or weight < 0.:
        raise ValueError('Ball XY weight must be finite and nonnegative')
    event=jnp.asarray(event,dtype=bool)
    speed=jnp.asarray(vxy_m_s)
    if event.ndim!=1 or event.shape!=speed.shape:
        raise ValueError('Event and ball speed must have identical (N,) shapes')
    return jnp.where(event,-float(weight)*(speed/.025)**2,0.).astype(jnp.float32)
