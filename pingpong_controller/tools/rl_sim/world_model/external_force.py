"""Known ball-force integration on the frozen residual-free physical model."""

import jax
import jax.numpy as jnp

from .full_prior_world import full_prior_physics_step
from .imagined_env import select_lanes
from .model_numerics import precision_scope
from .tangent_contact import tangent_contact_step


def forced_physics_step(robot,auxiliary,state,command,force_world_n,**options):
    """One1 ms step; no parameter or neural feature changes.

    In the contact equation, ball external acceleration subtracts from the
    racket-minus-ball free-acceleration term. Integrate F/m separately in the
    ball velocity/position. Event force therefore remains contact force only.
    The robot surrogate and all zero-force lanes retain their original result.
    """
    if force_world_n.shape!=state.ball.position_m.shape:
        raise ValueError('Expected external ball force(N,3) in newtons')
    baseline,trace=full_prior_physics_step(robot,auxiliary,state,command,**options)
    enabled=jnp.any(force_world_n!=0.,axis=-1)

    def correct(_):
        c=options['contact_contract'];dt=c.dt_s
        acceleration=force_world_n/c.mass_kg
        terms=trace['solver_terms'].at[:,:3].add(-acceleration)
        pose=trace['racket_before']
        racket=dict(geom_position_m=pose['geom_position'],geom_rotation_world=pose['rotation'],
            racket_linear_velocity_m_s=pose['velocity'],racket_angular_velocity_rad_s=pose['omega'],
            joint_q_rad=state.q_rad,delayed_q_command_rad=command['q'])
        with precision_scope(options.get('matmul_precision','highest')):
            ball,contact=tangent_contact_step(None,options['surface'],state.ball,racket,terms,
                ball_radius_m=options['ball_radius_m'],position_scale=options['position_scale'],
                contract=c,learned=False)
        ball=ball._replace(velocity_m_s=ball.velocity_m_s+acceleration*dt,
            position_m=ball.position_m+acceleration*dt*dt)
        contact=dict(contact,after=ball)
        after=baseline._replace(ball=select_lanes(enabled,ball,baseline.ball))
        changed=dict(trace,contact=select_lanes(enabled,contact,trace['contact']),
            solver_terms=jnp.where(enabled[:,None],terms,trace['solver_terms']))
        return after,changed

    return jax.lax.cond(jnp.any(enabled),correct,lambda _: (baseline,trace),operand=None)
