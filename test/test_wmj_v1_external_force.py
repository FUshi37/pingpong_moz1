"""Known external force changes free acceleration, not neural inputs or hit credit."""

import jax
import jax.numpy as jnp
import numpy as np
from test_wmj_v1_full_prior_world import components
from world_model.full_prior_world import full_prior_physics_step
from world_model.external_force import forced_physics_step
from world_model.action_effects import model_event_trace
from world_model.contact_geometry import surface_features
from world_model.soft_contact import soft_contact_increment


def test_zero_force_preserves_full_prior_and20ms_flight_impulse():
    robot,auxiliary,state,options=components()
    state=state._replace(ball=state.ball._replace(position_m=jnp.array([[0.,-.5,2.]]),
        velocity_m_s=jnp.zeros((1,3)),omega_world_rad_s=jnp.zeros((1,3))))
    command={k:jnp.zeros((1,7)) for k in ('q','dq','ddq')}
    zero=jnp.zeros((1,3))
    baseline=full_prior_physics_step(robot,auxiliary,state,command,**options)
    same=forced_physics_step(robot,auxiliary,state,command,zero,**options)
    for a,b in zip(jax.tree.leaves(baseline),jax.tree.leaves(same)):
        np.testing.assert_array_equal(a,b)
    force=jnp.array([[.006,-.008,.012]])
    run=jax.jit(lambda f:jax.lax.scan(lambda s,_:(
        forced_physics_step(robot,auxiliary,s,command,f,**options)[0],None),state,None,length=20)[0])
    forced=run(force);free=run(zero)
    dv=np.asarray(force/options['contact_contract'].mass_kg)*.020
    np.testing.assert_allclose(forced.ball.velocity_m_s-free.ball.velocity_m_s,dv,atol=2e-6)
    expected=np.asarray(force/options['contact_contract'].mass_kg)*(.001**2*20*21/2)
    np.testing.assert_allclose(forced.ball.position_m-free.ball.position_m,expected,atol=2e-6)
    np.testing.assert_array_equal(forced.q_rad,free.q_rad)


def test_force_contact_prior_and_event_impulse_use_relative_free_acceleration():
    robot,auxiliary,state,options=components()
    command={k:jnp.zeros((1,7)) for k in ('q','dq','ddq')}
    force=jnp.array([[.003,-.005,.01]])
    after,trace=forced_physics_step(robot,auxiliary,state,command,force,**options)
    _,base=full_prior_physics_step(robot,auxiliary,state,command,**options)
    pose=base['racket_before'];terms=base['solver_terms'];c=options['contact_contract']
    geometry=surface_features(options['surface'],state.ball.position_m,pose['geom_position'],pose['rotation'],.02)
    point_term=terms[:,:3]+jnp.cross(terms[:,3:],geometry['contact_position_world_m']-pose['geom_position'])
    values=dict(ball_position_m=state.ball.position_m,ball_velocity_m_s=state.ball.velocity_m_s,
        ball_omega_world_rad_s=state.ball.omega_world_rad_s,geom_position_m=pose['geom_position'],
        racket_linear_velocity_m_s=pose['velocity'],racket_angular_velocity_rad_s=pose['omega'])
    relative_acceleration=jnp.array([0.,0.,-9.81])+force/c.mass_kg-point_term
    expected=jax.vmap(lambda g,v,a:soft_contact_increment(c,g,**v,gravity_m_s2=a))(
        geometry,values,relative_acceleration)
    np.testing.assert_allclose(after.ball.velocity_m_s,
        state.ball.velocity_m_s+(jnp.array([0.,0.,-9.81])+force/c.mass_kg)*.001
        +expected.delta_velocity_world_m_s,atol=4e-7)
    event=model_event_trace(trace,surface=options['surface'],ball_radius_m=.02,contact_contract=c)
    np.testing.assert_allclose(event['contact_normal_force_n'],expected.force_contact_n[:,0],atol=2e-6)
