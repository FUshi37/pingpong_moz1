"""Privileged value features retain control memory without camera RNG leakage."""

import jax
import jax.numpy as jnp
import numpy as np

from test_wmj_v1_control_projection import fixture
from test_wmj_v1_world_control import setup_control
from test_wmj_v1_full_prior_world import components
from world_model.full_prior_world import FullPriorWorldState, measured_racket_pose
from world_model.observation_engine import init_world_observation
from world_model.task import init_task
from world_model.physical_events import init_physical_events
from world_model.critic_features import critic_features, critic_layout
from world_model.actor_critic import init_critic, frozen_value


def setup_value(fixture):
    chain,q,control,context,_=setup_control(fixture)
    _,_,physics,_=components();physics=physics._replace(q_rad=q)
    observation,_=init_world_observation(physics.ball.position_m,q,
        jax.random.split(jax.random.PRNGKey(0),1),
        pose_from_q=lambda x:measured_racket_pose(chain,x),ar1_enabled=False)
    world=FullPriorWorldState(physics,control,observation,measured_racket_pose(chain,q))
    return world,init_task(jnp.array([[0.,-.5,1.4]])),init_physical_events(1),context,jnp.zeros(1,jnp.int32)


def test_full_memory_is_sensitive_to_oldest_command_and_excludes_rng(fixture):
    args=setup_value(fixture);x=critic_features(*args);layout=critic_layout(*args)
    assert x.shape==(1,layout['dimension']) and np.isfinite(x).all()
    assert not any('rng' in r['name'] for r in layout['fields'])
    world=args[0];delay=dict(world.control.output_delay)
    delay['q']=delay['q'].at[0,0,0].add(.1)
    changed=world._replace(control=world.control._replace(output_delay=delay))
    difference=critic_features(changed,*args[1:])-x
    assert np.count_nonzero(difference)==1
    np.testing.assert_allclose(jnp.max(jnp.abs(difference)),.1/np.pi,atol=2e-7)
    camera=world.observation.camera._replace(rng=jax.random.split(jax.random.PRNGKey(9),1))
    changed=world._replace(observation=world.observation._replace(camera=camera))
    np.testing.assert_array_equal(x,critic_features(changed,*args[1:]))


def test_fresh_critic_value_zero_and_frozen_weight_input_gradient(fixture):
    x=critic_features(*setup_value(fixture));p=init_critic(jax.random.PRNGKey(2),x.shape[-1])
    np.testing.assert_array_equal(frozen_value(p,x),0.)
    # A fixed nonzero synthetic last layer tests the derivative path without
    # performing an optimizer update or claiming a trained value function.
    p=(*p[:-1],dict(w=jnp.ones_like(p[-1]['w'])*.01,b=p[-1]['b']))
    dp,dx=jax.grad(lambda p,x:frozen_value(p,x).sum(),argnums=(0,1))(p,x)
    assert all(not np.asarray(v).any() for v in jax.tree.leaves(dp))
    assert np.linalg.norm(dx)>1e-6 and np.isfinite(dx).all()
