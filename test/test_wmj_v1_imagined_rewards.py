"""Task shaping has directional gradients and no reward from padding/setup."""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))

from world_model.imagined_rewards import RewardContract, release_reward, control_reward


def test_release_reward_is_unique_masked_and_directional_far_from_goal():
    c = RewardContract()
    position = jnp.array([[0., -.5, 1.1]])
    velocity = jnp.array([[0., 0., 2.]])
    goal = jnp.array([[.3, -.5, 1.3]])
    def value(v):
        return release_reward(position, v, goal, jnp.array([True]), contract=c)['total'][0]
    gradient = jax.grad(value)(velocity)
    assert float(gradient[0,0]) > .1
    assert np.isfinite(gradient).all()
    off = release_reward(position, velocity, goal, jnp.array([False]), contract=c)
    assert all(float(x[0]) == 0. for x in off.values())


def test_horizontal_penalty_activates_only_near_the_goal():
    c = RewardContract()
    position = jnp.array([[0., -.5, 1.1]])
    velocity = jnp.array([[.1, 0., 2.]])
    apex = position + velocity*(2./9.81)
    apex = apex.at[:,2].add(-.5*9.81*(2./9.81)**2)
    near = release_reward(position,velocity,apex,jnp.array([True]),contract=c)
    far = release_reward(position,velocity,apex+jnp.array([[.4,0.,0.]]),jnp.array([True]),contract=c)
    assert float(near['steady_horizontal'][0]) < -.1
    assert abs(float(far['steady_horizontal'][0])) < 1e-12


def test_dense_rates_fail_once_and_padding_is_zero():
    zero=jnp.zeros((1,7)); alive=jnp.array([True]); failure=jnp.array([False])
    def sample(dt, valid=alive, failed=failure):
        return control_reward(zero,zero,zero,active=valid,physical_failure=failed,dt_s=dt,contract=RewardContract())
    np.testing.assert_allclose(sample(.005)['total']*2,sample(.01)['total'])
    assert float(sample(.005,failed=alive)['failure'][0]) == -10.
    assert all(float(x[0])==0 for x in sample(.005,valid=failure).values())
