"""Imagined targets distinguish reset, time limits and missing model data."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))

from world_model.actor_critic import (
    ImaginedBatch, lambda_returns, actor_return, average_gradients,
)


def batch(reward, discount, trace, valid=None):
    reward=jnp.asarray(reward,dtype=jnp.float32)[:,None]
    return ImaginedBatch(reward,jnp.asarray(discount)[:,None],jnp.asarray(trace)[:,None],
                        jnp.ones_like(reward,dtype=bool) if valid is None else jnp.asarray(valid)[:,None],
                        'imagination')


def test_lambda_targets_bootstrap_time_limit_but_do_not_cross_reset():
    b=batch([1.,100.],[.9,.9],[False,False])
    value=jnp.array([[2.],[50.]])
    target=lambda_returns(b,value,lambda_value=.99)
    np.testing.assert_allclose(target[:,0],[2.8,145.])
    assert not jax.grad(lambda v:lambda_returns(b,v,lambda_value=.99).sum())(value).any()
    terminal=batch([1.,100.],[0.,.9],[False,False])
    assert float(lambda_returns(terminal,value)[0,0])==1.


def test_actor_keeps_terminal_state_derivative_and_ignores_padding():
    b=batch([1.,0.],[.9,0.],[True,False],[True,False])
    def objective(v):
        return actor_return(b,jnp.array([[v],[999.]]))[0]
    np.testing.assert_allclose(jax.grad(objective)(2.),.9)
    np.testing.assert_allclose(objective(2.),2.8)
    with pytest.raises(ValueError):
        lambda_returns(b._replace(origin='teacher'),jnp.zeros((2,1)))


def test_microbatch_gradient_weighting_matches_logical_mean():
    x=jnp.arange(10,dtype=jnp.float32)
    loss=lambda p,a:jnp.mean((p*a-1.)**2)
    expected=jax.grad(loss)(.3,x)
    g=[jax.grad(loss)(.3,x[:3]),jax.grad(loss)(.3,x[3:])]
    actual=average_gradients(g,[3,7])
    np.testing.assert_allclose(actual,expected,atol=1e-6)
