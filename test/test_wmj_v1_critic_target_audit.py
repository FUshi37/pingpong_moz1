"""Independent finite-return mixtures cover termination, timeout and padding."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                      'pingpong_controller/tools/rl_sim'))
from world_model.actor_critic import ImaginedBatch, lambda_returns
from world_model.critic_targets import lambda_mixture_oracle


def test_n_step_mixture_matches_hand_calculated_window_edge():
    batch = ImaginedBatch(jnp.array([[1.], [2.]]), jnp.full((2, 1), .9),
                          jnp.ones((2, 1), bool), jnp.ones((2, 1), bool))
    values = jnp.array([[3.], [5.]])
    oracle = lambda_mixture_oracle(batch, values, .5)
    np.testing.assert_allclose(oracle[:, 0], [5.275, 6.5], atol=1e-6)
    np.testing.assert_allclose(lambda_returns(batch, values, lambda_value=.5), oracle, atol=1e-6)


def test_terminal_timeout_model_cut_and_padding_have_independent_values():
    rewards = jnp.array([[1., 1., 0., 1.], [-10., 999., 999., 999.], [999.]*4])
    discounts = jnp.array([[.9, .9, 0., .9], [0., 0., 0., 0.], [0.]*4])
    valid = jnp.array([[True]*4, [True, False, False, False], [False]*4])
    trace = jnp.array([[True, False, False, True], [False]*4, [False]*4])
    batch = ImaginedBatch(rewards, discounts, trace, valid)
    values = jnp.full((3, 4), 2.)
    oracle = lambda_mixture_oracle(batch, values, 1.)
    np.testing.assert_allclose(oracle[0], [-8., 2.8, 0., 2.8], atol=1e-6)
    assert not oracle[~np.asarray(valid)].any()
    np.testing.assert_allclose(lambda_returns(batch, values, lambda_value=1.), oracle, atol=1e-6)
    assert not jax.grad(lambda v: lambda_returns(batch, v).sum())(values).any()


def test_random_mixed_boundaries_match_independent_enumeration():
    rng = np.random.default_rng(741)
    rewards = rng.normal(size=(17, 5)).astype(np.float32)
    discounts = np.full_like(rewards, .9995)
    valid = np.ones_like(rewards, bool)
    trace = np.ones_like(valid)
    for lane, length in enumerate((1, 5, 9, 14, 17)):
        valid[length:, lane] = False
        trace[length-1:, lane] = False
        discounts[length:, lane] = 0.
        if lane % 2 == 0:
            discounts[length-1, lane] = 0.
    batch = ImaginedBatch(*map(jnp.asarray, (rewards, discounts, trace, valid)))
    values = jnp.asarray(rng.normal(size=rewards.shape), jnp.float32)
    for lam in (0., .99, 1.):
        np.testing.assert_allclose(lambda_returns(batch, values, lambda_value=lam),
            lambda_mixture_oracle(batch, values, lam), atol=5e-5, rtol=1e-5)
