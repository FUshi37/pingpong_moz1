"""Learned robot residuals recurse on their own states and retain action effects."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.neural import init_mlp, init_adam, adam_update
from world_model.robot_model import RobotScale, robot_step, robot_rollout
from world_model.model_losses import robot_execution_loss


def test_zero_robot_residual_is_known_inertial_baseline_and_rollout_recurses():
    parameters = init_mlp(jax.random.PRNGKey(4), (35, 128, 128, 14), zero_output=True)
    scale = RobotScale(jnp.zeros(35), jnp.ones(35), jnp.zeros(14), jnp.ones(14))
    q, dq = jnp.zeros((2, 7)), jnp.full((2, 7), .3)
    command = dict(q=q+.2, dq=dq, ddq=jnp.zeros_like(q))
    after_q, after_dq = robot_step(parameters, scale, q, dq, command)
    np.testing.assert_allclose(after_q, .0003, atol=1e-8)
    np.testing.assert_allclose(after_dq, dq)
    commands = jax.tree.map(lambda value: jnp.broadcast_to(value, (40,)+value.shape), command)
    (last_q, last_dq), trace = robot_rollout(parameters, scale, q, dq, commands)
    np.testing.assert_allclose(last_q, .012, atol=1e-7)
    np.testing.assert_allclose(last_dq, dq)
    assert trace[0].shape == (40, 2, 7)


def test_robot_fit_learns_command_effect_with_finite_differentiable_rollout():
    key = jax.random.PRNGKey(7)
    parameters = init_mlp(key, (35, 128, 128, 14), zero_output=True)
    optimizer = init_adam(parameters)
    q = jnp.zeros((64, 7))
    command_q = jax.random.uniform(key, (64, 7), minval=-.2, maxval=.2)
    commands = dict(q=command_q, dq=q, ddq=q)
    target_dq = command_q*.05
    scale = RobotScale(jnp.zeros(35), jnp.ones(35), jnp.zeros(14), jnp.full(14, .01))

    def loss(p):
        next_q, next_dq = robot_step(p, scale, q, q, commands)
        return jnp.mean(((next_dq-target_dq)/.01)**2)+jnp.mean((next_q/.01)**2)

    @jax.jit
    def update(p, opt):
        value, grad = jax.value_and_grad(loss)(p)
        p, opt, _ = adam_update(p, grad, opt, learning_rate=3e-4, max_gradient_norm=1.)
        return p, opt, value

    before = float(loss(parameters))
    for _ in range(320):
        parameters, optimizer, _ = update(parameters, optimizer)
    assert float(loss(parameters)) < before*.04
    derivative = jax.jacfwd(lambda command: robot_step(parameters, scale,
        q[:1], q[:1], dict(q=command, dq=q[:1], ddq=q[:1]))[1])(command_q[:1])
    assert np.isfinite(derivative).all()
    assert float(jnp.diag(derivative[0, :, 0, :]).mean()) > .03


def test_autoregressive_loss_never_corrects_with_later_teacher_states():
    parameters = init_mlp(jax.random.PRNGKey(12), (35, 128, 128, 14), zero_output=True)
    scale = RobotScale(jnp.zeros(35), jnp.ones(35), jnp.zeros(14), jnp.ones(14))
    q = jnp.zeros((10, 2, 7))
    batch = dict(q_before=q, dq_before=q, q_after=q, dq_after=q,
        commands=dict(q=q, dq=q, ddq=q), valid=jnp.ones((10, 2), bool))
    first, _ = robot_execution_loss(parameters, scale, batch, autoregressive=True)
    corrupted = {**batch, 'q_before': q.at[1:].set(2.)}
    second, _ = robot_execution_loss(parameters, scale, corrupted, autoregressive=True)
    assert float(first) == float(second) == 0.
    auxiliary, _ = robot_execution_loss(parameters, scale, corrupted, autoregressive=False)
    assert float(auxiliary) > 100.
