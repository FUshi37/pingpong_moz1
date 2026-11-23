"""Full-chain execution preserves the arm prefix and predicts support motion."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from world_model.neural import init_mlp
from world_model.robot_model import RobotScale, robot_features, robot_step
from world_model.robot_support_model import (
    expand_arm_parameters, support_features, support_residual_target,
    support_robot_step, support_robot_rollout,
)
from world_model.support_losses import support_execution_loss
from world_model.kinematics import ChainBody, Hinge, KinematicChain


def test_appended_support_features_and_warmstart_preserve_arm_prediction():
    key = jax.random.PRNGKey(24)
    q = jax.random.normal(key, (3, 13))*.2
    dq = q*.1
    command = dict(q=q[:, :7]+.01, dq=dq[:, :7], ddq=dq[:, :7]*.5)
    old = init_mlp(key, (35, 128, 128, 14))
    new = expand_arm_parameters(old)
    old_scale = RobotScale(jnp.zeros(35), jnp.ones(35), jnp.zeros(14), jnp.ones(14))
    new_scale = RobotScale(jnp.zeros(47), jnp.ones(47), jnp.zeros(26), jnp.ones(26))
    features = support_features(q, dq, command)
    np.testing.assert_array_equal(features[:, :35], robot_features(q[:, :7], dq[:, :7], command))
    np.testing.assert_array_equal(features[:, 35:41], q[:, 7:])
    np.testing.assert_array_equal(features[:, 41:47], dq[:, 7:])
    predicted = support_robot_step(new, new_scale, q, dq, command)
    baseline = robot_step(old, old_scale, q[:, :7], dq[:, :7], command)
    for actual, expected in zip(predicted, baseline):
        np.testing.assert_allclose(actual[:, :7], expected, atol=1e-6)
    np.testing.assert_allclose(predicted[0][:, 7:], q[:, 7:]+dq[:, 7:]*.001, atol=1e-8)
    np.testing.assert_array_equal(predicted[1][:, 7:], dq[:, 7:])


def test_support_outputs_are_state_deltas_not_new_policy_actions():
    params = init_mlp(jax.random.PRNGKey(1), (47, 128, 128, 26), zero_output=True)
    bias = jnp.zeros(26).at[14:20].set(.002).at[20:26].set(.1)
    params = params[:-1]+({**params[-1], 'b': bias},)
    scale = RobotScale(jnp.zeros(47), jnp.ones(47), jnp.zeros(26), jnp.ones(26))
    q, dq = jnp.zeros((2, 13)), jnp.zeros((2, 13))
    commands = {key: jnp.zeros((5, 2, 7)) for key in ('q', 'dq', 'ddq')}
    (end_q, end_dq), trace = support_robot_rollout(params, scale, q, dq, commands)
    np.testing.assert_array_equal(end_q[:, :7], 0.)
    np.testing.assert_allclose(end_q[:, 7:], .010+10*.0001, atol=1e-8)
    np.testing.assert_allclose(end_dq[:, 7:], .5, atol=1e-8)
    assert trace[0].shape == (5, 2, 13)
    labels = support_residual_target(q, dq, q+.002, dq+.1)
    np.testing.assert_allclose(labels[:, :7], .002)
    np.testing.assert_allclose(labels[:, 7:14], .1)
    np.testing.assert_allclose(labels[:, 14:20], .002)
    np.testing.assert_allclose(labels[:, 20:26], .1)


def test_support_rollout_never_reads_future_measured_waist_state():
    parameters = init_mlp(jax.random.PRNGKey(12), (47, 128, 128, 26), zero_output=True)
    scale = RobotScale(jnp.zeros(47), jnp.ones(47), jnp.zeros(26), jnp.ones(26))
    q = jnp.zeros((10, 2, 13))
    command = jnp.zeros((10, 2, 7))
    batch = dict(q_before=q, dq_before=q, q_after=q, dq_after=q,
        commands=dict(q=command, dq=command, ddq=command), valid=jnp.ones((10, 2), bool))
    before, _ = support_execution_loss(parameters, scale, batch, autoregressive=True)
    corrupted = {**batch, 'q_before': q.at[1:, :, 7:].set(.05)}
    after, _ = support_execution_loss(parameters, scale, corrupted, autoregressive=True)
    assert float(before) == float(after) == 0.
    auxiliary, _ = support_execution_loss(parameters, scale, corrupted, autoregressive=False)
    assert float(auxiliary) > 100.


def test_native_pose_supervision_trains_predictions_during_single_step_bootstrap():
    zero, eye = jnp.zeros(3), jnp.eye(3)
    joints = tuple(Hinge(i, 3, jnp.array([0., 0., 1.]), zero, 0., 0.) for i in range(13))
    point = jnp.array([.2, 0., 0.])
    chain = KinematicChain((ChainBody(zero, eye, joints),), tuple(range(13)),
                            point, eye, point, eye)
    parameters = init_mlp(jax.random.PRNGKey(15), (47, 128, 128, 26), zero_output=True)
    scale = RobotScale(jnp.zeros(47), jnp.ones(47), jnp.zeros(26), jnp.ones(26))
    q, command = jnp.zeros((5, 1, 13)), jnp.zeros((5, 1, 7))
    batch = dict(q_before=q, dq_before=q, q_after=q, dq_after=q,
        commands=dict(q=command, dq=command, ddq=command), valid=jnp.ones((5, 1), bool),
        racket_site_position_before_m=jnp.broadcast_to(point+jnp.array([0., .003, 0.]), (5, 1, 3)),
        racket_geom_rotation_before=jnp.broadcast_to(eye, (5, 1, 3, 3)),
        racket_linear_velocity_before_m_s=jnp.zeros((5, 1, 3)),
        racket_angular_velocity_before_rad_s=jnp.zeros((5, 1, 3)))
    gradient = jax.grad(lambda p: support_execution_loss(p, scale, batch,
        autoregressive=False, chain=chain)[0])(parameters)
    assert np.isfinite(gradient[-1]['b']).all()
    assert float(jnp.linalg.norm(gradient[-1]['b'])) > 1.
