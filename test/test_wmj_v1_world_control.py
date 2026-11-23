"""The composed control path retains projection, RMP memory and delayed commands."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from test_wmj_v1_control_projection import fixture
from recovered_rmp_jax import JointRMPParams, init_joint_rmp_state, joint_rmp_step
from world_model.control_projection import ProjectionContract, project_reference
from world_model.control_rmp import promote_rmp_indices


def setup_control(fixture):
    from world_model.world_control import ControlContract, ControlState, ControlContext
    env, chain, fixed, addresses, _, _ = fixture
    q = jnp.asarray(fixed[addresses][None], jnp.float32)
    rmp = promote_rmp_indices(init_joint_rmp_state(q[:, :7]))
    buffers = dict(q=jnp.broadcast_to(q[:, None, :7], (1, 19, 7)),
                   dq=jnp.zeros((1, 19, 7)), ddq=jnp.zeros((1, 19, 7)))
    state = ControlState(q[:, :7], jnp.zeros((1, 7)), rmp, buffers)
    context = ControlContext({}, jnp.ones(1), jnp.array([[18, 19, 19, 18, 18, 18, 18]]))
    contract = ControlContract(ProjectionContract.from_native(env), JointRMPParams(),
                               jnp.ones(7)*30., jnp.ones(7)*700., 1.)
    return chain, q, state, context, contract


def test_control_composition_matches_native_rmp_and_delay(fixture):
    from world_model.world_control import control_step
    chain, q, state, context, c = setup_control(fixture)
    raw = jnp.array([[2., -.1, .02, .1, -.2, .1, -2.]])
    actual, trace = jax.jit(lambda s,a:control_step(s, a, q, context, chain=chain, contract=c))(state, raw)
    projection = project_reference(chain, c.projection, q, state.reference_q_rad,
                                   jnp.clip(raw, -1., 1.)*c.projection.velocity_limit)
    native_rmp = state.rmp._replace(target_buffer_index=state.rmp.target_buffer_index[0],
                                    output_buffer_index=state.rmp.output_buffer_index[0])
    expected, *commands = joint_rmp_step(native_rmp, projection.reference_q,
        c.projection.q_lower, c.projection.q_upper, c.projection.velocity_limit,
        c.acceleration_limit_rad_s2, c.jerk_limit_rad_s3, c.rmp_params, .001,
        servo_steps=5, target_update_delay_steps=3, use_safety_bounds=True,
        use_jerk_limit=False, return_history=True)
    for left, right in zip(jax.tree.leaves(actual.rmp), jax.tree.leaves(promote_rmp_indices(expected))):
        np.testing.assert_allclose(left, right, atol=2e-6, rtol=1e-6)
    for key, command in zip(('q', 'dq', 'ddq'), commands):
        combined = np.concatenate((state.output_delay[key], np.swapaxes(command, 0, 1)), axis=1)
        reference = np.array([[combined[0, 19+i-int(context.output_delay_steps[0,j]), j]
                               for j in range(7)] for i in range(5)])
        np.testing.assert_allclose(trace['delayed_commands'][key][:, 0], reference, atol=1e-6)
        np.testing.assert_allclose(actual.output_delay[key], combined[:, -19:], atol=1e-6)
    np.testing.assert_allclose(actual.reference_q_rad, projection.reference_q, atol=1e-6)
    np.testing.assert_allclose(actual.previous_accepted_action,
        jnp.clip(projection.accepted_qvel/c.projection.velocity_limit, -1., 1.), atol=1e-6)


def test_control_memory_retains_action_effect_past_output_delay(fixture):
    from world_model.world_control import control_step
    chain, q, state, context, contract = setup_control(fixture)
    # Wide projection limits keep this derivative test in one smooth mode.
    contract = replace(contract, projection=replace(contract.projection,
        position_lower=jnp.ones(3)*-10., position_upper=jnp.ones(3)*10.,
        outer_position_lower=jnp.ones(3)*-10., outer_position_upper=jnp.ones(3)*10.,
        roll_pitch_limit=jnp.ones(2), outer_roll_pitch_limit=jnp.ones(2)))
    def run(value):
        def step(current, _):
            after, trace = control_step(current, jnp.zeros((1, 7)).at[0, 0].set(value),
                                       q, context, chain=chain, contract=contract)
            return after, trace['delayed_commands']['q']
        return jax.lax.scan(step, state, None, length=16)[1][-1, -1, 0, 0]
    value = jnp.float32(.02)
    gradient = jax.jit(jax.grad(run))(value)
    numerical = (jax.jit(run)(value+.002)-jax.jit(run)(value-.002))/.004
    assert float(gradient) > 1e-5
    np.testing.assert_allclose(gradient, numerical, rtol=.03, atol=2e-5)
