"""WMJ-only derivative adapter, with exactly the native RMP forward arithmetic.

The three functions below are source-pinned copies of their native counterparts.
Only the two safety-braking square roots receive a finite zero-branch derivative.
No target, limit, integration, filter, delay or simulated controller is changed.
The shared V324 implementation and all previous world paths remain untouched.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp

from recovered_rmp_jax import JointRMPParams, JointRMPState
from .control_projection import project_reference
from .control_rmp import RING_INDICES, apply_output_delay
from .world_control import ControlState, ControlContext, ControlContract


@jax.custom_jvp
def braking_sqrt(value):
    """Native sqrt forward; zero derivative for the clamped, nonpositive branch."""
    return jnp.sqrt(value)


@braking_sqrt.defjvp
def _braking_sqrt_jvp(primals, tangents):
    value, = primals
    tangent, = tangents
    positive = value > 0.
    denominator = jnp.sqrt(jnp.where(positive, value, 1.))
    return braking_sqrt(value), jnp.where(positive, .5*tangent/denominator, 0.)


def _verify_sources():
    """One import-time check, never in a control/reset/physics loop."""
    root = Path(__file__).resolve().parents[1]
    for relative, name, digest in SOURCE_FUNCTIONS:
        text = (root/relative).read_text()
        node = next(n for n in ast.parse(text).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        source = '\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'
        if hashlib.sha256(source.encode()).hexdigest() != digest:
            raise ValueError(f'Native function changed; re-audit WMJ derivative adapter: {name}')

SOURCE_FUNCTIONS = (('recovered_rmp_jax.py', 'joint_rmp_step', '29e294c66f67326b68c56ae01fffa5dbcf0e8fca95f9a60ef3886255205dfac8'), ('world_model/control_rmp.py', 'rmp_step_lanes', '147928dec08f3e2eb16996fa0d6aebcf41d9d94ddc0ac3b4e16019db8784cae2'), ('world_model/world_control.py', 'control_step', 'e8c5acc61943d8bf9c4cbed3e52375990c782dba8a8d7e4487575a09af60d40e'))
_verify_sources()


def joint_rmp_step(
    state: JointRMPState,
    position_target: jax.Array,
    q_min_rad: jax.Array,
    q_max_rad: jax.Array,
    qd_max_rad_s: jax.Array,
    qdd_max_rad_s2: jax.Array,
    qddd_max_rad_s3: jax.Array,
    params: JointRMPParams,
    dt_s: float,
    *,
    jnt_kp: jax.Array | None = None,
    jnt_kd: jax.Array | None = None,
    estimator_joint_process_var: jax.Array | None = None,
    estimator_joint_measure_var: jax.Array | None = None,
    joint_velocity_feedforward: jax.Array | None = None,
    joint_acceleration_feedforward: jax.Array | None = None,
    acceleration_weight: jax.Array | None = None,
    target_filter_length: jax.Array | None = None,
    servo_steps: int = 1,
    target_update_delay_steps: int = 0,
    target_updated: bool | jax.Array = True,
    use_safety_bounds: bool = True,
    use_jerk_limit: bool = False,
    return_history: bool = False,
) -> JointRMPState | tuple[JointRMPState, jax.Array, jax.Array, jax.Array]:
    """Advance the recovered joint RMP for one outer-loop command.

    Set ``servo_steps=5`` and ``dt_s=0.001`` for the 200 Hz policy / 1 kHz
    planner contract.  Returned histories have shape ``(servo_steps, n, 7)``.
    """

    if dt_s <= 0.0 or servo_steps <= 0:
        raise ValueError("dt_s and servo_steps must be positive")
    if not 0 <= int(target_update_delay_steps) < int(servo_steps):
        raise ValueError(
            "target_update_delay_steps must be in [0, servo_steps)"
        )

    target = jnp.asarray(position_target, dtype=state.q.dtype)
    q_min = jnp.asarray(q_min_rad, dtype=state.q.dtype)
    q_max = jnp.asarray(q_max_rad, dtype=state.q.dtype)
    qd_max = jnp.asarray(qd_max_rad_s, dtype=state.q.dtype)
    qdd_max = jnp.asarray(qdd_max_rad_s2, dtype=state.q.dtype)
    qddd_max = jnp.asarray(qddd_max_rad_s3, dtype=state.q.dtype)
    angle_gap = jnp.asarray(params.angle_gap_rad, dtype=state.q.dtype)
    period = float(dt_s)
    transition = jnp.asarray(
        [
            [1.0, period, 0.5 * period * period],
            [0.0, 1.0, period],
            [0.0, 0.0, 1.0],
        ],
        dtype=state.q.dtype,
    )
    dt2, dt3 = period**2, period**3
    dt4, dt5 = period**4, period**5
    base_process_covariance = jnp.asarray(
        [
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0, dt2 / 2.0, period],
        ],
        dtype=state.q.dtype,
    )
    process_var = jnp.asarray(
        params.estimator_joint_process_var
        if estimator_joint_process_var is None
        else estimator_joint_process_var,
        dtype=state.q.dtype,
    )
    measure_var = jnp.asarray(
        params.estimator_joint_measure_var
        if estimator_joint_measure_var is None
        else estimator_joint_measure_var,
        dtype=state.q.dtype,
    )
    process_covariance = process_var[..., None, None] * base_process_covariance
    kp = jnp.asarray(
        params.jnt_kp if jnt_kp is None else jnt_kp, dtype=state.q.dtype
    )
    kd = jnp.asarray(
        params.jnt_kd if jnt_kd is None else jnt_kd, dtype=state.q.dtype
    )
    velocity_feedforward = jnp.asarray(
        params.joint_velocity_feedforward
        if joint_velocity_feedforward is None
        else joint_velocity_feedforward,
        dtype=state.q.dtype,
    )
    acceleration_feedforward = jnp.asarray(
        params.joint_acceleration_feedforward
        if joint_acceleration_feedforward is None
        else joint_acceleration_feedforward,
        dtype=state.q.dtype,
    )
    acc_weight = jnp.asarray(
        params.acc_weight if acceleration_weight is None else acceleration_weight,
        dtype=state.q.dtype,
    )
    filter_length = state.target_q_buffer.shape[-2]
    selected_filter_length = None
    if target_filter_length is not None:
        selected_filter_length = jnp.clip(
            jnp.asarray(target_filter_length, dtype=jnp.int32), 1, filter_length
        )
    identity3 = jnp.eye(3, dtype=state.q.dtype)

    def one_servo_step(
        servo_index: int | jax.Array, value: JointRMPState
    ) -> JointRMPState:
        is_target_update = (
            servo_index == int(target_update_delay_steps)
        ) & jnp.asarray(target_updated, dtype=bool)
        elapsed_s = (
            jnp.minimum(value.ticks_since_target_update, 100)[..., None]
            * period
        )
        maximum_target_delta = qd_max * elapsed_s
        new_accepted_target = jnp.clip(
            value.accepted_target
            + jnp.clip(
                target - value.accepted_target,
                -maximum_target_delta,
                maximum_target_delta,
            ),
            q_min,
            q_max,
        )
        accepted_target = jnp.where(
            is_target_update, new_accepted_target, value.accepted_target
        )
        predicted_x = jnp.einsum("ij,...nj->...ni", transition, value.target_x)
        predicted_covariance = (
            jnp.einsum(
                "ij,...njk,lk->...nil",
                transition,
                value.target_covariance,
                transition,
            )
            + process_covariance
        )

        def correct_estimator(_: None) -> tuple[jax.Array, jax.Array]:
            innovation = accepted_target - predicted_x[..., 0]
            innovation_covariance = predicted_covariance[..., 0, 0] + measure_var
            gain = predicted_covariance[..., :, 0] / innovation_covariance[..., None]
            corrected_x = predicted_x + gain * innovation[..., None]
            kh = gain[..., :, None] * jnp.asarray(
                [1.0, 0.0, 0.0], dtype=state.q.dtype
            )
            left = identity3 - kh
            corrected_covariance = (
                left @ predicted_covariance @ jnp.swapaxes(left, -1, -2)
                + measure_var[..., None, None]
                * gain[..., :, None]
                * gain[..., None, :]
            )
            return corrected_x, corrected_covariance

        braking_acceleration = jnp.clip(
            -value.target_x[..., 1] / period, -qdd_max, qdd_max
        )
        decelerating_x = value.target_x.at[..., 2].set(
            jnp.where(
                jnp.abs(value.target_x[..., 1]) <= 1.0e-6,
                0.0,
                braking_acceleration,
            )
        )
        decelerated_x = jnp.einsum(
            "ij,...nj->...ni", transition, decelerating_x
        )

        def held_target_prediction(_: None) -> tuple[jax.Array, jax.Array]:
            predict = value.ticks_since_target_update <= 20
            return (
                jnp.where(predict[..., None, None], predicted_x, decelerated_x),
                predicted_covariance,
            )

        target_x, target_covariance = jax.lax.cond(
            is_target_update,
            correct_estimator,
            held_target_prediction,
            operand=None,
        )
        estimated_q = target_x[..., 0]
        estimated_qd = jnp.clip(target_x[..., 1], -qd_max, qd_max)
        estimated_qdd = target_x[..., 2]
        target_index = value.target_buffer_index
        target_q_buffer = value.target_q_buffer.at[..., target_index, :].set(
            estimated_q
        )
        target_qd_buffer = value.target_qd_buffer.at[..., target_index, :].set(
            estimated_qd
        )
        target_qdd_buffer = value.target_qdd_buffer.at[..., target_index, :].set(
            estimated_qdd
        )
        if selected_filter_length is None:
            filtered_q = jnp.mean(target_q_buffer, axis=-2)
            filtered_qd = jnp.mean(target_qd_buffer, axis=-2)
            filtered_qdd = jnp.mean(target_qdd_buffer, axis=-2)
        else:
            slots = jnp.arange(filter_length, dtype=jnp.int32)
            ages = (target_index - slots) % filter_length
            mask = ages < selected_filter_length[..., None]

            def recent_mean(buffer: jax.Array) -> jax.Array:
                recent = jnp.swapaxes(buffer, -2, -1)
                return jnp.sum(recent * mask, axis=-1) / selected_filter_length

            filtered_q = recent_mean(target_q_buffer)
            filtered_qd = recent_mean(target_qd_buffer)
            filtered_qdd = recent_mean(target_qdd_buffer)

        reference_qdd = (
            kp * (filtered_q - value.q)
            + kd * (velocity_feedforward * filtered_qd - value.qd)
            + acceleration_feedforward * filtered_qdd
        )
        reference_qdd = jnp.clip(reference_qdd, -qdd_max, qdd_max)
        qdd = reference_qdd / (1.0 + acc_weight)
        if use_safety_bounds:
            safe_min = q_min + angle_gap
            safe_max = q_max - angle_gap
            braking = 0.998 * qdd_max
            safe_velocity_upper = jnp.minimum(
                qd_max,
                braking_sqrt(2.0 * braking * jnp.maximum(safe_max - value.q, 0.0)),
            )
            safe_velocity_lower = jnp.maximum(
                -qd_max,
                -braking_sqrt(2.0 * braking * jnp.maximum(value.q - safe_min, 0.0)),
            )
            lower = jnp.maximum(
                -qdd_max, (safe_velocity_lower - value.qd) / period
            )
            upper = jnp.minimum(
                qdd_max, (safe_velocity_upper - value.qd) / period
            )
            if use_jerk_limit:
                lower = jnp.maximum(lower, value.qdd - qddd_max * period)
                upper = jnp.minimum(upper, value.qdd + qddd_max * period)
            feasible = lower <= upper
            emergency_recovery = jnp.clip(jnp.zeros_like(qdd), upper, lower)
            qdd = jnp.where(
                feasible, jnp.clip(qdd, lower, upper), emergency_recovery
            )
        q_next = value.q + (value.qd + 0.5 * period * qdd) * period
        qd_next = value.qd + period * qdd
        output_index = value.output_buffer_index
        output_q_buffer = value.output_q_buffer.at[..., output_index, :].set(q_next)
        output_qd_buffer = value.output_qd_buffer.at[..., output_index, :].set(
            qd_next
        )
        output_qdd_buffer = value.output_qdd_buffer.at[..., output_index, :].set(
            qdd
        )
        return JointRMPState(
            q=q_next,
            qd=qd_next,
            qdd=qdd,
            command_q=jnp.mean(output_q_buffer, axis=-2),
            command_qd=jnp.mean(output_qd_buffer, axis=-2),
            command_qdd=jnp.mean(output_qdd_buffer, axis=-2),
            accepted_target=accepted_target,
            ticks_since_target_update=jnp.where(
                is_target_update,
                jnp.ones_like(value.ticks_since_target_update),
                value.ticks_since_target_update + 1,
            ),
            target_x=target_x,
            target_covariance=target_covariance,
            target_q_buffer=target_q_buffer,
            target_qd_buffer=target_qd_buffer,
            target_qdd_buffer=target_qdd_buffer,
            target_buffer_index=(target_index + 1) % filter_length,
            output_q_buffer=output_q_buffer,
            output_qd_buffer=output_qd_buffer,
            output_qdd_buffer=output_qdd_buffer,
            output_buffer_index=(output_index + 1)
            % value.output_q_buffer.shape[-2],
        )

    if return_history:

        def scan_step(
            value: JointRMPState, servo_index: jax.Array
        ) -> tuple[JointRMPState, tuple[jax.Array, jax.Array, jax.Array]]:
            next_value = one_servo_step(servo_index, value)
            return next_value, (
                next_value.command_q,
                next_value.command_qd,
                next_value.command_qdd,
            )

        final_state, history = jax.lax.scan(
            scan_step,
            state,
            xs=jnp.arange(int(servo_steps), dtype=jnp.int32),
        )
        return final_state, history[0], history[1], history[2]
    return jax.lax.fori_loop(0, int(servo_steps), one_servo_step, state)


def rmp_step_lanes(state: JointRMPState, target, context: dict, *, q_min, q_max,
                   velocity_limit, acceleration_limit, jerk_limit, params):
    """Vmap unchanged native one-lane calls, including five internal 1 ms ticks.

    Independent branch seeds and physical resets may have different ring
    indices. A singleton native batch preserves its indexing arithmetic; no
    history rotation, clock guessing or controller-state reset is performed.
    Context keys are the native per-episode RMP keyword arrays with shape(N,7).
    """
    def one_lane(lane, reference, episode):
        native = JointRMPState(**{name: value if name in RING_INDICES else value[None]
                                  for name, value in lane._asdict().items()})
        next_state, q, dq, ddq = joint_rmp_step(native, reference[None],
            q_min, q_max, velocity_limit, acceleration_limit, jerk_limit, params, .001,
            **{name: value[None] for name, value in episode.items()},
            servo_steps=5, target_update_delay_steps=3, use_safety_bounds=True,
            use_jerk_limit=False, return_history=True)
        compact = JointRMPState(**{name: value if name in RING_INDICES else value[0]
                                  for name, value in next_state._asdict().items()})
        return compact, dict(q=q[:, 0], dq=dq[:, 0], ddq=ddq[:, 0])
    after, commands = jax.vmap(one_lane)(state, target, context)
    return after, jax.tree.map(lambda value: jnp.swapaxes(value, 0, 1), commands)


def control_step(state: ControlState, raw_action, measured_q_rad, context: ControlContext,
                 *, chain, contract: ControlContract):
    """Advance one200 Hz policy action using own measured/predicted13-joint state."""
    if (raw_action.shape != state.reference_q_rad.shape or raw_action.ndim != 2
            or raw_action.shape[-1] != 7 or context.action_scale_mult.shape != raw_action.shape[:1]
            or context.output_delay_steps.shape != raw_action.shape
            or set(state.output_delay) != {'q', 'dq', 'ddq'}):
        raise ValueError('Expected seven actions/references and matching per-lane control context')
    c = contract.projection
    clipped = jnp.clip(raw_action, -1., 1.)
    denominator = c.velocity_limit[None]*contract.action_velocity_scale*context.action_scale_mult[:, None]
    requested = jnp.clip(clipped*denominator, -c.velocity_limit[None], c.velocity_limit[None])
    projection = project_reference(chain, c, measured_q_rad, state.reference_q_rad, requested)
    accepted = jnp.clip(projection.accepted_qvel/jnp.maximum(denominator, 1e-6), -1., 1.)
    rmp, commands = rmp_step_lanes(state.rmp, projection.reference_q, context.rmp,
        q_min=c.q_lower, q_max=c.q_upper, velocity_limit=c.velocity_limit,
        acceleration_limit=contract.acceleration_limit_rad_s2,
        jerk_limit=contract.jerk_limit_rad_s3, params=contract.rmp_params)
    delayed, buffers = {}, {}
    for key in ('q', 'dq', 'ddq'):
        delayed[key], buffers[key] = apply_output_delay(state.output_delay[key], commands[key],
                                                         context.output_delay_steps)
    after = ControlState(projection.reference_q, accepted, rmp, buffers)
    return after, dict(raw_action=raw_action, clipped_action=clipped, accepted_action=accepted,
        requested_qvel_rad_s=requested, projection=projection,
        rmp_commands=commands, delayed_commands=delayed)
