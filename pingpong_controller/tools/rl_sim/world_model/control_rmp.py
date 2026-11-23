"""Native RMP with per-lane ring clocks and separately preserved output delay."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from recovered_rmp_jax import JointRMPState, joint_rmp_step


RING_INDICES = frozenset(('target_buffer_index', 'output_buffer_index'))


def promote_rmp_indices(state: JointRMPState) -> JointRMPState:
    """Convert a shared-phase native batch to independently restorable clocks."""
    size = state.q.shape[0]
    if state.q.ndim != 2 or state.q.shape[1] != 7:
        raise ValueError('Expected native right-arm state (N,7)')
    values = {}
    for name in RING_INDICES:
        value = getattr(state, name)
        if value.shape not in ((), (size,)):
            raise ValueError('Unknown RMP ring-index layout')
        values[name] = jnp.broadcast_to(value, (size,))
    return state._replace(**values)


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


def apply_output_delay(buffer, commands, per_joint_delay_steps):
    """Return (5,N,7) delayed commands and (N,D,7) retained history, unchanged."""
    length = buffer.shape[1]
    combined = jnp.concatenate((buffer, jnp.swapaxes(commands, 0, 1)), axis=1)
    indices = length+jnp.arange(commands.shape[0])[None, :, None]-per_joint_delay_steps[:, None, :]
    delayed = jnp.take_along_axis(combined, indices, axis=1)
    return jnp.swapaxes(delayed, 0, 1), combined[:, -length:, :]
