"""Contact and apex labels from actual 1 ms trajectories, independent of reward.

V324's trace contact arrays describe the interval at its pre-integration
configuration. A release therefore uses that interval's before-state as the
observed outgoing state; it does not add another gravity tick to the label.
The state persists across chunks and belongs in every complete seed.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class PhysicalEventState(NamedTuple):
    contact_active: Any
    impact_id: Any
    normal_impulse_n_s: Any
    contact_duration_steps: Any
    controlled_contact: Any


def init_physical_events(n_envs: int) -> PhysicalEventState:
    """Initialize only a fresh non-contact launch; mid-flight forks restore state."""
    off = jnp.zeros(n_envs, bool)
    integer = jnp.zeros(n_envs, jnp.int32)
    return PhysicalEventState(off, integer, jnp.zeros(n_envs, jnp.float32), integer, off)


def physical_event_substep(state: PhysicalEventState, trace: dict, *, valid,
                           initialization_active, dt_s: float = .001):
    """Observe one substep without modifying physics or inventing an impulse.

    valid excludes unusable numerical/other-collision samples and padding.
    A useful controlled release needs a contact begun after setup, positive
    integrated normal force and upward outgoing velocity. Geometric touches
    and all physical entry/release edges are still retained as separate labels.
    """
    overlap = trace['contact_overlap']
    entry = valid & overlap & ~state.contact_active
    release = valid & ~overlap & state.contact_active
    before_p = trace['ball_qpos_before'][..., :3]
    after_p = trace['ball_qpos_after'][..., :3]
    before_v = trace['ball_qvel_before'][..., :3]
    after_v = trace['ball_qvel_after'][..., :3]
    finite = (jnp.all(jnp.isfinite(before_p), axis=-1)
              & jnp.all(jnp.isfinite(after_p), axis=-1)
              & jnp.all(jnp.isfinite(before_v), axis=-1)
              & jnp.all(jnp.isfinite(after_v), axis=-1)
              & jnp.isfinite(trace['contact_normal_force_n']))
    usable = valid & finite
    entry &= finite
    release &= finite
    controlled_release = (release & state.controlled_contact & ~initialization_active
                          & (state.normal_impulse_n_s > 0.) & (before_v[..., 2] > 0.))
    force = jnp.maximum(jnp.nan_to_num(trace['contact_normal_force_n']), 0.)
    impulse = jnp.where(entry, 0., state.normal_impulse_n_s) + force*dt_s
    duration = jnp.where(entry, 0, state.contact_duration_steps) + 1
    controlled = jnp.where(entry, ~initialization_active, state.controlled_contact)
    candidate = PhysicalEventState(overlap, state.impact_id+entry.astype(jnp.int32),
        jnp.where(overlap, impulse, 0.), jnp.where(overlap, duration, 0),
        overlap & controlled & ~initialization_active)
    after = jax.tree.map(lambda old, new: jnp.where(usable, new, old), state, candidate)
    apex = (usable & ~initialization_active & ~overlap
            & (before_v[..., 2] > 0.) & (after_v[..., 2] <= 0.))
    fraction = jnp.clip(before_v[..., 2]/jnp.maximum(before_v[..., 2]-after_v[..., 2], 1e-12), 0., 1.)
    position = before_p+fraction[..., None]*(after_p-before_p)
    velocity = before_v+fraction[..., None]*(after_v-before_v)

    def masked(value, mask):
        return jnp.where(mask[..., None], value, 0.)

    events = dict(contact_entry=entry, contact_release=release,
        controlled_release=controlled_release, impact_id=after.impact_id,
        contact_active=after.contact_active, valid=usable,
        contact_duration_steps=jnp.where(release, state.contact_duration_steps, after.contact_duration_steps),
        release_normal_impulse_n_s=jnp.where(release, state.normal_impulse_n_s, 0.),
        release_position_m=masked(before_p, release), release_velocity_m_s=masked(before_v, release),
        actual_apex=apex, apex_position_m=masked(position, apex), apex_velocity_m_s=masked(velocity, apex),
        apex_fraction=jnp.where(apex, fraction, 0.))
    return after, events


def scan_physical_events(state: PhysicalEventState, trace: dict, *, valid,
                         initialization_active, dt_s: float = .001):
    """Label a time-major (substeps,N,...) trace, retaining cross-chunk contacts."""
    if valid.shape != trace['contact_overlap'].shape or initialization_active.shape != valid.shape:
        raise ValueError('Physical event masks must match the time/environment axes')

    def step(current, inputs):
        sample, available, setup = inputs
        return physical_event_substep(current, sample, valid=available,
                                      initialization_active=setup, dt_s=dt_s)

    return jax.lax.scan(step, state, (trace, valid, initialization_active))
