"""Same-clock coupled-physics errors; native labels select diagnostic events."""

from __future__ import annotations

import jax.numpy as jnp

from .contact_losses import ballistic_apex


def ball_metrics(trace: dict, batch: dict) -> tuple[dict, dict]:
    """Keep a teacher release even when the model misses it or becomes nonfinite."""
    before, actual = trace['before'], batch['before']
    outgoing = batch['valid'] & batch['controlled_release']
    endpoint_mask = batch['valid'][-1]
    metrics = dict(
        endpoint_position_m=(jnp.linalg.norm(trace['after'].position_m[-1]
            -batch['after'].position_m[-1], axis=-1), endpoint_mask),
        endpoint_velocity_m_s=(jnp.linalg.norm(trace['after'].velocity_m_s[-1]
            -batch['after'].velocity_m_s[-1], axis=-1), endpoint_mask),
        outgoing_velocity_m_s=(jnp.linalg.norm(before.velocity_m_s-actual.velocity_m_s, axis=-1), outgoing),
        next_apex_m=(jnp.linalg.norm(ballistic_apex(before.position_m, before.velocity_m_s)
            -ballistic_apex(actual.position_m, actual.velocity_m_s), axis=-1), outgoing))
    events = {}
    for edge, plural in (('entry', 'entries'), ('release', 'releases')):
        events[edge+'_missed'] = jnp.sum(batch['valid'] & batch[edge] & ~trace[edge])
        events[edge+'_extra'] = jnp.sum(batch['valid'] & trace[edge] & ~batch[edge])
        events['true_'+plural] = jnp.sum(batch['valid'] & batch[edge])
    return metrics, events


def precontact_mask(entry, valid, *, steps: int = 25):
    """Select before-states0..25 ms before an actual entry inside this window.

    Future native labels are used only to stratify errors, never as model
    inputs. No unobserved event beyond the right edge is invented.
    """
    future = jnp.pad(entry, ((0, steps), (0, 0)))
    mask = jnp.zeros_like(entry)
    for offset in range(steps+1):
        mask = mask | future[offset:offset+entry.shape[0]]
    return mask & valid


def robot_metrics(trace: dict, raw: dict, batch: dict) -> dict:
    """Compare predicted full-chain FK with the native preintegration labels."""
    pose = trace['racket_before']
    normal = raw['racket_geom_rotation_before'][..., :, 0]
    masks = dict(all=batch['valid'], precontact=precontact_mask(batch['entry'], batch['valid']))
    position = jnp.linalg.norm(pose['position']-raw['racket_site_position_before_m'], axis=-1)
    velocity = jnp.abs(jnp.sum((pose['velocity']-raw['racket_linear_velocity_before_m_s'])*normal, axis=-1))
    metrics = {}
    for scope, mask in masks.items():
        metrics[scope+'_racket_position_m'] = (position, mask)
        metrics[scope+'_racket_normal_velocity_m_s'] = (velocity, mask)
    for key, predicted, label in (
        ('joint_q_rmse_rad', trace['q_after_rad'][-1], raw['q_after'][-1]),
        ('joint_dq_rmse_rad_s', trace['dq_after_rad_s'][-1], raw['dq_after'][-1])):
        metrics[key] = (jnp.sqrt(jnp.mean((predicted-label)**2, axis=-1)), batch['valid'][-1])
    return metrics
