"""Optional single-step supervision alongside unchanged recursive contact loss."""

from __future__ import annotations

import math
from typing import Callable


def make_contact_objective(component_loss: Callable, *, single_step_weight: float) -> Callable:
    """Preserve the recursive branch and add separately evaluated teacher forcing.

    Labels enter the auxiliary training branch only. The physical transition,
    recursive state feedback and evaluation rollout remain unchanged.
    """
    if not math.isfinite(single_step_weight) or single_step_weight < 0:
        raise ValueError('Single-step auxiliary weight must be finite and nonnegative')
    if single_step_weight == 0:
        return component_loss

    def objective(parameters, surface, batch, *, ball_radius_m, autoregressive):
        loss, metrics = component_loss(parameters, surface, batch,
            ball_radius_m=ball_radius_m, autoregressive=autoregressive)
        if not autoregressive:
            return loss, metrics
        auxiliary, auxiliary_metrics = component_loss(parameters, surface, batch,
            ball_radius_m=ball_radius_m, autoregressive=False)
        total = loss+single_step_weight*auxiliary
        return total, dict(metrics, loss=total, recursive_loss=loss,
            auxiliary_single_step_loss=auxiliary,
            auxiliary_weighted_loss=single_step_weight*auxiliary,
            **{'single_step_'+key: value for key, value in auxiliary_metrics.items()})

    return objective


def validate_contact_resume_recipe(previous: dict, current: dict, *, completed: int) -> None:
    """Allow an auxiliary change only before the first recursive update.

    This permits a paired experiment from one identical bootstrap checkpoint;
    already consumed optimizer/loss/data semantics cannot be reinterpreted.
    """
    old, new = dict(previous), dict(current)
    old_weight = old.pop('single_step_aux_weight', 0.)
    new_weight = new.pop('single_step_aux_weight', 0.)
    if (completed < 0 or any(not math.isfinite(w) or w < 0
                             for w in (old_weight, new_weight))):
        raise ValueError('Invalid completed updates or auxiliary weight')
    if old != new:
        raise ValueError('Resume contact fit recipe changed')
    if old_weight != new_weight and completed > old['bootstrap_updates']:
        raise ValueError('Auxiliary loss changes already executed recursive updates')
