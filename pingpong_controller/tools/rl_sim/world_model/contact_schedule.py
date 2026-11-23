"""Contact fitting window schedules and exact consumed-data accounting."""

from __future__ import annotations

from .contact_fitting import validate_contact_resume_recipe
from .fit_schedule import controls_at, parse_schedule


def contact_schedule(recipe: dict) -> list[list[int]]:
    """Treat older constant-window checkpoints as an equivalent single stage."""
    values = recipe.get('window_schedule')
    text = None if values is None else ','.join(':'.join(map(str, row)) for row in values)
    return parse_schedule(text, bootstrap=recipe['bootstrap_updates'], initial=recipe['controls'])


def validate_contact_training_recipe(previous: dict, current: dict, *, completed: int) -> None:
    """Permit future window extension; keep past losses and optimizer semantics."""
    validate_contact_resume_recipe(
        {k: v for k, v in previous.items() if k != 'window_schedule'},
        {k: v for k, v in current.items() if k != 'window_schedule'}, completed=completed)
    old, new = contact_schedule(previous), contact_schedule(current)
    boundaries = {0} | {step for step, _ in old+new if step < completed}
    if any(controls_at(step, old) != controls_at(step, new)
           for step in boundaries if step < completed):
        raise ValueError('Window schedule changes already executed contact updates')


def contact_substep_uses(recipe: dict, *, completed: int) -> tuple[int, int]:
    """Return primary and extra auxiliary uses, not new collection counts."""
    if completed < 0:
        raise ValueError('Completed update count must be nonnegative')
    schedule = contact_schedule(recipe)
    batch = recipe.get('batch_windows', 256)

    def uses(start: int) -> int:
        if start >= completed:
            return 0
        edges = sorted({start, completed} | {step for step, _ in schedule
                                            if start < step < completed})
        return sum((right-left)*controls_at(left, schedule)*5*batch
                   for left, right in zip(edges, edges[1:]))

    auxiliary = uses(recipe['bootstrap_updates']) if recipe.get('single_step_aux_weight', 0.) > 0 else 0
    return uses(0), auxiliary
