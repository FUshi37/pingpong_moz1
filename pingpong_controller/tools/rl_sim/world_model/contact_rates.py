"""Explicit future learning-rate stages with preserved optimizer history."""

from __future__ import annotations

import math


def parse_rates(text: str | None, *, initial: float) -> list[list]:
    """Parse update:rate pairs; every schedule starts with the existing rate."""
    if not math.isfinite(initial) or initial <= 0:
        raise ValueError('Initial learning rate must be finite and positive')
    schedule = [[0, initial]]
    if text is not None:
        schedule = []
        for item in text.split(','):
            step, rate = item.split(':')
            schedule.append([int(step), float(rate)])
    if not schedule or schedule[0] != [0, initial]:
        raise ValueError('Rate schedule must start at update0 with the existing rate')
    for index, (step, rate) in enumerate(schedule):
        if step < 0 or not math.isfinite(rate) or rate <= 0:
            raise ValueError('Rate boundaries and learning rates must be valid')
        if index and step <= schedule[index-1][0]:
            raise ValueError('Rate boundaries must increase strictly')
    return schedule


def rate_at(iteration: int, schedule: list[list]) -> float:
    """Select the rate for one zero-based optimizer iteration."""
    rate = schedule[0][1]
    for boundary, value in schedule:
        if iteration < boundary:
            break
        rate = value
    return rate


def rate_schedule(recipe: dict) -> list[list]:
    values = recipe.get('learning_rate_schedule')
    text = None if values is None else ','.join(':'.join(map(str, row)) for row in values)
    return parse_rates(text, initial=recipe['learning_rate'])


def validate_rate_resume(previous: dict, current: dict, *, completed: int) -> None:
    """A new rate may begin at the next update, never in the consumed prefix."""
    if completed < 0:
        raise ValueError('Completed update count must be nonnegative')
    old, new = rate_schedule(previous), rate_schedule(current)
    boundaries = {0} | {step for step, _ in old+new if step < completed}
    if any(rate_at(step, old) != rate_at(step, new)
           for step in boundaries if step < completed):
        raise ValueError('Learning rate changes already executed optimizer updates')
