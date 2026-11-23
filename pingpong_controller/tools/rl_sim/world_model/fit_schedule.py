"""Explicit 8/16/32/64 control-window curriculum with compatible resume."""

from __future__ import annotations


def parse_schedule(text: str | None, *, bootstrap: int, initial: int) -> list[list[int]]:
    """Map zero-based optimizer iteration boundaries to physical window lengths."""
    schedule = ([[bootstrap, initial]] if text is None else
                [[int(value) for value in item.split(':')] for item in text.split(',')])
    if not schedule or schedule[0] != [bootstrap, initial]:
        raise ValueError('Window schedule must begin with the declared bootstrap and window')
    for i, row in enumerate(schedule):
        if len(row) != 2 or row[0] < 0 or row[1] not in (8, 16, 32, 64):
            raise ValueError('Expected nonnegative update:window pairs with windows 8/16/32/64')
        if i and (row[0] <= schedule[i-1][0] or row[1] != 2*schedule[i-1][1]):
            raise ValueError('Window stages must advance in order without skips or duplicates')
    return schedule


def controls_at(iteration: int, schedule: list[list[int]]) -> int:
    """Bootstrap uses the initial window as independent single-step labels."""
    controls = schedule[0][1]
    for boundary, window in schedule:
        if iteration < boundary:
            break
        controls = window
    return controls


def validate_resume_recipe(previous: dict, current: dict, *, completed: int) -> None:
    """Permit a future curriculum extension; never reinterpret consumed updates."""
    varying = {'window_schedule', 'control_window'}
    if ({key: value for key, value in previous.items() if key not in varying}
            != {key: value for key, value in current.items() if key not in varying}):
        raise ValueError('Resume optimizer/loss/data recipe changed')
    old = previous.get('window_schedule', [[previous['bootstrap_updates'], previous['control_window']]])
    new = current.get('window_schedule', [[current['bootstrap_updates'], current['control_window']]])
    if any(controls_at(step, old) != controls_at(step, new) for step in range(completed)):
        raise ValueError('Resume schedule changes an already executed update')
