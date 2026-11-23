"""Offline window and validity rules for frozen new-pose model diagnostics."""

from __future__ import annotations

import numpy as np


def prefix_validity(finite, other_collision, endpoint_failure, admitted) -> tuple:
    """Separate usable physical labels from the prefix before first task failure.

    Endpoint constraints are observed after substep four. An invalid interval
    itself is excluded; later recovery never reopens a successful task prefix.
    Post-failure finite diagnostics remain explicitly separate from task data.
    """
    finite = np.asarray(finite, bool)
    other_collision = np.asarray(other_collision, bool)
    endpoint_failure = np.asarray(endpoint_failure)
    admitted = np.asarray(admitted, bool)
    if (finite.ndim != 2 or finite.shape != other_collision.shape
            or finite.shape[0] != 5 * len(endpoint_failure)
            or endpoint_failure.shape[1:] != finite.shape[1:]
            or admitted.shape != finite.shape[1:]):
        raise ValueError('Expected aligned1ms labels and5ms endpoint failures')
    usable = finite & ~other_collision & admitted[None]
    failure = ~usable
    failure[4::5] |= endpoint_failure != 0
    prefix = ~np.maximum.accumulate(failure, axis=0) & admitted[None]
    return usable, prefix


def diagnostic_starts(controlled_release, horizon: int) -> np.ndarray:
    """Center on first observed release; keep misses at their real reset seed."""
    release = np.asarray(controlled_release, bool)
    if release.ndim != 2 or release.shape[0] % 5 or not 1 <= horizon <= release.shape[0] // 5:
        raise ValueError('Invalid native duration or prediction horizon')
    controls = release.shape[0] // 5
    first = np.argmax(release, axis=0) // 5
    return np.where(release.any(axis=0), np.clip(first - horizon // 2, 0, controls - horizon), 0)


def select_windows(values, starts, length: int):
    """Gather aligned time windows without clipping or wrapping unavailable rows."""
    values = np.asarray(values)
    starts = np.asarray(starts, dtype=np.int64)
    if (values.ndim < 2 or starts.shape != (values.shape[1],) or length < 1
            or np.any(starts < 0) or np.any(starts + length > len(values))):
        raise ValueError('Requested window is outside the actual recorded sequence')
    return values[starts[None] + np.arange(length)[:, None], np.arange(len(starts))[None]]
