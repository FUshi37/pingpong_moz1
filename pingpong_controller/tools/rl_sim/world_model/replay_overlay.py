"""Lossless replay differences against immutable, authenticated teacher arrays.

These are storage deltas only. Every decoded label is from the canonical new
replay, including small arm/ball changes caused by a different compiled graph.
No numerical threshold silently substitutes an old trajectory value.
"""

from __future__ import annotations

import numpy as np


def encode_overlay(base: dict, canonical: dict) -> tuple[dict, dict]:
    """Encode equal bytes by reference, other equal-layout bytes by XOR."""
    arrays, layout = {}, {}
    for name, value in canonical.items():
        value = np.ascontiguousarray(value)
        if value.dtype.kind not in 'biufc' or not np.isfinite(value).all():
            raise ValueError('Canonical replay labels must be finite numeric arrays')
        entry = dict(shape=list(value.shape), dtype=value.dtype.str)
        if name in base and base[name].shape == value.shape and base[name].dtype == value.dtype:
            difference = np.bitwise_xor(np.ascontiguousarray(base[name]).view(np.uint8),
                                        value.view(np.uint8))
            if np.any(difference):
                arrays[name] = difference
                entry['encoding'] = 'xor'
            else:
                entry['encoding'] = 'base'
        else:
            arrays[name] = value
            entry['encoding'] = 'literal'
        layout[name] = entry
    return arrays, layout


def decode_overlay(base: dict, arrays: dict, layout: dict, *, names=None) -> dict:
    """Reconstruct exact canonical values; callers authenticate both archives."""
    selected = set(layout) if names is None else set(names)
    if not selected or not selected.issubset(layout):
        raise ValueError('Unknown or empty replay stream selection')
    values = {}
    for name in selected:
        spec = layout[name]
        dtype, shape = np.dtype(spec['dtype']), tuple(spec['shape'])
        encoding = spec['encoding']
        if encoding in ('base', 'xor'):
            if name not in base or base[name].dtype != dtype or base[name].shape != shape:
                raise ValueError(f'Base replay layout mismatch: {name}')
            value = base[name]
            if encoding == 'xor':
                raw = np.ascontiguousarray(value).view(np.uint8)
                if arrays[name].dtype != np.uint8 or arrays[name].shape != raw.shape:
                    raise ValueError(f'XOR replay layout mismatch: {name}')
                value = np.bitwise_xor(raw, arrays[name]).view(dtype).reshape(shape)
        elif encoding == 'literal':
            value = arrays[name]
        else:
            raise ValueError('Unknown replay encoding')
        if value.dtype != dtype or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Canonical replay layout mismatch: {name}')
        values[name] = value
    return values
