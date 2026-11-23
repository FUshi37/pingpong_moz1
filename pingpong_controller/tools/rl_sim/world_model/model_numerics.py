"""Explicit learned-component precision, separate from native control arithmetic."""

from contextlib import nullcontext

import jax


def precision_scope(value):
    """Preserve ambient arithmetic for legacy models; scope full FP32 explicitly."""
    if value is None:
        return nullcontext()
    if value != 'highest':
        raise ValueError('Only legacy/default or full FP32 learned-component precision is supported')
    return jax.default_matmul_precision(value)


def validate_precision_transition(previous, current, *, start_stage):
    """A new explicit stage may change only legacy precision to full FP32."""
    old, new = dict(previous), dict(current)
    before, after = old.pop('matmul_precision',None), new.pop('matmul_precision',None)
    precision_scope(before)
    precision_scope(after)
    if old != new:
        raise ValueError('Precision stage cannot change another fitting recipe field')
    if before == after and not start_stage:
        return
    if not (start_stage and before is None and after == 'highest'):
        raise ValueError('Changed precision requires an explicit new full-FP32 stage')
