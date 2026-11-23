"""Lossless numerical seeds bound to an authenticated static context.

The caller stores static MJX model/geometry once per context, removes that
model from the per-seed tree, and supplies a matching template at restore.
Every remaining leaf, including scalar ring indices, RNG, contact history and
MJX warm-start data, is retained. No q/dq-only state approximation is made.
"""

from __future__ import annotations

import hashlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _leaf_signature(path, value: np.ndarray) -> dict:
    return dict(path=jax.tree_util.keystr(path), shape=list(value.shape), dtype=value.dtype.str,
                sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def numeric_tree_digest(tree) -> str:
    """Hash actual episode context leaves, including array shape, type and path."""
    digest = hashlib.sha256()
    leaves, _ = jax.tree_util.tree_flatten_with_path(jax.device_get(tree))
    for path, value in leaves:
        array = np.asarray(value)
        if array.dtype.kind not in 'biufc':
            raise ValueError('Numeric context leaves are required')
        digest.update(jax.tree_util.keystr(path).encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def pack_numeric_state(state: Any, *, context_sha256: str) -> tuple[dict, dict]:
    """Pack an exact full seed into named non-object arrays and integrity metadata."""
    arrays, signatures = {}, []
    leaves, _ = jax.tree_util.tree_flatten_with_path(jax.device_get(state))
    for index, (path, leaf) in enumerate(leaves):
        value = np.asarray(leaf)
        if value.dtype.kind not in 'biufc' or not np.isfinite(value).all():
            raise ValueError(f'Nonfinite or unsupported seed leaf {jax.tree_util.keystr(path)}')
        arrays[f'leaf_{index:06d}'] = value.copy()
        signatures.append(_leaf_signature(path, value))
    return arrays, dict(schema='wmj_numeric_seed_v1', context_sha256=context_sha256,
                        leaves=signatures, leaf_count=len(signatures))


def restore_numeric_state(template, arrays: dict, manifest: dict, *, context_sha256: str):
    """Verify every leaf before restoring into the same authenticated context."""
    if (manifest.get('schema') != 'wmj_numeric_seed_v1'
            or manifest.get('context_sha256') != context_sha256):
        raise ValueError('Seed context identity mismatch')
    leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    if (manifest['leaf_count'] != len(leaves) or len(manifest['leaves']) != len(leaves)
            or set(arrays) != {f'leaf_{i:06d}' for i in range(len(leaves))}):
        raise ValueError('Seed layout/leaf count mismatch')
    restored = []
    for index, ((path, example), expected) in enumerate(zip(leaves, manifest['leaves'])):
        value = np.asarray(arrays[f'leaf_{index:06d}'])
        if (_leaf_signature(path, value) != expected
                or value.shape != np.shape(example) or value.dtype != np.asarray(example).dtype):
            raise ValueError(f'Seed integrity/layout mismatch at {jax.tree_util.keystr(path)}')
        restored.append(jnp.asarray(value))
    return jax.tree_util.tree_unflatten(treedef, restored)


def concatenate_phase_group(states: list):
    """Join batched seeds only when every batch-shared scalar clock is identical."""
    if not states:
        raise ValueError('Cannot concatenate an empty phase group')
    reference = jax.tree.structure(states[0])
    if any(jax.tree.structure(s) != reference for s in states[1:]):
        raise ValueError('Seed context/tree mismatch')

    def join(*values):
        first = np.asarray(values[0])
        if first.ndim == 0:
            if any(not np.array_equal(first, np.asarray(v)) for v in values[1:]):
                raise ValueError('Incompatible shared buffer phase; group seeds by phase first')
            return jnp.asarray(first)
        if any(np.shape(v)[1:] != first.shape[1:] or np.asarray(v).dtype != first.dtype
               for v in values[1:]):
            raise ValueError('Seed lane shapes/dtypes disagree')
        return jnp.concatenate(values, axis=0)

    return jax.tree.map(join, *states)
