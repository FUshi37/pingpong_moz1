"""Unique, integrity-checked evidence writes inside the shared storage cap."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from .contracts import file_hash
from .resources import StorageBudget


def discard_file_cache(path: Path) -> None:
    """Release cache of this closed, fsynced artifact without modifying it.

    Large teacher archives otherwise charge duplicate filesystem cache to the
    training cgroup after the caller already owns their CPU arrays. This is a
    per-file hint, never a global cache drop or a relaxation of memory limits.
    """
    if not hasattr(os, 'posix_fadvise'):
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


def write_json_unique(path: Path, payload: dict, budget: StorageBudget) -> str:
    """Atomically publish a new JSON artifact without replacing any prior file."""
    path = path.absolute()
    if not path.resolve().is_relative_to(budget.root.resolve()):
        raise ValueError('Output must remain inside the registered WMJ budget root')
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Existing evidence is protected: {path}')
    data = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    budget.check(additional_bytes=2 * len(data) + 8192)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.wmj-write-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path = Path(temporary)
        json.loads(temporary_path.read_text(encoding='utf-8'))
        digest = file_hash(temporary_path)
        # link is atomic and refuses an existing destination, unlike replace.
        os.link(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return digest
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_array_archive_unique(path: Path, arrays: dict, budget: StorageBudget) -> dict:
    """Publish a compressed numeric chunk/seed without pickle or overwriting.

    Missing labels must be represented by finite placeholders plus validity
    masks. Callers save continuous streams and complete seeds, not duplicated
    flattened actor histories. Arrays remain on CPU; no full dataset goes to
    the GPU. The returned manifest is committed by the dataset index writer.
    """
    path = path.absolute()
    if not path.resolve().is_relative_to(budget.root.resolve()):
        raise ValueError('Output must remain inside the registered WMJ budget root')
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Existing evidence is protected: {path}')
    if not arrays:
        raise ValueError('A numeric archive must contain at least one array')
    values = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name):
            raise ValueError('Archive array names must be simple identifiers')
        array = np.asarray(value)
        if array.dtype.kind not in 'biufc':
            raise ValueError('Only numeric arrays are allowed; pickle is forbidden')
        if not np.isfinite(array).all():
            raise ValueError('Archive values must be finite; use validity masks for missing labels')
        values[name] = array
    uncompressed = sum(a.nbytes for a in values.values())
    # Deflate can slightly expand incompressible bytes. Reserve raw data plus
    # expansion, NPY/ZIP headers and allocated filesystem blocks before writing.
    reservation = uncompressed + uncompressed//100 + len(values)*8192 + 65536
    budget.check(additional_bytes=reservation)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.wmj-write-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path = Path(temporary)
        # Hash the completed bytes, then atomically publish with no replacement.
        digest = file_hash(temporary_path)
        os.link(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        discard_file_cache(path)
        return dict(schema='wmj_numeric_arrays_v1', sha256=digest,
                    stored_bytes=path.stat().st_size, uncompressed_bytes=uncompressed,
                    arrays={name: dict(shape=list(a.shape), dtype=a.dtype.str)
                            for name, a in values.items()})
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_array_archive(path: Path, manifest: dict, *, names=None,
                       release_cache=True) -> dict[str, np.ndarray]:
    """Authenticate the entire archive, decompressing only requested streams.

    Fitting a single dynamics component need not materialize every camera,
    contact and task stream. Partial loading never weakens file authentication.
    Repeated input readers may retain OS cache within their resource envelope.
    """
    if (manifest.get('schema') != 'wmj_numeric_arrays_v1'
            or file_hash(path) != manifest.get('sha256')):
        raise ValueError('Numeric archive integrity mismatch')
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(manifest['arrays']):
            raise ValueError('Numeric archive array layout mismatch')
        selected = set(manifest['arrays']) if names is None else set(names)
        if not selected or not selected.issubset(manifest['arrays']):
            raise ValueError('Unknown or empty archive stream selection')
        values = {}
        for name, expected in manifest['arrays'].items():
            if name not in selected:
                continue
            array = archive[name]
            if (array.dtype.kind not in 'biufc' or list(array.shape) != expected['shape']
                    or array.dtype.str != expected['dtype'] or not np.isfinite(array).all()):
                raise ValueError(f'Numeric archive array integrity mismatch: {name}')
            values[name] = array
    if release_cache:
        discard_file_cache(path)
    return values
