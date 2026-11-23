"""Frozen feature scale, phase-matched coverage and value-input derivatives."""

from __future__ import annotations

import gzip
import io
import json
import jax
import jax.numpy as jnp
import numpy as np

from .actor_critic import frozen_value


STAT_NAMES = ('mean', 'std', 'minimum', 'maximum', 'p01', 'p99')
ACTIVE_STD = 1e-6


def sequential_storage_reservation(arrays: dict, report: dict) -> dict:
    """Reserve the actual peak of two sequential atomic writes, plus log slack."""
    raw = sum(np.asarray(a).nbytes for a in arrays.values())
    archive_writer = raw+raw//100+len(arrays)*8192+65536
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    stored = buffer.tell()
    report_bytes = len(gzip.compress(json.dumps(report,separators=(',',':'),allow_nan=False).encode()))
    after_archive = stored+2*report_bytes+32768
    return dict(archive_writer_bytes=archive_writer, archive_compressed_bytes=stored,
        report_after_archive_bytes=after_archive, log_slack_bytes=32768,
        required_peak_bytes=max(archive_writer,after_archive)+32768)


def input_groups(layout: dict) -> dict:
    """An exhaustive disjoint reporting partition; never change network inputs."""
    groups = {name:[] for name in ('actor', 'physical', 'task', 'control')}
    for field in layout['fields']:
        name = field['name']
        if name == 'actor_observation':
            group = 'actor'
        elif name in ('joint_q', 'joint_dq') or name.startswith(('ball_', 'events.', 'pulse.')):
            group = 'physical'
        elif name == 'episode.elapsed' or name.startswith(('task.', 'sequence.')):
            group = 'task'
        elif name in ('reference_q', 'previous_accepted_action') or name.startswith(
                ('rmp.', 'output_delay.', 'context.', 'camera.')):
            group = 'control'
        else:
            raise ValueError('Unregistered diagnostic feature: '+name)
        groups[group].extend(range(field['start'], field['end']))
    values = np.concatenate([np.asarray(v, np.int32) for v in groups.values()])
    if not np.array_equal(np.sort(values), np.arange(layout['dimension'])):
        raise ValueError('Diagnostic groups must cover every feature exactly once')
    return {k:np.asarray(v, np.int32) for k,v in groups.items()}


def component_stats(inputs: np.ndarray) -> np.ndarray:
    value = np.asarray(inputs, np.float64)
    if value.ndim != 2 or not len(value) or not np.isfinite(value).all():
        raise ValueError('Finite feature rows required')
    return np.stack((value.mean(0), value.std(0), value.min(0), value.max(0),
                     np.quantile(value, .01, axis=0), np.quantile(value, .99, axis=0))).astype(np.float32)


def distribution_summary(inputs: np.ndarray, training_stats: np.ndarray, indices: np.ndarray) -> dict:
    """Training-only normalization, with constant dimensions reported separately."""
    value = np.asarray(inputs)[:, indices]
    mean, std, lo, hi, q01, q99 = np.asarray(training_stats)[:, indices]
    active = std > ACTIVE_STD
    constant_drift = np.abs(value[:, ~active]-mean[~active]) > ACTIVE_STD
    shift = np.abs(value.mean(0)[active]-mean[active])/std[active]
    return dict(width=len(indices), active_components=int(active.sum()),
        constant_components=int((~active).sum()), rms=float(np.sqrt(np.mean(value.astype(float)**2))),
        abs_p99=float(np.quantile(np.abs(value), .99)), abs_max=float(np.max(np.abs(value))),
        outside_training_range_fraction=float(np.mean((value < lo-ACTIVE_STD) | (value > hi+ACTIVE_STD))),
        outside_training_01_99_fraction=float(np.mean((value < q01-ACTIVE_STD) | (value > q99+ACTIVE_STD))),
        constant_drift_fraction=float(constant_drift.mean()) if constant_drift.size else 0.,
        mean_shift_std_median=float(np.median(shift)) if shift.size else None,
        mean_shift_std_p95=float(np.quantile(shift, .95)) if shift.size else None,
        mean_shift_std_max=float(shift.max()) if shift.size else None)


@jax.jit
def input_gradients(parameters, inputs):
    """Return dV/d(encoded feature), with network parameters explicitly frozen."""
    return jax.vmap(jax.grad(lambda x:frozen_value(parameters, x)))(inputs)


@jax.jit
def _nearest_batch(reference, train_source, train_phase, x, source, phase):
    squared = jnp.maximum(jnp.sum(x*x, axis=1)[:, None]
        + jnp.sum(reference*reference, axis=1)[None, :]
        - 2*(x @ reference.T), 0.)/reference.shape[1]
    valid = (source[:, None] != train_source[None, :]) & (phase[:, None] == train_phase[None, :])
    masked = jnp.where(valid, squared, jnp.inf)
    index = jnp.argmin(masked, axis=1)
    distance = jnp.sqrt(jnp.min(masked, axis=1))
    return distance, jnp.where(jnp.isfinite(distance), index, -1)


def nearest_training_rows(train: np.ndarray, query: np.ndarray,
                          train_source: np.ndarray, query_source: np.ndarray,
                          train_phase: np.ndarray, query_phase: np.ndarray,
                          active: np.ndarray, scale: np.ndarray, *, chunk: int = 128) -> tuple:
    """Nearest standardized RMS within phase, excluding the same source root."""
    if not np.any(active):
        return np.full(len(query), np.inf), np.full(len(query), -1, np.int32)
    mean = np.asarray(train, np.float64)[:, active].mean(0).astype(np.float32)
    normalized_train = (np.asarray(train)[:, active]-mean)/scale[active]
    normalized_query = (np.asarray(query)[:, active]-mean)/scale[active]
    reference = jnp.asarray(normalized_train, jnp.float32)
    train_source, train_phase = jnp.asarray(train_source), jnp.asarray(train_phase)

    distances, indices = [], []
    for start in range(0, len(query), chunk):
        n = min(chunk, len(query)-start)
        x = np.zeros((chunk, normalized_query.shape[1]), np.float32)
        source = np.full(chunk, -10000, np.int32); phase = np.full(chunk, -10000, np.int32)
        x[:n] = normalized_query[start:start+n]
        source[:n] = query_source[start:start+n]; phase[:n] = query_phase[start:start+n]
        distance, index = _nearest_batch(reference, train_source, train_phase,
            jnp.asarray(x), jnp.asarray(source), jnp.asarray(phase))
        distances.append(np.asarray(distance)[:n]); indices.append(np.asarray(index)[:n])
    distance, index = np.concatenate(distances).astype(float), np.concatenate(indices)
    present = index >= 0
    delta = (normalized_query[present].astype(float)
             - normalized_train[index[present]].astype(float))
    distance[present] = np.sqrt(np.mean(delta*delta, axis=1))
    return distance, index
