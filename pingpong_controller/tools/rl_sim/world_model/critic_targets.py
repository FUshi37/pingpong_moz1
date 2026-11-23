"""Read-only critic diagnostics with independent finite-return enumeration."""

from __future__ import annotations

import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import read_array_archive
from .contracts import authenticate, repository_root


def lambda_mixture_oracle(batch, next_values, lambda_value=.99):
    """Enumerate n-step returns and their mixture instead of a reverse recurrence."""
    reward, discount, trace, valid = map(np.asarray,
        (batch.reward, batch.discount, batch.trace_continue, batch.valid))
    next_values = np.asarray(next_values, float)
    result = np.zeros(reward.shape, np.float64)
    length, lanes = reward.shape
    for lane in range(lanes):
        for start in range(length):
            if not valid[start, lane]:
                continue
            accumulated, product, mixture_weight = 0., 1., 1.
            for end in range(start, length):
                accumulated += product * float(reward[end, lane])
                product *= float(discount[end, lane])
                n_step = accumulated + product * next_values[end, lane]
                last = (not trace[end, lane] or end + 1 == length
                        or not valid[end + 1, lane])
                result[start, lane] += mixture_weight * (1. if last else 1. - lambda_value) * n_step
                if last:
                    break
                mixture_weight *= lambda_value
    return result


def stats(values):
    values = np.asarray(values, float).reshape(-1)
    if not values.size:
        return dict(count=0)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite diagnostic values')
    return dict(count=int(values.size), mean=float(values.mean()), std=float(values.std()),
                minimum=float(values.min()), p50=float(np.median(values)),
                p95=float(np.quantile(values, .95)), maximum=float(values.max()))


def regression_stats(prediction, target):
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    if prediction.shape != target.shape:
        raise ValueError('Prediction/target layout differs')
    if not target.size:
        return dict(count=0)
    error = prediction - target
    variance = float(np.var(target))
    return dict(count=int(target.size), prediction=stats(prediction), target=stats(target),
                bias=float(error.mean()), rmse=float(np.sqrt(np.mean(error**2))),
                absolute_error=stats(np.abs(error)),
                explained_variance=float(1. - np.var(error) / variance) if variance > 1e-12 else None)


def checkpoint_critics_and_pool(reference, course_template):
    """Decode exact checkpoint leaves for inspection, never for candidate resume."""
    path = authenticate(repository_root(), reference)
    payload = json.loads(path.read_text())
    if payload['schema'] != 'wmj_v1_imagination_checkpoint':
        raise ValueError('Expected a full existing learner checkpoint')
    prefixes = ("['learner'].critic[", "['learner'].target_critic[",
                "['learner'].pool.environment")
    counters = ("['learner'].critic_optimizer.step", "['learner'].actor_updates",
                "['learner'].critic_updates")
    specs = {f'leaf_{i:06d}': row for i, row in enumerate(payload['state']['leaves'])
             if row['path'].startswith(prefixes) or row['path'] in counters}
    arrays = read_array_archive(path.parent / payload['archive_file'], payload['archive'], names=specs)
    by_path = {}
    for name, row in specs.items():
        value = np.asarray(arrays[name])
        if (list(value.shape) != row['shape'] or value.dtype.str != row['dtype']
                or hashlib.sha256(value.tobytes()).hexdigest() != row['sha256']
                or not np.isfinite(value).all()):
            raise ValueError('Checkpoint leaf failed integrity: ' + row['path'])
        by_path[row['path']] = value
    networks = {}
    for kind in ('critic', 'target_critic'):
        layers = []
        for i, width in enumerate(((2602, 256), (256, 256), (256, 1))):
            layer = {}
            for key in ('w', 'b'):
                value = by_path[f"['learner'].{kind}[{i}]['{key}']"]
                expected = width if key == 'w' else (width[1],)
                if value.shape != expected or value.dtype != np.float32:
                    raise ValueError('Critic shape/dtype changed')
                layer[key] = jnp.asarray(value)
            layers.append(layer)
        networks[kind] = tuple(layers)
    leaves, structure = jax.tree_util.tree_flatten_with_path(course_template)
    prefix = "['learner'].pool.environment"
    count = by_path[prefix + '.world.physics.q_rad'].shape[0]
    restored = []
    used = set()
    for key, example in leaves:
        name = prefix + jax.tree_util.keystr(key)
        value = by_path[name]
        expected = (count,) + example.shape[1:] if example.ndim else example.shape
        if value.shape != expected or value.dtype != np.asarray(example).dtype:
            raise ValueError('Saved pool layout differs: ' + name)
        restored.append(jnp.asarray(value))
        used.add(name)
    if used != {name for name in by_path if name.startswith(prefix)}:
        raise ValueError('Saved pool leaves missing or extra')
    return dict(networks, pool=jax.tree_util.tree_unflatten(structure, restored),
                metadata=payload['metadata'], runtime_signature=payload['runtime_signature'],
                counters={name.split('.')[-1]: int(by_path[name]) for name in counters},
                scope='Authenticated endpoint feature inspection only; no resumed simulation or learning')


def feature_summary(inputs, layout):
    inputs = np.asarray(inputs)
    if inputs.ndim != 2 or inputs.shape[1] != 2602 or not np.isfinite(inputs).all():
        raise ValueError('Invalid explicit critic features')
    rows = []
    for field in layout['fields']:
        block = inputs[:, field['start']:field['end']]
        rows.append(dict(name=field['name'], width=field['end']-field['start'],
                         abs_values=stats(np.abs(block)),
                         component_std=stats(block.std(axis=0)) if len(block) else dict(count=0)))
    return dict(rows=len(inputs), dimension=inputs.shape[1], blocks=rows)


def activation_summary(parameters, inputs):
    """Measure fixed input scale and tanh saturation without changing normalization."""
    value = jnp.asarray(inputs, jnp.float32)
    result = []
    for layer in parameters[:-1]:
        pre = value @ layer['w'] + layer['b']
        value = jnp.tanh(pre)
        result.append(dict(pre_rms=float(jnp.sqrt(jnp.mean(pre**2))),
                           saturated_fraction=float(jnp.mean(jnp.abs(value) > .99)),
                           mean_feature_std=float(jnp.std(value, axis=0).mean())))
    output = value @ parameters[-1]['w'] + parameters[-1]['b']
    return dict(hidden=result, prediction=stats(output),
                output_weight_l2=float(jnp.linalg.norm(parameters[-1]['w'])),
                output_bias=float(parameters[-1]['b'][0]))


def historical_learning(plan):
    rows = []
    for reference in plan['training_updates']:
        log = json.loads(authenticate(repository_root(), reference).read_text())
        steps = log['learning']['critic']
        rows.append(dict(reference=reference, update=log['update'], critic_updates=log['critic_updates'],
            valid=log['valid'], scheduled=log['scheduled_transitions'], terminated=log['terminated'],
            failure_fraction=log['terminated']/log['valid'],
            failure_bounds=log['physical_failure_bound_violations'],
            reward_terms=log['reward_terms_sum'], explained_variance=log['explained_variance_before'],
            critic_loss=[s['loss'] for s in steps], critic_gradient_norm=[s['gradient_norm'] for s in steps],
            critic_clipping_multiplier=[min(1., 1./max(s['gradient_norm'], 1e-12)) for s in steps],
            sampled_rows=[s['samples'] for s in steps], teacher_rows=log['learning']['teacher_rows']))
    rows.sort(key=lambda row: row['update'])
    if [row['update'] for row in rows] != list(range(1, 31)):
        raise ValueError('Expected all30actual historical updates')
    return dict(rows=rows, clipping_threshold=1., target_ema_rate=.01,
                target_ema_calls_per_actor=1, critic_updates_per_actor=4,
                initial_target_weight_after_30_ema=.99**30,
                scope='Actual historical training logs; current free-angle/reward/model candidate was never fitted')
