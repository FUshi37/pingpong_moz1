"""Private follow-up: paired projection branches and auxiliary input rounding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .contracts import file_hash
from .module_gradient_tools import distribution, probe_direction


def compare_precisions(fp32: tuple, fp64: tuple) -> dict:
    """Compare the same source lanes, excluding changed branches explicitly."""
    a, da, ma = map(np.asarray, fp32)
    b, db, mb = map(np.asarray, fp64)
    n = len(a)
    changed = np.any(ma.reshape(n, -1) != mb.reshape(n, -1), axis=-1)
    norm_a, norm_b = np.linalg.norm(da, axis=-1), np.linalg.norm(db, axis=-1)
    magnitude = np.maximum(norm_a, norm_b)
    informative = magnitude > 1e-5
    common = ~changed & informative
    relative = np.linalg.norm(da-db, axis=-1)/np.maximum(magnitude, 1e-8)
    return dict(changed_branch_indices=np.flatnonzero(changed).tolist(),
        common_informative_indices=np.flatnonzero(common).tolist(),
        fp32_modes=ma.astype(int).tolist(), fp64_modes=mb.astype(int).tolist(),
        fp32_tangent_norm=norm_a.tolist(), fp64_tangent_norm=norm_b.tolist(),
        centered_output_abs_difference=distribution(np.abs(a-b)),
        common_branch_tangent_relative_error=distribution(relative[common]))


def input_rounding_rows(plus: Any, minus: Any, direction: Any,
                        scale: Any, epsilon: float) -> list[dict]:
    """Expose loss of raw-feature perturbations relative to their training scale."""
    plus, minus, direction, scale = map(lambda x: np.asarray(x, float),
                                        (plus, minus, direction, scale))
    observed = (plus-minus)/(2*epsilon*scale)
    return [dict(feature_index=i,
        unchanged_fraction=float(np.mean(plus[:, i] == minus[:, i])),
        normalized_direction_error_rms=float(np.sqrt(np.mean((observed[:, i]-direction[:, i])**2))),
        normalized_direction_rms=float(np.sqrt(np.mean(direction[:, i]**2))))
        for i in range(plus.shape[1])]


def run_details(data: dict, plan: dict, reference: Path, report: dict,
                guard: Any) -> None:
    """Reuse exactly the first probe sources and frozen models; no rollout edits."""
    from . import neural
    from .audit_module_gradients import checkpoint, component_functions, normalized_directions
    from .robot_support_model import support_features

    previous = json.loads(reference.read_text())
    if (previous.get('status') != 'FROZEN_MODULE_GRADIENT_DIAGNOSIS_COMPLETE'
            or previous['plan'] != report['plan']
            or previous['source_initial_state_sha256'] != report['source_initial_state_sha256']
            or previous['capture'] != report['capture']
            or previous['parameters_before_sha256'] != report['parameters_before_sha256']):
        raise ValueError('Follow-up must reproduce the same original module capture')
    report['followup_from'] = dict(path=str(reference), sha256=file_hash(reference))
    pairs = {}
    for high in (False, True):
        with jax.enable_x64(high):
            functions = component_functions(data, high_precision=high, seed=plan['seed'])
            for name in ('projection', 'control8steps_frozen_encoder'):
                checkpoint(guard)
                function = functions[name]
                dtype = jnp.float64 if high else jnp.float32
                forward = jax.jit(function)
                derivative = jax.jit(lambda a: jax.jvp(function, (a,),
                                      (jnp.ones_like(a),), has_aux=True))
                zero = jnp.asarray(0., dtype)
                value, modes = jax.device_get(forward(zero))
                primal, tangent, derivative_modes = jax.device_get(derivative(zero))
                if not np.array_equal(modes, derivative_modes):
                    raise ValueError('Changed projection branch in derivative primal')
                pairs.setdefault(name, []).append((value, tangent, modes))
                report.setdefault('followup_primal_difference', {})[f'{high}/{name}'] = distribution(value-primal)
                forward.clear_cache()
                derivative.clear_cache()
        jax.clear_caches()
    report['paired_precisions'] = {name: compare_precisions(*pair) for name, pair in pairs.items()}
    for name, result in report['paired_precisions'].items():
        result['changed_branch_bank_rows'] = [plan['source_indices'][i]
                                              for i in result['changed_branch_indices']]
        print(f'[DETAIL] {name} changed_precision_branches={result["changed_branch_indices"]}', flush=True)

    cp = data['contact']
    features = support_features(cp['q'], cp['dq'], cp['commands'])
    scale = data['auxiliary']['scale']
    parameters = data['auxiliary']['parameters']
    direction = jnp.asarray(normalized_directions(plan['seed']+2, features.shape))
    names = ([f'RightArm_q_{i}' for i in range(7)]
             +[f'RightArm_dq_{i}' for i in range(7)]
             +[f'delayed_q_minus_q_{i}' for i in range(7)]
             +[f'delayed_dq_{i}' for i in range(7)]
             +[f'delayed_ddq_{i}' for i in range(7)]
             +[f'LegWaist_q_{i}' for i in range(6)]
             +[f'LegWaist_dq_{i}' for i in range(6)])
    report['auxiliary_input_rounding'] = {}
    @jax.jit
    def feature_pair(epsilon):
        delta = epsilon*direction*scale.input_scale
        return features+delta, features-delta
    for epsilon in (.0001, .001, .01):
        plus, minus = jax.device_get(feature_pair(jnp.asarray(epsilon, jnp.float32)))
        rows = input_rounding_rows(plus, minus, direction, scale.input_scale, epsilon)
        for row, name in zip(rows, names):
            row['feature_name'] = name
            row['input_scale'] = float(scale.input_scale[row['feature_index']])
        report['auxiliary_input_rounding'][str(epsilon)] = rows
    feature_pair.clear_cache()

    # This bypass and mixed arithmetic exist only in diagnostic closures.
    normalized = (features-scale.input_mean)/scale.input_scale
    def normalized_mlp(alpha):
        with jax.default_matmul_precision('highest'):
            y = neural.mlp(parameters, normalized+alpha*direction)
        return y, jnp.zeros((len(features), 1), jnp.int32)
    report['auxiliary_isolation'] = dict(normalized_input_fp32=probe_direction(
        normalized_mlp, plan['epsilons'], dtype=jnp.float32, check=lambda: checkpoint(guard)))
    with jax.enable_x64(True):
        feature64, mean64, scale64, direction64 = [jnp.asarray(v, jnp.float64)
            for v in (features, scale.input_mean, scale.input_scale, direction)]
        def normalize64_mlp32(alpha):
            z = (feature64+alpha*direction64*scale64-mean64)/scale64
            with jax.default_matmul_precision('highest'):
                y = neural.mlp(parameters, z)  # The original MLP explicitly casts to FP32.
            return y, jnp.zeros((len(features), 1), jnp.int32)
        report['auxiliary_isolation']['normalization64_mlp32'] = probe_direction(
            normalize64_mlp32, plan['epsilons'], dtype=jnp.float64,
            check=lambda: checkpoint(guard))
    print('[DETAIL] auxiliary input rounding and private normalization isolation complete', flush=True)
    report['status'] = 'FROZEN_MODULE_GRADIENT_DETAILS_COMPLETE'
