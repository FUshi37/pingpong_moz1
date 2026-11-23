"""Audit WMJ inputs and real-checkpoint migration without starting training."""

from __future__ import annotations

import argparse
from datetime import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np

from .artifacts import write_json_unique
from .contracts import config_hash, file_hash, load_config, repository_root
from .resources import StorageBudget


def audit_actor(cfg: dict) -> dict:
    """Compare the actual released actor and its expanded input on fixed inputs."""
    import jax
    import jax.numpy as jnp
    from .migration import actor_mean, expand_actor_history, load_v321_actor

    original = load_v321_actor(repository_root(), cfg['lineage']['actor_source'])
    expanded = expand_actor_history(original)
    rng = np.random.default_rng(20261105)
    observations = rng.normal(0., .3, (1024, 72)).astype(np.float32)
    observations[:, 69:71] = [0., -.49]
    added = rng.normal(0., .3, (1024, cfg['policy']['actor_obs_dim']-72)).astype(np.float32)
    augmented = np.concatenate((observations, added), axis=-1)
    old = jax.jit(actor_mean)(original, jnp.asarray(observations))
    new = jax.jit(actor_mean)(expanded, jnp.asarray(augmented))
    jax.block_until_ready((old, new))
    max_error = float(np.max(np.abs(np.asarray(old) - np.asarray(new))))
    copied = all(np.array_equal(np.asarray(expanded[k][name])[:value.shape[0]], value)
                 for k, layer in original.items() for name, value in layer.items())
    passed = copied and max_error <= 5e-6
    if not passed:
        raise ValueError(f'Actual checkpoint actor migration failed: error={max_error}')
    return dict(status='PASS', samples=1024, seed=20261105, source='actual_V321_checkpoint',
        source_sha256=cfg['lineage']['actor_source']['sha256'], max_raw_action_abs_error=max_error,
        preregistered_abs_tolerance=5e-6, original_weight_prefix_exact=copied,
        actor_input_dim=cfg['policy']['actor_obs_dim'],
        added_rows_zero=bool(np.all(np.asarray(expanded['l1']['w'][72:]) == 0.)),
        output_activation='linear', backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True,
                        help='New report path under the registered WMJ output root')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    root = repository_root()
    resource = cfg['resources']
    budget = StorageBudget(root / resource['output_root'],
                           resource['max_new_bytes'], resource['min_free_bytes'])
    storage = budget.check(additional_bytes=1024 * 1024)
    config_path = args.config or root / 'pingpong_controller/tools/rl_2real/WMJ_V1_CONFIG.json'
    paths = [*Path(__file__).parent.glob('*.py'), config_path,
             root / 'AGENTS.md', root / 'TRAINING_CURRICULA.md']
    report = dict(
        schema='wmj_input_audit_v1', training_version='WMJ-V1',
        timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        command=shlex.join([sys.executable, '-m', 'world_model.preflight',
                           *(sys.argv[1:] if argv is None else argv)]),
        cwd=str(Path.cwd()), output_path=str(args.output.resolve()),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                             cwd=root, text=True).strip(),
        source_hashes={str(p.relative_to(root)): file_hash(p) for p in paths},
        config_hash=config_hash(cfg), storage=storage,
        dependencies={p: version(p) for p in ('jax', 'jaxlib', 'mujoco', 'numpy')},
        requested_jax_platform=os.environ.get('JAX_PLATFORMS'),
        authenticated_inputs=cfg['authenticated_inputs'],
        actor_migration=audit_actor(cfg),
        qualification_scope='authenticated_inputs_and_numerical_actor_migration_only',
        ready_for_training=False,
        remaining_qualification=[
            'full_native_state_restore_and_common_command_replay',
            'compiled_geometry_FK_and_domain_atlas',
            'WMJ_specific_topology_v2_CUDA_pairing',
            'global_dataset_and_stratified_world_model_readiness',
            'pure_imagination_gradients_and_end_to_end_restore',
            'GPU_performance_and_full_task_training',
        ],
        hardware_test_status='NOT_RUN',
    )
    digest = write_json_unique(args.output, report, budget)
    print(json.dumps(dict(input_audit='PASS', ready_for_training=False,
                          report=str(args.output.resolve()), sha256=digest), ensure_ascii=False))


if __name__ == '__main__':
    main()
