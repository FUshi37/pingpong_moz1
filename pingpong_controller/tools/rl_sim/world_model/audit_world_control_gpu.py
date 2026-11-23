"""Guarded GPU counterpart of the full-seed known-control replay audit."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import jax

from .artifacts import write_json_unique
from .audit_world_control import audit
from .contracts import config_hash, file_hash, load_config
from .runtime_guard import RunGuard
from .support_data import SupportReplayReader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support-replay', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--seed', type=int, default=2026110904)
    parser.add_argument('--max-wall-s', type=float, default=900.)
    parser.add_argument('--component-diagnostics', action='store_true')
    args = parser.parse_args(argv)
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, args.max_wall_s) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        paths = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_native_seed_control_gpu_audit_v1', training_version='WMJ-V1',
            status='INCOMPLETE', config=cfg, config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), invocation=sys.argv,
            source_sha256={p.name:file_hash(p) for p in paths},
            scope='native_recorded_q_feedback_not_learned_physics',
            actor_updates=0, critic_updates=0, world_model_ready=False)
        write_json_unique(args.output_dir/'manifest.json',
            {**report, 'source_snapshot':{p.name:p.read_text() for p in paths}}, guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA GPU is required')
            report['devices'] = [str(device) for device in jax.devices()]
            reader = SupportReplayReader(args.support_replay, guard.budget, max_bytes=0)
            result = audit(cfg, reader, seed=args.seed, guard=guard, component_diagnostics=args.component_diagnostics)
            tolerance = dict(reference_q_rad=.0001, accepted_normalized=.002,
                delayed_q_rad=.0001, delayed_dq_rad_s=.001, delayed_ddq_rad_s2=.02)
            passed = all(value['nonfinite'] == 0 and value['maximum'] <= tolerance[key]
                         for key,value in result['overall'].items())
            report.update(validation=result, dataset_sha256=reader.identity,
                reference_maximum_tolerances=tolerance,
                status='CONTROL_REPLAY_PASS' if passed else 'CONTROL_REPLAY_NOT_PASSED')
        except InterruptedError as exc:
            report.update(status='INTERRUPTED', error=str(exc))
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            digest = write_json_unique(args.output_dir/'result.json', report, guard.budget)
            print(dict(path=str(args.output_dir), sha256=digest, status=report['status']), flush=True)


if __name__ == '__main__':
    main()
