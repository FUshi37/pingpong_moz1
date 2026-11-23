"""Opt-in final-only checkpoint policy and synchronized runtime accounting."""
from pathlib import Path
from argparse import Namespace
import os
import time


def validate_ram_logging(args: Namespace) -> None:
    """Fail closed if final-only online logging could persist an SDK spool."""
    if not (getattr(args, 'final_model_only', False) and getattr(args, 'wandb', False)):
        return
    for name in ('WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR', 'TMPDIR'):
        value = os.environ.get(name)
        if not value or not Path(value).resolve().is_relative_to('/dev/shm'):
            raise ValueError(f'Final-only online logging requires {name} in /dev/shm')
    if args.wandb_mode != 'online' or not args.wandb_metrics_only:
        raise ValueError('Final-only logging requires online, metrics-only W&B')


def checkpoint_destination(path: Path, args: Namespace, extra: dict | None) -> Path | None:
    """Permit one final native-validated model; leave historical saving unchanged."""
    if not getattr(args, 'final_model_only', False):
        return path
    extra = extra or {}
    index = getattr(args, 'final_converged_stage_index', None)
    name = getattr(args, 'final_converged_stage_name', None)
    if (index is None or name is None or extra.get('stage_index') != index
            or extra.get('stage_name') != name or path.name != f'{index:02d}_{name}.pkl'):
        return None
    row = extra.get('last_row') or {}
    if row.get('convergence/stage_converged') != 1. or row.get('advance_eval/passed') != 1.:
        return None
    if row.get('retention_eval/required', 0.) and row.get('retention_eval/passed') != 1.:
        return None
    target = args.save_dir / 'mjx_curriculum_final.pkl'
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite final model: {target}')
    return target


def final_checkpoint_metadata(extra: dict) -> dict:
    """Keep final identity/acceptance only; exclude rolling histories and replay data."""
    keys = ('stage_index', 'stage_name', 'stage_update', 'global_update', 'last_row')
    return {key: extra[key] for key in keys} | {'all_stages_converged': True}


class RunTiming:
    """Account wall time including compilation; asynchronous kernels need barriers."""

    def __init__(self, start: float | None = None):
        self.start = time.perf_counter() if start is None else start
        self.stage_start = self.start
        self.stage_setup_s = 0.
        self.rollout = self.ppo = self.evaluation = 0.
        self.previous_log_s = 0.

    def start_stage(self, index: int, now: float | None = None) -> None:
        self.stage_index = index
        self.stage_start = time.perf_counter() if now is None else now
        self.stage_setup_s = 0.

    def record(self, *, update_start: float, rollout_s: float, ppo_s: float,
               evaluation_s: float, global_step: int, first_update: bool,
               now: float | None = None) -> dict[str, float]:
        now = time.perf_counter() if now is None else now
        if first_update:
            self.stage_setup_s = update_start - self.stage_start
        self.rollout += rollout_s
        self.ppo += ppo_s
        self.evaluation += evaluation_s
        wall = now - self.start
        return {
            'perf/run_wall_s': wall,
            'perf/run_wall_hours': wall / 3600.,
            'perf/stage_wall_s': now - self.stage_start,
            'perf/stage_setup_s': self.stage_setup_s,
            'perf/rollout_s': rollout_s,
            'perf/ppo_s': ppo_s,
            'perf/evaluation_s': evaluation_s,
            'perf/update_wall_s': now - update_start,
            'perf/update_other_s': max(0., now-update_start-rollout_s-ppo_s-evaluation_s),
            'perf/rollout_s_total': self.rollout,
            'perf/ppo_s_total': self.ppo,
            'perf/evaluation_s_total': self.evaluation,
            'perf/other_wall_s_total': wall-self.rollout-self.ppo-self.evaluation,
            'perf/effective_sps': global_step / max(wall, 1e-9),
            'perf/first_update_includes_compile': float(first_update),
            'perf/previous_wandb_log_s': self.previous_log_s,
        }
