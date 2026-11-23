"""Shared bounded single-GPU worker resources for WMJ fitting and evaluation."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import threading
import time

from .audit_gpu_replay import _gpu_sample
from .contracts import repository_root
from .resources import StorageBudget, WallBudget


class RunGuard:
    """Honor supervisor signals at the next save boundary; never broaden limits."""

    def __init__(self,cfg,gpu_uuid,output_dir,max_wall_s,*,startup_reserve_bytes=32*2**20):
        resource=cfg['resources']
        if (os.environ.get('PP_MEMORY_SCOPE_ACTIVE')!='YES'
                or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu_uuid
                or os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')!='false'):
            raise ValueError('Use a cgroup-guarded single-UUID launcher without preallocation')
        if gpu_uuid not in resource['allowed_gpu_uuids'] or not 0<max_wall_s<=43200:
            raise ValueError('Unregistered GPU or invalid wall budget')
        available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                           if line.startswith('MemAvailable:')))*1024
        if available<resource['startup_min_host_available_gib']*2**30:
            raise RuntimeError('Insufficient startup host memory')
        self.budget=StorageBudget(repository_root()/resource['output_root'],resource['max_new_bytes'],resource['min_free_bytes'])
        # This is a write estimate, not the hard quota/free-space protection.
        # Pure frozen evaluation does not create a data or learner archive.
        if not isinstance(startup_reserve_bytes,int) or startup_reserve_bytes<2**20:
            raise ValueError('A positive explicit startup write reservation is required')
        self.budget.check(additional_bytes=startup_reserve_bytes)
        if not output_dir.resolve().is_relative_to(self.budget.root.resolve()):
            raise ValueError('Worker output must remain inside the WMJ storage budget')
        initial=_gpu_sample(gpu_uuid)
        if initial['other_compute_pids'] or initial['temperature_c']>=78 or initial['memory_free_mib']<12*1024:
            raise RuntimeError(f'GPU is unavailable: {initial}')
        self.gpu_uuid=gpu_uuid
        self.samples=[initial]
        self.started=time.monotonic()
        self.wall=WallBudget(self.started,max_wall_s)
        self.stop_reason=None
        self._closed=threading.Event()
        self._previous={}

    def __enter__(self):
        def requested(signum,frame):
            self.stop_reason=self.stop_reason or f'signal_{signal.Signals(signum).name}'
        self._previous={sig:signal.signal(sig,requested) for sig in (signal.SIGINT,signal.SIGTERM)}
        self._monitor=threading.Thread(target=self._watch,daemon=True)
        self._monitor.start()
        return self

    def _watch(self):
        while not self._closed.wait(5.):
            try:
                value=_gpu_sample(self.gpu_uuid)
                self.samples.append(value)
                if value['other_compute_pids'] or value['temperature_c']>=78:
                    self.stop_reason=self.stop_reason or 'gpu_resource_limit'
                if self.wall.remaining_s()<=10.:
                    self.stop_reason=self.stop_reason or 'wall_budget'
            except Exception as exc:
                self.stop_reason=self.stop_reason or f'resource_monitor_error:{exc}'

    def check(self):
        if self.stop_reason:
            raise InterruptedError(self.stop_reason)
        self.wall.check(reserve_s=10.)

    def __exit__(self,*exc):
        self._closed.set()
        self._monitor.join(timeout=6.)
        for sig,handler in self._previous.items():
            signal.signal(sig,handler)

    def report(self):
        return dict(wall_s=time.monotonic()-self.started,stop_reason=self.stop_reason,gpu_samples=self.samples)
