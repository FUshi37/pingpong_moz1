"""Non-destructive storage and wall-clock limits shared by WMJ commands."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import time


class StorageLimitError(RuntimeError):
    """The registered budget cannot accommodate the next write."""


@dataclass(frozen=True)
class StorageBudget:
    root: Path
    max_bytes: int
    min_free_bytes: int

    def used_bytes(self) -> int:
        """Count allocated evidence under the dedicated WMJ output root."""
        if self.root.is_symlink():
            raise StorageLimitError('Output budget root must not be a symlink')
        total = 0
        if not self.root.exists():
            return total
        for directory, names, files in os.walk(self.root, followlinks=False):
            for name in names + files:
                path = Path(directory) / name
                if path.is_symlink():
                    raise StorageLimitError(f'Symlink escapes storage accounting: {path}')
            for name in files:
                stat = (Path(directory) / name).stat()
                total += max(stat.st_size, stat.st_blocks * 512)
        return total

    def check(self, *, additional_bytes: int = 0, free_bytes: int | None = None) -> dict:
        if additional_bytes < 0:
            raise ValueError('Write reservation cannot be negative')
        used = self.used_bytes()
        probe = self.root
        while not probe.exists():
            probe = probe.parent
        free = shutil.disk_usage(probe).free if free_bytes is None else free_bytes
        if used + additional_bytes > self.max_bytes:
            raise StorageLimitError('WMJ output budget would be exceeded; ask before collecting more')
        if free - additional_bytes < self.min_free_bytes:
            raise StorageLimitError('Required free disk reserve would be consumed; ask before continuing')
        return dict(used_bytes=used, free_bytes=free, reserved_bytes=additional_bytes)


@dataclass(frozen=True)
class WallBudget:
    started_monotonic_s: float
    max_wall_s: float

    def remaining_s(self) -> float:
        return max(0.0, self.max_wall_s - (time.monotonic() - self.started_monotonic_s))

    def check(self, reserve_s: float = 0.0) -> None:
        if self.remaining_s() <= reserve_s:
            raise TimeoutError('Total run wall-clock budget exhausted; save without advancing course')
