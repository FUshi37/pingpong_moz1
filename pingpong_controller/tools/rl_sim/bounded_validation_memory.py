"""Release compilation caches around sequential, opt-in validation probes."""

from collections.abc import Callable
import ctypes
import fcntl
import gc
import os
from pathlib import Path
import tempfile
from typing import TypeVar

import jax

Result = TypeVar("Result")


def release_caches() -> None:
    """Drop JAX staging caches and return unused glibc arenas to the host."""
    jax.clear_caches()
    gc.collect()
    library = ctypes.CDLL(None)
    if hasattr(library, "malloc_trim"):
        library.malloc_trim.argtypes = [ctypes.c_size_t]
        library.malloc_trim.restype = ctypes.c_int
        library.malloc_trim(0)


def release_startup_caches(
    *, enabled: bool, stage_update: int, initial_update: int,
) -> bool:
    """Release initialization executables once after the first complete update."""
    if not enabled or stage_update != initial_update + 1:
        return False
    release_caches()
    return True


def call_with_released_caches(
    evaluate: Callable[..., Result], *args, **kwargs,
) -> Result:
    """Keep identical inputs/results while avoiding resident probe caches."""
    lock_path = Path(os.environ.get(
        "V224_VALIDATION_LOCK_PATH",
        str(Path(tempfile.gettempdir()) / f"pingpong_v224_validation_{os.getuid()}.lock"),
    ))
    release_caches()
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return evaluate(*args, **kwargs)
        finally:
            release_caches()
