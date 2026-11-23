"""Pure validation may reserve its actual writes without changing storage limits."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))


@pytest.mark.parametrize('requested', [None, 16 * 2**20])
def test_validation_reservation_preserves_hard_storage_budget(monkeypatch, tmp_path, requested):
    import world_model.runtime_guard as module
    from world_model.resources import StorageLimitError
    calls = []
    class Budget:
        def __init__(self, root, maximum, minimum):
            self.root = root
            assert maximum == 6 * 2**30 and minimum == 8 * 2**30
        def check(self, *, additional_bytes):
            calls.append(additional_bytes)
            raise StorageLimitError('Hard quota reached')
    original = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda p, *a, **k:
        'MemAvailable: 22000000 kB\n' if str(p) == '/proc/meminfo' else original(p, *a, **k))
    monkeypatch.setattr(module, 'StorageBudget', Budget)
    monkeypatch.setattr(module, 'repository_root', lambda: tmp_path)
    monkeypatch.setenv('PP_MEMORY_SCOPE_ACTIVE', 'YES')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'test-gpu')
    monkeypatch.setenv('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    cfg = dict(resources=dict(allowed_gpu_uuids=['test-gpu'], startup_min_host_available_gib=13.5,
        output_root='outputs', max_new_bytes=6 * 2**30, min_free_bytes=8 * 2**30))
    kwargs = {} if requested is None else dict(startup_reserve_bytes=requested)
    with pytest.raises(StorageLimitError, match='Hard quota'):
        module.RunGuard(cfg, 'test-gpu', tmp_path / 'outputs/run', 60., **kwargs)
    assert calls == [32 * 2**20 if requested is None else requested]
