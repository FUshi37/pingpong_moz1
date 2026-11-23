"""WMJ-only PSI headroom gate retains the independent capacity guards."""

import json
from pathlib import Path
import sys

import pytest

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))
import training_memory_scope as guard


@pytest.mark.parametrize('current,available,expected', [
    (1.0, 20.0, None),
    (7.999, 8.001, None),
    (8.0, 20.0, 'sustained_memory_pressure'),
    (1.0, 8.0, 'sustained_memory_pressure'),
    (2.966, 11.113, None),
])
def test_psi_gate_requires_either_headroom_boundary(current, available, expected):
    pressure = {'full': {'avg10': 8.0}}
    kwargs = dict(memory_pressure=pressure, pressure_streak=3)
    assert guard.pressure_reason(
        int(current * guard.GIB), int(available * guard.GIB),
        wmj_v1_conditional_psi=True, **kwargs,
    ) == expected
    # Every pre-existing caller retains the independent PSI stop.
    assert guard.pressure_reason(
        int(current * guard.GIB), int(available * guard.GIB), **kwargs,
    ) == 'sustained_memory_pressure'


def test_capacity_host_and_high_events_remain_independent_of_psi():
    kwargs = dict(wmj_v1_conditional_psi=True,
                  scope_stop_bytes=int(9.7 * guard.GIB),
                  host_stop_streak_samples=2)
    assert guard.pressure_reason(
        int(9.7 * guard.GIB), 20 * guard.GIB,
        scope_stop_streak=3, **kwargs,
    ) == 'scope_memory_stop'
    assert guard.pressure_reason(
        guard.GIB, int(5.99 * guard.GIB), host_stop_streak=2, **kwargs,
    ) == 'host_available_low'
    assert guard.pressure_reason(
        guard.GIB, 20 * guard.GIB, memory_high_events_delta=1, **kwargs,
    ) == 'scope_memory_high_event'
    assert guard.pressure_reason(
        8 * guard.GIB, 20 * guard.GIB,
        memory_pressure={'full': {'avg10': 8.0}}, pressure_streak=2, **kwargs,
    ) is None


def test_wmj_flag_survives_systemd_boundary_and_is_explicit(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(guard, 'run_named_scope',
                        lambda command, unit: captured.update(command=command) or 0)
    monkeypatch.setattr(sys, 'argv', [
        'guard', '--wmj-v1-conditional-psi', '--memory-high-gib', '10',
        '--memory-max-gib', '11', '--memory-stop-gib', '9.7',
        '--host-stop-available-gib', '6', '--host-stop-streak-samples', '2',
        '--log-dir', str(tmp_path / 'scope'), '--', '/bin/true',
    ])
    assert guard.main() == 0
    assert captured['command'].count('--wmj-v1-conditional-psi') == 1
    for name in ('run_wmj_v1_fit_robot.sh', 'run_wmj_v1_collect.sh'):
        assert '--wmj-v1-conditional-psi' in (SIM / name).read_text()
    cfg = json.loads((SIM.parent / 'rl_2real/WMJ_V1_CONFIG.json').read_text())
    assert cfg['resources']['psi_stop_policy'] == {
        'mode': 'wmj_v1_conditional', 'scope_at_least_gib': 8.0,
        'host_available_at_most_gib': 8.0, 'otherwise': 'log_warning',
    }
    assert cfg['resources']['startup_min_host_available_gib'] == 13.5
