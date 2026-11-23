"""The approved resume changes resource limits, never the learning contract."""

import copy
from pathlib import Path
import sys

import pytest

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))


def test_resume_psi_is_explicit_and_leaves_capacity_guards_intact(monkeypatch):
    import training_memory_scope as guard
    g = guard.GIB
    pressure = dict(memory_pressure={'full': {'avg10': 5.48}}, pressure_streak=3)
    old = dict(wmj_v1_conditional_psi=True, **pressure)
    new = dict(old, wmj_v1_fixed_critic_resume_psi=True)
    assert guard.pressure_reason(int(3.88*g), int(10.66*g), **old) is None
    # Reproduce the historical13.5GiB default and its isolated8GiB override.
    monkeypatch.setattr(guard, 'WMJ_V1_PSI_HOST_AVAILABLE_BYTES', int(13.5*g))
    assert guard.pressure_reason(int(3.88*g), int(10.66*g), **old) == 'sustained_memory_pressure'
    assert guard.pressure_reason(int(3.88*g), int(10.66*g), **new) is None
    assert guard.pressure_reason(8*g, 20*g, **new) == 'sustained_memory_pressure'
    assert guard.pressure_reason(3*g, 8*g, **new) == 'sustained_memory_pressure'
    assert guard.pressure_reason(3*g, int(8.01*g), **new) is None
    assert guard.pressure_reason(3*g, 5*g, **new) == 'host_available_low'
    assert guard.pressure_reason(3*g, 20*g, memory_high_events_delta=1, **new) == 'scope_memory_high_event'
    assert guard.pressure_reason(int(9.7*g), 20*g, scope_stop_bytes=int(9.7*g),
                                 scope_stop_streak=3, **new) == 'scope_memory_stop'


def test_resume_psi_scope_forwarding_and_wrong_command_rejection(monkeypatch, tmp_path):
    import training_memory_scope as guard
    captured = {}
    monkeypatch.setattr(guard, 'run_named_scope', lambda cmd, unit: captured.update(command=cmd) or 0)
    argv = ['guard', '--wmj-v1-conditional-psi', '--wmj-v1-fixed-critic-resume-psi',
        '--log-dir', str(tmp_path/'scope'), '--', sys.executable,
        '-m', 'world_model.train_fixed_critic', '--resume', '/tmp/parent.json']
    monkeypatch.setattr(sys, 'argv', argv)
    assert guard.main() == 0
    assert captured['command'].count('--wmj-v1-fixed-critic-resume-psi') == 1
    monkeypatch.setattr(sys, 'argv', argv[:-6]+['/bin/true'])
    with pytest.raises(SystemExit):
        guard.main()


def test_resume_budget_requires_exact_cap_and_registered_extension():
    from world_model.contracts import load_config, validate_config
    cfg = load_config()
    cfg['resources']['max_new_bytes'] = 6710886400
    cfg['critic_fixed_data_fit']['resource_resume_plan'] = {'path':'registered'}
    validate_config(cfg)
    for cap in (6710886399, 6710886401):
        bad = copy.deepcopy(cfg); bad['resources']['max_new_bytes'] = cap
        with pytest.raises(ValueError, match='Storage budget'):
            validate_config(bad)
    bad = copy.deepcopy(cfg); bad['critic_fixed_data_fit'].pop('resource_resume_plan')
    with pytest.raises(ValueError, match='Storage budget'):
        validate_config(bad)


def test_resume_bridge_rejects_learning_or_checkpoint_changes():
    import json
    from world_model.contracts import load_config, authenticate, repository_root
    from world_model.train_fixed_critic import check_plan, check_resource_continuation
    cfg = load_config(); cfg['critic_fixed_data_fit']['authorized'] = True
    # The completed old run pins its original guard; the later user-approved
    # globalWMJ8GiB policy must not silently rerun that closed historical plan.
    with pytest.raises(ValueError, match='guard changed'):
        check_plan(cfg)
    plan = json.loads(authenticate(repository_root(), cfg['critic_fixed_data_fit']['plan']).read_text())
    assert plan['resource_continuation']['completed_updates'] == 176
    assert plan['max_phase_bytes'] == 48*2**20
    changed = copy.deepcopy(plan); changed['recipe']['critic_lr'] *= 2
    with pytest.raises(ValueError, match='learning'):
        check_resource_continuation(changed)
    changed = copy.deepcopy(plan); changed['resource_continuation']['completed_updates'] = 180
    with pytest.raises(ValueError, match='resume'):
        check_resource_continuation(changed)
