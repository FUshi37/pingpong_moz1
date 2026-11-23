"""Fail-closed resource limits and early stop decisions without allocating RAM."""

from pathlib import Path
import signal
import sys

import pytest

RL_SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(RL_SIM))


def test_scope_limits_must_be_real_and_bounded(tmp_path):
    from training_memory_scope import GIB, SCOPE_STOP_BYTES, verify_scope

    for name, value in {'memory.max': 12 * GIB, 'memory.high': 10 * GIB,
                        'memory.swap.max': 0}.items():
        (tmp_path / name).write_text(str(value))
    verify_scope(tmp_path)
    assert SCOPE_STOP_BYTES < 10 * GIB
    (tmp_path / 'memory.high').write_text(str(11 * GIB))
    with pytest.raises(ValueError):
        verify_scope(tmp_path)
    (tmp_path / 'memory.high').write_text(str(10 * GIB))
    (tmp_path / 'memory.max').write_text('max')
    with pytest.raises(ValueError):
        verify_scope(tmp_path)
    (tmp_path / 'memory.max').write_text(str(12 * GIB))
    (tmp_path / 'memory.swap.max').write_text('1024')
    with pytest.raises(ValueError):
        verify_scope(tmp_path)


def test_scope_accepts_explicit_13gib_high_14gib_max_contract(tmp_path):
    from training_memory_scope import GIB, ScopeMemoryLimits, verify_scope

    limits = ScopeMemoryLimits.from_gib(high_gib=13.0, max_gib=14.0,
                                        stop_gib=12.5)
    for name, value in {
        'memory.max': 14 * GIB,
        'memory.high': 13 * GIB,
        'memory.swap.max': 0,
    }.items():
        (tmp_path / name).write_text(str(value))
    verify_scope(tmp_path, limits)

    (tmp_path / 'memory.high').write_text(str(14 * GIB))
    with pytest.raises(ValueError, match='unsafe scope limits'):
        verify_scope(tmp_path, limits)


def test_outer_wrapper_forwards_explicit_memory_contract(monkeypatch, tmp_path):
    import training_memory_scope as guard

    captured = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        guard,
        'run_named_scope',
        lambda command, unit: captured.update(command=command, unit=unit) or 0,
    )
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'guard', '--memory-high-gib', '13', '--memory-max-gib', '14',
            '--memory-stop-gib', '12.5', '--host-stop-available-gib', '4',
            '--host-stop-streak-samples', '3',
            '--log-dir', str(tmp_path / 'log'),
            '--', '/bin/true',
        ],
    )
    assert guard.main() == 0
    command = captured['command']
    assert 'MemoryHigh=13G' in command
    assert 'MemoryMax=14G' in command
    assert command.count('--memory-high-gib') == 1
    assert command[command.index('--memory-high-gib') + 1] == '13'
    assert command[command.index('--memory-max-gib') + 1] == '14'
    assert command[command.index('--memory-stop-gib') + 1] == '12.5'
    assert command.count('--host-stop-available-gib') == 1
    assert command[command.index('--host-stop-available-gib') + 1] == '4'
    assert command.count('--host-stop-streak-samples') == 1
    assert command[command.index('--host-stop-streak-samples') + 1] == '3'


def test_pressure_stops_before_hard_limit_and_covers_host_pressure():
    from training_memory_scope import (
        GIB,
        PRESSURE_STREAK_SAMPLES,
        SCOPE_STOP_STREAK_SAMPLES,
        parse_memory_events,
        parse_memory_pressure,
        pressure_reason,
    )

    assert pressure_reason(8 * GIB, 12 * GIB) is None
    assert pressure_reason(9 * GIB, 12 * GIB) is None
    assert pressure_reason(
        int(9.5 * GIB),
        12 * GIB,
        scope_stop_streak=SCOPE_STOP_STREAK_SAMPLES - 1,
    ) is None
    assert pressure_reason(
        int(9.5 * GIB),
        12 * GIB,
        scope_stop_streak=SCOPE_STOP_STREAK_SAMPLES,
    ) == 'scope_memory_stop'
    assert pressure_reason(5 * GIB, 7 * GIB) is None
    assert pressure_reason(5 * GIB, 5 * GIB) == 'host_available_low'
    assert pressure_reason(
        5 * GIB,
        int(5.75 * GIB),
        host_stop_available_bytes=5 * GIB,
    ) is None
    assert pressure_reason(
        5 * GIB,
        int(4.99 * GIB),
        host_stop_available_bytes=5 * GIB,
    ) == 'host_available_low'
    assert pressure_reason(
        5 * GIB,
        int(3.99 * GIB),
        host_stop_available_bytes=4 * GIB,
        host_stop_streak=2,
        host_stop_streak_samples=3,
    ) is None
    assert pressure_reason(
        5 * GIB,
        int(3.99 * GIB),
        host_stop_available_bytes=4 * GIB,
        host_stop_streak=3,
        host_stop_streak_samples=3,
    ) == 'host_available_low'
    pressure = parse_memory_pressure(
        'some avg10=11.50 avg60=3.20 avg300=0.70 total=10\n'
        'full avg10=5.25 avg60=1.10 avg300=0.20 total=4'
    )
    assert pressure['some']['avg10'] == pytest.approx(11.5)
    assert pressure['full']['avg10'] == pytest.approx(5.25)
    assert pressure_reason(
        8 * GIB,
        12 * GIB,
        memory_pressure=pressure,
        pressure_streak=PRESSURE_STREAK_SAMPLES - 1,
    ) is None
    assert pressure_reason(
        8 * GIB,
        12 * GIB,
        memory_pressure=pressure,
        pressure_streak=PRESSURE_STREAK_SAMPLES,
    ) == 'sustained_memory_pressure'
    assert parse_memory_events('low 0\nhigh 17\nmax 0\noom 0')['high'] == 17
    assert pressure_reason(
        8 * GIB,
        12 * GIB,
        memory_high_events_delta=1,
    ) == 'scope_memory_high_event'


def test_scope_stop_threshold_is_below_kernel_throttle(tmp_path):
    from training_memory_scope import GIB, verify_scope

    for name, value in {
        'memory.max': 12 * GIB,
        'memory.high': 9 * GIB,
        'memory.swap.max': 0,
    }.items():
        (tmp_path / name).write_text(str(value))
    with pytest.raises(ValueError, match='unsafe scope limits'):
        verify_scope(tmp_path)


def test_graceful_checkpoint_window_precedes_forced_termination():
    from training_memory_scope import (
        MEMORY_CHECKPOINT_SIGKILL_GRACE_S,
        MEMORY_CHECKPOINT_SIGTERM_GRACE_S,
        SIGKILL_GRACE_S,
        SIGTERM_GRACE_S,
    )

    assert SIGTERM_GRACE_S >= 10.0
    assert SIGKILL_GRACE_S >= SIGTERM_GRACE_S + 10.0
    assert MEMORY_CHECKPOINT_SIGTERM_GRACE_S >= 120.0
    assert (
        MEMORY_CHECKPOINT_SIGKILL_GRACE_S
        >= MEMORY_CHECKPOINT_SIGTERM_GRACE_S + 60.0
    )


def test_both_current_launchers_require_verified_scope():
    for name in ['run_gpu0_fixed_contact_plane_vision_robust_qvel_rmp_v38_from_scratch.sh',
                 'run_gpu1_fixed_contact_plane_vision_robust_qvel_rmp_v39_resume_v9_actor.sh']:
        source = (RL_SIM / name).read_text()
        assert 'training_memory_scope.py' in source
        assert '--verify-only' in source
        assert '--clear-jax-caches-between-stages' in source


def test_scoped_fixed_contact_launchers_do_not_report_old_unsafe_scope_limit():
    scoped_launchers = []
    for path in RL_SIM.glob('run_gpu*fixed_contact*.sh'):
        source = path.read_text()
        if 'training_memory_scope.py' in source:
            scoped_launchers.append(path)
            assert '10.25 GiB stop' not in source, path.name
    assert scoped_launchers


def test_gpu0_sequence_evidence_path_crosses_named_scope_explicitly():
    from training_memory_scope import training_child_environment

    evidence = "/tmp/v67/curriculum_progress.csv"
    forwarded = training_child_environment({
        "GPU0_V67_PROGRESS_CSV": evidence,
        "UNRELATED_SECRET": "must-not-cross",
    })
    assert forwarded == (f"GPU0_V67_PROGRESS_CSV={evidence}",)


def test_v95_sequence_evidence_and_v96_probe_flag_cross_scope():
    from training_memory_scope import training_child_environment

    evidence = "/tmp/v95/curriculum_progress.csv"
    forwarded = training_child_environment({
        "GPU0_V95_PROGRESS_CSV": evidence,
        "V96_EXPERIMENTAL_PROBE": "YES",
        "UNRELATED_SECRET": "must-not-cross",
    })
    assert forwarded == (
        f"GPU0_V95_PROGRESS_CSV={evidence}",
        "V96_EXPERIMENTAL_PROBE=YES",
    )


def test_v97_sequence_evidence_and_v98_probe_flag_cross_scope():
    from training_memory_scope import training_child_environment

    evidence = "/tmp/v97/curriculum_progress.csv"
    forwarded = training_child_environment({
        "GPU0_V97_PROGRESS_CSV": evidence,
        "V97_EXPERIMENTAL_PROBE": "YES",
        "V98_EXPERIMENTAL_PROBE": "YES",
        "UNRELATED_SECRET": "must-not-cross",
    })
    assert forwarded == (
        f"GPU0_V97_PROGRESS_CSV={evidence}",
        "V97_EXPERIMENTAL_PROBE=YES",
        "V98_EXPERIMENTAL_PROBE=YES",
    )


def test_v99_sequence_evidence_and_v100_probe_flag_cross_scope():
    from training_memory_scope import training_child_environment

    evidence = "/tmp/v99/curriculum_progress.csv"
    forwarded = training_child_environment({
        "GPU0_V99_PROGRESS_CSV": evidence,
        "V99_EXPERIMENTAL_PROBE": "YES",
        "V100_EXPERIMENTAL_PROBE": "YES",
        "UNRELATED_SECRET": "must-not-cross",
    })
    assert forwarded == (
        f"GPU0_V99_PROGRESS_CSV={evidence}",
        "V100_EXPERIMENTAL_PROBE=YES",
        "V99_EXPERIMENTAL_PROBE=YES",
    )


def test_v101_probe_flag_crosses_named_scope():
    from training_memory_scope import training_child_environment

    forwarded = training_child_environment({
        "STAGE_REEXEC_EXPECTED_SHA256": "abc123",
        "V101_EXPERIMENTAL_PROBE": "YES",
        "UNRELATED_SECRET": "must-not-cross",
    })
    assert forwarded == (
        "STAGE_REEXEC_EXPECTED_SHA256=abc123",
        "V101_EXPERIMENTAL_PROBE=YES",
    )


def test_gpu1_1024_batch_and_resume_contract_cross_named_scope():
    from training_memory_scope import training_child_environment

    forwarded = training_child_environment({
        "GPU1_BATCH_PRESET": "v136_1024x128",
        "GPU1_RESUME_MODE": "passed_stage_boundary",
        "N_STEPS": "128",
        "UNRELATED_SECRET": "must-not-cross",
    })
    assert forwarded == (
        "GPU1_BATCH_PRESET=v136_1024x128",
        "GPU1_RESUME_MODE=passed_stage_boundary",
        "N_STEPS=128",
    )


def test_v176_cmdp_contract_crosses_named_scope():
    from training_memory_scope import training_child_environment

    forwarded = training_child_environment({
        "CMDP_ENABLED": "YES",
        "CMDP_MIGRATE_COST_CRITICS": "YES",
        "CMDP_COST_MODE": "fixed_contact_quality",
        "UNRELATED_SECRET": "must-not-cross",
    })
    assert forwarded == (
        "CMDP_COST_MODE=fixed_contact_quality",
        "CMDP_ENABLED=YES",
        "CMDP_MIGRATE_COST_CRITICS=YES",
    )


def test_predictive_next_stage_recovery_crosses_named_scope():
    from training_memory_scope import training_child_environment

    forwarded = training_child_environment({
        "RECOVER_NEXT_STAGE_REEXEC": "YES",
        "STAGE_REEXEC_START_STAGE": "v146_g0_arrival_local",
        "WANDB_SOURCE_TAG": "resume-passed-stage",
        "UNRELATED_SECRET": "must-not-cross",
    })
    assert forwarded == (
        "RECOVER_NEXT_STAGE_REEXEC=YES",
        "STAGE_REEXEC_START_STAGE=v146_g0_arrival_local",
        "WANDB_SOURCE_TAG=resume-passed-stage",
    )


def test_fixed_v99_v101_launchers_use_dual_gpu_headroom_and_recovery() -> None:
    for name in (
        'run_gpu0_fixed_contact_plane_vision_robust_qvel_rmp_v99_1024x128_resume_v97_u500.sh',
        'run_gpu1_fixed_contact_plane_vision_robust_qvel_rmp_v101_1024x128_resume_v98_u1978.sh',
    ):
        source = (RL_SIM / name).read_text()
        assert '--memory-high-gib 13 --memory-max-gib 14' in source
        assert '--host-stop-available-gib 4' in source
        assert '--host-stop-streak-samples 3' in source
        assert 'RECOVER_STAGE_REEXEC=${RECOVER_STAGE_REEXEC:-NO}' in source
        assert 'STAGE_REEXEC_EXPECTED_SHA256' in source
        assert '--resume-curriculum-state' in source
        assert 'memory_guard_stage_reexec_' in source


def test_guard_interrupts_only_owned_child_and_records_stop(tmp_path, monkeypatch):
    import training_memory_scope as guard

    ready = tmp_path / 'ready'
    stopped = tmp_path / 'stopped'
    child = tmp_path / 'child.py'
    child.write_text(
        'from pathlib import Path\nimport signal, time\n'
        f'def stop(*args):\n Path({str(stopped)!r}).touch()\n raise SystemExit(0)\n'
        'signal.signal(signal.SIGINT, stop)\n'
        f'Path({str(ready)!r}).touch()\n'
        'while True: time.sleep(.01)\n')
    monkeypatch.setattr(guard, 'scope_path', lambda: tmp_path)
    monkeypatch.setattr(guard, 'verify_scope', lambda _: None)
    monkeypatch.setattr(guard, 'memory_sample', lambda _: {
        'scope_current_bytes': (10 if ready.exists() else 1) * guard.GIB,
        'host_available_bytes': 12 * guard.GIB,
        'memory_events': 'low 0\nhigh 0\nmax 0\noom 0',
        'memory_pressure': (
            'some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n'
            'full avg10=0.00 avg60=0.00 avg300=0.00 total=0'
        ),
    })
    direct_signals = []
    group_signals = []
    real_kill = guard.os.kill
    real_killpg = guard.os.killpg
    monkeypatch.setattr(
        guard.os,
        'kill',
        lambda pid, signum: (
            direct_signals.append((pid, signum)), real_kill(pid, signum)
        )[1],
    )
    monkeypatch.setattr(
        guard.os,
        'killpg',
        lambda pid, signum: (
            group_signals.append((pid, signum)), real_killpg(pid, signum)
        )[1],
    )
    assert guard.supervise([sys.executable, str(child)], tmp_path) == 75
    assert stopped.exists()
    assert 'scope_memory_stop' in (tmp_path / 'memory.jsonl').read_text()
    assert any(signum == signal.SIGINT for _, signum in direct_signals)
    assert not any(signum == signal.SIGINT for _, signum in group_signals)


def test_guard_keeps_log_pipe_alive_through_sigterm_escalation(
    tmp_path, monkeypatch
):
    import training_memory_scope as guard

    ready = tmp_path / 'ready'
    term_seen = tmp_path / 'term_seen'
    child = tmp_path / 'slow_child.py'
    child.write_text(
        'from pathlib import Path\nimport signal, time\n'
        'signal.signal(signal.SIGINT, lambda *_: None)\n'
        f'def term(*_):\n Path({str(term_seen)!r}).touch()\n raise SystemExit(0)\n'
        'signal.signal(signal.SIGTERM, term)\n'
        f'Path({str(ready)!r}).touch()\n'
        'while True: time.sleep(.01)\n'
    )
    monkeypatch.setattr(guard, 'scope_path', lambda: tmp_path)
    monkeypatch.setattr(guard, 'verify_scope', lambda _: None)
    monkeypatch.setattr(guard, 'MEMORY_CHECKPOINT_SIGTERM_GRACE_S', 0.05)
    monkeypatch.setattr(guard, 'MEMORY_CHECKPOINT_SIGKILL_GRACE_S', 0.5)
    monkeypatch.setattr(guard, 'SAMPLE_PERIOD_S', 0.01)
    monkeypatch.setattr(guard, 'memory_sample', lambda _: {
        'scope_current_bytes': (10 if ready.exists() else 1) * guard.GIB,
        'host_available_bytes': 12 * guard.GIB,
        'memory_events': 'low 0\nhigh 0\nmax 0\noom 0',
        'memory_pressure': (
            'some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n'
            'full avg10=0.00 avg60=0.00 avg300=0.00 total=0'
        ),
    })
    direct_signals = []
    group_signals = []
    real_kill = guard.os.kill
    real_killpg = guard.os.killpg
    monkeypatch.setattr(
        guard.os,
        'kill',
        lambda pid, signum: (
            direct_signals.append((pid, signum)), real_kill(pid, signum)
        )[1],
    )
    monkeypatch.setattr(
        guard.os,
        'killpg',
        lambda pid, signum: (
            group_signals.append((pid, signum)), real_killpg(pid, signum)
        )[1],
    )

    assert guard.supervise([sys.executable, str(child)], tmp_path) == 75
    assert term_seen.exists()
    assert any(signum == signal.SIGINT for _, signum in direct_signals)
    assert any(signum == signal.SIGTERM for _, signum in direct_signals)
    assert not any(
        signum in {signal.SIGINT, signal.SIGTERM}
        for _, signum in group_signals
    )


def test_fake_scope_flag_cannot_bypass_kernel_verification(monkeypatch):
    import training_memory_scope as guard

    monkeypatch.setenv('PP_MEMORY_SCOPE_ACTIVE', 'YES')
    monkeypatch.setattr(sys, 'argv', ['guard', '--verify-only'])
    def reject(_):
        raise ValueError('unbounded scope')
    monkeypatch.setattr(guard, 'verify_scope', reject)
    with pytest.raises(ValueError, match='unbounded'):
        guard.main()


def test_outer_scope_signal_targets_the_named_cgroup(monkeypatch, tmp_path):
    """A terminal interrupt must not orphan the inside supervisor/trainer."""
    import training_memory_scope as guard

    calls = []

    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 130
            return self.returncode

    monkeypatch.setattr(guard.subprocess, 'Popen', lambda command: FakeProcess())
    monkeypatch.setattr(
        guard.subprocess,
        'run',
        lambda command, **kwargs: calls.append(command),
    )
    monkeypatch.setattr(guard.time, 'sleep', lambda _seconds: None)

    process = FakeProcess()
    guard.stop_scope_process_tree(process, 'pp-training-test.service', signal.SIGINT)
    assert calls[0] == [
        'systemctl', '--user', 'kill', '--kill-whom=all',
        '--signal=SIGINT', 'pp-training-test.service',
    ]
    assert process.returncode == 130


def test_outer_wrapper_uses_waitable_transient_service(monkeypatch, tmp_path):
    import training_memory_scope as guard

    captured = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        guard,
        'run_named_scope',
        lambda command, unit: captured.update(command=command, unit=unit) or 0,
    )
    monkeypatch.setattr(
        sys,
        'argv',
        ['guard', '--log-dir', str(tmp_path / 'log'), '--', '/bin/true'],
    )
    assert guard.main() == 0
    command = captured['command']
    assert '--scope' not in command
    assert '--wait' in command
    assert '--pipe' in command
    assert '--service-type=exec' in command
    assert f'--working-directory={tmp_path}' in command
    assert captured['unit'].endswith('.service')


def test_outer_wrapper_passes_only_training_control_environment(
    monkeypatch, tmp_path
):
    import training_memory_scope as guard

    captured = {}
    monkeypatch.setenv('CONFIRM_GPU0_READY', 'YES')
    monkeypatch.setenv('ACKNOWLEDGE_INCOMPLETE_RMP_DR_EVIDENCE', 'YES')
    monkeypatch.setenv('RUN_DIR', '/tmp/fixed-run')
    monkeypatch.setenv('WANDB_API_KEY', 'must-not-be-forwarded')
    monkeypatch.setenv('UNRELATED_VALUE', 'must-not-be-forwarded')
    monkeypatch.setattr(
        guard,
        'run_named_scope',
        lambda command, unit: captured.update(command=command, unit=unit) or 0,
    )
    monkeypatch.setattr(
        sys,
        'argv',
        ['guard', '--log-dir', str(tmp_path / 'log'), '--', '/bin/true'],
    )
    assert guard.main() == 0
    command = captured['command']
    forwarded = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == '--child-env'
    ]
    assert 'CONFIRM_GPU0_READY=YES' in forwarded
    assert 'ACKNOWLEDGE_INCOMPLETE_RMP_DR_EVIDENCE=YES' in forwarded
    assert 'RUN_DIR=/tmp/fixed-run' in forwarded
    assert not any(value.startswith('WANDB_API_KEY=') for value in forwarded)
    assert not any(value.startswith('UNRELATED_VALUE=') for value in forwarded)
