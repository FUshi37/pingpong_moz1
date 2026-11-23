"""Lossless recovery preserves evidence and the sampled evaluation contract."""

import gzip
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                      'pingpong_controller/tools/rl_sim'))


def test_lossless_archive_exact_round_trip_and_no_overwrite(tmp_path):
    from world_model.recover_reward_direction import write_gzip_json
    from world_model.resources import StorageBudget
    payload = dict(values=[1.2345678901234567, -0.0, 1e-30],
                   missing=None, failed=[True, False], source='原始证据')
    budget = StorageBudget(tmp_path, 1024 * 1024, 0)
    path = tmp_path / 'sampled.json.gz'
    digest = write_gzip_json(path, payload, budget)
    raw = (json.dumps(payload, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')) + '\n').encode()
    assert gzip.decompress(path.read_bytes()) == raw
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        write_gzip_json(path, {'changed': True}, budget)
    assert gzip.decompress(path.read_bytes()) == raw
    assert [p.name for p in tmp_path.iterdir()] == ['sampled.json.gz']


def test_compressed_reservation_precedes_any_write(tmp_path):
    from world_model.recover_reward_direction import write_gzip_json
    from world_model.resources import StorageBudget, StorageLimitError
    with pytest.raises(StorageLimitError):
        write_gzip_json(tmp_path / 'sampled.json.gz', {'values': [0] * 10000},
                        StorageBudget(tmp_path, 8192, 0))
    assert list(tmp_path.iterdir()) == []


def test_sampled_recovery_changes_only_private_pairing_and_storage():
    from world_model.recover_reward_direction import sampled_audit, sampled_collector
    from world_model import audit_reward_direction, reward_direction_rollout
    collector = sampled_collector()
    assert 'sampled' in collector.__code__.co_consts
    assert 'deterministic' not in collector.__code__.co_consts
    assert 'deterministic' in reward_direction_rollout.collect_branches.__code__.co_consts
    audit = sampled_audit()
    assert '.json.gz' in audit.__code__.co_consts and 24576 in audit.__code__.co_consts
    assert '.json' in audit_reward_direction.audit.__code__.co_consts
    assert 49152 in audit_reward_direction.audit.__code__.co_consts
    assert audit.__globals__['objective_function'] is audit_reward_direction.objective_function
    assert collector.__globals__['make_native_step'] is reward_direction_rollout.make_native_step


def test_recomputed_scalars_do_not_claim_full_gradient_equivalence():
    from world_model.recover_reward_direction import gradient_recomputation_report
    old = dict(norm=21.23526, objective=-7.672395,
               normalized_direction_sha256='original', per_source_objectives=[[1., 2.]],
               valid_controls=1616, release_count=27, physical_failures=24)
    new = dict(old, norm=21.23398, objective=-7.672136,
               normalized_direction_sha256='recomputed', per_source_objectives=[[1., 2.00001]])
    report = gradient_recomputation_report(old, new)
    assert not report['direction_hash_equal']
    assert not report['full_vector_comparison_available']
    assert report['norm_relative_difference'] < .0001
    assert report['objective_absolute_difference'] > .0001
    with pytest.raises(ValueError):
        gradient_recomputation_report(old, dict(new, norm=float('nan')))
