"""Native evaluator keeps its released baseline unless an explicit actor is supplied."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))


def test_baseline_evaluation_selection_is_unchanged_and_override_requires_identity(monkeypatch):
    import world_model.evaluate_frozen as module
    source = {'path': 'original', 'sha256': 'original_hash'}
    expected = {'sentinel': object()}
    monkeypatch.setattr(module, 'load_v321_actor', lambda root, ref: ref)
    monkeypatch.setattr(module, 'expand_actor_history', lambda value: expected if value == source else None)
    actor, reference = module.select_evaluation_actor({'lineage': {'actor_source': source}})
    assert actor is expected and reference == source
    candidate = {name: {'w': np.zeros(shape, np.float32), 'b': np.zeros(shape[1], np.float32)}
                 for name, shape in [('l1', (969, 256)), ('l2', (256, 256)), ('out', (256, 7))]}
    with pytest.raises(ValueError):
        module.select_evaluation_actor({}, actor_override=candidate)
    ref = dict(path='candidate', sha256='a' * 64)
    actor, reference = module.select_evaluation_actor({}, actor_override=candidate, actor_reference=ref)
    assert actor is candidate and reference == ref
