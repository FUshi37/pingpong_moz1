"""Numerical fitting stages cannot silently reinterpret a saved checkpoint."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.model_numerics import precision_scope, validate_precision_transition


def test_precision_stage_is_explicit_and_other_recipe_fields_are_preserved():
    original = dict(learning_rate=.0003, controls=8)
    candidate = dict(original, matmul_precision='highest')
    validate_precision_transition(original, original, start_stage=False)
    with pytest.raises(ValueError):
        validate_precision_transition(original, candidate, start_stage=False)
    validate_precision_transition(original, candidate, start_stage=True)
    validate_precision_transition(candidate, candidate, start_stage=False)
    for changed in (dict(candidate, controls=16), dict(candidate,learning_rate=.0001), original):
        with pytest.raises(ValueError):
            validate_precision_transition(candidate, changed, start_stage=True)
    with pytest.raises(ValueError):
        precision_scope('bfloat16')
