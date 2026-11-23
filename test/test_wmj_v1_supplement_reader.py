"""Supplement diagnostics keep final-test and invalid initial states separate."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from world_model.supplement_data import diagnostic_batches


def test_diagnostic_batches_exclude_test_and_only_pad_computational_lanes():
    groups = [dict(split=s, initial_geometry_admitted=valid) for s, valid in
              [('train', True), ('test', True), ('validation', True), ('train', False), ('train', True)]]
    batches, excluded = diagnostic_batches(groups, width=4)
    assert excluded == {'test': [1], 'invalid_initial_geometry': [3]}
    assert len(batches) == 1
    assert batches[0]['useful'] == 3
    np.testing.assert_array_equal(batches[0]['lanes'], [0, 2, 4, 0])


def test_diagnostic_batches_reject_an_unknown_partition():
    with pytest.raises(ValueError, match='partition'):
        diagnostic_batches([dict(split='unchecked', initial_geometry_admitted=True)])
