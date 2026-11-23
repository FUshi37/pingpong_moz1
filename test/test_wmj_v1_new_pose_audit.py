"""New-pose diagnostics retain failures and never manufacture eligible windows."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from world_model.new_pose_audit import diagnostic_starts, prefix_validity, select_windows


def test_failure_prefix_never_reopens_and_unknown_never_qualifies():
    finite = np.ones((10, 3), bool)
    finite[7, 0] = False
    failure = np.zeros((2, 3), int)
    failure[0, 1] = 5
    physics, prefix = prefix_validity(finite, np.zeros_like(finite), failure, [True, True, False])
    np.testing.assert_array_equal(prefix[:, 0], [1] * 7 + [0] * 3)
    np.testing.assert_array_equal(prefix[:, 1], [1] * 4 + [0] * 6)
    assert not prefix[:, 2].any()
    assert physics[8, 0] and physics[:, 1].all()
    assert not physics[:, 2].any()


def test_windows_use_first_actual_release_and_retain_no_release_lanes():
    release = np.zeros((640, 3), bool)
    release[403, 0] = True
    release[7, 1] = True
    starts = diagnostic_starts(release, 8)
    np.testing.assert_array_equal(starts, [76, 0, 0])
    np.testing.assert_array_equal(diagnostic_starts(release, 128), [0, 0, 0])
    values = np.arange(129 * 3).reshape(129, 3)
    selected = select_windows(values, starts, 9)
    for lane in range(3):
        np.testing.assert_array_equal(selected[:, lane], values[starts[lane]:starts[lane]+9, lane])
    with pytest.raises(ValueError):
        select_windows(values, [122, 0, 0], 9)
