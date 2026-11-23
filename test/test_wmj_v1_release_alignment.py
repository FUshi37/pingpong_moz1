"""Release alignment reports missing modes instead of hiding them in quantiles."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from world_model.release_alignment import compare_releases


def test_delayed_release_uses_its_own_outgoing_state_and_keeps_missing_events():
    native = np.zeros((10, 2), bool)
    native[3, :] = True
    predicted = np.zeros_like(native)
    predicted[5, 0] = True
    p = np.zeros((10, 2, 3))
    v = np.zeros_like(p)
    v[3, :, 2] = 2.
    model_v = np.zeros_like(v)
    model_v[5, 0, 2] = 2.
    r = compare_releases(native, predicted, p, v, p, model_v, ['a', 'b'])
    assert r['paired_events'] == 1 and r['missing_events'] == 1
    assert r['extra_events'] == 0
    assert r['velocity_error_m_s']['maximum'] == 0.
    assert r['cases'][0]['pairs'][0]['time_shift_ms'] == 2
    assert not r['cases'][1]['same_release_count']


def test_extra_release_and_nonfinite_predictions_remain_visible():
    mask = np.zeros((8, 1), bool)
    mask[2, 0] = True
    extra = mask.copy()
    extra[4, 0] = True
    p = np.zeros((8, 1, 3)); v = np.ones_like(p)
    bad = v.copy(); bad[2, 0] = np.nan
    r = compare_releases(mask, extra, p, v, p, bad, ['a'])
    assert r['extra_events'] == 1 and r['nonfinite_pairs'] == 1
    assert r['velocity_error_m_s']['nonfinite'] == 1
    assert r['same_count_velocity_error_m_s']['count'] == 0
