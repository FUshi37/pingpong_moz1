"""Replay overlays reconstruct canonical arrays without mixing teacher states."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from world_model.replay_overlay import encode_overlay, decode_overlay


def test_overlay_is_bitwise_lossless_for_small_and_large_replay_changes():
    base = dict(q=np.arange(24, dtype=np.float32).reshape(4, 6),
                event=np.zeros((4, 2), bool), clock=np.arange(4, dtype=np.int32))
    canonical = {key: value.copy() for key, value in base.items()}
    canonical['q'][1, 2] = np.nextafter(canonical['q'][1, 2], np.float32(9))
    canonical['q'][3, 4] = -35.
    canonical['event'][2, 1] = True
    canonical['support_q'] = np.ones((4, 6), np.float32)
    arrays, layout = encode_overlay(base, canonical)
    assert layout['q']['encoding'] == 'xor'
    assert layout['clock']['encoding'] == 'base'
    restored = decode_overlay(base, arrays, layout)
    for name, value in canonical.items():
        assert restored[name].tobytes() == value.tobytes()
        assert restored[name].dtype == value.dtype
    subset = decode_overlay(base, arrays, layout, names=['q'])
    assert set(subset) == {'q'}
    with pytest.raises(ValueError, match='layout'):
        decode_overlay({**base, 'q': base['q'].astype(np.float64)}, arrays, layout)
