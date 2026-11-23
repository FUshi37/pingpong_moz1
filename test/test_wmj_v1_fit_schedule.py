"""Window continuation preserves every already-executed optimizer update."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from world_model.fit_schedule import parse_schedule, controls_at, validate_resume_recipe


def test_progressive_windows_and_resume_before_new_stage():
    schedule = parse_schedule('256:8,1024:16,2048:32,4096:64', bootstrap=256, initial=8)
    assert [controls_at(i, schedule) for i in (0, 256, 1023, 1024, 4096)] == [8, 8, 8, 16, 64]
    old = dict(control_window=8, bootstrap_updates=256, learning_rate=3e-4)
    new = {**old, 'window_schedule': schedule}
    validate_resume_recipe(old, new, completed=512)
    validate_resume_recipe(old, new, completed=1024)
    with pytest.raises(ValueError, match='already executed'):
        validate_resume_recipe(old, new, completed=1025)
    with pytest.raises(ValueError, match='optimizer/loss'):
        validate_resume_recipe(old, {**new, 'learning_rate': 1e-4}, completed=512)


@pytest.mark.parametrize('text', ['256:8,256:16', '256:16', '256:8,100:16',
                                  '256:8,300:64', '256:8,300:9', '256:8,300:8'])
def test_invalid_or_skipped_stages_are_rejected(text):
    with pytest.raises(ValueError):
        parse_schedule(text, bootstrap=256, initial=8)
