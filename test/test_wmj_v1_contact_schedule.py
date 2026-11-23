"""Contact-window continuation preserves every already consumed training step."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))


def test_contact_window_extension_preserves_bootstrap_and_rejects_past_changes():
    from world_model.contact_schedule import contact_schedule, validate_contact_training_recipe
    from world_model.fit_schedule import controls_at

    old = dict(bootstrap_updates=2048, controls=8, single_step_aux_weight=256., learning_rate=3e-4)
    new = dict(old, window_schedule=[[2048, 8], [8192, 16], [12288, 32]])
    validate_contact_training_recipe(old, new, completed=8192)
    assert all(controls_at(i, contact_schedule(old)) == controls_at(i, contact_schedule(new))
               for i in range(8192))
    assert controls_at(8192, contact_schedule(new)) == 16
    with pytest.raises(ValueError, match='executed'):
        validate_contact_training_recipe(old, new, completed=8193)
    with pytest.raises(ValueError, match='executed'):
        validate_contact_training_recipe(new, old, completed=8193)
    with pytest.raises(ValueError, match='recipe'):
        validate_contact_training_recipe(old, dict(new, learning_rate=1e-4), completed=8192)
    with pytest.raises(ValueError):
        contact_schedule(dict(old, window_schedule=[[2048, 8], [8192, 32]]))


def test_contact_sampling_cost_counts_actual_windows_and_auxiliary_uses():
    from world_model.contact_schedule import contact_substep_uses

    old = dict(bootstrap_updates=2, controls=8)
    assert contact_substep_uses(old, completed=6) == (6*256*8*5, 0)
    mixed = dict(old, single_step_aux_weight=256., window_schedule=[[2, 8], [4, 16]])
    assert contact_substep_uses(mixed, completed=6) == (
        (4*8+2*16)*256*5, (2*8+2*16)*256*5)
