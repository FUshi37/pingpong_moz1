"""Learning-rate continuation must not rewrite consumed optimizer steps."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))


def test_rate_schedule_matches_legacy_prefix_and_rejects_reinterpretation():
    from world_model.contact_rates import parse_rates, rate_at, validate_rate_resume
    old = dict(learning_rate=3e-4)
    schedule = parse_rates('0:0.0003,8192:0.0001', initial=3e-4)
    new = dict(old, learning_rate_schedule=schedule)
    validate_rate_resume(old, new, completed=8192)
    assert rate_at(8191, schedule) == 3e-4
    assert rate_at(8192, schedule) == 1e-4
    assert all(rate_at(i, parse_rates(None, initial=3e-4)) == rate_at(i, schedule)
               for i in range(8192))
    with pytest.raises(ValueError, match='executed'):
        validate_rate_resume(old, new, completed=8193)
    with pytest.raises(ValueError, match='executed'):
        validate_rate_resume(new, old, completed=8193)
    validate_rate_resume(new, new, completed=9216)


@pytest.mark.parametrize('text', ['1:0.0003', '0:0.0001', '0:0.0003,0:0.0001',
    '0:0.0003,100:0', '0:0.0003,100:nan', '0:0.0003,100:inf'])
def test_rate_schedule_rejects_invalid_or_changed_initial_rates(text):
    from world_model.contact_rates import parse_rates
    with pytest.raises(ValueError):
        parse_rates(text, initial=3e-4)
