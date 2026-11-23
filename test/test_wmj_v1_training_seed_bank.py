"""Training sources stay independent, unique and physically admitted."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from world_model.training_seed_bank import (
    balanced_selection, incoming_admitted, initial_candidates,
)


def test_precontact_candidates_require_train_validity_and_real_contact_lead():
    groups = [dict(split=s, region_id=i, root_seed_family=f'r{i}')
              for i, s in enumerate(['train', 'validation', 'train', 'train'])]
    events = np.zeros((160, 4), bool)
    events[99, :2] = True
    events[39, 2] = True
    events[119, 3] = True
    rows, counts = initial_candidates(groups, np.array([True, True, True, False]),
                                      events, trial=2, minimum_lead_s=.08)
    assert [r['lane'] for r in rows] == [0]
    assert rows[0]['time_to_contact_s'] == pytest.approx(.1)
    assert counts['train'] == 3
    assert counts['eligible'] == 1


def test_balancing_keeps_rare_regions_without_duplicating_root_families():
    rows = [dict(split='train', region_id=region, root_seed_family=f'{region}_{i}')
            for region, count in [(0, 1), (1, 10), (2, 10)] for i in range(count)]
    picked, report = balanced_selection(rows, count=9, seed=42)
    assert len({r['root_seed_family'] for r in picked}) == 9
    assert len([r for r in picked if r['region_id'] == 0]) == 1
    assert len([r for r in picked if r['region_id'] == 1]) == 4
    assert report['regions'] == 3
    for row in picked:
        expected = 1. if row['region_id'] == 0 else .4
        assert row['conditional_inclusion_probability'] == expected
    with pytest.raises(ValueError, match='shortage'):
        balanced_selection(rows, count=22, seed=42)
    with pytest.raises(ValueError, match='train'):
        balanced_selection([dict(rows[0], split='validation')], count=1, seed=42)
    with pytest.raises(ValueError, match='unique'):
        balanced_selection([rows[0], rows[0]], count=2, seed=42)


def test_incoming_ball_need_not_be_directly_over_racket_but_geometry_is_required():
    audit = dict(self_collision=dict(status='PASS'), initial_ball_clearance=dict(status='PASS'),
                 bounds=True, drop_geometry=dict(inside_face=False))
    assert incoming_admitted(audit)
    assert not incoming_admitted(dict(audit, self_collision=dict(status='UNKNOWN')))
    assert not incoming_admitted(dict(audit, initial_ball_clearance=dict(status='FAIL')))
    assert not incoming_admitted(dict(audit, bounds=False))
