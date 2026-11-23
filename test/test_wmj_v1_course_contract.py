"""Reject unapproved sequence layouts before entering the compiled course."""

import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.contracts import load_config
from world_model.course_learner import validate_course_config


def test_course_entry_rejects_changed_sequence_or_critic_contract():
    cfg = load_config()
    validate_course_config(cfg)
    for mutate in (
        lambda c:c['imagination_implementation']['critic'].update(input_dimension=2596),
        lambda c:c['curriculum']['stages'][3]['training_sequence'].update(goals_per_group=5),
        lambda c:c['samplers']['P_start']['course_counts']['S01'].update(precontact=128),
    ):
        changed = copy.deepcopy(cfg)
        mutate(changed)
        with pytest.raises(ValueError):
            validate_course_config(changed)
