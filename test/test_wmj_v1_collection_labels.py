"""Failed tasks remain useful physics data with distinct initialization labels."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.contracts import load_config
from world_model.collection_labels import label_task_bounds


def test_out_of_view_initialization_and_later_failure_are_distinct():
    cfg = load_config()
    states = dict(q_rad=np.zeros((3, 2, 7)),
        ball_qpos=np.broadcast_to(np.array([0., -.5, 1.35, 1., 0., 0., 0.]), (3, 2, 7)).copy(),
        racket_position_m=np.broadcast_to(np.array([0., -.5, 1.075]), (3, 2, 3)).copy(),
        racket_rotation_world=np.broadcast_to(np.eye(3), (3, 2, 3, 3)))
    states['ball_qpos'][0, 0, 0] = .45
    states['racket_position_m'][2, 1, 1] = -.27
    result = label_task_bounds(cfg, states, reference_rotation=np.eye(3),
                               joint_lower_rad=-np.ones(7), joint_upper_rad=np.ones(7))
    np.testing.assert_array_equal(result['initial_task_eligible'], [False, True])
    np.testing.assert_array_equal(result['first_task_bound_failure_control'], [0, 2])
    np.testing.assert_array_equal(result['within_task_bounds'], [[False, True], [True, True], [True, False]])
    assert not result['physics_invalid'].any()
