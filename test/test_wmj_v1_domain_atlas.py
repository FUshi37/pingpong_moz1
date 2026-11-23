"""Independent geometric checks before any policy selects evaluation targets."""

from pathlib import Path
import sys

import mujoco
import numpy as np

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model.domain_atlas import collision_pairs, check_self_collision, solve_ik


def test_disabled_collision_masks_do_not_hide_robot_self_intersection():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="torso"><geom type="box" size=".2 .2 .2" contype="0" conaffinity="0"/></body>
      <body name="arm" pos=".3 0 0"><joint type="hinge"/>
        <geom type="box" size=".2 .2 .2" contype="0" conaffinity="0"/>
        <geom type="box" size=".2 .2 .2" contype="0" conaffinity="0"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pairs = collision_pairs(model, moving_body_ids=[2], excluded_body_ids=[])
    assert len(pairs) == 1
    result = check_self_collision(model, data, pairs)
    assert result['status'] == 'FAIL'
    assert result['minimum_distance_m'] < -.09


def test_concave_mesh_hull_alarm_is_not_called_actual_self_collision():
    # Two disconnected pieces have an empty central gap. MuJoCo's single-mesh
    # convex hull fills that gap; the sphere sits wholly inside the gap.
    vertices = np.array([[x, y, z] for center in (-.3, .3)
                         for x in (center-.05, center+.05)
                         for y in (-.1, .1) for z in (-.1, .1)])
    text = ' '.join(str(x) for x in vertices.ravel())
    cube_faces = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6],
        [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],
        [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])
    faces = ' '.join(str(x) for x in np.concatenate((cube_faces, cube_faces+8)).ravel())
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><asset>
      <mesh name="nonconvex" vertex="{text}" face="{faces}"/></asset><worldbody>
      <body name="torso"><geom type="mesh" mesh="nonconvex"/></body>
      <body name="arm" pos=".01 .02 0"><joint/><geom type="sphere" size=".04"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result = check_self_collision(model, data, [(0, 1)])
    assert result['status'] == 'UNKNOWN'
    assert result['interpenetrations'] == []
    assert len(result['unresolved_mesh_hull_overlaps']) == 1
    assert result['minimum_distance_m'] < -.01


def test_zero_distance_is_not_a_clearance_certificate():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><geom type="box" size=".1 .1 .1"/></body>
      <body pos=".2 0 0"><joint/><geom type="box" size=".1 .1 .1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result = check_self_collision(model, data, [(0, 1)])
    assert result['status'] == 'UNKNOWN'
    assert len(result['unresolved_zero_distance']) == 1


def test_ik_uses_joint_limits_and_never_calls_solver_failure_unreachable():
    model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
      <worldbody><body pos="0 0 1"><joint type="hinge" axis="0 0 1" range="-2 2"/>
      <geom type="sphere" size=".03"/>
      <body pos=".3 0 0"><joint type="hinge" axis="0 0 1" range="-2 2"/>
      <geom type="sphere" size=".03"/><site pos=".2 0 0"/>
      </body></body></worldbody></mujoco>''')
    target = [.3*np.cos(.5)+.2*np.cos(-.2), .3*np.sin(.5)+.2*np.sin(-.2), 1.]
    result = solve_ik(model, model.qpos0, [0, 1], 0, np.eye(3), np.asarray(target),
        np.array([.3, -.4]), np.full(2, -2.), np.full(2, 2.))
    assert result['status'] == 'IK_FEASIBLE'
    assert np.linalg.norm(np.asarray(result['position_m'])-target) < .0005
    assert np.all(np.abs(result['q_rad']) <= 2.)
    impossible = solve_ik(model, model.qpos0, [0, 1], 0, np.eye(3), np.array([5., 5., 1.]),
        np.array([.3, -.4]), np.full(2, -2.), np.full(2, 2.), max_nfev=15)
    assert impossible['status'] == 'UNKNOWN'
    assert 'unreachable' not in impossible['reason']
