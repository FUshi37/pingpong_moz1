"""Geometric diagnostics must use the compiled mesh frame exactly once."""

from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model.audit_collision_geometry import mesh_components, mesh_world
from world_model.mesh_surface import triangle_pairs_overlap, audit_surface_intersections


def test_compiled_mesh_transform_preserves_noncentered_source_vertices():
    original = np.array([[1., 0., 0.], [1.2, 0., 0.], [1., .3, 0.], [1., 0., .4]])
    vertices = ' '.join(str(x) for x in original.ravel())
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <asset><mesh name="tetra" vertex="{vertices}"/></asset>
      <worldbody><body pos=".1 .2 .3" quat=".7071067811865476 0 0 .7071067811865476">
      <geom type="mesh" mesh="tetra" pos=".05 -.02 .04"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    actual, faces = mesh_world(model, data, 0)
    expected = (original + [.05, -.02, .04]) @ np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]) + [.1, .2, .3]
    distances = np.linalg.norm(actual[:, None]-expected[None], axis=-1)
    assert np.max(np.min(distances, axis=1)) < 1e-7
    assert faces.shape == (4, 3)


def test_original_components_preserve_open_surfaces_without_hole_repair():
    vertices = np.zeros((7, 3))
    faces = np.array([[0, 1, 2], [2, 1, 3], [4, 5, 6]])
    before = faces.copy()
    assert mesh_components(vertices, faces) == [
        dict(vertices=4, faces=2), dict(vertices=3, faces=1)]
    np.testing.assert_array_equal(faces, before)


@pytest.mark.parametrize('other,intersects', [
    ([[.2, .2, -1.], [.2, .2, 1.], [.6, .2, 0.]], True),
    ([[2., 0., 0.], [3., 0., 0.], [2., 1., 0.]], False),
    ([[.2, .2, 0.], [.4, .2, 0.], [.2, .4, 0.]], True),
    ([[0., 0., .001], [1., 0., .001], [0., 1., .001]], False),
    ([[1., 0., 0.], [2., 0., 0.], [1., 1., 0.]], True),
])
def test_triangle_crossing_coplanar_and_touching_cases(other, intersects):
    first = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    second = np.array([other], dtype=float)
    hit, degenerate = triangle_pairs_overlap(first, second, tolerance_m=1e-10)
    assert bool(hit[0]) is intersects
    assert not degenerate[0]
    # Rigid transforms must not change the geometric answer.
    rotation = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))[0]
    again, _ = triangle_pairs_overlap(first @ rotation+[2., 3., -1.],
                                     second @ rotation+[2., 3., -1.], tolerance_m=1e-10)
    np.testing.assert_array_equal(hit, again)


def test_bvh_matches_all_triangle_pairs_and_preserves_unknown_degeneracy():
    rng = np.random.default_rng(7)
    first, second = rng.normal(size=(30, 3, 3)), rng.normal(size=(35, 3, 3))+.5
    a = np.repeat(first, len(second), axis=0)
    b = np.tile(second, (len(first), 1, 1))
    expected, _ = triangle_pairs_overlap(a, b)
    result = audit_surface_intersections(first, second, leaf_size=4)
    assert result['unseparated_triangle_pairs'] == int(np.count_nonzero(expected))
    assert result['status'] == 'NOT_SEPARATED_AT_TOLERANCE'
    first[0] = 0.
    unknown = audit_surface_intersections(first[:1], second)
    assert unknown['status'] == 'UNKNOWN_DEGENERATE_INPUT'


def test_disjoint_surfaces_and_budget_stop_are_distinct():
    a = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    assert audit_surface_intersections(a, a+[0., 0., .01])['status'] == 'SURFACES_DISJOINT'
    stopped = audit_surface_intersections(a, a, max_candidate_pairs=0)
    assert stopped['status'] == 'UNKNOWN_QUERY_BUDGET'
