"""Original surfaces, containment and ambiguity govern offline seed admission."""

from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model.initialization_geometry import MeshTopology, refine_mesh_pair


def box(half_size=1., center=(0., 0., 0.)):
    vertices = np.array([[x, y, z] for x in (-half_size, half_size)
                         for y in (-half_size, half_size) for z in (-half_size, half_size)])
    vertices += center
    faces = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6],
        [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],
        [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])
    return vertices, faces


def classify(a, b):
    return refine_mesh_pair(MeshTopology(*a), a[0], MeshTopology(*b), b[0])


def test_disconnected_original_parts_do_not_fill_the_gap_between_them():
    left, faces = box(.1, (-.4, 0., 0.))
    right, _ = box(.1, (.4, 0., 0.))
    first = np.concatenate((left, right)), np.concatenate((faces, faces+8))
    result = classify(first, box(.05))
    assert result['status'] == 'PASS'
    assert result['reason'] == 'all_original_components_have_verified_separators'
    assert len(result['component_separators']) == 2


def test_closed_containment_is_failure_without_any_surface_crossing():
    result = classify(box(), box(.1))
    assert result['surface_check']['status'] == 'SURFACES_DISJOINT'
    assert result['status'] == 'FAIL'
    assert result['reason'] == 'closed_mesh_containment'


def test_open_containing_shell_is_unknown_and_is_never_repaired():
    vertices, faces = box()
    original = faces[:-1].copy()
    result = classify((vertices, original), box(.1))
    assert result['status'] == 'UNKNOWN'
    assert result['reason'] == 'unresolved_containment_in_open_or_inconsistent_mesh'
    np.testing.assert_array_equal(original, faces[:-1])


def test_clear_open_meshes_can_pass_with_a_verified_separating_plane():
    a, fa = box(.1)
    b, fb = box(.1, (.5, 0., 0.))
    result = classify((a, fa[:-1]), (b, fb[:-1]))
    assert result['status'] == 'PASS'


def test_crossing_original_surfaces_fail():
    result = classify(box(), box(.4, (.9, .1, .1)))
    assert result['status'] == 'FAIL'
    assert result['reason'] == 'original_surface_intersection'


def test_closed_hollow_shell_does_not_fill_its_inner_cavity():
    outer, fo = box(1.)
    inner, fi = box(.5)
    hollow = np.concatenate((outer, inner)), np.concatenate((fo, fi[:, ::-1]+8))
    result = classify(hollow, box(.1))
    assert result['status'] == 'PASS'
    assert result['reason'] == 'closed_meshes_have_no_containment'


def test_inconsistent_face_orientation_cannot_certify_containment():
    a, faces = box()
    faces[0] = faces[0, ::-1]
    assert classify((a, faces), box(.1))['status'] == 'UNKNOWN'


def test_offline_refinement_does_not_modify_native_collision_masks_or_geometry():
    from world_model.initialization_geometry import InitializationGeometryAuditor
    a, f = box(.1, (-.4, 0., 0.))
    b, _ = box(.1, (.4, 0., 0.))
    vertices = np.concatenate((a, b)); faces = np.concatenate((f, f+8))
    inner, fi = box(.05)
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><asset>
      <mesh name="outer" vertex="{' '.join(map(str,vertices.ravel()))}" face="{' '.join(map(str,faces.ravel()))}"/>
      <mesh name="inner" vertex="{' '.join(map(str,inner.ravel()))}" face="{' '.join(map(str,fi.ravel()))}"/>
      </asset><worldbody><body><geom type="mesh" mesh="outer"/></body>
      <body pos=".001 .002 0"><joint/><geom type="mesh" mesh="inner"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    names = ('geom_contype', 'geom_conaffinity', 'mesh_vert', 'mesh_face', 'geom_pos', 'geom_quat')
    before = {name: np.array(getattr(model, name)) for name in names}
    result = InitializationGeometryAuditor(model, [(0, 1)]).audit(data)
    assert result['status'] == 'PASS'
    assert result['execution_scope'] == 'offline_initialization_only'
    for name in names:
        np.testing.assert_array_equal(getattr(model, name), before[name])
