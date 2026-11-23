"""A vertical serve must intersect the actual finite, tilted racket face."""

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.contact_surface import surface_from_vertices
from world_model.frozen_initialization import vertical_drop_geometry


def test_stationary_drop_uses_finite_face_and_radius_with_tilt():
    vertices=np.array([[x,y,z] for x in (-.075,.075) for y in (-.075,.075) for z in (-.001,.001)])
    surface=surface_from_vertices(vertices,normal_axis=2)
    angle=np.deg2rad(2.);rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    position=np.array([0.,-.5,1.075])
    hit=vertical_drop_geometry(surface,position,rotation,np.array([.03,-.5,1.4]),.02)
    assert hit['inside_face'] and hit['positive_drop']
    np.testing.assert_allclose(np.array(hit['ball_center_at_contact_m'])[:2],[.03,-.5],atol=1e-12)
    point=np.array(hit['contact_surface_point_m'])
    local=rotation.T@(point-position)
    np.testing.assert_allclose(local[2],.001,atol=1e-9)
    miss=vertical_drop_geometry(surface,position,rotation,np.array([.1,-.5,1.4]),.02)
    assert not miss['inside_face']
