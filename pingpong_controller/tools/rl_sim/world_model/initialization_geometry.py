"""Offline-only seed geometry admission; never imported by a rollout step.

Original triangle surfaces are authoritative. Component separators are checked
against every original vertex; an open mesh needs no invented closure when
such a separator proves disjointness. Unresolved open-mesh containment stays
UNKNOWN. No simulator model, collision mask or triangle is modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError

from .mesh_surface import audit_surface_intersections


@dataclass
class MeshTopology:
    vertices: np.ndarray
    faces: np.ndarray
    components: list = field(init=False)
    closed_consistently_oriented: bool = field(init=False)
    boundary_edges: int = field(init=False)
    nonmanifold_edges: int = field(init=False)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, np.float64)
        self.faces = np.asarray(self.faces, np.int32)
        if (self.vertices.ndim != 2 or self.vertices.shape[1] != 3
                or self.faces.ndim != 2 or self.faces.shape[1] != 3
                or not len(self.faces) or not np.isfinite(self.vertices).all()
                or np.any(self.faces < 0) or np.any(self.faces >= len(self.vertices))):
            raise ValueError('Invalid original triangle mesh')
        edges = np.concatenate((self.faces[:, :2], self.faces[:, 1:], self.faces[:, [2, 0]]))
        graph = coo_matrix((np.ones(len(edges)), edges.T),
                           shape=(len(self.vertices), len(self.vertices))).tocsr()
        _, labels = connected_components(graph, directed=False)
        self.components = []
        for label in np.unique(labels[self.faces[:, 0]]):
            ids = np.flatnonzero(labels == label)
            try:
                hull = ids[ConvexHull(self.vertices[ids]).vertices]
            except QhullError:
                hull = ids
            self.components.append(dict(vertex_ids=ids, hull_ids=hull))
        _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
        balance = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1., -1.))
        self.boundary_edges = int(np.count_nonzero(counts == 1))
        self.nonmanifold_edges = int(np.count_nonzero(counts > 2))
        self.closed_consistently_oriented = bool(np.all(counts == 2) and np.all(balance == 0))


def _separator(first, second, hull_first, hull_second, tolerance_m):
    lo_a, hi_a = first.min(axis=0), first.max(axis=0)
    lo_b, hi_b = second.min(axis=0), second.max(axis=0)
    candidates = np.concatenate((lo_a-hi_b, lo_b-hi_a))
    axis = int(np.argmax(candidates))
    normal = np.zeros(3)
    if candidates[axis] > tolerance_m:
        normal[axis % 3] = 1. if axis < 3 else -1.
        method = 'axis_separation'
    else:
        origin = (lo_a+hi_a+lo_b+hi_b)/4
        va, vb = hull_first-origin, hull_second-origin
        inequalities = np.concatenate((np.column_stack((-va, np.ones((len(va), 2)))),
            np.column_stack((vb, -np.ones(len(vb)), np.ones(len(vb))))))
        solution = linprog([0., 0., 0., 0., -1.], A_ub=inequalities,
            b_ub=np.zeros(len(inequalities)), bounds=[(-1., 1.)]*3+[(None, None)]*2,
            method='highs')
        if not solution.success or np.linalg.norm(solution.x[:3]) <= 1e-12:
            return None
        normal = solution.x[:3]/np.linalg.norm(solution.x[:3])
        method = 'linear_program_separator'
    # The solver proposes a plane. Every original vertex must verify it.
    minimum_a, maximum_b = float(np.min(first @ normal)), float(np.max(second @ normal))
    gap = minimum_a-maximum_b
    if not np.isfinite(gap) or gap <= tolerance_m:
        return None
    return dict(method=method, normal=normal.tolist(), minimum_projection_first_m=minimum_a,
                maximum_projection_second_m=maximum_b, clearance_lower_bound_m=gap)


def _winding_number(point, triangles):
    """Signed solid-angle sum, following Jacobson et al., SIGGRAPH 2013.

    Reference: https://igl.ethz.ch/projects/winding-number/ . Only closed,
    consistently oriented inputs with unambiguous integral results qualify.
    """
    relative = triangles-point
    a, b, c = relative[:, 0], relative[:, 1], relative[:, 2]
    lengths = np.linalg.norm(relative, axis=-1)
    if np.min(lengths) <= 1e-12:
        return None
    determinant = np.einsum('ij,ij->i', a, np.cross(b, c))
    denominator = (lengths.prod(axis=1)
        + np.einsum('ij,ij->i', a, b)*lengths[:, 2]
        + np.einsum('ij,ij->i', b, c)*lengths[:, 0]
        + np.einsum('ij,ij->i', c, a)*lengths[:, 1])
    value = float(np.arctan2(determinant, denominator).sum()/(2*np.pi))
    return value if np.isfinite(value) else None


def refine_mesh_pair(first, world_first, second, world_second, *, tolerance_m=1e-9):
    """Audit original mesh surfaces, then disjointness or closed containment."""
    a, b = np.asarray(world_first), np.asarray(world_second)
    if a.shape != first.vertices.shape or b.shape != second.vertices.shape:
        raise ValueError('World mesh shape differs from original topology')
    ta, tb = a[first.faces], b[second.faces]
    surface = audit_surface_intersections(ta, tb, tolerance_m=tolerance_m)
    result = dict(status='UNKNOWN', surface_check=surface, component_separators=[],
        topology=[dict(closed_consistently_oriented=t.closed_consistently_oriented,
                       boundary_edges=t.boundary_edges, nonmanifold_edges=t.nonmanifold_edges)
                  for t in (first, second)], additional_clearance_m=0.)
    if surface['status'] == 'NOT_SEPARATED_AT_TOLERANCE':
        exact = audit_surface_intersections(ta, tb, tolerance_m=0.)
        result['zero_margin_surface_check'] = exact
        if exact['status'] == 'NOT_SEPARATED_AT_TOLERANCE':
            return dict(result, status='FAIL', reason='original_surface_intersection')
        return dict(result, reason='surface_contact_within_numerical_tolerance')
    if surface['status'] != 'SURFACES_DISJOINT':
        return dict(result, reason='incomplete_or_degenerate_surface_query')
    unresolved = []
    for i, ca in enumerate(first.components):
        for j, cb in enumerate(second.components):
            proof = _separator(a[ca['vertex_ids']], b[cb['vertex_ids']],
                               a[ca['hull_ids']], b[cb['hull_ids']], tolerance_m)
            if proof is None:
                unresolved.append([i, j])
            else:
                result['component_separators'].append(dict(components=[i, j], **proof))
    if not unresolved:
        return dict(result, status='PASS', reason='all_original_components_have_verified_separators')
    result['unseparated_components'] = unresolved
    if not (first.closed_consistently_oriented and second.closed_consistently_oriented):
        return dict(result, reason='unresolved_containment_in_open_or_inconsistent_mesh')
    winding = [[_winding_number(vertices[c['vertex_ids'][0]], other)
                for c in topology.components]
               for topology, vertices, other in ((first, a, tb), (second, b, ta))]
    result['containment_winding_numbers'] = winding
    values = [v for direction in winding for v in direction]
    if any(v is None or min(abs(v), abs(abs(v)-1.)) > 1e-5 for v in values):
        return dict(result, reason='ambiguous_closed_mesh_winding')
    if any(abs(v) > .5 for v in values):
        return dict(result, status='FAIL', reason='closed_mesh_containment')
    return dict(result, status='PASS', reason='closed_meshes_have_no_containment')


class InitializationGeometryAuditor:
    """CPU preparation tool with cached topology, absent from runtime state."""

    def __init__(self, model, pairs, *, tolerance_m=1e-9):
        self.model, self.pairs, self.tolerance_m = model, pairs, tolerance_m
        self._meshes = {}

    def _mesh(self, geom, data):
        mesh_id = int(self.model.geom_dataid[geom])
        if mesh_id not in self._meshes:
            m = self.model
            vs, vn = int(m.mesh_vertadr[mesh_id]), int(m.mesh_vertnum[mesh_id])
            fs, fn = int(m.mesh_faceadr[mesh_id]), int(m.mesh_facenum[mesh_id])
            self._meshes[mesh_id] = MeshTopology(m.mesh_vert[vs:vs+vn], m.mesh_face[fs:fs+fn])
        topology = self._meshes[mesh_id]
        world = topology.vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        return topology, world

    def audit(self, data):
        from .domain_atlas import check_self_collision
        broad = check_self_collision(self.model, data, self.pairs)
        refinements = []
        for alarm in broad['unresolved_mesh_hull_overlaps']+broad['unresolved_zero_distance']:
            a, b = alarm['geoms']
            if all(self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH for g in (a, b)):
                first, wa = self._mesh(a, data)
                second, wb = self._mesh(b, data)
                fine = refine_mesh_pair(first, wa, second, wb, tolerance_m=self.tolerance_m)
            else:
                fine = dict(status='UNKNOWN', reason='unsupported_original_mesh_primitive_refinement')
            refinements.append(dict(geoms=[a, b], bodies=alarm['bodies'], result=fine))
        statuses = [r['result']['status'] for r in refinements]
        status = ('FAIL' if broad['interpenetrations'] or 'FAIL' in statuses else
                  'UNKNOWN' if broad['unsupported'] or not self.pairs or 'UNKNOWN' in statuses else 'PASS')
        return dict(status=status, execution_scope='offline_initialization_only',
            broad_phase=broad, original_geometry_refinements=refinements,
            additional_clearance_m=0., trajectory_safety_certified=False)
