"""Bounded original-triangle diagnostics, independent of convex hull collision.

No hole filling, surface simplification or allowed-pair deletion is performed.
A disjoint-surface result alone does not rule out solid containment and is not
a hardware collision certificate. Degenerate inputs remain explicitly unknown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def triangle_pairs_overlap(first, second, *, tolerance_m=1e-9):
    """Test paired (N,3,3) triangles using normalized separating axes.

    Include in-plane edge normals so coplanar separated triangles do not
    produce false intersections. Tolerance is a conservative distance margin.
    """
    first, second = np.asarray(first, float), np.asarray(second, float)
    if first.shape != second.shape or first.ndim != 3 or first.shape[1:] != (3, 3):
        raise ValueError('Paired triangles require matching (N,3,3) arrays')
    if tolerance_m < 0 or not np.isfinite(tolerance_m):
        raise ValueError('Invalid distance tolerance')
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('Nonfinite triangles')
    ea = np.roll(first, -1, axis=1)-first
    eb = np.roll(second, -1, axis=1)-second
    na, nb = np.cross(ea[:, 0], ea[:, 1]), np.cross(eb[:, 0], eb[:, 1])
    degenerate = (np.linalg.norm(na, axis=1) <= 1e-16) | (np.linalg.norm(nb, axis=1) <= 1e-16)
    axes = np.concatenate((na[:, None], nb[:, None],
        np.cross(ea[:, :, None], eb[:, None]).reshape(-1, 9, 3),
        np.cross(na[:, None], ea), np.cross(nb[:, None], eb)), axis=1)
    norms = np.linalg.norm(axes, axis=-1)
    usable = norms > 1e-16
    axes = axes/np.maximum(norms[..., None], 1e-300)
    origin = first[:, :1]
    pa = np.einsum('nav,nkv->nak', axes, first-origin)
    pb = np.einsum('nav,nkv->nak', axes, second-origin)
    gap = np.maximum(pa.min(axis=-1)-pb.max(axis=-1), pb.min(axis=-1)-pa.max(axis=-1))
    separated = np.any((gap > tolerance_m) & usable, axis=1)
    return ~separated, degenerate


@dataclass
class _Node:
    lower: np.ndarray
    upper: np.ndarray
    count: int
    ids: np.ndarray | None = None
    children: tuple | None = None


def _tree(lower, upper, ids, leaf_size):
    node = _Node(lower[ids].min(axis=0), upper[ids].max(axis=0), len(ids))
    if len(ids) <= leaf_size:
        node.ids = ids
    else:
        centers = (lower[ids]+upper[ids])/2
        axis = int(np.argmax(np.ptp(centers, axis=0)))
        middle = len(ids)//2
        order = np.argpartition(centers[:, axis], middle)
        node.children = (_tree(lower, upper, ids[order[:middle]], leaf_size),
                         _tree(lower, upper, ids[order[middle:]], leaf_size))
    return node


def audit_surface_intersections(first, second, *, tolerance_m=1e-9,
                                leaf_size=16, max_candidate_pairs=2_000_000):
    """Traverse all possibly intersecting triangle pairs with bounded work."""
    first, second = np.asarray(first, float), np.asarray(second, float)
    for triangles in (first, second):
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0:
            raise ValueError('Nonempty (N,3,3) triangle arrays are required')
        if not np.isfinite(triangles).all():
            raise ValueError('Nonfinite triangles')
    if leaf_size < 1 or max_candidate_pairs < 0 or tolerance_m < 0:
        raise ValueError('Invalid diagnostic budget or tolerance')
    degenerate_counts = [int(np.count_nonzero(np.linalg.norm(
        np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1) <= 1e-16))
        for t in (first, second)]
    lo_a, hi_a = first.min(axis=1), first.max(axis=1)
    lo_b, hi_b = second.min(axis=1), second.max(axis=1)
    stack = [(_tree(lo_a, hi_a, np.arange(len(first)), leaf_size),
              _tree(lo_b, hi_b, np.arange(len(second)), leaf_size))]
    candidates = visited = intersections = 0
    witnesses = []
    exceeded = False
    while stack:
        a, b = stack.pop()
        visited += 1
        if np.any(np.maximum(a.lower-b.upper, b.lower-a.upper) > tolerance_m):
            continue
        if a.ids is not None and b.ids is not None:
            gap = np.maximum(lo_a[a.ids, None]-hi_b[b.ids], lo_b[b.ids]-hi_a[a.ids, None])
            i, j = np.nonzero(np.all(gap <= tolerance_m, axis=-1))
            ia, ib = a.ids[i], b.ids[j]
            if candidates+len(ia) > max_candidate_pairs:
                exceeded = True
                break
            candidates += len(ia)
            if len(ia):
                hit, degenerate = triangle_pairs_overlap(first[ia], second[ib], tolerance_m=tolerance_m)
                hit = hit & ~degenerate
                intersections += int(np.count_nonzero(hit))
                if len(witnesses) < 128:
                    witnesses.extend(np.column_stack((ia[hit], ib[hit]))[:128-len(witnesses)].tolist())
        elif a.children is not None and (b.children is None or a.count >= b.count):
            stack.extend((child, b) for child in a.children)
        else:
            stack.extend((a, child) for child in b.children)
    status = ('UNKNOWN_QUERY_BUDGET' if exceeded else
              'NOT_SEPARATED_AT_TOLERANCE' if intersections else
              'UNKNOWN_DEGENERATE_INPUT' if any(degenerate_counts) else 'SURFACES_DISJOINT')
    return dict(status=status, tolerance_m=tolerance_m,
        triangle_counts=[len(first), len(second)], degenerate_triangle_counts=degenerate_counts,
        visited_node_pairs=visited, tested_triangle_pairs=candidates,
        unseparated_triangle_pairs=intersections, example_unseparated_triangle_ids=witnesses,
        query_complete=not exceeded, max_candidate_pairs=max_candidate_pairs,
        scope='original_triangle_surfaces_only; solid_containment_and_hardware_not_certified')
