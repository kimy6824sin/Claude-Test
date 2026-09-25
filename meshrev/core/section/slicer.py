"""Planar sections of triangle meshes (the raw material of mesh sketches).

Algorithm (exact, vectorised):

1. Signed distances ``s_i = (p_i - o)·n`` of all vertices to the plane. Vertices
   with ``|s_i| < eps`` are *symbolically perturbed* to ``+eps`` so no vertex
   lies on the plane; every cut triangle then has exactly two crossing edges
   and degenerate cases (plane through vertices/edges/faces) disappear.
2. Each crossing edge ``(a, b)`` yields the point
   ``p = p_a + t (p_b - p_a)`` with ``t = s_a / (s_a - s_b)``. The edge is keyed
   by its sorted vertex pair, so the two triangles sharing an edge produce the
   *same* node - chaining is topologically exact instead of distance based.
3. Every cut triangle is a link between its two edge nodes. Walking the node
   graph from degree-1 nodes yields open chains (mesh boundary / holes),
   the remaining cycles are closed loops.
4. Open chains whose ends lie within ``gap_tolerance`` are bridged (scan holes,
   cracks); a chain whose own ends meet is closed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

from meshrev.core.mesh.processing import ensure_triangles, triangle_array
from meshrev.core.types import FloatArray, Plane


@dataclass(eq=False)
class SectionCurve:
    """Polylines where a plane cuts a mesh. ``closed[i]`` tells whether polyline i loops."""

    plane: Plane
    polylines: list[FloatArray] = field(default_factory=list)
    closed: list[bool] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.polylines

    def to_2d(self) -> list[FloatArray]:
        """Polylines in the plane's ``(u, v)`` sketch coordinates."""
        return [self.plane.to_local(p) for p in self.polylines]

    @property
    def total_length(self) -> float:
        total = 0.0
        for pts, is_closed in zip(self.polylines, self.closed, strict=True):
            seg = np.diff(np.vstack([pts, pts[:1]]) if is_closed else pts, axis=0)
            total += float(np.linalg.norm(seg, axis=1).sum())
        return total

    def to_polydata(self) -> pv.PolyData:
        if self.is_empty:
            return pv.PolyData()
        points, lines, offset = [], [], 0
        for pts, is_closed in zip(self.polylines, self.closed, strict=True):
            ids = list(range(offset, offset + len(pts)))
            if is_closed:
                ids.append(offset)
            lines.append([len(ids), *ids])
            points.append(pts)
            offset += len(pts)
        return pv.PolyData(np.vstack(points), lines=np.concatenate(lines))


def _cut_segments(points: FloatArray, tris: np.ndarray, plane: Plane):
    """Intersection nodes (one per crossing edge) and links (one per cut triangle)."""
    s = (points - plane.origin) @ plane.normal
    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    eps = 1e-12 * scale
    s = np.where(np.abs(s) < eps, eps, s)  # symbolic perturbation: nothing lies on the plane
    above = s > 0
    side = above[tris]
    cut = side.any(axis=1) & ~side.all(axis=1)
    tris = tris[cut]
    side = side[cut]
    if not len(tris):
        return np.empty((0, 3)), np.empty((0, 2), dtype=np.int64)
    # the vertex alone on its side is the apex; its two edges are the crossing ones
    apex_col = np.where(side.sum(axis=1) == 1, np.argmax(side, axis=1), np.argmin(side, axis=1))
    rows = np.arange(len(tris))
    apex = tris[rows, apex_col]
    other1 = tris[rows, (apex_col + 1) % 3]
    other2 = tris[rows, (apex_col + 2) % 3]
    edges = np.concatenate([np.column_stack([apex, other1]), np.column_stack([apex, other2])])
    edges.sort(axis=1)
    keys = edges[:, 0] * np.int64(len(points)) + edges[:, 1]
    unique_keys, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    a, b = edges[first, 0], edges[first, 1]
    t = s[a] / (s[a] - s[b])
    nodes = points[a] + t[:, None] * (points[b] - points[a])
    n = len(tris)
    links = np.column_stack([inverse[:n], inverse[n:]])
    return nodes, links


def _chain(n_nodes: int, links: np.ndarray) -> list[tuple[list[int], bool]]:
    """Walk the node graph into ordered chains ``(node ids, closed)``."""
    neighbours: list[list[int]] = [[] for _ in range(n_nodes)]
    for i, (p, q) in enumerate(links):
        neighbours[p].append(i)
        neighbours[q].append(i)
    used = np.zeros(len(links), dtype=bool)
    chains: list[tuple[list[int], bool]] = []

    def walk(start: int) -> tuple[list[int], bool]:
        path = [start]
        node = start
        while True:
            nxt = next((k for k in neighbours[node] if not used[k]), None)
            if nxt is None:
                return path, False
            used[nxt] = True
            p, q = links[nxt]
            node = q if p == node else p
            if node == start:
                return path, True
            path.append(node)

    degree = np.array([len(nb) for nb in neighbours])
    for start in np.flatnonzero(degree % 2 == 1):  # chain ends (mesh boundary)
        if any(not used[k] for k in neighbours[start]):
            chains.append(walk(int(start)))
    for start in range(n_nodes):  # remaining closed loops
        if any(not used[k] for k in neighbours[start]):
            chains.append(walk(start))
    return chains


def _dedupe(pts: FloatArray, closed: bool, tol: float) -> FloatArray:
    """Drop consecutive (near) duplicate points produced by cuts close to vertices."""
    if len(pts) < 2:
        return pts
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > tol]
    pts = pts[keep]
    if closed and len(pts) > 2 and np.linalg.norm(pts[0] - pts[-1]) <= tol:
        pts = pts[:-1]
    return pts


def bridge_gaps(
    polylines: list[FloatArray], closed: list[bool], gap_tolerance: float
) -> tuple[list[FloatArray], list[bool]]:
    """Join open chains whose end points are closer than ``gap_tolerance``.

    Greedy: repeatedly connect the globally closest pair of free ends (a chain
    with both of its own ends close is closed). Ends further apart than the
    tolerance stay open - they are reported instead of being guessed.
    """
    polys = [p for p in polylines]
    flags = list(closed)
    while True:
        open_ids = [i for i, c in enumerate(flags) if not c and len(polys[i]) > 1]
        if not open_ids:
            break
        ends = []  # (chain, is_tail, point)
        for i in open_ids:
            ends.append((i, False, polys[i][0]))
            ends.append((i, True, polys[i][-1]))
        pts = np.array([e[2] for e in ends])
        dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        np.fill_diagonal(dist, np.inf)
        for k in range(0, len(ends), 2):  # a chain's own two ends may only close it
            if len(polys[ends[k][0]]) < 3:
                dist[k, k + 1] = dist[k + 1, k] = np.inf
        k1, k2 = np.unravel_index(np.argmin(dist), dist.shape)
        if dist[k1, k2] > gap_tolerance:
            break
        (i, tail_i, _), (j, tail_j, _) = ends[k1], ends[k2]
        if i == j:
            flags[i] = True
            continue
        a = polys[i] if tail_i else polys[i][::-1]  # a ends at the joint
        b = polys[j] if not tail_j else polys[j][::-1]  # b starts at the joint
        polys[i] = np.vstack([a, b])
        del polys[j], flags[j]
    return polys, flags


def slice_mesh(
    mesh: pv.PolyData,
    plane: Plane,
    gap_tolerance: float | None = None,
    min_points: int = 3,
) -> SectionCurve:
    """Cut ``mesh`` with ``plane`` into ordered 3D polylines.

    ``gap_tolerance``: open chains closer than this are bridged; ``None`` uses
    three times the median segment length, ``0`` disables bridging. Chains with
    fewer than ``min_points`` points (slivers at tangential cuts) are dropped.
    """
    tri_mesh = ensure_triangles(mesh)
    points = np.asarray(tri_mesh.points, dtype=np.float64)
    nodes, links = _cut_segments(points, triangle_array(tri_mesh), plane)
    curve = SectionCurve(plane=plane)
    if not len(links):
        return curve
    seg = np.linalg.norm(nodes[links[:, 0]] - nodes[links[:, 1]], axis=1)
    median_seg = float(np.median(seg)) if len(seg) else 0.0
    dedupe_tol = 1e-9 * max(float(np.ptp(points, axis=0).max()), 1.0)
    polylines, closed = [], []
    for ids, is_closed in _chain(len(nodes), links):
        pts = _dedupe(nodes[ids], is_closed, dedupe_tol)
        polylines.append(pts)
        closed.append(is_closed)
    gap = 3.0 * median_seg if gap_tolerance is None else gap_tolerance
    if gap > 0:
        polylines, closed = bridge_gaps(polylines, closed, gap)
    for pts, is_closed in zip(polylines, closed, strict=True):
        if len(pts) >= min_points:
            curve.polylines.append(pts)
            curve.closed.append(is_closed)
    return curve


def slice_parallel(
    mesh: pv.PolyData, plane: Plane, offsets: Sequence[float], **kwargs
) -> list[SectionCurve]:
    """Sections at ``plane`` shifted by each offset along its normal."""
    return [slice_mesh(mesh, plane.offset(off), **kwargs) for off in offsets]
