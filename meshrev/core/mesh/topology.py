"""Vectorised triangle-mesh topology and per-face geometry.

:class:`MeshGeometry` is the numeric view of a triangle mesh used by the
recognition algorithms: plain numpy arrays plus lazily computed adjacency.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from meshrev.core.mesh.processing import ensure_triangles, triangle_array
from meshrev.core.types import BBox


class MeshGeometry:
    """Points ``(V, 3)`` and triangles ``(F, 3)`` with cached derived quantities."""

    def __init__(self, points: np.ndarray, faces: np.ndarray) -> None:
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must be an (F, 3) triangle index array")

    @classmethod
    def from_polydata(cls, mesh: pv.PolyData) -> MeshGeometry:
        mesh = ensure_triangles(mesh)
        return cls(np.asarray(mesh.points), triangle_array(mesh))

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def n_points(self) -> int:
        return len(self.points)

    # -- per-face quantities ----------------------------------------------------------
    @cached_property
    def _face_cross(self) -> np.ndarray:
        p = self.points[self.faces]
        return np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])

    @cached_property
    def face_areas(self) -> np.ndarray:
        return 0.5 * np.linalg.norm(self._face_cross, axis=1)

    @cached_property
    def face_normals(self) -> np.ndarray:
        """Unit normals from the triangle winding (zero for degenerate faces)."""
        cross = self._face_cross
        length = np.linalg.norm(cross, axis=1, keepdims=True)
        return np.divide(cross, length, out=np.zeros_like(cross), where=length > 0)

    @cached_property
    def face_centroids(self) -> np.ndarray:
        return self.points[self.faces].mean(axis=1)

    @cached_property
    def bbox(self) -> BBox:
        return BBox.from_points(self.points)

    @cached_property
    def diagonal(self) -> float:
        return self.bbox.diagonal

    @cached_property
    def mean_edge_length(self) -> float:
        p = self.points[self.faces]
        edges = np.linalg.norm(p - np.roll(p, 1, axis=1), axis=2)
        return float(edges.mean())

    # -- per-vertex quantities ----------------------------------------------------------
    @cached_property
    def vertex_faces(self) -> sp.csr_matrix:
        """Sparse ``(V, F)`` incidence matrix."""
        rows = self.faces.ravel()
        cols = np.repeat(np.arange(self.n_faces), 3)
        data = np.ones(rows.size, dtype=np.float64)
        return sp.csr_matrix((data, (rows, cols)), shape=(self.n_points, self.n_faces))

    @cached_property
    def vertex_normals(self) -> np.ndarray:
        """Area weighted vertex normals."""
        summed = self.vertex_faces @ (self.face_normals * self.face_areas[:, None])
        length = np.linalg.norm(summed, axis=1, keepdims=True)
        return np.divide(summed, length, out=np.zeros_like(summed), where=length > 0)

    @cached_property
    def vertex_areas(self) -> np.ndarray:
        """One third of the incident face areas (Voronoi-free barycentric area)."""
        return (self.vertex_faces @ self.face_areas) / 3.0

    # -- adjacency ------------------------------------------------------------------------
    @cached_property
    def _edge_table(self) -> tuple[np.ndarray, np.ndarray]:
        """Face pairs sharing an edge and the shared edge's vertex ids.

        Manifold edges yield one pair; an edge shared by k > 2 faces links the k
        faces in a chain (k - 1 pairs), which keeps the graph connected.
        """
        f = self.faces
        half = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        owner = np.tile(np.arange(self.n_faces), 3)
        half.sort(axis=1)
        key = half[:, 0] * np.int64(self.n_points) + half[:, 1]
        order = np.argsort(key, kind="stable")
        key, owner, half = key[order], owner[order], half[order]
        same = key[1:] == key[:-1]
        pairs = np.column_stack([owner[:-1][same], owner[1:][same]])
        return pairs, half[:-1][same]

    @property
    def face_pairs(self) -> np.ndarray:
        """``(P, 2)`` indices of edge-adjacent faces."""
        return self._edge_table[0]

    @property
    def pair_edges(self) -> np.ndarray:
        """``(P, 2)`` vertex ids of the edge shared by each face pair."""
        return self._edge_table[1]

    @cached_property
    def pair_edge_lengths(self) -> np.ndarray:
        e = self.pair_edges
        return np.linalg.norm(self.points[e[:, 0]] - self.points[e[:, 1]], axis=1)

    @cached_property
    def pair_normal_cos(self) -> np.ndarray:
        """Cosine of the dihedral (normal-to-normal) angle of every face pair."""
        n = self.face_normals
        a, b = self.face_pairs.T
        return np.clip(np.einsum("ij,ij->i", n[a], n[b]), -1.0, 1.0)

    def adjacency(
        self,
        pair_mask: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> sp.csr_matrix:
        """Symmetric sparse face adjacency, optionally restricted to ``pair_mask``."""
        pairs = self.face_pairs if pair_mask is None else self.face_pairs[pair_mask]
        w = np.ones(len(pairs)) if weights is None else np.asarray(weights, dtype=np.float64)
        if pair_mask is not None and weights is not None and len(w) != len(pairs):
            w = w[pair_mask]
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        data = np.concatenate([w, w])
        return sp.csr_matrix((data, (rows, cols)), shape=(self.n_faces, self.n_faces))

    def components(
        self, pair_mask: np.ndarray | None = None, face_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Connected component label per face (``-1`` for faces outside ``face_mask``)."""
        mask = np.ones(len(self.face_pairs), dtype=bool) if pair_mask is None else pair_mask.copy()
        if face_mask is not None:
            a, b = self.face_pairs.T
            mask &= face_mask[a] & face_mask[b]
        _, labels = connected_components(self.adjacency(mask), directed=False)
        labels = labels.astype(np.int64)
        if face_mask is not None:
            labels[~face_mask] = -1
            kept = labels >= 0
            _, labels[kept] = np.unique(labels[kept], return_inverse=True)
        return labels

    def face_vertices(self, face_ids: np.ndarray) -> np.ndarray:
        """Sorted unique vertex ids used by ``face_ids``."""
        used = np.zeros(self.n_points, dtype=bool)
        used[self.faces[np.asarray(face_ids, dtype=np.int64)].ravel()] = True
        return np.flatnonzero(used)

    def boundary_pair_mask(self, labels: np.ndarray) -> np.ndarray:
        """Face pairs whose faces carry different labels."""
        a, b = self.face_pairs.T
        return labels[a] != labels[b]
