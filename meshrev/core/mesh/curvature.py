"""Per-face principal curvature estimation (normal cycle / edge tensor method).

Every edge e shared by two faces contributes the tensor

    E_e = beta_e * |e| * (ê ⊗ ê)

where beta_e is the signed dihedral angle across the edge and ê its unit
direction (Cohen-Steiner & Morvan 2003). Summing E_e over a neighbourhood B and
dividing by area(B) gives a tensor whose two tangential eigenvalues converge to
the principal curvatures (its eigenvectors are the principal directions rotated
by 90°, which the classifier does not need). Coplanar triangle pairs (beta = 0,
e.g. the diagonal of a planar quad) correctly contribute nothing, and edges
sharper than ``sharp_angle_deg`` are skipped so ring-groove corners do not bleed
into the neighbouring faces. The neighbourhood is enlarged by area weighted
averaging over smooth neighbours (``smoothing_iterations``) to suppress noise.

Sign convention: with outward normals a convex surface has positive curvature
(sphere of radius R: k1 = k2 = +1/R); a bore (hole) has k = -1/R.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from meshrev.core.mesh.topology import MeshGeometry


@dataclass(eq=False)
class FaceCurvature:
    k_max: np.ndarray  # algebraically larger principal curvature per face
    k_min: np.ndarray
    smooth_pairs: np.ndarray  # boolean mask over MeshGeometry.face_pairs

    @property
    def mean(self) -> np.ndarray:
        return 0.5 * (self.k_max + self.k_min)

    @property
    def gaussian(self) -> np.ndarray:
        return self.k_max * self.k_min

    @property
    def abs_max(self) -> np.ndarray:
        return np.maximum(np.abs(self.k_max), np.abs(self.k_min))

    @property
    def abs_min(self) -> np.ndarray:
        return np.minimum(np.abs(self.k_max), np.abs(self.k_min))


def _tangent_frames(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.zeros_like(normals)
    use_x = np.abs(normals[:, 0]) < 0.9
    helper[use_x, 0] = 1.0
    helper[~use_x, 1] = 1.0
    u = np.cross(helper, normals)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-300)
    v = np.cross(normals, u)
    return u, v


def face_curvature(
    geom: MeshGeometry, sharp_angle_deg: float = 30.0, smoothing_iterations: int = 2
) -> FaceCurvature:
    f = geom.n_faces
    n = geom.face_normals
    smooth = geom.pair_normal_cos >= np.cos(np.radians(sharp_angle_deg))
    a, b = geom.face_pairs[smooth].T
    edges = geom.pair_edges[smooth]
    e_vec = geom.points[edges[:, 1]] - geom.points[edges[:, 0]]
    e_len = np.linalg.norm(e_vec, axis=1)
    e_dir = e_vec / np.maximum(e_len, 1e-300)[:, None]
    beta = np.arccos(geom.pair_normal_cos[smooth])
    # convex (positive) when the normal turns in the direction of travel: dn·dp > 0
    offset = geom.face_centroids[b] - geom.face_centroids[a]
    beta = np.where(np.einsum("ij,ij->i", offset, n[b] - n[a]) >= 0, beta, -beta)
    edge_tensor = (0.5 * beta * e_len)[:, None, None] * np.einsum("ei,ej->eij", e_dir, e_dir)
    edge_tensor = edge_tensor.reshape(-1, 9)
    tensor = np.zeros((f, 9))
    for faces in (a, b):  # each face receives half of each of its smooth edges
        for k in range(9):
            tensor[:, k] += np.bincount(faces, weights=edge_tensor[:, k], minlength=f)
    area = geom.face_areas.copy()
    if smoothing_iterations > 0:
        adjacency = geom.adjacency(smooth) + sp.identity(f, format="csr")
        for _ in range(smoothing_iterations):
            tensor = adjacency @ tensor
            area = adjacency @ area
    tensor = (tensor / np.maximum(area, 1e-300)[:, None]).reshape(f, 3, 3)
    u, v = _tangent_frames(n)
    s11 = np.einsum("fi,fij,fj->f", u, tensor, u)
    s12 = np.einsum("fi,fij,fj->f", u, tensor, v)
    s22 = np.einsum("fi,fij,fj->f", v, tensor, v)
    mean = 0.5 * (s11 + s22)
    radius = np.sqrt((0.5 * (s11 - s22)) ** 2 + s12**2)
    return FaceCurvature(k_max=mean + radius, k_min=mean - radius, smooth_pairs=smooth)
