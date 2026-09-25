"""Mesh preparation and clean-up operations (thin, documented wrappers over VTK)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv

from meshrev.core.types import BBox

ORIGINAL_CELL_ID = "orig_cell_id"


def ensure_triangles(mesh: pv.DataSet) -> pv.PolyData:
    """Return a triangle-only surface: extracts the surface, triangulates and drops
    degenerate cells, lines and vertices."""
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    if mesh.n_lines or mesh.n_verts or not mesh.is_all_triangles:
        mesh = mesh.triangulate()
        mesh = pv.PolyData(mesh.points, faces=mesh.faces)  # drop lines/verts
    return mesh


def clean(mesh: pv.PolyData, tolerance: float = 0.0) -> pv.PolyData:
    """Merge coincident points (exact by default) and remove degenerate triangles."""
    cleaned = mesh.clean(
        tolerance=tolerance, absolute=True, polys_to_lines=False, lines_to_points=False
    )
    return ensure_triangles(cleaned)


def prepare_mesh(mesh: pv.DataSet) -> pv.PolyData:
    """Standard import pipeline: triangulate + merge duplicated vertices."""
    return clean(ensure_triangles(mesh))


def display_mesh(mesh: pv.PolyData, feature_angle: float = 30.0) -> pv.PolyData:
    """Copy of ``mesh`` with split point normals for smooth shading.

    Vertices are split along edges sharper than ``feature_angle`` so ring grooves
    and lands stay crisp in smooth mode. The cell array ``orig_cell_id`` maps the
    rendered cells back to the source mesh (used by picking and highlighting).
    """
    src = mesh.copy(deep=False)
    src.cell_data[ORIGINAL_CELL_ID] = np.arange(src.n_cells, dtype=np.int64)
    return src.compute_normals(
        cell_normals=False,
        point_normals=True,
        split_vertices=True,
        feature_angle=feature_angle,
        consistent_normals=False,
        auto_orient_normals=False,
    )


def triangle_array(mesh: pv.PolyData) -> np.ndarray:
    """``(F, 3)`` vertex indices of a triangle-only mesh."""
    return mesh.faces.reshape(-1, 4)[:, 1:]


def signed_volume(mesh: pv.PolyData) -> float:
    """Signed enclosed volume; positive when the triangle winding points outwards."""
    tri = mesh.points[triangle_array(mesh)].astype(np.float64)
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def orient_outward(mesh: pv.PolyData) -> pv.PolyData:
    """Flip the winding of a closed mesh whose normals point inwards."""
    return mesh.flip_faces() if signed_volume(mesh) < 0 else mesh


def decimate(mesh: pv.PolyData, reduction: float, preserve_topology: bool = True) -> pv.PolyData:
    """Reduce the triangle count by ``reduction`` (0..1) keeping feature edges."""
    return mesh.decimate_pro(
        reduction, feature_angle=30.0, preserve_topology=preserve_topology, splitting=False
    )


def smooth(mesh: pv.PolyData, iterations: int = 20, pass_band: float = 0.1) -> pv.PolyData:
    """Windowed-sinc smoothing (low shrinkage; keeps sharp feature edges)."""
    return mesh.smooth_taubin(
        n_iter=iterations, pass_band=pass_band, feature_smoothing=False, boundary_smoothing=False
    )


def fill_holes(mesh: pv.PolyData, max_hole_size: float) -> pv.PolyData:
    """Close boundary loops whose circumference-scale size is below ``max_hole_size``."""
    return ensure_triangles(mesh.fill_holes(max_hole_size))


@dataclass(frozen=True)
class MeshStats:
    n_points: int
    n_faces: int
    bbox: BBox
    area: float
    volume: float | None
    n_boundary_edges: int
    n_non_manifold_edges: int

    @property
    def is_closed(self) -> bool:
        return self.n_boundary_edges == 0 and self.n_non_manifold_edges == 0


def mesh_statistics(mesh: pv.PolyData) -> MeshStats:
    boundary = mesh.extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False
    ).n_cells
    non_manifold = mesh.extract_feature_edges(
        boundary_edges=False, feature_edges=False, manifold_edges=False, non_manifold_edges=True
    ).n_cells
    closed = boundary == 0 and non_manifold == 0
    return MeshStats(
        n_points=mesh.n_points,
        n_faces=mesh.n_cells,
        bbox=BBox.from_bounds(mesh.bounds),
        area=float(mesh.area),
        volume=abs(float(mesh.volume)) if closed and mesh.n_cells else None,
        n_boundary_edges=boundary,
        n_non_manifold_edges=non_manifold,
    )
