"""Accuracy analysis (Design X "Accuracy Analyzer"): mesh ↔ CAD deviation.

Distance field
--------------
For every query point ``p`` (scan vertices, or dense samples of the CAD
surface) the signed shortest Euclidean distance to the reference surface ``S``
is computed:

    d(p) = s(p) · min_{q ∈ S} |p - q|,   s(p) = +1 outside the solid, -1 inside.

The reference B-Rep is tessellated with a small chordal deflection (default
0.02 % of its size, at most a few µm on piston parts) so that the triangle
surface is within that deflection of the exact CAD faces. The closest point on
the triangles is found exactly with a static cell locator (VTK
``vtkImplicitPolyDataDistance``); the sign comes from the angle-weighted
pseudo-normal of the closest feature (face / edge / vertex), which is robust
even at sharp CAD edges (Bærentzen & Aanæs 2005).

Sign convention (as in Design X): **negative = the measured surface lies inside
the CAD (material missing / CAD too big)**, **positive = measured surface
outside the CAD (excess material / CAD too small)**. For the reverse direction
(CAD → mesh) the sign is taken with respect to the measured mesh.

Statistics use only points within ``max_range`` of the reference ("maximum
deviation" in Design X) so regions deliberately not modelled - e.g. pin bosses
on a revolved piston body - are reported as *out of range* instead of
polluting RMS / std.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from meshrev.core.types import FloatArray

Direction = Literal["mesh_to_cad", "cad_to_mesh"]


# ======================================================================================
# Distance field
# ======================================================================================
def signed_distances(points: FloatArray, surface: pv.PolyData) -> FloatArray:
    """Signed shortest distance of ``points`` to the closed triangle ``surface``.

    Negative inside the surface, positive outside (the surface must be closed and
    consistently oriented; tessellated B-Rep solids are).
    """
    pts = np.ascontiguousarray(points, dtype=np.float64)
    implicit = vtk.vtkImplicitPolyDataDistance()
    implicit.SetInput(surface.triangulate() if not surface.is_all_triangles else surface)
    values = vtk.vtkDoubleArray()
    implicit.FunctionValue(numpy_to_vtk(pts, deep=True), values)
    return vtk_to_numpy(values).astype(np.float64).copy()


def closest_points(points: FloatArray, surface: pv.PolyData) -> FloatArray:
    """Closest points on ``surface`` (for deviation vectors / labels)."""
    pts = np.ascontiguousarray(points, dtype=np.float64)
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    out = np.empty_like(pts)
    closest = [0.0, 0.0, 0.0]
    cell_id, sub_id, dist2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
    for i, p in enumerate(pts):
        locator.FindClosestPoint(p, closest, cell_id, sub_id, dist2)
        out[i] = closest
    return out


def sample_surface(surface: pv.PolyData, spacing: float, seed: int = 0) -> FloatArray:
    """Area-uniform random samples on a triangle surface, about one per
    ``spacing²`` (used to measure CAD → mesh: CAD faces with no scan data)."""
    tri = surface.triangulate()
    faces = tri.faces.reshape(-1, 4)[:, 1:]
    p = np.asarray(tri.points)[faces]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    n = int(max(1, min(2_000_000, area.sum() / max(spacing, 1e-9) ** 2)))
    rng = np.random.default_rng(seed)
    which = rng.choice(len(faces), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    bary = np.column_stack([1 - s, s * (1 - r2), s * r2])
    return np.einsum("ni,nij->nj", bary, p[which])


# ======================================================================================
# Result and statistics
# ======================================================================================
@dataclass(eq=False)
class DeviationStats:
    count: int  # points inside max_range (used for the statistics)
    out_of_range: int
    max_positive: float
    max_negative: float
    mean: float
    std: float
    rms: float
    mean_abs: float
    within_tolerance: float  # share of in-range points with |d| <= tolerance
    above_tolerance: float  # share with d > +tolerance
    below_tolerance: float  # share with d < -tolerance
    p95_abs: float

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(eq=False)
class DeviationResult:
    """Signed deviations of ``points`` from a reference surface."""

    points: FloatArray
    distances: FloatArray
    tolerance: float
    max_range: float
    direction: Direction = "mesh_to_cad"
    weights: FloatArray | None = None  # area per point (None = count based)
    stats: DeviationStats = field(init=False)

    def __post_init__(self) -> None:
        self.stats = compute_stats(self.distances, self.tolerance, self.max_range, self.weights)

    @property
    def in_range(self) -> np.ndarray:
        return np.abs(self.distances) <= self.max_range

    # compatibility with the former analysis.deviation API
    @property
    def rms(self) -> float:
        return self.stats.rms

    @property
    def max_abs(self) -> float:
        return max(self.stats.max_positive, -self.stats.max_negative)

    @property
    def within_tolerance_ratio(self) -> float:
        return self.stats.within_tolerance

    def summary(self) -> dict[str, float]:
        return self.stats.as_dict()

    def histogram(self, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
        d = self.distances[self.in_range]
        return np.histogram(d, bins=bins, range=(-self.max_range, self.max_range))

    def report(self) -> dict[str, str]:
        """Human readable metrics for the property panel / reports."""
        s = self.stats
        total = max(len(self.distances), 1)
        return {
            "方向": "网格 → CAD" if self.direction == "mesh_to_cad" else "CAD → 网格",
            "公差": f"±{self.tolerance:.4f}",
            "最大显示范围": f"±{self.max_range:.4f}",
            "统计点数": f"{s.count:,}" + ("（面积加权）" if self.weights is not None else ""),
            "超范围点": f"{s.out_of_range:,} ({100 * s.out_of_range / total:.1f}%)",
            "最大正偏差": f"{s.max_positive:+.4f}",
            "最大负偏差": f"{s.max_negative:+.4f}",
            "平均偏差": f"{s.mean:+.4f}",
            "标准差": f"{s.std:.4f}",
            "RMS": f"{s.rms:.4f}",
            "平均绝对偏差": f"{s.mean_abs:.4f}",
            "95% |偏差| ≤": f"{s.p95_abs:.4f}",
            "公差内": f"{100 * s.within_tolerance:.2f}%",
            "超上公差 (+)": f"{100 * s.above_tolerance:.2f}%",
            "超下公差 (−)": f"{100 * s.below_tolerance:.2f}%",
        }


def compute_stats(
    distances: FloatArray, tolerance: float, max_range: float, weights: FloatArray | None = None
) -> DeviationStats:
    """Deviation metrics on the points within ``max_range``.

    With ``weights`` (surface area represented by each point) every statistic
    is area weighted: dense regions (threads, fillets tessellated finely) do not
    dominate the result, as they would with plain vertex counts.

    mean = Σw·d / Σw,  std = sqrt(Σw (d - mean)² / Σw),  RMS = sqrt(Σw d² / Σw)
    (RMS² = std² + mean²: RMS also captures a systematic offset).
    """
    d = np.asarray(distances, dtype=np.float64)
    w_all = np.ones_like(d) if weights is None else np.asarray(weights, dtype=np.float64)
    keep = np.abs(d) <= max_range
    used, w = d[keep], w_all[keep]
    n = len(used)
    if n == 0 or w.sum() <= 0:
        return DeviationStats(0, len(d), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    w = w / w.sum()
    mean = float(w @ used)
    order = np.argsort(np.abs(used))
    cum = np.cumsum(w[order])
    p95 = float(np.abs(used[order])[min(np.searchsorted(cum, 0.95), n - 1)])
    return DeviationStats(
        count=n,
        out_of_range=int(len(d) - n),
        max_positive=float(max(used.max(), 0.0)),
        max_negative=float(min(used.min(), 0.0)),
        mean=mean,
        std=float(math.sqrt(max(w @ (used - mean) ** 2, 0.0))),
        rms=float(math.sqrt(w @ used**2)),
        mean_abs=float(w @ np.abs(used)),
        within_tolerance=float(w @ (np.abs(used) <= tolerance)),
        above_tolerance=float(w @ (used > tolerance)),
        below_tolerance=float(w @ (used < -tolerance)),
        p95_abs=p95,
    )


def vertex_areas(mesh: pv.PolyData) -> FloatArray:
    """Area represented by each vertex: one third of its adjacent triangle areas."""
    tri = mesh.triangulate() if not mesh.is_all_triangles else mesh
    faces = tri.faces.reshape(-1, 4)[:, 1:]
    p = np.asarray(tri.points)[faces]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    return np.bincount(faces.ravel(), weights=np.repeat(area / 3.0, 3), minlength=tri.n_points)


def reference_surface(reference, tessellation: float | None = None) -> pv.PolyData:
    """Triangle surface of a reference: a ``CadBody`` is re-tessellated finely,
    a ``pv.PolyData`` is used as is."""
    if isinstance(reference, pv.PolyData):
        return reference
    kernel = reference.kernel
    data = reference.to_polydata()
    size = float(np.linalg.norm(np.ptp(np.asarray(data.points), axis=0))) if data.n_points else 1.0
    deflection = tessellation or max(2e-4 * size, 1e-4)
    return kernel.tessellate(reference.shape, deflection)


def compute_deviation(
    measured: pv.PolyData,
    reference,
    tolerance: float = 0.05,
    max_range: float | None = None,
    direction: Direction = "mesh_to_cad",
    tessellation: float | None = None,
    sample_spacing: float | None = None,
) -> DeviationResult:
    """Deviation between a measured mesh and a reference (``CadBody`` or surface).

    * ``mesh_to_cad``: every scan vertex against the CAD (the usual "3D compare");
    * ``cad_to_mesh``: dense samples of the CAD surface against the scan, which
      reveals CAD faces without scan support (e.g. an invented fillet).
    ``max_range`` defaults to ``10 × tolerance``.
    """
    max_range = max_range or 10.0 * tolerance
    surface = reference_surface(reference, tessellation)
    if direction == "mesh_to_cad":
        points = np.asarray(measured.points, dtype=np.float64)
        distances = signed_distances(points, surface)
        weights = vertex_areas(measured) if measured.n_cells else None
    else:
        size = float(np.linalg.norm(np.ptp(np.asarray(surface.points), axis=0)))
        points = sample_surface(surface, sample_spacing or max(size / 400.0, 1e-3))
        # sign relative to the scan: CAD outside the scanned part -> positive
        distances = signed_distances(points, measured)
        weights = None  # area-uniform samples already
    return DeviationResult(
        points, distances, float(tolerance), float(max_range), direction, weights
    )


# ======================================================================================
# Colour map (Design X style: blue - green - red)
# ======================================================================================
def deviation_lut(tolerance: float, max_range: float, bands: int = 15) -> pv.LookupTable:
    """Banded diverging colour table over ``[-max_range, +max_range]``.

    Bands inside ``±tolerance`` are green; below it runs cyan → blue (negative,
    material missing), above it yellow → red (positive, excess material). Values
    beyond the range are shown grey (``above/below range colour``).
    """
    bands = max(3, bands | 1)  # odd: one band centred on zero
    edges = np.linspace(-max_range, max_range, bands + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    colors = np.array([_band_color(c, tolerance, max_range) for c in centers])
    lut = pv.LookupTable()
    lut.values = np.column_stack([np.round(colors * 255), np.full(bands, 255)]).astype(np.uint8)
    lut.scalar_range = (-max_range, max_range)
    lut.below_range_color = (0.55, 0.55, 0.60, 1.0)
    lut.above_range_color = (0.55, 0.55, 0.60, 1.0)
    return lut


def _band_color(value: float, tolerance: float, max_range: float) -> np.ndarray:
    if abs(value) <= tolerance:
        return np.array([0.10, 0.80, 0.25])  # in tolerance: green
    t = min(1.0, (abs(value) - tolerance) / max(max_range - tolerance, 1e-12))
    if value > 0:  # yellow -> orange -> red
        return np.array([1.0, 0.95 - 0.95 * t, 0.10 * (1 - t)])
    return np.array([0.10 * (1 - t), 0.85 - 0.75 * t, 1.0])  # cyan -> blue


def deviation_colors(
    distances: FloatArray, tolerance: float, max_range: float, bands: int = 15
) -> np.ndarray:
    """``(N, 3)`` uint8 colours of the banded map (grey outside the range)."""
    d = np.asarray(distances, dtype=np.float64)
    bands = max(3, bands | 1)
    edges = np.linspace(-max_range, max_range, bands + 1)
    idx = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, bands - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    table = np.array([_band_color(c, tolerance, max_range) for c in centers])
    rgb = table[idx]
    rgb[np.abs(d) > max_range] = (0.55, 0.55, 0.60)
    return (rgb * 255).astype(np.uint8)


def render_heatmap(
    result: DeviationResult,
    mesh: pv.PolyData | None = None,
    plotter: pv.Plotter | None = None,
    bands: int = 15,
) -> pv.Plotter:
    """Stand-alone PyVista false-colour heat map with a colour bar (scripts/reports)."""
    plotter = plotter or pv.Plotter()
    if mesh is not None and result.direction == "mesh_to_cad":
        data = mesh.copy(deep=False)
        data.point_data["偏差"] = result.distances
    else:
        data = pv.PolyData(result.points)
        data.point_data["偏差"] = result.distances
    plotter.add_mesh(
        data,
        scalars="偏差",
        cmap=deviation_lut(result.tolerance, result.max_range, bands),
        clim=(-result.max_range, result.max_range),
        smooth_shading=True,
        point_size=4,
        scalar_bar_args={
            "title": "deviation [mm]",
            "n_labels": 7,
            "fmt": "%+.3f",
            "vertical": True,
        },
    )
    return plotter
