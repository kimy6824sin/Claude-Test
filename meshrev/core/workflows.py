"""Automatic reverse-engineering pipelines, graded by the accuracy analyzer.

Three strategies cover most piston-engine parts (Claude.md: parts are built from
extrusions, revolutions and boolean cuts):

* :func:`reconstruct_revolved` - pistons, liners, pulleys, flywheels: the axis
  of rotation is the candidate (RANSAC cylinders, dominant flat-face normal)
  about which most of the surface is rotationally consistent; half sections on
  planes through it are revolved (the most accurate angle wins), then concave
  cylinders perpendicular to the axis (pin bores) are cut if they improve the
  accuracy.
* :func:`reconstruct_prismatic` - flanges, brackets, covers: 2.5D "layered
  extrusion". The extrusion direction is the normal of the largest flat faces;
  their heights split the part into layers (split further where the section
  changes); each layer's section is fitted (lines/arcs/circles, holes) and
  extruded over the layer; all layers are fused.
* :func:`reconstruct_hybrid` - mixed parts (a ring gear = revolved dish +
  extruded teeth, a piston = revolved crown + prismatic skirt): zone-wise best
  of the two above, see its docstring.

Every result carries a :class:`~meshrev.core.deviation.DeviationResult`, so the
quality of a reconstruction is always measured, never assumed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

from meshrev.core.bodies import CadBody
from meshrev.core.cad.kernel import BooleanOp, get_kernel
from meshrev.core.deviation import DeviationResult, compute_deviation
from meshrev.core.features.sketching import plane_through_axis
from meshrev.core.mesh.topology import MeshGeometry
from meshrev.core.primitives import (
    CylinderPrimitive,
    MeshPrimitive,
    PrimitiveType,
    RansacOptions,
    SegmentationOptions,
    auto_segment,
    detect_mesh_primitives,
    merge_coaxial_cylinders,
)
from meshrev.core.section.sketch import Line2D, Sketch, loop_polyline
from meshrev.core.section.slicer import slice_mesh
from meshrev.core.sketch_fit import (
    SketchFitOptions,
    fit_section,
    half_profile,
    polygon_area,
)
from meshrev.core.types import Axis, Plane, orthonormal_basis

log = logging.getLogger(__name__)


@dataclass(eq=False)
class Reconstruction:
    """A reverse-engineered solid with its measured accuracy."""

    method: str
    body: CadBody
    deviation: DeviationResult
    steps: list[str] = field(default_factory=list)

    @property
    def volume(self) -> float:
        return float(self.body.kernel.volume(self.body.shape))

    @property
    def valid(self) -> bool:
        return bool(self.body.kernel.is_valid(self.body.shape))

    def score(self) -> float:
        """Share of scan points within tolerance (the headline accuracy number)."""
        s = self.deviation.stats
        n_all = s.count + s.out_of_range
        return s.within_tolerance * s.count / max(n_all, 1)


def _grade(mesh: pv.PolyData, body: CadBody, tolerance: float, max_range: float) -> DeviationResult:
    return compute_deviation(mesh, body, tolerance=tolerance, max_range=max_range)


# ======================================================================================
# Revolved parts
# ======================================================================================
def fit_axis_origin(
    mesh: pv.PolyData, direction: np.ndarray, start: np.ndarray | None = None, iterations: int = 4
) -> Axis:
    """Least-squares axis position for a fixed direction ``a``.

    With m_i = a x n_i the coplanarity condition of :func:`revolution_score` is
    linear in the axis origin c:  m_i . c = m_i . p_i.  The area-weighted normal
    equations are solved (rank 2: c is free along a, fixed by the minimum-norm
    solution and moved to the part's centroid height).  Iteratively reweighted:
    faces whose normal line misses the current axis by more than twice the median
    residual (teeth flanks, pockets) are dropped, so non-rotational features do not
    drag the axis."""
    geom = MeshGeometry.from_polydata(mesh)
    a = direction / np.linalg.norm(direction)
    p, w = geom.face_centroids, geom.face_areas
    m = np.cross(a, geom.face_normals)
    rhs = np.einsum("ij,ij->i", m, p)
    centroid = (p * w[:, None]).sum(axis=0) / w.sum()
    c = centroid if start is None else np.asarray(start, float)
    keep = np.ones(len(p), bool)
    for _ in range(iterations):
        mw = m[keep] * w[keep, None]
        c_new, *_ = np.linalg.lstsq(mw.T @ m[keep], mw.T @ rhs[keep], rcond=None)
        c = c_new - ((c_new - centroid) @ a) * a
        resid = np.abs(m @ c - rhs)
        keep = resid <= 2.0 * max(float(np.median(resid)), 1e-9)
    return Axis(c, a)


def revolution_score(mesh: pv.PolyData, axis: Axis, angle_tol_deg: float = 3.0) -> float:
    """Area share of the surface that is consistent with rotation about ``axis``.

    Every normal line of a surface of revolution meets its axis, i.e. the three
    vectors a (axis direction), n (normal) and p - c (point relative to the axis
    origin) are coplanar: det[a, n, p - c] = (a x n) . (p - c) = 0.  Dividing by
    |p - c| turns the triple product into the sine of an angle, which is compared
    against ``angle_tol_deg``.  Faces on the axis are ignored (undefined)."""
    geom = MeshGeometry.from_polydata(mesh)
    a = axis.direction / np.linalg.norm(axis.direction)
    rel = geom.face_centroids - np.asarray(axis.origin, float)
    dist = np.linalg.norm(rel, axis=1)
    ok = dist > 1e-9
    sine = np.abs(np.einsum("ij,ij->i", np.cross(a, geom.face_normals), rel))
    sine[ok] /= dist[ok]
    good = ok & (sine <= math.sin(math.radians(angle_tol_deg)))
    area = geom.face_areas
    return float(area[good].sum() / max(area.sum(), 1e-12))


def main_axis(mesh: pv.PolyData) -> tuple[Axis, float, str]:
    """Axis of rotation of the part: ``(axis, revolution score, source)``.

    Candidates are the RANSAC cylinders of the part and the dominant flat-face
    normal (end faces of a disc are perpendicular to its axis); each is refined
    with :func:`fit_axis_origin` and the one about which the largest share of the
    surface is rotationally consistent wins (:func:`revolution_score`).  This
    rejects spurious cylinders fitted to gear teeth or free-form patches, which a
    largest-support rule alone would pick."""
    geom = MeshGeometry.from_polydata(mesh)
    found = detect_mesh_primitives(
        geom, (PrimitiveType.CYLINDER,), RansacOptions(min_support=100), max_primitives=6
    )
    candidates = [
        (
            Axis(c.primitive.axis_point, c.primitive.axis_direction),
            f"圆柱 r={c.primitive.radius:.3f}",
        )
        for c in found
    ]
    candidates.append((Axis(np.zeros(3), dominant_direction(mesh)), "主平面法向"))
    best = None
    for axis, source in candidates:
        for refined in (axis, fit_axis_origin(mesh, axis.direction, axis.origin)):
            score = revolution_score(mesh, refined)
            if best is None or score > best[1]:
                best = (refined, score, source)
    return best


def _radial(points: np.ndarray, axis: Axis) -> tuple[np.ndarray, np.ndarray]:
    """(height, radius) of points in the cylindrical frame of ``axis``."""
    a = axis.direction / np.linalg.norm(axis.direction)
    rel = np.asarray(points, float) - np.asarray(axis.origin, float)
    h = rel @ a
    return h, np.linalg.norm(rel - np.outer(h, a), axis=1)


def cross_holes(
    mesh: pv.PolyData,
    axis: Axis,
    min_angle_deg: float = 60.0,
    max_radius_ratio: float = 0.45,
    interior_ratios: tuple[float, ...] = (0.7, 0.8, 0.9),
) -> list[CylinderPrimitive]:
    """Concave cylinders roughly perpendicular to the main axis (pin bores, oil
    holes); coaxial pieces (both bosses) are merged.

    Auto segmentation finds them on clean meshes.  On scans whose smooth blends
    make segmentation merge the bosses into one free-form region, RANSAC is run
    on the part's interior (radius < ``interior_ratios`` x outer radius), where
    the bores are the dominant cylinders."""
    geom = MeshGeometry.from_polydata(mesh)
    _, r_vertices = _radial(geom.points, axis)
    outer = float(np.percentile(r_vertices, 99))
    total = float(geom.face_areas.sum())

    def accept(cyl: CylinderPrimitive) -> bool:
        return (
            bool(cyl.concave)
            and math.degrees(cyl.axis.angle_to(axis)) >= min_angle_deg
            and cyl.radius <= max_radius_ratio * outer
        )

    seg = auto_segment(geom, SegmentationOptions())
    pieces = [
        MeshPrimitive(region.fit, region.face_ids)
        for region in seg.regions_of_type(PrimitiveType.CYLINDER)
        if accept(region.primitive)
    ]
    if not pieces:
        _, r_faces = _radial(geom.face_centroids, axis)
        # RANSAC is stochastic and greedy: several interior shells make it robust
        for ratio in interior_ratios:
            found = detect_mesh_primitives(
                geom,
                (PrimitiveType.CYLINDER,),
                RansacOptions(min_support=50),
                8,
                face_mask=r_faces < ratio * outer,
            )
            pieces += [item for item in found if accept(item.primitive)]
    merged = merge_coaxial_cylinders(geom, pieces, angle_tol_deg=2.0)
    # a real bore is supported by a reasonable area
    return [m.primitive for m in merged if float(geom.face_areas[m.face_ids].sum()) > 2e-3 * total]


def revolve_sections(
    mesh: pv.PolyData,
    axis: Axis,
    angles_deg: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0),
    tolerance: float = 0.1,
    max_range: float = 1.0,
    steps: list[str] | None = None,
) -> list[Reconstruction]:
    """One graded revolve per half-section angle (sections that fail are skipped)."""
    kernel = get_kernel()
    out = []
    for angle in angles_deg:
        plane = plane_through_axis(axis, angle)
        curve = slice_mesh(mesh, plane)
        if curve.is_empty:
            continue
        half = half_profile(fit_section(curve))
        sketches = half.to_sketches()
        if not sketches:
            continue
        u, _ = plane.basis()
        try:
            solid = kernel.fuse_all(
                [kernel.revolve(s, Axis(plane.origin, u), 360.0) for s in sketches]
            )
        except Exception as exc:  # noqa: BLE001 - try the next section angle
            log.info("revolve at %s° failed: %s", angle, exc)
            continue
        body = CadBody(solid, f"旋转体 {angle:.0f}°", kernel=kernel)
        result = Reconstruction(
            "revolve",
            body,
            _grade(mesh, body, tolerance, max_range),
            list(steps or []) + [f"半截面 @ {angle:.0f}°: {half.counts()}"],
        )
        log.info("revolve @%s°: score %.3f", angle, result.score())
        out.append(result)
    return out


def cut_cross_holes(
    mesh: pv.PolyData,
    rec: Reconstruction,
    axis: Axis,
    tolerance: float = 0.1,
    max_range: float = 1.0,
) -> Reconstruction:
    """Boolean-cut the concave cylinders perpendicular to ``axis`` (pin bores).

    Candidates are tried largest first and each cut is kept only if it improves
    the measured accuracy - on scans, RANSAC may also return a relief or chamfer
    cylinder next to the real bore, and the deviation decides between them."""
    kernel = get_kernel()
    holes = sorted(cross_holes(mesh, axis), key=lambda c: -c.radius * c.length)
    if not holes:
        return rec
    best = rec
    _, r_all = _radial(np.asarray(mesh.points), axis)
    reach = float(r_all.max()) + 0.05 * float(mesh.length)
    for hole in holes:
        d = hole.axis_direction / np.linalg.norm(hole.axis_direction)
        # signed offset of the measured bore from the main axis along its own axis
        s = float((hole.axis_point - np.asarray(axis.origin, float)) @ d)
        if abs(s) < 0.25 * hole.length:  # spans both sides: through all
            t0, t1 = -reach, reach
        else:  # one boss only: from its inner end outwards (the other may differ)
            side = math.copysign(1.0, s)
            t0, t1 = sorted((s - side * 0.5 * hole.length, side * reach))
        centre = hole.axis_point + (0.5 * (t0 + t1) - s) * d
        tool = kernel.cylinder(Axis(centre, d), hole.radius, t1 - t0)
        shape = kernel.boolean(best.body.shape, tool, BooleanOp.CUT)
        body = CadBody(shape, f"{rec.body.name} + 横孔", kernel=kernel)
        trial = Reconstruction(
            rec.method if rec.method.endswith("+holes") else f"{rec.method}+holes",
            body,
            _grade(mesh, body, tolerance, max_range),
            [*best.steps, f"横向孔: r={hole.radius:.3f} 轴 {np.round(d, 3)}"],
        )
        if trial.score() > best.score():
            best = trial
        else:
            best.steps.append(f"  (候选孔 r={hole.radius:.3f} 未改善精度，未切除)")
    return best


def reconstruct_revolved(
    mesh: pv.PolyData,
    axis: Axis | None = None,
    angles_deg: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0),
    tolerance: float = 0.1,
    max_range: float = 1.0,
    cut_holes: bool = True,
) -> Reconstruction:
    steps: list[str] = []
    if axis is None:
        axis, score, source = main_axis(mesh)
        steps.append(
            f"主轴 ({source}): 方向 {np.round(axis.direction, 4)}, 回转一致面积 {100 * score:.1f}%"
        )
    candidates = revolve_sections(mesh, axis, angles_deg, tolerance, max_range, steps)
    if not candidates:
        raise ValueError("no usable section through the axis")
    best = max(candidates, key=Reconstruction.score)
    return cut_cross_holes(mesh, best, axis, tolerance, max_range) if cut_holes else best


# ======================================================================================
# Prismatic parts (2.5D)
# ======================================================================================
def dominant_direction(
    mesh: pv.PolyData, cos_limit: float = math.cos(math.radians(3.0))
) -> np.ndarray:
    """Normal direction carrying the most flat-face area (the extrusion axis)."""
    geom = MeshGeometry.from_polydata(mesh)
    n, a = geom.face_normals, geom.face_areas
    order = np.argsort(-a)
    candidates = [np.eye(3)[k] for k in range(3)]
    candidates += [n[i] for i in order[:: max(1, len(order) // 200)][:200]]
    best, best_area = None, -1.0
    for d in candidates:
        area = float(a[np.abs(n @ d) >= cos_limit].sum())
        if area > best_area:
            best, best_area = d, area
    # refine as the area-weighted mean of the aligned normals
    aligned = np.abs(n @ best) >= cos_limit
    signs = np.sign(n[aligned] @ best)[:, None]
    d = (n[aligned] * signs * a[aligned, None]).sum(axis=0)
    return d / np.linalg.norm(d)


def step_heights(
    mesh: pv.PolyData,
    direction: np.ndarray,
    min_area_ratio: float = 0.01,
    merge: float | None = None,
) -> list[float]:
    """Heights (along ``direction``) of flat faces perpendicular to it: the steps
    that bound the layers of a 2.5D part. Always includes the part's extent."""
    geom = MeshGeometry.from_polydata(mesh)
    n, a, c = geom.face_normals, geom.face_areas, geom.face_centroids
    h = c @ direction
    flat = np.abs(n @ direction) >= math.cos(math.radians(3.0))
    lo, hi = float((geom.points @ direction).min()), float((geom.points @ direction).max())
    merge = merge or max(1e-3 * (hi - lo), 0.05)
    order = np.argsort(h[flat])
    hs, areas = h[flat][order], a[flat][order]
    levels = []
    start = 0
    for i in range(1, len(hs) + 1):
        if i == len(hs) or hs[i] - hs[i - 1] > merge:
            cluster_area = areas[start:i].sum()
            if cluster_area >= min_area_ratio * a.sum():
                levels.append(float(np.average(hs[start:i], weights=areas[start:i])))
            start = i
    levels = sorted(set([lo, hi, *levels]))
    out = [levels[0]]
    for value in levels[1:]:
        if value - out[-1] > 2 * merge:
            out.append(value)
        else:
            out[-1] = max(out[-1], value) if value == hi else out[-1]
    if out[-1] < hi - 2 * merge:
        out.append(hi)
    else:
        out[-1] = hi
    return out


def _section_distance(curve_pts: list[np.ndarray], polygons: list[np.ndarray]) -> float:
    """How far a section is from the fitted polygons (2D): the 95th percentile of
    point-to-edge distances over the *large* loops. Small loops (tapped holes,
    whose thread section changes with height by design, oil holes) and isolated
    outliers must not trigger a split."""
    if not curve_pts or not polygons:
        return math.inf
    lengths = [float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in curve_pts]
    big = [p for p, n in zip(curve_pts, lengths, strict=True) if n >= 0.15 * max(lengths)]
    edges = np.vstack([np.stack([p, np.roll(p, -1, axis=0)], axis=1) for p in polygons])
    a, b = edges[:, 0], edges[:, 1]
    ab = b - a
    denom = np.maximum((ab**2).sum(axis=1), 1e-300)
    dists = []
    for pts in big:
        sub = pts[:: max(1, len(pts) // 400)]
        t = np.clip(((sub[:, None] - a) * ab).sum(axis=2) / denom, 0.0, 1.0)
        closest = a + t[..., None] * ab
        dists.append(np.linalg.norm(sub[:, None] - closest, axis=2).min(axis=1))
    return float(np.percentile(np.concatenate(dists), 95))


def _layer_sketches(mesh, base, d, x_axis, h0, h1, fit_options, tolerance, depth, steps):
    """Sketches for layer [h0, h1]; split in two while the sections at the layer
    ends differ from the mid-section fit by more than ``tolerance`` (drafted,
    lofted or curved walls become a fine staircase)."""
    mid = Plane(base + 0.5 * (h0 + h1) * d, d, x_axis=x_axis)
    curve = slice_mesh(mesh, mid)
    if curve.is_empty:
        return []
    result = fit_section(curve, fit_options)
    sketches = result.to_sketches()
    thickness = h1 - h0
    if depth > 0 and thickness > 1.0 and sketches:
        polygons = [loop_polyline(loop.entities) for loop in result.loops if loop.closed]
        # quartile sections: edge fillets / chamfers at the layer ends do not
        # count, only drafted or lofted walls (the section changes along height)
        worst = 0.0
        for h in (h0 + 0.25 * thickness, h1 - 0.25 * thickness):
            end = slice_mesh(mesh, Plane(base + h * d, d, x_axis=x_axis))
            worst = max(worst, _section_distance(end.to_2d(), polygons))
        if worst > tolerance:
            hm = 0.5 * (h0 + h1)
            return _layer_sketches(
                mesh, base, d, x_axis, h0, hm, fit_options, tolerance, depth - 1, steps
            ) + _layer_sketches(
                mesh, base, d, x_axis, hm, h1, fit_options, tolerance, depth - 1, steps
            )
    steps.append(f"层 [{h0:.3f}, {h1:.3f}]: {result.counts()}")
    out = []
    for sketch in sketches:
        sketch.plane = Plane(base + h0 * d, d, x_axis=x_axis)
        out.append((sketch, thickness))
    return out


def reconstruct_prismatic(
    mesh: pv.PolyData,
    direction: np.ndarray | None = None,
    tolerance: float = 0.1,
    max_range: float = 1.0,
    fit_options: SketchFitOptions | None = None,
    max_depth: int = 4,
    min_layer: float = 0.3,
) -> Reconstruction:
    kernel = get_kernel()
    d = dominant_direction(mesh) if direction is None else np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    heights = step_heights(mesh, d)
    # thin slivers (chamfer / fillet bands) join the neighbouring layer
    merged = [heights[0]]
    for h in heights[1:]:
        if h - merged[-1] < min_layer and len(merged) > 1:
            merged[-1] = h
        elif h - merged[-1] >= min_layer or len(merged) == 1:
            merged.append(h)
    heights = merged
    x_axis, _ = orthonormal_basis(d)
    center = np.asarray(mesh.center)
    base = center - (center @ d) * d
    steps = [f"拉伸方向 {np.round(d, 4)}", f"台阶高度 {np.round(heights, 3).tolist()}"]
    solids = []
    tol = fit_options.tolerance if fit_options and fit_options.tolerance else None
    for h0, h1 in zip(heights[:-1], heights[1:], strict=True):
        layers: dict[float, list] = {}
        for sketch, thickness in _layer_sketches(
            mesh, base, d, x_axis, h0, h1, fit_options, tolerance, max_depth, steps
        ):
            start = float((sketch.plane.origin - base) @ d)
            layers.setdefault(start, []).append((sketch, thickness))
        for start, items in layers.items():
            thickness = items[0][1]
            layer = [kernel.extrude(sk, d, t) for sk, t in items]
            if all(kernel.is_valid(x) for x in layer):
                solids.extend(layer)
                continue
            # a fitted loop self-intersects (tight feature): refit this layer finer,
            # then as a pure polygon (always simple) - never drop material
            mid = Plane(base + (start + 0.5 * thickness) * d, d, x_axis=x_axis)
            curve = slice_mesh(mesh, mid)
            repaired = None
            for opts in (
                SketchFitOptions(tolerance=0.5 * tol if tol else None),
                SketchFitOptions(tolerance=tol, max_radius_ratio=0.0, hole_circularity=0.0),
            ):
                result = fit_section(curve, opts)
                candidate = []
                for sk in result.to_sketches():
                    sk.plane = Plane(base + start * d, d, x_axis=x_axis)
                    candidate.append(kernel.extrude(sk, d, thickness))
                if candidate and all(kernel.is_valid(x) for x in candidate):
                    repaired = candidate
                    break
            if repaired is None:
                steps.append(f"  ⚠ 层 {start:.3f} 轮廓无法修复，已跳过")
                continue
            steps.append(f"  层 {start:.3f}: 轮廓自相交，已用更细公差/折线重拟合")
            solids.extend(repaired)
    if not solids:
        raise ValueError("no closed section in any layer")
    fused = kernel.fuse_all(solids)
    cleaned = kernel.clean(fused)
    # face unification occasionally breaks a many-layer fuse: keep the valid one
    shape = cleaned if kernel.is_valid(cleaned) or not kernel.is_valid(fused) else fused
    body = CadBody(shape, "分层拉伸体", kernel=kernel)
    return Reconstruction("layered extrude", body, _grade(mesh, body, tolerance, max_range), steps)


# ======================================================================================
# Hybrid: zone-wise best of several reconstructions
# ======================================================================================
def grid_region_loops(
    mask: np.ndarray, row_edges: np.ndarray, col_edges: np.ndarray
) -> list[np.ndarray]:
    """Boundary loops of the union of the ``True`` cells of a rectilinear grid.

    Cell (i, j) spans ``row_edges[i:i+2] x col_edges[j:j+2]`` in (x, y).  Every
    cell contributes its four edges counter-clockwise; an edge shared by two
    selected cells appears once in each direction and cancels, so what remains
    is exactly the region boundary (outer loops CCW, holes CW).  The directed
    edges are chained into loops and collinear vertices removed."""
    edges: dict[tuple, tuple] = {}
    for i, j in zip(*np.nonzero(mask), strict=True):
        corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
        for p, q in zip(corners, corners[1:] + corners[:1], strict=True):
            if (q, p) in edges:
                del edges[(q, p)]
            else:
                edges[(p, q)] = True
    outgoing: dict[tuple, list] = {}
    for p, q in edges:
        outgoing.setdefault(p, []).append(q)
    loops = []
    while outgoing:
        start = next(iter(outgoing))
        loop, p, prev = [start], start, None
        while True:
            options = outgoing[p]
            if len(options) > 1 and prev is not None:
                # pinch vertex (cells touching diagonally): turn left, i.e. keep
                # hugging the current cell, so the two regions stay separate loops
                din = (p[0] - prev[0], p[1] - prev[1])
                options.sort(key=lambda q: din[0] * (q[1] - p[1]) - din[1] * (q[0] - p[0]))
            q = options.pop()
            if not outgoing[p]:
                del outgoing[p]
            if q == start:
                break
            loop.append(q)
            prev, p = p, q
        pts = np.array([[row_edges[i], col_edges[j]] for i, j in loop], float)
        prev, nxt = np.roll(pts, 1, axis=0), np.roll(pts, -1, axis=0)
        e0, e1 = pts - prev, nxt - pts
        turn = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
        loops.append(pts[np.abs(turn) > 1e-12])
    return loops


def loops_to_sketches(loops: list[np.ndarray], plane: Plane) -> list[Sketch]:
    """Polygon loops (outer CCW, holes CW) to line sketches, holes assigned to the
    outer loop that contains them."""
    outers = [lp for lp in loops if polygon_area(lp) > 0]
    holes = [lp for lp in loops if polygon_area(lp) < 0]

    def lines(lp: np.ndarray) -> list:
        return [
            Line2D(tuple(map(float, p)), tuple(map(float, q)))
            for p, q in zip(lp, np.roll(lp, -1, axis=0), strict=True)
        ]

    sketches = []
    for outer in outers:
        inner = [h for h in holes if _point_in_polygon(_inside_point(h), outer)]
        sketches.append(Sketch(plane, lines(outer), [lines(h) for h in inner]))
    return sketches


def _inside_point(loop: np.ndarray) -> np.ndarray:
    """A point just inside a polygon: off the first edge's midpoint, to the left
    for a CCW loop and to the right for a CW (hole) loop."""
    p, q = loop[0], loop[1]
    d = q - p
    left = np.array([-d[1], d[0]])
    sign = 1.0 if polygon_area(loop) > 0 else -1.0
    return 0.5 * (p + q) + sign * 1e-6 * left / np.linalg.norm(d)


def _point_in_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
    """Even-odd ray casting."""
    x, y = point
    inside = False
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def reconstruct_hybrid(
    mesh: pv.PolyData,
    candidates: list[Reconstruction],
    axis: Axis,
    tolerance: float = 0.1,
    max_range: float = 1.0,
    radial_bins: int = 48,
    axial_bins: int = 24,
    min_share: float = 0.0003,
    switch_gain: float = 0.1,
) -> Reconstruction:
    """Zone-wise combination of candidate solids (Design X "混合建模" style).

    Real parts are rarely one feature type: a ring gear is a revolved dish with
    extruded teeth, a piston a revolved crown with a prismatic, cut-away skirt.
    Space is partitioned in the cylindrical frame of ``axis`` into *axial slabs*
    (between the step heights perpendicular to the axis) and *radial bins*.  In
    every cell the candidate with the largest area of scan points within
    tolerance wins (the per-point deviations already exist, so the choice costs
    no extra geometry).  Adjacent bins with the same winner merge into annular
    zones Z; the result is  U_c ( candidate_c ∩ U_{Z won by c} Z ),  i.e. every
    candidate contributes only where it is measurably the most accurate.
    Empty cells inherit the slab's (or the globally) best candidate, and the outer
    zones are padded so that no material is clipped at the part's extent."""
    kernel = get_kernel()
    ref = candidates[0].deviation
    for cand in candidates:
        if len(cand.deviation.points) != len(ref.points):
            raise ValueError("candidates must be graded on the same scan points")
    a = axis.direction / np.linalg.norm(axis.direction)
    o = np.asarray(axis.origin, float)
    o = o - (o @ a) * a  # heights are absolute projections, as in step_heights
    rel = ref.points - o
    h = rel @ a
    r = np.linalg.norm(rel - np.outer(h, a), axis=1)
    w = ref.weights if ref.weights is not None else np.ones(len(h))
    good = (
        np.stack(
            [
                (np.abs(c.deviation.distances) <= tolerance) & np.isfinite(c.deviation.distances)
                for c in candidates
            ]
        ).astype(float)
        * w
    )
    global_best = int(np.argmax([c.score() for c in candidates]))

    # axial slabs: the step heights, subdivided so that no slab exceeds 1/axial_bins
    steps_h = step_heights(mesh, a)
    max_slab = (steps_h[-1] - steps_h[0]) / axial_bins
    heights = [steps_h[0]]
    for h1 in steps_h[1:]:
        n = max(1, math.ceil((h1 - heights[-1]) / max_slab - 1e-9))
        heights += list(np.linspace(heights[-1], h1, n + 1)[1:])
    r_max = float(r.max())
    r_edges = np.linspace(0.0, r_max, radial_bins + 1)
    slab_of = np.clip(np.searchsorted(heights, h, side="right") - 1, 0, len(heights) - 2)
    bin_of = np.clip(np.searchsorted(r_edges, r, side="right") - 1, 0, radial_bins - 1)
    total = float(w.sum())
    n_slabs = len(heights) - 1
    grid = np.empty((n_slabs, radial_bins), int)
    for k in range(n_slabs):
        in_slab = slab_of == k
        score = np.zeros((len(candidates), radial_bins))
        mass = np.zeros(radial_bins)
        np.add.at(mass, bin_of[in_slab], w[in_slab])
        for c in range(len(candidates)):
            np.add.at(score[c], bin_of[in_slab], good[c, in_slab])
        slab_best = int(np.argmax(score.sum(axis=1))) if mass.sum() > 0 else global_best
        # switch away from the slab's best only for a clear gain (seams cost accuracy)
        winner = np.argmax(score, axis=0)
        gain = score[winner, np.arange(radial_bins)] - score[slab_best]
        winner = np.where(gain > switch_gain * mass, winner, slab_best)
        labels = np.where(mass >= min_share * total, winner, -1)
        # fill sparse bins from the neighbours, else the slab's best
        for i in np.flatnonzero(labels < 0):
            left = labels[:i][labels[:i] >= 0]
            labels[i] = left[-1] if left.size else slab_best
        grid[k] = labels
    # zone boundaries in the (h, r) half plane; the outer ones are padded
    pad = 0.05 * float(mesh.length)
    h_edges = np.array(heights, float)
    h_edges[0] -= pad
    h_edges[-1] += pad
    r_edges = r_edges.copy()
    r_edges[-1] = r_max + pad
    plane_normal = np.cross(a, orthonormal_basis(a)[0])
    plane = Plane(o, plane_normal, x_axis=a)
    pieces = []
    steps = [f"混合建模: {len(candidates)} 个候选, {n_slabs} 轴向层 × {radial_bins} 径向环"]
    for label in np.unique(grid):
        loops = grid_region_loops(grid == label, h_edges, r_edges)
        # each zone is a revolved rectilinear (h, r) profile: one clean solid, no
        # fusing of many coincident-faced rings (fragile in OCC booleans)
        # each zone is a revolved rectilinear (h, r) profile, intersected with the
        # candidate separately (zones are disjoint; fusing them first can fail).
        # The common part is symmetric, but OCC's classification of a large
        # faceted solid occasionally fails in one argument order (almost nothing
        # is returned); both orders are computed and the larger one kept.
        shape = candidates[label].body.shape
        for sketch in loops_to_sketches(loops, plane):
            zone = kernel.revolve(sketch, Axis(o, a), 360.0)
            piece = max(
                (
                    kernel.boolean(zone, shape, BooleanOp.INTERSECT),
                    kernel.boolean(shape, zone, BooleanOp.INTERSECT),
                ),
                key=kernel.volume,
            )
            if kernel.volume(piece) > 0:
                pieces.append(piece)
        share = float((grid == label).mean())
        steps.append(f"  {candidates[label].body.name}: {100 * share:.1f}% 的区域单元")
    fused = kernel.fuse_all(pieces)
    cleaned = kernel.clean(fused)
    shape = cleaned if kernel.is_valid(cleaned) or not kernel.is_valid(fused) else fused
    body = CadBody(shape, "混合重建体", kernel=kernel)
    return Reconstruction("hybrid", body, _grade(mesh, body, tolerance, max_range), steps)


def reconstruct(
    mesh: pv.PolyData, tolerance: float = 0.1, max_range: float = 1.0, hybrid: bool = True
) -> list[Reconstruction]:
    """Try every strategy and return the successful ones, best first.

    Revolve (every section angle) and layered extrusion are graded; when the part
    has a rotation axis, their zone-wise hybrid is built too (with the cross holes
    cut afterwards)."""
    results: list[Reconstruction] = []
    revolves: list[Reconstruction] = []
    axis = None
    try:
        axis, score, source = main_axis(mesh)
        note = (
            f"主轴 ({source}): 方向 {np.round(axis.direction, 4)}, 回转一致面积 {100 * score:.1f}%"
        )
        revolves = revolve_sections(
            mesh, axis, tolerance=tolerance, max_range=max_range, steps=[note]
        )
        if revolves:
            best = max(revolves, key=Reconstruction.score)
            results.append(cut_cross_holes(mesh, best, axis, tolerance, max_range))
    except Exception as exc:  # noqa: BLE001 - a strategy may not apply
        log.info("revolve failed: %s", exc)
    try:
        results.append(reconstruct_prismatic(mesh, tolerance=tolerance, max_range=max_range))
    except Exception as exc:  # noqa: BLE001
        log.info("prismatic failed: %s", exc)
    candidates = [max(revolves, key=Reconstruction.score)] if revolves else []
    candidates += [r for r in results if r.method == "layered extrude"]
    if hybrid and axis is not None and len(candidates) > 1:
        try:
            mixed = reconstruct_hybrid(mesh, candidates, axis, tolerance, max_range)
            results.append(cut_cross_holes(mesh, mixed, axis, tolerance, max_range))
        except Exception as exc:  # noqa: BLE001
            log.info("hybrid failed: %s", exc)
    return sorted(results, key=lambda r: -r.score())
