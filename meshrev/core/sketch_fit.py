"""2D profile fitting for mesh sketches (Design X "Mesh Sketch").

Input: section polylines (``slice_mesh``) expressed in the sketch plane's 2D
frame. Output: structured, CAD-ready entities - ``Line2D(p1, p2)``,
``Arc2D(center, radius, start_angle, end_angle, ccw)``, ``Circle2D(center, r)`` -
organised in loops with an outer/hole hierarchy (:class:`SketchFitResult`,
:meth:`SketchFitResult.to_sketches` feeds ``CadKernel.extrude/revolve``).

Pipeline per section
--------------------
1. **Loop hierarchy** - containment depth of every closed loop (even-odd rule):
   depth 0 = outer boundary, 1 = hole (pin bore wall, bolt hole), 2 = island ...
   Outer loops are oriented counter-clockwise, holes clockwise. Loops shorter
   than ``min_loop_length`` are dropped as noise; open chains (sections of open
   meshes, unbridgeable gaps) are fitted as open profiles and reported.
2. **Denoising & ordering** - the slicer already returns ordered points. They
   are densified/thinned to a uniform spacing ``h`` *keeping every original
   vertex* (so true corners survive), and isolated spikes (points far from the
   chord of their neighbours) are removed.
3. **Segmentation** - strong corners (turning angle > ``corner_angle_deg``) are
   hard breaks. Between them a greedy maximal-extent search (exponential +
   binary search) grows the longest line or arc that stays within
   ``tolerance``; breakpoints between neighbours are then moved to minimise the
   total squared error (tangent line/arc transitions have no corner), and
   adjacent collinear lines / co-circular arcs are merged.
4. **Fitting** - total least squares lines, algebraic (Kasa) + geometric
   (Levenberg-Marquardt) circles, both with a RANSAC fallback for outliers.
5. **Topology** - consecutive entities share one exact vertex: the
   intersection of the two fitted curves nearest to the break (or a projection
   for tangent/parallel neighbours). Arcs are rebuilt through (start vertex,
   fitted mid point, end vertex), so loops close to machine precision.
6. **Design intent** - lines within ``snap_angle_deg`` of the sketch axes are
   made exactly parallel to them (measured noise is not design).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import least_squares

from meshrev.core.section.sketch import (
    Arc2D,
    Circle2D,
    Line2D,
    Sketch,
    SketchEntity,
    loop_polyline,
)
from meshrev.core.section.slicer import SectionCurve
from meshrev.core.types import FloatArray, Plane

# Public aliases matching the notation Line(p1, p2), Arc(center, r, start, end)
Line = Line2D
Arc = Arc2D
Circle = Circle2D

Kind = Literal["line", "arc"]
TWO_PI = 2.0 * math.pi


# ======================================================================================
# Primitive fits (2D)
# ======================================================================================
@dataclass(frozen=True)
class LineFit:
    """Infinite line ``point + t * direction`` (unit direction)."""

    point: FloatArray
    direction: FloatArray

    def residuals(self, pts: FloatArray) -> FloatArray:
        """Signed perpendicular distances: ``(p - p0) × d`` (2D cross product)."""
        rel = pts - self.point
        return rel[:, 0] * self.direction[1] - rel[:, 1] * self.direction[0]

    def project(self, pts: FloatArray) -> FloatArray:
        t = (pts - self.point) @ self.direction
        return self.point + t[:, None] * self.direction


@dataclass(frozen=True)
class CircleFit:
    center: FloatArray
    radius: float

    def residuals(self, pts: FloatArray) -> FloatArray:
        """Signed radial distances ``|p - c| - r``."""
        return np.linalg.norm(pts - self.center, axis=1) - self.radius

    def project(self, pts: FloatArray) -> FloatArray:
        rel = pts - self.center
        norm = np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-300)
        return self.center + self.radius * rel / norm


def fit_line_lsq(points: FloatArray, weights: FloatArray | None = None) -> LineFit:
    """Total least squares line.

    Minimises ``Σ w_i ((p_i - c) · n)²`` over unit normals ``n``: the optimum
    passes through the weighted centroid ``c`` and ``n`` is the eigenvector of
    the smallest eigenvalue of the covariance ``Σ w (p - c)(p - c)ᵀ`` (so the
    direction is the principal eigenvector). Unlike ``y = ax + b`` regression
    this is orientation independent and handles vertical lines.
    """
    pts = np.asarray(points, dtype=np.float64)
    w = np.ones(len(pts)) if weights is None else np.asarray(weights, dtype=np.float64)
    c = (w[:, None] * pts).sum(axis=0) / w.sum()
    rel = pts - c
    cov = (rel * w[:, None]).T @ rel
    _, vecs = np.linalg.eigh(cov)
    d = vecs[:, 1]
    if d @ (pts[-1] - pts[0]) < 0:  # direction follows the point order
        d = -d
    return LineFit(c, d)


def fit_circle_algebraic(points: FloatArray) -> CircleFit:
    """Kasa fit: ``x² + y² = 2 a x + 2 b y + c`` is *linear* in ``(a, b, c)``.

    Solved by linear least squares; centre ``(a, b)``, ``r = sqrt(c + a² + b²)``.
    Exact for noise-free data, slightly biased for short noisy arcs - used as
    the initial value of :func:`fit_circle_lsq` and inside the greedy search.
    """
    pts = np.asarray(points, dtype=np.float64)
    shift = pts.mean(axis=0)
    rel = pts - shift
    a = np.column_stack([2 * rel, np.ones(len(rel))])
    b = (rel**2).sum(axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(a, b, rcond=None)
    r = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
    return CircleFit(np.array([cx, cy]) + shift, r)


def fit_circle_lsq(points: FloatArray, initial: CircleFit | None = None) -> CircleFit:
    """Geometric circle fit minimising ``Σ (|p_i - c| - r)²``.

    Levenberg-Marquardt on ``(cx, cy, r)`` with the analytic Jacobian
    ``∂res/∂c = -(p - c)/|p - c|``, ``∂res/∂r = -1``, started from the Kasa fit.
    This is the maximum-likelihood estimate for isotropic Gaussian noise.
    """
    pts = np.asarray(points, dtype=np.float64)
    init = initial or fit_circle_algebraic(pts)
    if len(pts) < 4:
        return init

    def res(x: FloatArray) -> FloatArray:
        return np.linalg.norm(pts - x[:2], axis=1) - x[2]

    def jac(x: FloatArray) -> FloatArray:
        rel = pts - x[:2]
        dist = np.maximum(np.linalg.norm(rel, axis=1), 1e-300)
        return np.column_stack([-rel / dist[:, None], -np.ones(len(pts))])

    sol = least_squares(res, np.r_[init.center, init.radius], jac=jac, method="lm", max_nfev=100)
    return CircleFit(sol.x[:2], abs(float(sol.x[2])))


def circle_from_3_points(a: FloatArray, b: FloatArray, c: FloatArray) -> CircleFit | None:
    """Circumcircle: the centre is the intersection of two perpendicular bisectors,
    ``2 (b - a)·x = |b|² - |a|²`` and ``2 (c - a)·x = |c|² - |a|²``."""
    m = 2.0 * np.array([b - a, c - a])
    det = np.linalg.det(m)
    if abs(det) < 1e-12 * max(1.0, float(np.abs(m).max())) ** 2:
        return None  # collinear
    rhs = np.array([b @ b - a @ a, c @ c - a @ a])
    center = np.linalg.solve(m, rhs)
    return CircleFit(center, float(np.linalg.norm(a - center)))


def ransac_line_2d(
    points: FloatArray, threshold: float, iterations: int = 200, seed: int = 0
) -> tuple[LineFit, np.ndarray]:
    """RANSAC line: hypotheses from random point pairs, consensus = points within
    ``threshold``; the best consensus set is refitted by total least squares."""
    pts = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    best = np.ones(len(pts), dtype=bool)
    best_count = -1
    for _ in range(iterations):
        i, j = rng.choice(len(pts), 2, replace=False)
        d = pts[j] - pts[i]
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            continue
        mask = np.abs(LineFit(pts[i], d / n).residuals(pts)) <= threshold
        if mask.sum() > best_count:
            best, best_count = mask, int(mask.sum())
    fit = fit_line_lsq(pts[best])
    return fit, np.abs(fit.residuals(pts)) <= threshold


def ransac_circle_2d(
    points: FloatArray, threshold: float, iterations: int = 300, seed: int = 0
) -> tuple[CircleFit, np.ndarray]:
    """RANSAC circle from random point triples (circumcircle), refined geometrically."""
    pts = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    best = np.ones(len(pts), dtype=bool)
    best_count = -1
    for _ in range(iterations):
        idx = rng.choice(len(pts), 3, replace=False)
        circle = circle_from_3_points(*pts[idx])
        if circle is None:
            continue
        mask = np.abs(circle.residuals(pts)) <= threshold
        if mask.sum() > best_count:
            best, best_count = mask, int(mask.sum())
    fit = fit_circle_lsq(pts[best])
    return fit, np.abs(fit.residuals(pts)) <= threshold


def robust_fit(points: FloatArray, kind: Kind, tol: float) -> LineFit | CircleFit:
    """Least squares; if outliers make it exceed ``tol`` while a clear majority is
    within tolerance, refit on the RANSAC consensus set instead."""
    fit: LineFit | CircleFit = fit_line_lsq(points) if kind == "line" else fit_circle_lsq(points)
    if np.max(np.abs(fit.residuals(points))) <= tol or len(points) < 8:
        return fit
    ransac = ransac_line_2d if kind == "line" else ransac_circle_2d
    alt, inliers = ransac(points, tol)
    if inliers.mean() >= 0.6 and inliers.sum() >= (2 if kind == "line" else 4):
        return alt
    return fit


# ======================================================================================
# Denoising, resampling, corners
# ======================================================================================
def polygon_area(points: FloatArray) -> float:
    """Signed area (shoelace formula); positive for counter-clockwise loops."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def point_in_polygon(point: FloatArray, polygon: FloatArray) -> bool:
    """Even-odd ray casting along +x."""
    x, y = point
    xi, yi = polygon[:, 0], polygon[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    crosses = (yi > y) != (yj > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at = xi + (y - yi) * (xj - xi) / (yj - yi)
    return bool(np.count_nonzero(crosses & (x < x_at)) % 2)


def estimate_noise(points: FloatArray, closed: bool) -> float:
    """Robust point noise sigma from chord distances of the *original* vertices.

    For each vertex the distance to the chord of its two neighbours is taken.
    On a straight noisy edge this is a linear combination of three noise terms
    (std ≈ sqrt(1.5) σ); on a curve the sagitta ``s²/(8r)`` adds to it but is
    small for dense sections. Unlike second differences it does not depend on
    uneven vertex spacing. The median absolute value gives σ via the MAD rule.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 5:
        return 0.0
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    if not closed:
        prev, cur, nxt = prev[1:-1], pts[1:-1], nxt[1:-1]
    else:
        cur = pts
    chord = nxt - prev
    length = np.linalg.norm(chord, axis=1)
    valid = length > 1e-12
    rel = cur - prev
    dist = np.abs(rel[:, 0] * chord[:, 1] - rel[:, 1] * chord[:, 0])[valid] / length[valid]
    if not len(dist):
        return 0.0
    return float(np.median(dist) / (0.6745 * math.sqrt(1.5)))


def densify(points: FloatArray, closed: bool, spacing: float) -> tuple[FloatArray, np.ndarray]:
    """Uniform-ish spacing that *keeps all original vertices*: long segments get
    interpolated points, runs of points closer than ``spacing / 3`` are thinned.

    Returns the points and a mask of the original (measured) vertices: inserted
    points lie on chords, i.e. *inside* curved surfaces by up to the facet
    sagitta, so they help segmentation but are excluded from the final fits.
    """
    pts = np.asarray(points, dtype=np.float64)
    ring = np.vstack([pts, pts[:1]]) if closed else pts
    out = [ring[0]]
    orig = [True]
    for a, b in zip(ring[:-1], ring[1:], strict=True):
        n = int(math.floor(float(np.linalg.norm(b - a)) / spacing))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / (n + 1))
            orig.append(False)
        out.append(b)
        orig.append(True)
    if closed:
        out, orig = out[:-1], orig[:-1]
    out = np.array(out)
    orig = np.array(orig)
    keep = [0]
    for i in range(1, len(out)):
        if np.linalg.norm(out[i] - out[keep[-1]]) >= spacing / 3.0:
            keep.append(i)
    if not closed and keep[-1] != len(out) - 1:
        keep[-1] = len(out) - 1
    return out[keep], orig[keep]


def remove_spikes(points: FloatArray, closed: bool, tol: float) -> FloatArray:
    """Drop isolated out-and-back outliers (scan spikes).

    A vertex is a spike if its distance to the chord of its two neighbours is
    larger than ``3 tol`` *and* larger than that chord: a real corner deviates by
    at most about half the chord, an excursion that returns to the same place
    by much more. Two adjacent candidates are kept (that is a feature).
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 7:
        return pts
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    chord = nxt - prev
    length = np.linalg.norm(chord, axis=1)
    rel = pts - prev
    dist = np.abs(rel[:, 0] * chord[:, 1] - rel[:, 1] * chord[:, 0]) / np.maximum(length, 1e-300)
    reach = np.minimum(np.linalg.norm(rel, axis=1), np.linalg.norm(pts - nxt, axis=1))
    spike = (dist > 3.0 * tol) & (np.minimum(dist, reach) > length)
    spike &= ~np.roll(spike, 1) & ~np.roll(spike, -1)
    if not closed:
        spike[0] = spike[-1] = False
    return pts[~spike]


def turning_angles(points: FloatArray, closed: bool, window: int) -> FloatArray:
    """Angle between ``p_i - p_{i-k}`` and ``p_{i+k} - p_i`` (radians, unsigned)."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    idx = np.arange(n)
    if closed:
        a = pts[(idx - window) % n]
        b = pts[(idx + window) % n]
    else:
        a = pts[np.clip(idx - window, 0, n - 1)]
        b = pts[np.clip(idx + window, 0, n - 1)]
    v1, v2 = pts - a, b - pts
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos = np.einsum("ij,ij->i", v1, v2) / np.maximum(n1 * n2, 1e-300)
    angles = np.arccos(np.clip(cos, -1.0, 1.0))
    angles[(n1 < 1e-12) | (n2 < 1e-12)] = 0.0
    return angles


def detect_corners(
    points: FloatArray, closed: bool, angle_deg: float, window: int = 2
) -> list[int]:
    """Indices of sharp corners: turning angle above the threshold and a local
    maximum within ``±window`` (non-maximum suppression)."""
    angles = turning_angles(points, closed, window)
    n = len(points)
    threshold = math.radians(angle_deg)
    corners = []
    for i in np.flatnonzero(angles > threshold):
        lo, hi = i - window, i + window + 1
        if closed:
            neigh = angles[np.arange(lo, hi) % n]
        else:
            if i == 0 or i == n - 1:
                continue
            neigh = angles[max(lo, 0) : min(hi, n)]
        if angles[i] >= neigh.max() and not (corners and i - corners[-1] <= window):
            corners.append(int(i))
    return corners


# ======================================================================================
# Segmentation
# ======================================================================================
@dataclass
class SketchFitOptions:
    tolerance: float | None = None  # max point deviation; None -> automatic
    corner_angle_deg: float = 35.0  # turning angle of a sharp corner (hard break)
    min_arc_sweep_deg: float = 4.0
    min_arc_points: int = 6
    max_radius_ratio: float = 3.0  # arcs larger than ratio * profile size become lines
    snap_angle_deg: float = 0.5  # 0 disables axis snapping of lines
    min_loop_length: float | None = None  # None -> 20 * tolerance
    min_line_length: float | None = None  # None -> 1.5 * tolerance
    min_feature_length: float | None = None  # chamfer/noise entities collapse; None -> 5 * tol
    hole_circularity: float = 0.03  # small loops this round (RMS / r) become circles
    small_hole_ratio: float = 0.25  # outer loops: round only if radius <= ratio * section size


@dataclass
class _Segment:
    start: int  # index of the first point
    end: int  # index of the last point (shared with the next segment)
    kind: Kind
    fit: LineFit | CircleFit | None = None


class _Fitter:
    """Fits and validity tests on one ordered point list."""

    def __init__(
        self,
        pts: FloatArray,
        tol: float,
        options: SketchFitOptions,
        size: float,
        orig: np.ndarray | None = None,
    ):
        self.pts = pts
        self.orig = np.ones(len(pts), dtype=bool) if orig is None else orig
        self.tol = tol
        self.options = options
        self.max_radius = options.max_radius_ratio * max(size, tol)

    def fit(self, i: int, j: int, kind: Kind, exact: bool = False) -> LineFit | CircleFit | None:
        seg = self.pts[i : j + 1]
        if kind == "line":
            return fit_line_lsq(seg) if len(seg) >= 2 else None
        if len(seg) < 3:
            return None
        circle = fit_circle_lsq(seg) if exact else fit_circle_algebraic(seg)
        if not (0 < circle.radius <= self.max_radius):
            return None
        return circle

    def sweep(self, circle: CircleFit, i: int, j: int) -> float:
        rel = self.pts[i : j + 1] - circle.center
        ang = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
        return float(abs(ang[-1] - ang[0]))

    def ok(self, i: int, j: int, kind: Kind) -> bool:
        fit = self.fit(i, j, kind)
        if fit is None:
            return False
        if kind == "arc":
            if j - i + 1 < self.options.min_arc_points:
                return False
            sweep = self.sweep(fit, i, j)
            if sweep < math.radians(self.options.min_arc_sweep_deg):
                return False
            # an arc must bulge measurably: sagitta r (1 - cos(sweep / 2)) >= tol,
            # otherwise a straight edge plus a rounded end would pass as a flat arc
            if sweep < math.pi and fit.radius * (1.0 - math.cos(sweep / 2)) < self.tol:
                return False
        return float(np.max(np.abs(fit.residuals(self.pts[i : j + 1])))) <= self.tol

    def sse(self, i: int, j: int, kind: Kind) -> float:
        fit = self.fit(i, j, kind)
        if fit is None:
            return math.inf
        return float(np.sum(fit.residuals(self.pts[i : j + 1]) ** 2))

    def max_extent(self, i: int, limit: int, kind: Kind) -> int | None:
        """Largest ``j <= limit`` such that points ``i..j`` fit ``kind``.

        Probes spans of doubling length (a short noisy arc may fail while a longer
        one is well conditioned, so the first probe is not decisive), then binary
        searches between the longest passing and the first failing probe.
        """
        first = i + (self.options.min_arc_points - 1 if kind == "arc" else 1)
        if first > limit:
            return None
        good: int | None = None
        bad: int | None = None
        span = first - i
        while True:
            probe = min(i + span, limit)
            if self.ok(i, probe, kind):
                good = probe
            elif good is not None:
                bad = probe
                break
            if probe == limit:
                break
            span *= 2
        if good is None:
            return None
        if bad is None:
            return good
        while bad - good > 1:
            mid = (good + bad) // 2
            if self.ok(i, mid, kind):
                good = mid
            else:
                bad = mid
        return good


def _segment_span(f: _Fitter, i0: int, i1: int) -> list[_Segment]:
    """Greedy longest-primitive segmentation of points ``i0..i1``."""
    segments: list[_Segment] = []
    i = i0
    while i < i1:
        j_line = f.max_extent(i, i1, "line") or (i + 1)
        j_arc = f.max_extent(i, i1, "arc")
        if j_arc is not None and j_arc > j_line:
            segments.append(_Segment(i, j_arc, "arc"))
            i = j_arc
        else:
            segments.append(_Segment(i, j_line, "line"))
            i = j_line
    return segments


def _refine_breaks(f: _Fitter, segments: list[_Segment]) -> None:
    """Move each breakpoint to minimise the summed squared error of both sides
    (tangent line/arc transitions have no corner to lock onto)."""
    for a, b in zip(segments[:-1], segments[1:], strict=True):
        min_a = 1 if a.kind == "line" else f.options.min_arc_points - 1
        min_b = 1 if b.kind == "line" else f.options.min_arc_points - 1
        lo = max(a.start + min_a, a.end - 12)
        hi = min(b.end - min_b, a.end + 12)
        if hi <= lo:
            continue
        best, best_cost = a.end, f.sse(a.start, a.end, a.kind) + f.sse(b.start, b.end, b.kind)
        for k in range(lo, hi + 1):
            cost = f.sse(a.start, k, a.kind) + f.sse(k, b.end, b.kind)
            if cost < best_cost - 1e-15:
                best, best_cost = k, cost
        a.end = b.start = best


def _merge_segments(f: _Fitter, segments: list[_Segment], closed: bool) -> list[_Segment]:
    """Merge neighbours of the same kind when one primitive fits both (collinear
    lines, co-circular arcs), including the wrap-around pair of closed loops."""
    changed = True
    while changed and len(segments) > 1:
        changed = False
        for k in range(len(segments) - 1):
            a, b = segments[k], segments[k + 1]
            if a.kind == b.kind and f.ok(a.start, b.end, a.kind):
                segments[k] = _Segment(a.start, b.end, a.kind)
                del segments[k + 1]
                changed = True
                break
    return segments


def _drop_tiny_lines(f: _Fitter, segments: list[_Segment], min_len: float) -> list[_Segment]:
    """Remove very short lines at noisy corners; neighbours absorb their points."""
    out = list(segments)
    k = 0
    while k < len(out) and len(out) > 1:
        s = out[k]
        chord = float(np.linalg.norm(f.pts[s.end] - f.pts[s.start]))
        if s.kind == "line" and chord < min_len and s.end - s.start <= 3:
            if k > 0:
                out[k - 1].end = s.end
            else:
                out[k + 1].start = s.start
            del out[k]
            continue
        k += 1
    return out


# ======================================================================================
# Entity construction
# ======================================================================================
def _gap(pts: FloatArray, segments: list[_Segment], k: int) -> float:
    """Half the distance bridged between segment ``k - 1`` and ``k``."""
    return 0.5 * float(np.linalg.norm(pts[segments[k - 1].end] - pts[segments[k].start]))


def _intersections(a: LineFit | CircleFit, b: LineFit | CircleFit) -> list[FloatArray]:
    """All intersection points of two fitted curves (line/line, line/circle,
    circle/circle) - closed form."""
    if isinstance(a, LineFit) and isinstance(b, LineFit):
        cross = a.direction[0] * b.direction[1] - a.direction[1] * b.direction[0]
        if abs(cross) < math.sin(math.radians(2.0)):
            return []  # (nearly) parallel: no stable intersection
        rel = b.point - a.point
        t = (rel[0] * b.direction[1] - rel[1] * b.direction[0]) / cross
        return [a.point + t * a.direction]
    if isinstance(a, CircleFit) and isinstance(b, LineFit):
        a, b = b, a
    if isinstance(a, LineFit) and isinstance(b, CircleFit):
        # |p0 + t d - c|² = r²  ->  t² + 2 t d·(p0 - c) + |p0 - c|² - r² = 0
        rel = a.point - b.center
        half_b = float(a.direction @ rel)
        disc = half_b * half_b - (float(rel @ rel) - b.radius**2)
        if disc < 0:
            return []
        root = math.sqrt(disc)
        return [a.point + t * a.direction for t in (-half_b - root, -half_b + root)]
    # circle / circle: radical line
    d_vec = b.center - a.center
    d = float(np.linalg.norm(d_vec))
    if d < 1e-12 or d > a.radius + b.radius or d < abs(a.radius - b.radius):
        return []
    x = (d * d + a.radius**2 - b.radius**2) / (2 * d)
    h = math.sqrt(max(a.radius**2 - x * x, 0.0))
    base = a.center + x * d_vec / d
    perp = np.array([-d_vec[1], d_vec[0]]) / d
    return [base + h * perp, base - h * perp]


def _tangent_point(a: LineFit | CircleFit, b: LineFit | CircleFit) -> FloatArray | None:
    """Contact point if the curves are (nearly) tangent: for a line and a circle the
    foot of the perpendicular from the centre (it stays exactly on the line), for
    two circles the point on the line of centres."""
    if isinstance(a, CircleFit) and isinstance(b, LineFit):
        a, b = b, a
    if isinstance(a, LineFit) and isinstance(b, CircleFit):
        foot = a.project(b.center[None])[0]
        gap = abs(float(np.linalg.norm(foot - b.center)) - b.radius)
        return foot if gap <= 0.02 * b.radius else None
    if isinstance(a, CircleFit) and isinstance(b, CircleFit):
        d_vec = b.center - a.center
        d = float(np.linalg.norm(d_vec))
        if d < 1e-12:
            return None
        external = abs(d - (a.radius + b.radius))
        internal = abs(d - abs(a.radius - b.radius))
        if min(external, internal) > 0.02 * min(a.radius, b.radius):
            return None
        sign = 1.0 if external <= internal or a.radius >= b.radius else -1.0
        return a.center + sign * a.radius * d_vec / d
    return None


def _crossing_angle(a: LineFit | CircleFit, b: LineFit | CircleFit, p: FloatArray) -> float:
    """Angle (radians, 0..pi/2) between the two curves' tangents at ``p``."""

    def tangent(curve: LineFit | CircleFit) -> FloatArray:
        if isinstance(curve, LineFit):
            return curve.direction
        r = p - curve.center
        t = np.array([-r[1], r[0]])
        return t / max(float(np.linalg.norm(t)), 1e-300)

    return float(np.arccos(min(1.0, abs(float(tangent(a) @ tangent(b))))))


def _vertex(
    a: LineFit | CircleFit, b: LineFit | CircleFit, near: FloatArray, tol: float
) -> FloatArray:
    """Shared vertex of consecutive entities.

    * curves that cross at a clear angle (> 8°): the intersection nearest to the
      measured breakpoint;
    * tangent continuation (line into fillet, arc into arc - by far the most
      common transition on machined parts): the exact tangent point;
    * otherwise (parallel lines, no intersection): mean of the projections of
      the breakpoint onto both curves.
    """
    candidates = _intersections(a, b)
    if candidates:
        best = min(candidates, key=lambda p: float(np.linalg.norm(p - near)))
        if np.linalg.norm(best - near) <= 5.0 * tol and _crossing_angle(a, b, best) > math.radians(
            8.0
        ):
            return best
    touch = _tangent_point(a, b)
    if touch is not None and np.linalg.norm(touch - near) <= 20.0 * tol:
        return touch
    if candidates:
        best = min(candidates, key=lambda p: float(np.linalg.norm(p - near)))
        if np.linalg.norm(best - near) <= 5.0 * tol:
            return best
    return 0.5 * (a.project(near[None])[0] + b.project(near[None])[0])


def _snap_line(fit: LineFit, snap_deg: float) -> LineFit:
    """Design intent: make nearly axis-parallel lines exactly parallel."""
    if snap_deg <= 0:
        return fit
    for axis in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
        cos = float(fit.direction @ axis)
        if abs(cos) >= math.cos(math.radians(snap_deg)):
            return LineFit(fit.point, axis * math.copysign(1.0, cos))
    return fit


def _arc_through(start: FloatArray, mid: FloatArray, end: FloatArray) -> Arc2D | None:
    """Arc from ``start`` via ``mid`` to ``end``; direction from the orientation of
    the triangle (positive cross product = counter-clockwise)."""
    circle = circle_from_3_points(start, mid, end)
    if circle is None:
        return None
    c = circle.center
    (ax, ay), (bx, by) = mid - start, end - mid
    ccw = ax * by - ay * bx > 0  # z of the 2D cross product
    a0 = math.atan2(start[1] - c[1], start[0] - c[0])
    a1 = math.atan2(end[1] - c[1], end[0] - c[0])
    return Arc2D((float(c[0]), float(c[1])), circle.radius, a0, a1, ccw)


def _entities(
    f: _Fitter, segments: list[_Segment], closed: bool
) -> tuple[list[SketchEntity], float]:
    pts, tol = f.pts, f.tol
    fits: list[LineFit | CircleFit] = []
    for s in segments:
        span = pts[s.start : s.end + 1]
        measured = span[f.orig[s.start : s.end + 1]]
        if len(measured) >= (5 if s.kind == "arc" else 3):
            span = measured  # fit the measured vertices, not the chord points
        fit = robust_fit(span, s.kind, tol)
        if isinstance(fit, LineFit):
            fit = _snap_line(fit, f.options.snap_angle_deg)
        fits.append(fit)
    n = len(segments)
    vertices: list[FloatArray] = []
    for k in range(n):
        if k == 0 and not closed:
            vertices.append(fits[0].project(pts[segments[0].start][None])[0])
        else:
            prev = fits[k - 1] if k > 0 else fits[-1]
            # segments are contiguous unless a small feature was collapsed between them
            near = 0.5 * (pts[segments[k - 1].end] + pts[segments[k].start])
            vertices.append(_vertex(prev, fits[k], near, max(tol, _gap(pts, segments, k))))
    if not closed:
        vertices.append(fits[-1].project(pts[segments[-1].end][None])[0])
    entities: list[SketchEntity] = []
    for k, s in enumerate(segments):
        v0 = vertices[k]
        v1 = vertices[(k + 1) % len(vertices)] if closed else vertices[k + 1]
        seg_pts = pts[s.start : s.end + 1]
        entity: SketchEntity | None = None
        if s.kind == "arc":
            circle = fits[k]
            mid = circle.project(seg_pts[len(seg_pts) // 2][None])[0]
            entity = _arc_through(v0, mid, v1)
            if entity is not None:
                # nearly flat arcs are lines (same sagitta rule as the greedy search)
                chord = float(np.linalg.norm(v1 - v0))
                sag = entity.radius - math.sqrt(max(entity.radius**2 - (chord / 2) ** 2, 0.0))
                if entity.sweep < math.pi and sag < tol:
                    entity = None
        if entity is None:
            entity = Line2D((float(v0[0]), float(v0[1])), (float(v1[0]), float(v1[1])))
        entities.append(entity)
    return entities, loop_deviation(entities, pts)


def entity_distance(entity: SketchEntity, pts: FloatArray) -> FloatArray:
    """Euclidean distance of points to a finite entity (segment / arc / circle)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if isinstance(entity, Line2D):
        return entity.distance(pts)
    c = np.asarray(entity.center)
    rel = pts - c
    radial = np.abs(np.linalg.norm(rel, axis=1) - entity.radius)
    if isinstance(entity, Circle2D):
        return radial
    # inside the angular range of the arc: radial distance, else nearest end point
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    offset = (ang - entity.start_angle) if entity.ccw else (entity.start_angle - ang)
    inside = np.mod(offset, TWO_PI) <= entity.sweep + 1e-12
    ends = np.minimum(
        np.linalg.norm(pts - np.asarray(entity.start), axis=1),
        np.linalg.norm(pts - np.asarray(entity.end), axis=1),
    )
    return np.where(inside, radial, ends)


def loop_deviation(entities: list[SketchEntity], pts: FloatArray) -> float:
    """Max over points of the distance to the nearest entity of the loop."""
    if not entities or not len(pts):
        return 0.0
    dist = np.min([entity_distance(e, pts) for e in entities], axis=0)
    return float(dist.max())


# ======================================================================================
# Loops and sections
# ======================================================================================
@dataclass(eq=False)
class ProfileLoop:
    entities: list[SketchEntity]
    closed: bool
    depth: int  # containment depth: even = material boundary, odd = hole
    points: FloatArray  # denoised 2D points the entities were fitted to
    max_deviation: float
    area: float = 0.0  # signed area of the input points after orientation
    parent: int | None = None

    @property
    def is_hole(self) -> bool:
        return self.closed and self.depth % 2 == 1

    def counts(self) -> dict[str, int]:
        return {
            "line": sum(isinstance(e, Line2D) for e in self.entities),
            "arc": sum(isinstance(e, Arc2D) for e in self.entities),
            "circle": sum(isinstance(e, Circle2D) for e in self.entities),
        }


@dataclass(eq=False)
class SketchFitResult:
    plane: Plane
    loops: list[ProfileLoop] = field(default_factory=list)
    tolerance: float = 0.0

    @property
    def entities(self) -> list[SketchEntity]:
        return [e for loop in self.loops for e in loop.entities]

    def counts(self) -> dict[str, int]:
        total = {"line": 0, "arc": 0, "circle": 0}
        for loop in self.loops:
            for key, value in loop.counts().items():
                total[key] += value
        return total

    @property
    def max_deviation(self) -> float:
        return max((loop.max_deviation for loop in self.loops), default=0.0)

    def to_sketches(self) -> list[Sketch]:
        """One sketch per material region: an outer loop with its direct holes."""
        sketches = []
        for k, loop in enumerate(self.loops):
            if not loop.closed or loop.is_hole:
                continue
            holes = [h.entities for h in self.loops if h.parent == k and h.is_hole]
            sketches.append(Sketch(self.plane, list(loop.entities), holes))
        return sketches

    def describe(self, limit: int = 60) -> list[str]:
        lines = []
        for k, loop in enumerate(self.loops):
            role = "开放" if not loop.closed else ("孔" if loop.is_hole else "外轮廓")
            lines.append(f"L{k} [{role}] 偏差 {loop.max_deviation:.4f}")
            lines.extend(f"    {e!r}" for e in loop.entities)
        return lines[:limit]


def _whole_circle(
    raw: FloatArray,
    pts: FloatArray,
    closed: bool,
    tol: float,
    options: SketchFitOptions,
    size: float,
    section_size: float = math.inf,
) -> Circle2D | None:
    """Design intent for holes and round sections.

    A closed loop is one circle if (a) the geometric circle fit is within the
    tolerance widened by the tessellation sagitta ``L²/(8r)`` of the longest
    mesh chords (coarse STL facets are not features), or (b) it is round within
    ``hole_circularity`` (RMS / r) and it is a hole, or an outer loop that is small
    (r <= ``small_hole_ratio`` x ``section_size``) - e.g. a tapped hole, whose
    thread flanks make the raw section non-circular.  Large outer loops never get
    this licence: the teeth of a ring gear are only a few percent of its radius
    but are the feature itself.
    """
    if not closed or len(pts) < options.min_arc_points:
        return None
    circle = fit_circle_lsq(raw if len(raw) >= options.min_arc_points else pts)
    if not 0 < circle.radius <= options.max_radius_ratio * size:
        return None
    res = np.abs(circle.residuals(pts))
    # the loop must actually go around the centre (not a C shape)
    rel = raw - circle.center
    ang = np.sort(np.mod(np.arctan2(rel[:, 1], rel[:, 0]), TWO_PI))
    gaps = np.diff(np.r_[ang, ang[0] + TWO_PI])
    if gaps.max() > math.radians(60.0):
        return None
    chords = np.linalg.norm(np.diff(np.vstack([raw, raw[:1]]), axis=0), axis=1)
    sagitta = float(np.percentile(chords, 95)) ** 2 / (8.0 * circle.radius)
    rms = float(np.sqrt(np.mean(res**2)))
    exact = res.max() <= tol + sagitta
    round_hole = (
        circle.radius <= options.small_hole_ratio * section_size
        and rms <= options.hole_circularity * circle.radius
        and res.max() <= 3.0 * options.hole_circularity * circle.radius
    )
    if exact or round_hole:
        return Circle2D((float(circle.center[0]), float(circle.center[1])), circle.radius)
    return None


def _collapse_small_features(
    f: _Fitter, segments: list[_Segment], closed: bool, min_len: float
) -> list[_Segment]:
    """Remove short entities (scan-rounded or bevelled corners, noise) whose two
    neighbours meet at a clear angle close by: the corner becomes sharp, as
    designed. Real small fillets longer than ``min_len`` are kept."""
    changed = True
    while changed and len(segments) > 2:
        changed = False
        n = len(segments)
        for k in range(n):
            if not closed and k in (0, n - 1):
                continue
            s = segments[k]
            length = float(np.linalg.norm(f.pts[s.end] - f.pts[s.start]))
            if s.kind == "arc":
                circle = f.fit(s.start, s.end, "arc", exact=True)
                if circle is not None:
                    length = circle.radius * f.sweep(circle, s.start, s.end)
            if length >= min_len:
                continue
            prev, nxt = segments[k - 1], segments[(k + 1) % n]
            a = f.fit(prev.start, prev.end, prev.kind, exact=True)
            b = f.fit(nxt.start, nxt.end, nxt.kind, exact=True)
            if a is None or b is None:
                continue
            near = f.pts[(s.start + s.end) // 2]
            hits = [p for p in _intersections(a, b) if np.linalg.norm(p - near) <= 2 * min_len]
            if not hits or _crossing_angle(a, b, hits[0]) < math.radians(20.0):
                continue
            # the removed span's points are dropped from both neighbours' fits; on a
            # closed loop index 0 and the last index are the same point (wrap)
            if k != 0:
                prev.end = s.start
            if k != n - 1:
                nxt.start = s.end
            if prev.end - prev.start < 1 or nxt.end - nxt.start < 1:
                continue
            del segments[k]
            changed = True
            break
    return segments


def _segment_loop(
    pts: FloatArray,
    closed: bool,
    tol: float,
    options: SketchFitOptions,
    size: float,
    orig: np.ndarray | None = None,
) -> tuple[_Fitter, list[_Segment]]:
    """Corner split + greedy segmentation + breakpoint refinement + merging.

    Closed loops are handled on the point list extended by its first point, so
    the last segment ends exactly where the first one starts.
    """
    corners = detect_corners(pts, closed, options.corner_angle_deg)
    if closed:
        mask = None if orig is None else np.r_[orig, orig[:1]]
        f = _Fitter(np.vstack([pts, pts[:1]]), tol, options, size, mask)
        breaks = sorted({0, *corners, len(pts)})
    else:
        f = _Fitter(pts, tol, options, size, orig)
        breaks = sorted({0, *corners, len(pts) - 1})
    segments: list[_Segment] = []
    for a, b in zip(breaks[:-1], breaks[1:], strict=True):
        if b > a:
            segments.extend(_segment_span(f, a, b))
    _refine_breaks(f, segments)
    segments = _merge_segments(f, segments, closed)
    segments = _drop_tiny_lines(f, segments, options.min_line_length or 1.5 * tol)
    segments = _collapse_small_features(
        f, segments, closed, options.min_feature_length or 5.0 * tol
    )
    return f, segments


def fit_polyline(
    points: FloatArray,
    closed: bool,
    tol: float,
    options: SketchFitOptions | None = None,
    size: float | None = None,
    is_hole: bool | None = None,
) -> tuple[list[SketchEntity], FloatArray, float]:
    """Fit one ordered 2D polyline; returns ``(entities, denoised points, max deviation)``."""
    options = options or SketchFitOptions()
    raw = np.asarray(points, dtype=np.float64)
    # the round-hole licence: holes and lone loops always, outer loops only if small
    licence_size = math.inf if is_hole in (True, None) else (size or math.inf)
    size = size or float(np.linalg.norm(np.ptp(raw, axis=0)))
    seg_len = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    spacing = float(np.clip(np.median(seg_len) if len(seg_len) else tol, 0.5 * tol, 2.0 * tol))
    raw = remove_spikes(raw, closed, tol)
    pts, orig = densify(raw, closed, spacing)
    if len(pts) < 3:
        return [], pts, 0.0

    # a whole closed loop on one circle: bolt hole, bore, cylinder section
    circle = _whole_circle(raw, pts, closed, tol, options, size, licence_size)
    if circle is not None:
        return [circle], pts, loop_deviation([circle], pts)

    if closed:
        # start at a corner (or the sharpest turn) so that no segment wraps around
        corners = detect_corners(pts, True, options.corner_angle_deg)
        start = corners[0] if corners else int(np.argmax(turning_angles(pts, True, 2)))
        pts = np.roll(pts, -start, axis=0)
        orig = np.roll(orig, -start)
    f, segments = _segment_loop(pts, closed, tol, options, size, orig)
    if closed and len(segments) >= 2 and segments[0].kind == segments[-1].kind:
        # the start was not a real break if one primitive spans across it: restart
        # the loop at the beginning of the last segment
        n = len(pts)
        across = np.vstack([pts[segments[-1].start : n], pts[: segments[0].end + 1]])
        if _Fitter(across, tol, options, size).ok(0, len(across) - 1, segments[0].kind):
            shift = segments[-1].start
            pts = np.roll(pts, -shift, axis=0)
            orig = np.roll(orig, -shift)
            f, segments = _segment_loop(pts, closed, tol, options, size, orig)
    entities, max_dev = _entities(f, segments, closed)
    return entities, pts, max_dev


def auto_tolerance(polylines: list[FloatArray], closed: list[bool]) -> float:
    """``max(4 σ_noise, 0.1 % of the section size)``; σ is the length weighted
    median of the per-loop estimates. The floor keeps tessellation facets and
    scan noise from being modelled as features (design intent over raw data)."""
    all_pts = np.vstack(polylines)
    size = float(np.linalg.norm(np.ptp(all_pts, axis=0)))
    noise = [estimate_noise(p, c) for p, c in zip(polylines, closed, strict=True)]
    weights = [len(p) for p in polylines]
    order = np.argsort(noise)
    cum = np.cumsum(np.asarray(weights)[order])
    median = float(np.asarray(noise)[order][np.searchsorted(cum, 0.5 * cum[-1])])
    return max(4.0 * median, 1e-3 * size, 1e-6)


def fit_profiles_2d(
    polylines: list[FloatArray],
    closed: list[bool],
    plane: Plane,
    options: SketchFitOptions | None = None,
) -> SketchFitResult:
    """Fit already projected 2D polylines (see :func:`fit_section`)."""
    options = options or SketchFitOptions()
    polylines = [np.asarray(p, dtype=np.float64) for p in polylines]
    if not polylines:
        return SketchFitResult(plane, [], options.tolerance or 0.0)
    tol = options.tolerance or auto_tolerance(polylines, closed)
    min_loop = options.min_loop_length or 20.0 * tol
    size = float(np.linalg.norm(np.ptp(np.vstack(polylines), axis=0)))

    items = []
    for pts, is_closed in zip(polylines, closed, strict=True):
        length = float(
            np.linalg.norm(
                np.diff(np.vstack([pts, pts[:1]]) if is_closed else pts, axis=0), axis=1
            ).sum()
        )
        if length < min_loop or len(pts) < 3:
            continue  # noise loop
        items.append((pts, is_closed, polygon_area(pts) if is_closed else 0.0))

    # containment hierarchy of closed loops (larger loops can contain smaller ones)
    closed_ids = [k for k, it in enumerate(items) if it[1]]
    depth = {k: 0 for k in range(len(items))}
    parent: dict[int, int | None] = {k: None for k in range(len(items))}
    for k in closed_ids:
        containers = [
            m
            for m in closed_ids
            if m != k
            and abs(items[m][2]) > abs(items[k][2])
            and point_in_polygon(items[k][0][0], items[m][0])
        ]
        depth[k] = len(containers)
        if containers:
            parent[k] = min(containers, key=lambda m: abs(items[m][2]))

    result = SketchFitResult(plane, [], tol)
    index_of: dict[int, int] = {}
    order = sorted(range(len(items)), key=lambda k: (depth[k], -abs(items[k][2])))
    for k in order:
        pts, is_closed, area = items[k]
        if is_closed:
            want_ccw = depth[k] % 2 == 0  # outer CCW, holes CW
            if (area > 0) != want_ccw:
                pts = pts[::-1]
                area = -area
        entities, clean, max_dev = fit_polyline(
            pts, is_closed, tol, options, size, is_hole=depth[k] % 2 == 1 if is_closed else None
        )
        if not entities:
            continue
        index_of[k] = len(result.loops)
        result.loops.append(ProfileLoop(entities, is_closed, depth[k], clean, max_dev, area))
    for k, idx in index_of.items():
        if parent[k] is not None and parent[k] in index_of:
            result.loops[idx].parent = index_of[parent[k]]
    return result


def fit_section(curve: SectionCurve, options: SketchFitOptions | None = None) -> SketchFitResult:
    """Fit a mesh section: project to the plane's 2D frame, then :func:`fit_profiles_2d`."""
    return fit_profiles_2d(curve.to_2d(), list(curve.closed), curve.plane, options)


def sketch_to_world(result: SketchFitResult, arc_segments: int = 48) -> list[FloatArray]:
    """3D polylines of all fitted loops (display)."""
    return [
        result.plane.to_world(loop_polyline(loop.entities, arc_segments)) for loop in result.loops
    ]


# ======================================================================================
# Half profiles for revolved features
# ======================================================================================
def split_loop_at_axis(
    points: FloatArray, side: float = 1.0, eps: float = 1e-9
) -> list[FloatArray]:
    """Part(s) of a closed 2D loop with ``side * v >= 0`` (the axis is ``v = 0``).

    A revolve needs the *half* section on one side of the axis. The loop is cut
    where it crosses the axis; every piece on the kept side runs from an entry
    crossing to an exit crossing. For a simple polygon, the stretches of the
    axis inside the polygon lie between consecutive crossings sorted along ``u``
    (pairs (0,1), (2,3), ... - even-odd rule), so each piece is closed by
    following the axis to the partner crossing and continuing with the piece
    that starts there, until the region is closed.
    """
    pts = np.asarray(points, dtype=np.float64)
    v = side * pts[:, 1]
    inside = v >= -eps
    if inside.all():
        return [pts]
    if not inside.any():
        return []
    n = len(pts)
    start = int(np.argmin(inside))  # an outside vertex
    order = np.r_[start:n, 0:start]
    pts, v, inside = pts[order], v[order], inside[order]

    crossings: list[FloatArray] = []
    chains: list[tuple[int, int, list[FloatArray]]] = []  # (entry id, exit id, points)
    current: list[FloatArray] | None = None
    entry = -1
    for i in range(n):
        a, b = i, (i + 1) % n
        if inside[a] != inside[b]:
            t = v[a] / (v[a] - v[b]) if v[a] != v[b] else 0.0
            c = pts[a] + t * (pts[b] - pts[a])
            c = np.array([c[0], 0.0])
            crossings.append(c)
            cid = len(crossings) - 1
            if inside[b]:  # entering the kept side
                current, entry = [c], cid
            else:  # leaving
                assert current is not None
                current.append(c)
                chains.append((entry, cid, current))
                current = None
        if inside[b] and current is not None:
            current.append(pts[b])
    # pair crossings along the axis (even-odd)
    by_u = sorted(range(len(crossings)), key=lambda k: crossings[k][0])
    partner = {}
    for k in range(0, len(by_u) - 1, 2):
        partner[by_u[k]] = by_u[k + 1]
        partner[by_u[k + 1]] = by_u[k]
    chain_from_entry = {c[0]: c for c in chains}
    used: set[int] = set()
    regions: list[FloatArray] = []
    for chain in chains:
        if chain[0] in used:
            continue
        region: list[FloatArray] = []
        cur = chain
        while cur is not None and cur[0] not in used:
            used.add(cur[0])
            region.extend(cur[2])
            nxt_entry = partner.get(cur[1])
            cur = chain_from_entry.get(nxt_entry) if nxt_entry is not None else None
        pts_region = np.array(region)
        keep = np.r_[True, np.linalg.norm(np.diff(pts_region, axis=0), axis=1) > eps]
        pts_region = pts_region[keep]
        if len(pts_region) >= 3 and abs(polygon_area(pts_region)) > eps:
            regions.append(pts_region)
    return regions


def half_profile(
    result: SketchFitResult, side: float = 1.0, options: SketchFitOptions | None = None
) -> SketchFitResult:
    """Refit the part of a sketch on one side of its ``u`` axis (``v = 0``).

    For a sketch made on a plane *through* a datum axis (``plane_through_axis``)
    the ``u`` axis is the rotation axis, so the result is the revolve profile.
    The measured loop points are split at the axis and fitted again with the
    same tolerance; the cut along the axis becomes a straight line.
    """
    loops, flags = [], []
    for loop in result.loops:
        if not loop.closed:
            continue
        for piece in split_loop_at_axis(loop.points, side):
            loops.append(piece)
            flags.append(True)
    opts = options or SketchFitOptions(tolerance=result.tolerance)
    if opts.tolerance is None:
        opts.tolerance = result.tolerance
    return fit_profiles_2d(loops, flags, result.plane, opts)
