"""Geometric primitive recognition for piston engine parts.

More than 80 % of a piston's surface is planes (crown, ring-groove flanks, boss
faces, gasket faces) and cylinders (skirt, ring lands, pin bore, bushing bores).
This module provides the recognition chain used by the feature tree:

1. **Least-squares fitting** (:func:`fit_plane`, :func:`fit_cylinder`,
   :func:`fit_sphere`): weighted PCA for planes; normal-covariance axis +
   algebraic circle initialisation followed by Levenberg-Marquardt refinement of
   the true geometric distance for cylinders (high precision axes, e.g. the pin
   bore) and spheres.
2. **Normal estimation** (:func:`estimate_normals`) for point clouds, and face /
   vertex normals for meshes (:class:`~meshrev.core.mesh.topology.MeshGeometry`).
3. **RANSAC extraction** (:func:`ransac_plane`, :func:`ransac_cylinder`,
   :func:`detect_primitives`) in the spirit of Schnabel et al. 2007: minimal
   samples use point *and normal* (3 points for a plane, 2 oriented points for a
   cylinder), localised sampling, compatibility by distance *and* normal angle,
   largest connected component, least-squares refinement.
4. **Auto segmentation** (:func:`auto_segment`): region growing on the face
   graph bounded by sharp edges (normal continuity) and curvature jumps,
   primitive classification of every region, RANSAC splitting of regions that
   leaked across smooth transitions, merging of compatible neighbours and
   absorption of slivers - comparable to Design X "Auto Segment".

Conventions: planes are ``n·x + d = 0`` with unit ``n``; cylinder residuals are
``distance_to_axis - radius`` (positive outside); a *concave* cylinder/sphere is
a hole/bowl (outward surface normals point towards the axis/centre).
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Literal

import numpy as np
import pyvista as pv
import scipy.sparse as sp
from numpy.typing import ArrayLike
from scipy.optimize import least_squares
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from meshrev.core.mesh.curvature import face_curvature
from meshrev.core.mesh.topology import MeshGeometry
from meshrev.core.types import Axis, FloatArray, Plane, as_points, normalize, orthonormal_basis

log = logging.getLogger(__name__)

IntArray = np.ndarray
MeshLike = pv.PolyData | MeshGeometry


# ======================================================================================
# Primitive types
# ======================================================================================
class PrimitiveType(str, Enum):
    PLANE = "plane"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    FREEFORM = "freeform"

    @property
    def label(self) -> str:
        return {
            PrimitiveType.PLANE: "平面",
            PrimitiveType.CYLINDER: "圆柱面",
            PrimitiveType.SPHERE: "球面",
            PrimitiveType.FREEFORM: "自由曲面",
        }[self]


def _canonical_direction(direction: FloatArray) -> FloatArray:
    """Flip ``direction`` so its dominant component is positive (deterministic axes)."""
    d = normalize(direction)
    return -d if d[np.argmax(np.abs(d))] < 0 else d


def _fmt(vec: ArrayLike, digits: int = 5) -> str:
    return "(" + ", ".join(f"{float(x):.{digits}f}" for x in np.ravel(vec)) + ")"


class Primitive(ABC):
    type: ClassVar[PrimitiveType]

    @abstractmethod
    def distance(self, points: ArrayLike) -> FloatArray:
        """Signed geometric distance of points to the surface."""

    @abstractmethod
    def normals_at(self, points: ArrayLike) -> FloatArray:
        """Unit surface normals at the points' projections (orientation arbitrary)."""

    @abstractmethod
    def describe(self) -> dict[str, str]: ...


@dataclass(eq=False)
class PlanePrimitive(Primitive):
    """Plane ``normal · x + d = 0``; ``center`` is the centroid of the fitted data."""

    type: ClassVar[PrimitiveType] = PrimitiveType.PLANE
    normal: FloatArray
    d: float
    center: FloatArray | None = None

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=np.float64)
        length = float(np.linalg.norm(n))
        self.normal = n / length
        self.d = float(self.d) / length
        if self.center is None:
            self.center = -self.d * self.normal
        else:  # keep the reference point exactly on the plane
            c = np.asarray(self.center, dtype=np.float64)
            self.center = c - (c @ self.normal + self.d) * self.normal

    @property
    def equation(self) -> tuple[float, float, float, float]:
        a, b, c = (float(x) for x in self.normal)
        return a, b, c, self.d

    @property
    def plane(self) -> Plane:
        return Plane(self.center, self.normal)

    def distance(self, points: ArrayLike) -> FloatArray:
        return as_points(points) @ self.normal + self.d

    def normals_at(self, points: ArrayLike) -> FloatArray:
        return np.broadcast_to(self.normal, as_points(points).shape).copy()

    def describe(self) -> dict[str, str]:
        a, b, c, d = self.equation
        terms = [f"{a:.6f}x"] + [
            f"{'-' if v < 0 else '+'} {abs(v):.6f}{var}" for v, var in ((b, "y"), (c, "z"), (d, ""))
        ]
        return {
            "平面方程": " ".join(terms) + " = 0",
            "法向": _fmt(self.normal, 6),
            "中心点": _fmt(self.center, 4),
        }


@dataclass(eq=False)
class CylinderPrimitive(Primitive):
    """Cylinder around the line through ``axis_point`` along ``axis_direction``.

    ``axis_point`` is the midpoint of the fitted data's extent along the axis and
    ``length`` that extent, so the pair describes the finite measured cylinder.
    """

    type: ClassVar[PrimitiveType] = PrimitiveType.CYLINDER
    axis_point: FloatArray
    axis_direction: FloatArray
    radius: float
    length: float = 0.0
    concave: bool | None = None  # True for holes/bores

    def __post_init__(self) -> None:
        self.axis_point = np.asarray(self.axis_point, dtype=np.float64)
        self.axis_direction = normalize(self.axis_direction)
        self.radius = float(self.radius)
        self.length = float(self.length)

    @property
    def axis(self) -> Axis:
        return Axis(self.axis_point, self.axis_direction)

    def axis_endpoints(self, margin: float = 0.0) -> tuple[FloatArray, FloatArray]:
        half = 0.5 * self.length + margin
        return (
            self.axis_point - half * self.axis_direction,
            self.axis_point + half * self.axis_direction,
        )

    def _radial(self, points: ArrayLike) -> FloatArray:
        rel = as_points(points) - self.axis_point
        return rel - np.outer(rel @ self.axis_direction, self.axis_direction)

    def distance(self, points: ArrayLike) -> FloatArray:
        return np.linalg.norm(self._radial(points), axis=1) - self.radius

    def normals_at(self, points: ArrayLike) -> FloatArray:
        radial = self._radial(points)
        return radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-300)

    def describe(self) -> dict[str, str]:
        info = {
            "轴线方向": _fmt(self.axis_direction, 6),
            "轴线中心点": _fmt(self.axis_point, 4),
            "半径": f"{self.radius:.5f}",
            "直径": f"{2 * self.radius:.5f}",
            "轴向长度": f"{self.length:.3f}",
        }
        if self.concave is not None:
            info["类型"] = "孔 (内圆柱)" if self.concave else "轴 (外圆柱)"
        return info


@dataclass(eq=False)
class SpherePrimitive(Primitive):
    type: ClassVar[PrimitiveType] = PrimitiveType.SPHERE
    center: FloatArray
    radius: float
    concave: bool | None = None

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64)
        self.radius = float(self.radius)

    def distance(self, points: ArrayLike) -> FloatArray:
        return np.linalg.norm(as_points(points) - self.center, axis=1) - self.radius

    def normals_at(self, points: ArrayLike) -> FloatArray:
        rel = as_points(points) - self.center
        return rel / np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-300)

    def describe(self) -> dict[str, str]:
        info = {"球心": _fmt(self.center, 4), "半径": f"{self.radius:.5f}"}
        if self.concave is not None:
            info["类型"] = "凹球面" if self.concave else "凸球面"
        return info


@dataclass(eq=False)
class FitResult:
    """A fitted primitive with the residuals of the data it was fitted to."""

    primitive: Primitive
    residuals: FloatArray
    inliers: IntArray | None = None  # sample indices (RANSAC) or face ids (mesh fits)
    iterations: int = 0

    @property
    def type(self) -> PrimitiveType:
        return self.primitive.type

    @property
    def n_points(self) -> int:
        return len(self.residuals)

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2))) if self.n_points else 0.0

    @property
    def max_error(self) -> float:
        return float(np.max(np.abs(self.residuals))) if self.n_points else 0.0

    def trimmed_rms(self, keep: float = 0.9) -> float:
        if not self.n_points:
            return 0.0
        r = np.sort(np.abs(self.residuals))[: max(1, int(math.ceil(keep * self.n_points)))]
        return float(np.sqrt(np.mean(r**2)))

    def inlier_ratio(self, tolerance: float) -> float:
        return float(np.mean(np.abs(self.residuals) <= tolerance)) if self.n_points else 0.0

    def describe(self) -> dict[str, str]:
        info = {"基元类型": self.type.label}
        info.update(self.primitive.describe())
        info["拟合点数"] = f"{self.n_points:,}"
        info["RMS 偏差"] = f"{self.rms:.5f}"
        info["RMS 偏差 (90%)"] = f"{self.trimmed_rms(0.9):.5f}"
        info["最大偏差"] = f"{self.max_error:.5f}"
        return info


# ======================================================================================
# Normal estimation
# ======================================================================================
def estimate_normals(
    points: ArrayLike,
    k: int = 16,
    orient: Literal["outward"] | ArrayLike | None = "outward",
) -> FloatArray:
    """PCA normals from the ``k`` nearest neighbours of every point.

    ``orient="outward"`` flips normals away from the cloud centroid (fine for
    closed, roughly convex parts); an array is interpreted as a viewpoint the
    normals should face; ``None`` leaves the sign arbitrary.
    """
    pts = as_points(points)
    k = int(min(max(k, 3), len(pts)))
    _, idx = cKDTree(pts).query(pts, k=k)
    neigh = pts[idx]
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered)
    _, vecs = np.linalg.eigh(cov)
    normals = vecs[:, :, 0]
    if orient is not None:
        if isinstance(orient, str):
            reference = pts - pts.mean(axis=0)
        else:
            reference = np.asarray(orient, dtype=np.float64) - pts
        flip = np.einsum("ij,ij->i", normals, reference) < 0
        normals[flip] *= -1.0
    return normals


# ======================================================================================
# Least-squares fitting
# ======================================================================================
def _weights(n: int, weights: ArrayLike | None) -> FloatArray:
    if weights is None:
        return np.ones(n)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(w) != n:
        raise ValueError("weights must have one entry per point")
    return np.maximum(w, 0.0)


def _subsample(n: int, max_points: int | None, seed: int = 0) -> IntArray | None:
    if max_points is None or n <= max_points:
        return None
    return np.sort(np.random.default_rng(seed).choice(n, max_points, replace=False))


def fit_plane(
    points: ArrayLike, weights: ArrayLike | None = None, normals: ArrayLike | None = None
) -> FitResult:
    """Weighted total least squares plane (PCA). ``normals`` only choose the sign."""
    pts = as_points(points)
    if len(pts) < 3:
        raise ValueError("a plane needs at least 3 points")
    w = _weights(len(pts), weights)
    center = (w[:, None] * pts).sum(axis=0) / w.sum()
    _, _, vt = np.linalg.svd((pts - center) * np.sqrt(w)[:, None], full_matrices=False)
    normal = vt[-1]
    if normals is not None:
        if (np.asarray(normals, dtype=np.float64) * w[:, None]).sum(axis=0) @ normal < 0:
            normal = -normal
    else:
        normal = _canonical_direction(normal)
    primitive = PlanePrimitive(normal, -float(normal @ center), center=center)
    return FitResult(primitive, primitive.distance(pts))


def fit_circle_2d(xy: ArrayLike, weights: ArrayLike | None = None) -> tuple[float, float, float]:
    """Algebraic (Kasa) circle fit; returns ``(cx, cy, r)``."""
    xy = np.asarray(xy, dtype=np.float64)
    w = np.sqrt(_weights(len(xy), weights))
    a = np.column_stack([2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))]) * w[:, None]
    b = (xy**2).sum(axis=1) * w
    (cx, cy, c), *_ = np.linalg.lstsq(a, b, rcond=None)
    return float(cx), float(cy), float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))


def axis_from_normals(
    normals: ArrayLike, weights: ArrayLike | None = None
) -> tuple[FloatArray, float]:
    """Cylinder axis = direction most orthogonal to all normals.

    Returns the axis and a conditioning value in [0, 0.5]: the share of the
    second eigenvalue, ~0 for (nearly) planar normal sets where the axis is
    undefined, up to 0.5 for a full 360° cylinder.
    """
    n = np.asarray(normals, dtype=np.float64)
    w = _weights(len(n), weights)
    vals, vecs = np.linalg.eigh((n * w[:, None]).T @ n)
    total = max(float(vals.sum()), 1e-300)
    return vecs[:, 0], float(vals[1] / total)


def _initial_cylinder(
    points: FloatArray, axis: FloatArray, weights: FloatArray | None = None
) -> CylinderPrimitive:
    u, v = orthonormal_basis(axis)
    w = _weights(len(points), weights)
    origin = (w[:, None] * points).sum(axis=0) / w.sum()
    rel = points - origin
    cx, cy, r = fit_circle_2d(np.column_stack([rel @ u, rel @ v]), w)
    return CylinderPrimitive(origin + cx * u + cy * v, axis, r)


def _finalize_cylinder(
    points: FloatArray, point_on_axis: FloatArray, direction: FloatArray, radius: float
) -> CylinderPrimitive:
    direction = _canonical_direction(direction)
    t = (points - point_on_axis) @ direction
    t0, t1 = float(t.min()), float(t.max())
    center = point_on_axis + 0.5 * (t0 + t1) * direction
    return CylinderPrimitive(center, direction, abs(radius), length=t1 - t0)


def _orientation_concave(
    primitive: CylinderPrimitive | SpherePrimitive, points: FloatArray, normals: FloatArray | None
) -> bool | None:
    if normals is None or len(normals) == 0:
        return None
    outward = primitive.normals_at(points)
    return bool(np.einsum("ij,ij->i", outward, np.asarray(normals)).mean() < 0)


def fit_cylinder(
    points: ArrayLike,
    weights: ArrayLike | None = None,
    normals: ArrayLike | None = None,
    initial: CylinderPrimitive | None = None,
    robust_scale: float | None = None,
    max_points: int | None = 60000,
) -> FitResult:
    """High precision cylinder fit minimising ``|dist_to_axis - r|``.

    Initialisation (in order of preference): ``initial``; the axis orthogonal to
    the ``normals``; otherwise the best of the three PCA directions. The
    5-parameter Levenberg-Marquardt refinement (axis tilt, axis offset, radius)
    uses a soft-L1 loss when ``robust_scale`` is given (outlier tolerant).
    """
    pts = as_points(points)
    if len(pts) < 6:
        raise ValueError("a cylinder needs at least 6 points")
    w = _weights(len(pts), weights)
    nrm = None if normals is None else np.asarray(normals, dtype=np.float64)
    if initial is None:
        if nrm is not None:
            axis, _ = axis_from_normals(nrm, w)
            initial = _initial_cylinder(pts, axis, w)
        else:
            centered = pts - pts.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            guesses = [_initial_cylinder(pts, d, w) for d in vt]
            initial = min(guesses, key=lambda c: float(np.mean(c.distance(pts) ** 2)))

    sub = _subsample(len(pts), max_points)
    fp, fw = (pts, w) if sub is None else (pts[sub], w[sub])
    a0 = initial.axis_direction
    u0, v0 = orthonormal_basis(a0)
    origin = initial.axis_point
    sqrt_w = np.sqrt(fw / fw.mean())

    def unpack(x: FloatArray) -> tuple[FloatArray, FloatArray, float]:
        a = a0 + x[0] * u0 + x[1] * v0
        a = a / np.linalg.norm(a)
        return origin + x[2] * u0 + x[3] * v0, a, x[4]

    def residuals(x: FloatArray) -> FloatArray:
        p0, a, r = unpack(x)
        return sqrt_w * (np.linalg.norm(np.cross(fp - p0, a), axis=1) - r)

    def jacobian(x: FloatArray) -> FloatArray:
        # rho = |d - (d·a) a| with d = p - p0; d rho/d p0 = -q̂ (radial unit vector),
        # d rho/d a = -(d·a) d / rho projected orthogonally to a (a is renormalised)
        p0, a, _ = unpack(x)
        w_len = np.linalg.norm(a0 + x[0] * u0 + x[1] * v0)
        d = fp - p0
        along = d @ a
        q = d - along[:, None] * a
        rho = np.maximum(np.linalg.norm(q, axis=1), 1e-300)
        q_hat = q / rho[:, None]
        da_alpha = (u0 - (u0 @ a) * a) / w_len
        da_beta = (v0 - (v0 @ a) * a) / w_len
        jac = np.empty((len(fp), 5))
        jac[:, 0] = -along * (d @ da_alpha) / rho
        jac[:, 1] = -along * (d @ da_beta) / rho
        jac[:, 2] = -(q_hat @ u0)
        jac[:, 3] = -(q_hat @ v0)
        jac[:, 4] = -1.0
        return jac * sqrt_w[:, None]

    x0 = np.array([0.0, 0.0, 0.0, 0.0, initial.radius])
    if robust_scale:
        sol = least_squares(
            residuals,
            x0,
            jac=jacobian,
            method="trf",
            loss="soft_l1",
            f_scale=robust_scale,
            x_scale="jac",
            max_nfev=100,
        )
    else:
        sol = least_squares(residuals, x0, jac=jacobian, method="lm", x_scale="jac", max_nfev=200)
    p0, a, r = unpack(sol.x)
    primitive = _finalize_cylinder(pts, p0, a, r)
    primitive.concave = _orientation_concave(primitive, pts, nrm)
    return FitResult(primitive, primitive.distance(pts), iterations=int(sol.nfev))


def fit_sphere(
    points: ArrayLike,
    weights: ArrayLike | None = None,
    normals: ArrayLike | None = None,
    robust_scale: float | None = None,
    max_points: int | None = 60000,
) -> FitResult:
    """Algebraic sphere fit refined by Levenberg-Marquardt on ``|p - c| - r``."""
    pts = as_points(points)
    if len(pts) < 4:
        raise ValueError("a sphere needs at least 4 points")
    w = _weights(len(pts), weights)
    sw = np.sqrt(w)
    shift = pts.mean(axis=0)
    rel = pts - shift
    a = np.column_stack([2 * rel, np.ones(len(rel))]) * sw[:, None]
    b = (rel**2).sum(axis=1) * sw
    (cx, cy, cz, c), *_ = np.linalg.lstsq(a, b, rcond=None)
    center0 = np.array([cx, cy, cz])
    r0 = float(np.sqrt(max(c + center0 @ center0, 1e-300)))
    sub = _subsample(len(pts), max_points)
    fp, fw = (rel, w) if sub is None else (rel[sub], w[sub])
    sqrt_w = np.sqrt(fw / fw.mean())

    def residuals(x: FloatArray) -> FloatArray:
        return sqrt_w * (np.linalg.norm(fp - x[:3], axis=1) - x[3])

    def jacobian(x: FloatArray) -> FloatArray:
        d = fp - x[:3]
        dist = np.maximum(np.linalg.norm(d, axis=1), 1e-300)
        return np.column_stack([-d / dist[:, None], -np.ones(len(fp))]) * sqrt_w[:, None]

    x0 = np.r_[center0, r0]
    if robust_scale:
        sol = least_squares(
            residuals,
            x0,
            jac=jacobian,
            method="trf",
            loss="soft_l1",
            f_scale=robust_scale,
            x_scale="jac",
            max_nfev=100,
        )
    else:
        sol = least_squares(residuals, x0, jac=jacobian, method="lm", x_scale="jac", max_nfev=200)
    primitive = SpherePrimitive(sol.x[:3] + shift, abs(sol.x[3]))
    nrm = None if normals is None else np.asarray(normals, dtype=np.float64)
    primitive.concave = _orientation_concave(primitive, pts, nrm)
    return FitResult(primitive, primitive.distance(pts), iterations=int(sol.nfev))


# ======================================================================================
# RANSAC
# ======================================================================================
@dataclass
class RansacOptions:
    """Parameters of the normal-aware RANSAC.

    Distances default to fractions of the data's bounding box diagonal so that
    the same options work for millimetre and metre data.
    """

    distance_threshold: float | None = None  # epsilon; None -> 0.004 * diagonal
    normal_threshold_deg: float = 20.0  # alpha: max normal deviation of inliers
    min_support: int = 100  # minimal number of inlier samples
    max_iterations: int = 2000  # hypotheses per extracted shape
    batch_size: int = 128
    confidence: float = 0.999
    eval_samples: int = 10000  # random subset used to score hypotheses
    local_radius: float | None = None  # localised sampling; None -> 0.15 * diagonal
    connectivity: bool = True  # keep the largest connected inlier component
    cluster_epsilon: float | None = None  # point clouds: None -> 3 * median spacing
    min_radius: float = 0.0
    max_radius: float | None = None  # None -> diagonal
    min_normal_angle_deg: float = 5.0  # cylinder samples need diverging normals
    refine: bool = True
    seed: int | None = 0


@dataclass(eq=False)
class _Samples:
    points: FloatArray
    normals: FloatArray
    weights: FloatArray
    adjacency: sp.csr_matrix | None = None
    _spacing: float | None = field(default=None, repr=False)

    _diagonal: float | None = field(default=None, repr=False)

    @property
    def diagonal(self) -> float:
        if self._diagonal is None:
            extent = self.points.max(axis=0) - self.points.min(axis=0)
            self._diagonal = float(np.linalg.norm(extent))
        return self._diagonal

    @property
    def spacing(self) -> float:
        """Typical distance between neighbouring samples."""
        if self._spacing is None:
            n = len(self.points)
            sub = _subsample(n, 5000, seed=1)
            sample = self.points if sub is None else self.points[sub]
            d, _ = cKDTree(sample).query(sample, k=2)
            # surface data: spacing scales with 1/sqrt(density)
            self._spacing = float(np.median(d[:, 1])) * math.sqrt(len(sample) / n)
        return self._spacing


def _make_samples(points, normals, weights, adjacency) -> _Samples:
    pts = as_points(points)
    nrm = np.asarray(normals, dtype=np.float64)
    if nrm.shape != pts.shape:
        raise ValueError("normals must match points")
    length = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, length, out=np.zeros_like(nrm), where=length > 0)
    return _Samples(pts, nrm, _weights(len(pts), weights), adjacency)


def _largest_component(samples: _Samples, idx: IntArray, eps: float) -> IntArray:
    if len(idx) < 2:
        return idx
    if samples.adjacency is not None:
        graph = samples.adjacency[idx][:, idx]
    else:
        pairs = cKDTree(samples.points[idx]).query_pairs(eps, output_type="ndarray")
        graph = sp.coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx), len(idx))
        )
    _, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels, weights=samples.weights[idx])
    return idx[labels == np.argmax(counts)]


def _draw_local(
    samples: _Samples, pool: IntArray, rng: np.random.Generator, n: int, count: int, radius: float
) -> IntArray:
    """``(n, count)`` sample indices; columns 1.. are drawn near column 0 when possible."""
    first = rng.choice(pool, n)
    cands = rng.choice(pool, (n, 48))
    dist = np.linalg.norm(samples.points[cands] - samples.points[first][:, None], axis=2)
    near = (dist < radius) & (cands != first[:, None])
    priority = rng.random(cands.shape) + near  # near candidates first, random order
    order = np.argsort(-priority, axis=1)[:, : count - 1]
    return np.column_stack([first, np.take_along_axis(cands, order, axis=1)])


def _plane_hypotheses(samples: _Samples, idx: IntArray, cos_alpha: float):
    p = samples.points[idx]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    length = np.linalg.norm(n, axis=1)
    valid = length > 1e-12 * max(samples.diagonal, 1.0) ** 2
    n = n / np.maximum(length, 1e-300)[:, None]
    for j in range(3):
        valid &= np.abs(np.einsum("ij,ij->i", samples.normals[idx[:, j]], n)) >= cos_alpha
    d = -np.einsum("ij,ij->i", n, p[:, 0])
    return valid, (n, d)


def _plane_scores(model, pe, ne, we, eps, cos_alpha):
    n, d = model
    dist = np.abs(pe @ n.T + d)
    ok = (dist < eps) & (np.abs(ne @ n.T) >= cos_alpha)
    return we @ ok


def _plane_inliers(model_single, pts, nrm, eps, cos_alpha):
    n, d = model_single
    return (np.abs(pts @ n + d) < eps) & (np.abs(nrm @ n) >= cos_alpha)


def _cylinder_hypotheses(
    samples: _Samples,
    idx: IntArray,
    cos_alpha: float,
    options: RansacOptions,
    max_radius: float,
    eps: float,
):
    p0, p1 = samples.points[idx[:, 0]], samples.points[idx[:, 1]]
    n0, n1 = samples.normals[idx[:, 0]], samples.normals[idx[:, 1]]
    a = np.cross(n0, n1)
    la = np.linalg.norm(a, axis=1)
    valid = la >= np.sin(np.radians(options.min_normal_angle_deg))
    a = a / np.maximum(la, 1e-300)[:, None]
    # intersect the normal lines p_i + t n_i inside the plane orthogonal to the axis
    q0 = p0 - np.einsum("ij,ij->i", p0, a)[:, None] * a
    q1 = p1 - np.einsum("ij,ij->i", p1, a)[:, None] * a
    delta = q1 - q0
    c01 = np.einsum("ij,ij->i", n0, n1)
    det = np.maximum(1.0 - c01**2, 1e-300)
    d0 = np.einsum("ij,ij->i", n0, delta)
    d1 = np.einsum("ij,ij->i", n1, delta)
    t = (d0 - c01 * d1) / det
    s = (c01 * d0 - d1) / det
    center = 0.5 * ((q0 + t[:, None] * n0) + (q1 + s[:, None] * n1))
    radius = 0.5 * (np.abs(t) + np.abs(s))
    valid &= np.abs(np.abs(t) - np.abs(s)) < 2 * eps
    valid &= (radius >= options.min_radius) & (radius <= max_radius)
    return valid, (a, center, radius)


def _cylinder_distance_and_cos(model, pe, ne, pe_sq, ne_pe):
    a, c, r = model
    along = pe @ a.T - np.einsum("ij,ij->i", c, a)  # (m, B)
    v_sq = pe_sq[:, None] - 2 * pe @ c.T + np.einsum("ij,ij->i", c, c)
    rho = np.sqrt(np.maximum(v_sq - along**2, 1e-300))
    n_dot_v = ne_pe[:, None] - ne @ c.T
    n_dot_radial = (n_dot_v - (ne @ a.T) * along) / rho
    return np.abs(rho - r), np.abs(n_dot_radial)


def _cylinder_scores(model, pe, ne, we, eps, cos_alpha):
    dist, cos = _cylinder_distance_and_cos(
        model, pe, ne, np.einsum("ij,ij->i", pe, pe), np.einsum("ij,ij->i", ne, pe)
    )
    return we @ ((dist < eps) & (cos >= cos_alpha))


def _cylinder_inliers(model_single, pts, nrm, eps, cos_alpha):
    a, c, r = model_single
    model = (a[None], c[None], np.array([r]))
    dist, cos = _cylinder_distance_and_cos(
        model, pts, nrm, np.einsum("ij,ij->i", pts, pts), np.einsum("ij,ij->i", nrm, pts)
    )
    return (dist[:, 0] < eps) & (cos[:, 0] >= cos_alpha)


def _ransac(
    samples: _Samples,
    kind: PrimitiveType,
    options: RansacOptions,
    available: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> FitResult | None:
    rng = rng or np.random.default_rng(options.seed)
    diag = samples.diagonal
    eps = options.distance_threshold or 0.004 * diag
    cos_alpha = float(np.cos(np.radians(options.normal_threshold_deg)))
    radius = options.local_radius or 0.15 * diag
    max_radius = options.max_radius or diag
    pool = np.flatnonzero(available) if available is not None else np.arange(len(samples.points))
    if len(pool) < max(options.min_support, 3):
        return None
    eval_idx = (
        pool
        if len(pool) <= options.eval_samples
        else rng.choice(pool, options.eval_samples, replace=False)
    )
    pe, ne, we = samples.points[eval_idx], samples.normals[eval_idx], samples.weights[eval_idx]
    total = float(we.sum())
    minimal = 3 if kind is PrimitiveType.PLANE else 2

    best_score, best_model, iterations = 0.0, None, 0
    while iterations < options.max_iterations:
        idx = _draw_local(samples, pool, rng, options.batch_size, minimal, radius)
        if kind is PrimitiveType.PLANE:
            valid, model = _plane_hypotheses(samples, idx, cos_alpha)
            scorer = _plane_scores
        else:
            valid, model = _cylinder_hypotheses(samples, idx, cos_alpha, options, max_radius, eps)
            scorer = _cylinder_scores
        iterations += options.batch_size
        if valid.any():
            model = tuple(m[valid] for m in model)
            scores = scorer(model, pe, ne, we, eps, cos_alpha)
            j = int(np.argmax(scores))
            if scores[j] > best_score:
                best_score, best_model = float(scores[j]), tuple(m[j] for m in model)
        ratio = best_score / total if total > 0 else 0.0
        if ratio > 0:
            miss = max(1e-12, 1.0 - ratio**minimal)
            needed = math.log(1.0 - options.confidence) / math.log(miss) if miss < 1 else math.inf
            if iterations >= needed:
                break
    if best_model is None:
        return None

    inlier_fn = _plane_inliers if kind is PrimitiveType.PLANE else _cylinder_inliers
    cluster_eps = options.cluster_epsilon or 3.0 * samples.spacing

    def collect(model) -> IntArray:
        mask = inlier_fn(model, samples.points[pool], samples.normals[pool], eps, cos_alpha)
        inl = pool[mask]
        if options.connectivity and len(inl):
            inl = _largest_component(samples, inl, cluster_eps)
        return inl

    inliers = collect(best_model)
    if len(inliers) < options.min_support:
        return None
    fit = None
    rounds = 2 if options.refine else 0
    for _ in range(rounds + 1):
        pts, nrm, w = samples.points[inliers], samples.normals[inliers], samples.weights[inliers]
        if kind is PrimitiveType.PLANE:
            fit = fit_plane(pts, w, nrm)
            prim = fit.primitive
            model = (prim.normal, prim.d)
        else:
            if len(inliers) < 6:
                return None
            fit = fit_cylinder(pts, w, nrm, robust_scale=eps)
            prim = fit.primitive
            if not options.min_radius <= prim.radius <= max_radius:
                return None
            model = (prim.axis_direction, prim.axis_point, prim.radius)
        if not options.refine:
            break
        updated = collect(model)
        if len(updated) < options.min_support:
            break
        if np.array_equal(updated, inliers):
            break
        inliers = updated
    fit = FitResult(
        fit.primitive,
        fit.primitive.distance(samples.points[inliers]),
        inliers=inliers,
        iterations=iterations,
    )
    return fit


def ransac_plane(
    points: ArrayLike,
    normals: ArrayLike,
    options: RansacOptions | None = None,
    weights: ArrayLike | None = None,
    adjacency: sp.spmatrix | None = None,
) -> FitResult | None:
    """Best plane of a point set (normals required); ``None`` if nothing is found."""
    samples = _make_samples(points, normals, weights, adjacency)
    return _ransac(samples, PrimitiveType.PLANE, options or RansacOptions())


def ransac_cylinder(
    points: ArrayLike,
    normals: ArrayLike,
    options: RansacOptions | None = None,
    weights: ArrayLike | None = None,
    adjacency: sp.spmatrix | None = None,
) -> FitResult | None:
    """Best cylinder of a point set (normals required); ``None`` if nothing is found."""
    samples = _make_samples(points, normals, weights, adjacency)
    return _ransac(samples, PrimitiveType.CYLINDER, options or RansacOptions())


RANSAC_TYPES = (PrimitiveType.PLANE, PrimitiveType.CYLINDER)


def _detect(
    samples: _Samples,
    types: Sequence[PrimitiveType],
    options: RansacOptions,
    max_primitives: int,
    available: np.ndarray | None = None,
) -> list[FitResult]:
    unsupported = set(types) - set(RANSAC_TYPES)
    if unsupported:
        raise ValueError(f"RANSAC supports planes and cylinders, not {sorted(unsupported)}")
    rng = np.random.default_rng(options.seed)
    available = np.ones(len(samples.points), bool) if available is None else available.copy()
    results: list[FitResult] = []
    while len(results) < max_primitives:
        candidates = [r for kind in types if (r := _ransac(samples, kind, options, available, rng))]
        if not candidates:
            break
        best = max(candidates, key=lambda r: float(samples.weights[r.inliers].sum()))
        results.append(best)
        available[best.inliers] = False
    return results


def detect_primitives(
    points: ArrayLike,
    normals: ArrayLike,
    types: Sequence[PrimitiveType] = RANSAC_TYPES,
    options: RansacOptions | None = None,
    weights: ArrayLike | None = None,
    adjacency: sp.spmatrix | None = None,
    max_primitives: int = 20,
) -> list[FitResult]:
    """Sequential RANSAC: repeatedly extract the best supported primitive of any
    requested type and remove its inliers, until nothing with ``min_support``
    samples remains. Results are ordered by support (largest first)."""
    samples = _make_samples(points, normals, weights, adjacency)
    return _detect(samples, types, options or RansacOptions(), max_primitives)


# ======================================================================================
# Mesh level API
# ======================================================================================
def as_geometry(mesh: MeshLike) -> MeshGeometry:
    return mesh if isinstance(mesh, MeshGeometry) else MeshGeometry.from_polydata(mesh)


def _mesh_samples(geom: MeshGeometry) -> _Samples:
    return _Samples(geom.face_centroids, geom.face_normals, geom.face_areas, geom.adjacency())


def fit_mesh_faces(
    mesh: MeshLike,
    face_ids: ArrayLike,
    kind: PrimitiveType,
    robust_scale: float | None = None,
    max_points: int | None = 60000,
) -> FitResult:
    """Fit ``kind`` to the vertices of ``face_ids`` (area weighted).

    Vertices of a scan lie on the measured surface while face centroids of a
    curved surface lie slightly inside it, so vertices give the more accurate
    (sub-micron on clean data) primitive. Face normals initialise the cylinder
    axis and decide the orientation (hole vs boss).
    """
    geom = as_geometry(mesh)
    selected = np.zeros(geom.n_faces, dtype=bool)
    selected[np.asarray(face_ids, dtype=np.int64)] = True
    faces = np.flatnonzero(selected)
    vertices = geom.face_vertices(faces)
    pts = geom.points[vertices]
    w = geom.vertex_areas[vertices]
    fn, fa, fc = geom.face_normals[faces], geom.face_areas[faces], geom.face_centroids[faces]
    kind = PrimitiveType(kind)
    if kind is PrimitiveType.PLANE:
        fit = fit_plane(pts, w)
        if robust_scale:  # re-weighted refit without gross outliers (e.g. bevelled edges)
            for _ in range(2):
                keep = np.abs(fit.residuals) <= 3.0 * robust_scale
                if keep.sum() < 3 or keep.all():
                    break
                refit = fit_plane(pts[keep], w[keep])
                fit = FitResult(refit.primitive, refit.primitive.distance(pts))
        if (fn * fa[:, None]).sum(axis=0) @ fit.primitive.normal < 0:
            prim = fit.primitive
            fit = FitResult(PlanePrimitive(-prim.normal, -prim.d, prim.center), -fit.residuals)
    elif kind is PrimitiveType.CYLINDER:
        axis, _ = axis_from_normals(fn, fa)
        fit = fit_cylinder(
            pts,
            w,
            initial=_initial_cylinder(pts, axis, w),
            robust_scale=robust_scale,
            max_points=max_points,
        )
        fit.primitive.concave = _orientation_concave(fit.primitive, fc, fn)
    elif kind is PrimitiveType.SPHERE:
        fit = fit_sphere(pts, w, robust_scale=robust_scale, max_points=max_points)
        fit.primitive.concave = _orientation_concave(fit.primitive, fc, fn)
    else:
        raise ValueError(f"cannot fit {kind}")
    fit.inliers = faces
    return fit


@dataclass(eq=False)
class MeshPrimitive:
    """A primitive found on a mesh: vertex-refined fit + supporting faces."""

    fit: FitResult
    face_ids: IntArray

    @property
    def type(self) -> PrimitiveType:
        return self.fit.type

    @property
    def primitive(self) -> Primitive:
        return self.fit.primitive


def _cylinders_compatible(
    a: CylinderPrimitive, b: CylinderPrimitive, angle_tol: float, dist_tol: float, radius_tol: float
) -> bool:
    if a.axis.angle_to(b.axis) > angle_tol:
        return False
    if abs(a.radius - b.radius) > radius_tol:
        return False
    return bool(
        a.axis.distance([b.axis_point])[0] <= dist_tol
        and b.axis.distance([a.axis_point])[0] <= dist_tol
    )


def merge_coaxial_cylinders(
    mesh: MeshLike,
    primitives: Sequence[MeshPrimitive],
    angle_tol_deg: float = 1.0,
    distance_tolerance: float | None = None,
    radius_tolerance: float | None = None,
    robust_scale: float | None = None,
) -> list[MeshPrimitive]:
    """Merge cylinders sharing axis and radius (e.g. the two pin-boss bores) and
    refit them together, which roughly doubles the axis baseline and accuracy."""
    geom = as_geometry(mesh)
    items = list(primitives)
    tol = distance_tolerance or 2e-3 * geom.diagonal
    rtol = radius_tolerance or 2e-3 * geom.diagonal
    merged = True
    while merged:
        merged = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a.type is not PrimitiveType.CYLINDER or b.type is not PrimitiveType.CYLINDER:
                    continue
                if not _cylinders_compatible(
                    a.primitive, b.primitive, np.radians(angle_tol_deg), tol, rtol
                ):
                    continue
                faces = np.union1d(a.face_ids, b.face_ids)
                fit = fit_mesh_faces(geom, faces, PrimitiveType.CYLINDER, robust_scale=robust_scale)
                items[i] = MeshPrimitive(fit, faces)
                del items[j]
                merged = True
                break
            if merged:
                break
    return items


def detect_mesh_primitives(
    mesh: MeshLike,
    types: Sequence[PrimitiveType] = RANSAC_TYPES,
    options: RansacOptions | None = None,
    max_primitives: int = 20,
    merge_coaxial: bool = True,
    face_mask: np.ndarray | None = None,
    fit_tolerance: float | None = None,
) -> list[MeshPrimitive]:
    """RANSAC on face centroids/normals (area weighted, mesh connectivity), then
    each primitive is refitted on the vertices of its faces.

    ``fit_tolerance`` is the noise scale of the robust vertex refit (default:
    :func:`auto_fit_tolerance`); it is much smaller than the RANSAC threshold so
    bevelled borders captured by RANSAC do not bias the final parameters.
    """
    geom = as_geometry(mesh)
    samples = _mesh_samples(geom)
    options = options or RansacOptions()
    found = _detect(samples, types, options, max_primitives, available=face_mask)
    scale = fit_tolerance or auto_fit_tolerance(geom)
    result = [
        MeshPrimitive(fit_mesh_faces(geom, r.inliers, r.type, robust_scale=scale), r.inliers)
        for r in found
    ]
    if merge_coaxial:
        result = merge_coaxial_cylinders(geom, result, robust_scale=scale)
    return result


def extract_planes(
    mesh: MeshLike, options: RansacOptions | None = None, max_count: int = 20
) -> list[MeshPrimitive]:
    """Planes of a mesh with their equation ``ax + by + cz + d = 0`` and faces.

    Planes and cylinders are detected together so that narrow strips of a
    cylinder (which lie within tolerance of a tangent plane) are claimed by the
    cylinder; only the planes are returned.
    """
    found = detect_mesh_primitives(mesh, RANSAC_TYPES, options, max_count + 40, False)
    return [item for item in found if item.type is PrimitiveType.PLANE][:max_count]


def extract_cylinders(
    mesh: MeshLike,
    options: RansacOptions | None = None,
    max_count: int = 20,
    merge_coaxial: bool = True,
) -> list[MeshPrimitive]:
    """Cylinders of a mesh (axis, centre, radius); coaxial pieces (e.g. the two
    pin-boss bores) are merged into one high precision cylinder."""
    found = detect_mesh_primitives(mesh, RANSAC_TYPES, options, max_count + 40, merge_coaxial)
    return [item for item in found if item.type is PrimitiveType.CYLINDER][:max_count]


# ======================================================================================
# Auto segmentation
# ======================================================================================
@dataclass
class SegmentationOptions:
    sharp_angle_deg: float = 30.0  # edges with a larger dihedral angle always split regions
    curvature_tolerance: float = 0.3  # allowed relative curvature change between neighbours
    curvature_noise: float | None = None  # absolute curvature noise [1/mm]; None -> estimated
    smoothing_iterations: int = 3  # curvature tensor smoothing (noise suppression)
    fit_tolerance: float | None = None  # RMS tolerance for primitives; None -> automatic
    min_inlier_ratio: float = 0.8  # share of vertices within 2.5 * fit_tolerance
    min_region_faces: int = 30
    min_region_area_ratio: float = 5e-4  # of the total mesh area
    detect_spheres: bool = True
    split_freeform: bool = True  # RANSAC-split regions that leaked across transitions
    merge_regions: bool = True
    max_radius_ratio: float = 3.0  # cylinders/spheres larger than ratio * diagonal are rejected


@dataclass(eq=False)
class Region:
    id: int
    face_ids: IntArray
    type: PrimitiveType
    fit: FitResult | None
    area: float

    @property
    def primitive(self) -> Primitive | None:
        return self.fit.primitive if self.fit is not None else None

    def describe(self) -> dict[str, str]:
        info = {
            "区域": f"R{self.id}",
            "面片数": f"{len(self.face_ids):,}",
            "面积": f"{self.area:.3f}",
        }
        if self.fit is not None:
            info.update(self.fit.describe())
        else:
            info["基元类型"] = self.type.label
        return info


TYPE_HUES = {
    PrimitiveType.PLANE: 0.58,  # blue
    PrimitiveType.CYLINDER: 0.33,  # green
    PrimitiveType.SPHERE: 0.08,  # orange
    PrimitiveType.FREEFORM: 0.90,  # pink
}


def _hsv_to_rgb(h: FloatArray, s: FloatArray, v: FloatArray) -> FloatArray:
    h = np.mod(h, 1.0) * 6.0
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    table = np.stack(
        [
            np.stack([v, t, p], -1),
            np.stack([q, v, p], -1),
            np.stack([p, v, t], -1),
            np.stack([p, q, v], -1),
            np.stack([t, p, v], -1),
            np.stack([v, p, q], -1),
        ]
    )
    return table[i, np.arange(len(i))]


@dataclass(eq=False)
class SegmentationResult:
    labels: IntArray  # region id per face
    regions: list[Region]
    fit_tolerance: float
    options: SegmentationOptions | None = None

    def region(self, region_id: int) -> Region:
        return self.regions[region_id]

    def faces_of(self, region_ids: Sequence[int]) -> IntArray:
        if not len(region_ids):
            return np.empty(0, dtype=np.int64)
        return np.concatenate([self.regions[i].face_ids for i in region_ids])

    def regions_of_type(self, kind: PrimitiveType) -> list[Region]:
        return [r for r in self.regions if r.type is kind]

    def type_counts(self) -> dict[PrimitiveType, int]:
        return {kind: len(self.regions_of_type(kind)) for kind in PrimitiveType}

    def region_colors(self, scheme: Literal["region", "type"] = "region") -> np.ndarray:
        """``(R, 3)`` uint8 colours. ``type`` uses one hue family per primitive type."""
        n = len(self.regions)
        ids = np.arange(n, dtype=np.float64)
        if scheme == "type":
            hue = np.array([TYPE_HUES[r.type] for r in self.regions])
            hue = hue + 0.035 * np.sin(ids * 2.1)
            sat = 0.45 + 0.2 * ((ids * 0.618) % 1.0)
            val = 0.78 + 0.18 * ((ids * 0.382) % 1.0)
        else:
            hue = (ids * 0.61803398875 + 0.12) % 1.0
            sat = np.full(n, 0.55)
            val = np.where(ids % 2 == 0, 0.95, 0.80)
        return (_hsv_to_rgb(hue, sat, val) * 255).astype(np.uint8)

    def face_colors(self, scheme: Literal["region", "type"] = "region") -> np.ndarray:
        return self.region_colors(scheme)[self.labels]

    def summary(self) -> dict[str, str]:
        counts = self.type_counts()
        return {
            "区域数": str(len(self.regions)),
            **{kind.label: str(counts[kind]) for kind in PrimitiveType},
            "拟合公差": f"{self.fit_tolerance:.4f}",
        }


def auto_fit_tolerance(geom: MeshGeometry) -> float:
    """Default RMS tolerance: 0.025 % of the diagonal, at least 2 % of an edge."""
    return max(2.5e-4 * geom.diagonal, 0.02 * geom.mean_edge_length)


class _Classifier:
    def __init__(self, geom: MeshGeometry, tol: float, options: SegmentationOptions) -> None:
        self.geom = geom
        self.tol = tol
        self.options = options
        self.max_radius = options.max_radius_ratio * geom.diagonal

        self.cos_normal = math.cos(math.radians(max(10.0, options.sharp_angle_deg / 2)))

    def accept(self, fit: FitResult, faces: IntArray) -> bool:
        """Distance *and* normal compatibility of the region with the primitive."""
        ratio = self.options.min_inlier_ratio
        if fit.inlier_ratio(2.5 * self.tol) < ratio or fit.trimmed_rms(ratio) > self.tol:
            return False
        return self.normal_ratio(fit.primitive, faces) >= ratio

    def normal_ratio(self, primitive: Primitive, faces: IntArray) -> float:
        geom = self.geom
        expected = primitive.normals_at(geom.face_centroids[faces])
        cos = np.abs(np.einsum("ij,ij->i", expected, geom.face_normals[faces]))
        area = geom.face_areas[faces]
        return float(area[cos >= self.cos_normal].sum() / max(area.sum(), 1e-300))

    def explains(self, fit: FitResult, faces: IntArray) -> bool:
        """True if ``fit`` also describes this subset (used before merging regions)."""
        geom = self.geom
        vertices = geom.face_vertices(faces)
        dist = np.abs(fit.primitive.distance(geom.points[vertices]))
        ratio = self.options.min_inlier_ratio
        return (
            float(np.mean(dist <= 2.5 * self.tol)) >= ratio
            and self.normal_ratio(fit.primitive, faces) >= ratio
        )

    def _fit(self, faces: IntArray, kind: PrimitiveType) -> FitResult | None:
        geom = self.geom
        if kind is PrimitiveType.CYLINDER:
            _, spread = axis_from_normals(geom.face_normals[faces], geom.face_areas[faces])
            if spread < 1e-5:
                return None  # flat normal set: the axis is undefined
        try:
            fit = fit_mesh_faces(geom, faces, kind, robust_scale=self.tol, max_points=20000)
        except (ValueError, np.linalg.LinAlgError):
            return None
        if getattr(fit.primitive, "radius", 0.0) > self.max_radius:
            return None
        return fit

    def classify(self, faces: IntArray) -> tuple[PrimitiveType, FitResult | None]:
        """Simplest accepted primitive; a curved model replaces the plane only if it
        is accepted too and at least halves the robust RMS (a narrow strip of a big
        cylinder is within tolerance of its tangent plane)."""
        if len(self.geom.face_vertices(faces)) < 8:
            return PrimitiveType.FREEFORM, None
        kinds = [PrimitiveType.PLANE, PrimitiveType.CYLINDER]
        if self.options.detect_spheres:
            kinds.append(PrimitiveType.SPHERE)
        keep = self.options.min_inlier_ratio
        chosen: FitResult | None = None
        for kind in kinds:
            if chosen is not None and chosen.trimmed_rms(keep) <= 0.25 * self.tol:
                break  # already (near) exact; nothing can be significantly better
            fit = self._fit(faces, kind)
            if fit is None or not self.accept(fit, faces):
                continue
            if chosen is None or fit.trimmed_rms(keep) <= 0.5 * chosen.trimmed_rms(keep):
                chosen = fit
        if chosen is None:
            return PrimitiveType.FREEFORM, None
        return chosen.type, chosen


def _pairwise_curvature_mask(geom, curvature, options, smooth) -> np.ndarray:
    a, b = geom.face_pairs.T
    d_max = np.abs(curvature.k_max[a] - curvature.k_max[b])
    d_min = np.abs(curvature.k_min[a] - curvature.k_min[b])
    if options.curvature_noise is not None:
        noise = options.curvature_noise
    else:
        sample = np.concatenate([d_max[smooth], d_min[smooth]])
        noise = 4.0 * 1.4826 * float(np.median(sample)) if sample.size else 0.0
    floor = noise + 1.0 / (50.0 * geom.diagonal)
    rel = options.curvature_tolerance
    scale_max = np.maximum(np.abs(curvature.k_max[a]), np.abs(curvature.k_max[b]))
    scale_min = np.maximum(np.abs(curvature.k_min[a]), np.abs(curvature.k_min[b]))
    return smooth & (d_max <= floor + rel * scale_max) & (d_min <= floor + rel * scale_min)


def auto_segment(
    mesh: MeshLike,
    options: SegmentationOptions | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> SegmentationResult:
    """Partition a triangle mesh into primitive regions (Design X "Auto Segment").

    1. Region growing over edge-adjacent faces; an edge is crossed only if its
       dihedral angle is below ``sharp_angle_deg`` (normal continuity) and the
       principal curvatures on both sides agree (curvature continuity).
    2. Every region is classified plane / cylinder / sphere / freeform by robust
       fitting on its vertices (``fit_tolerance``); adjacent regions whose union
       still fits one primitive are merged.
    3. Large freeform regions (e.g. a plane and a cylinder joined by a tangent
       fillet, or noisy zones) are split by RANSAC restricted to the region.
    4. Primitive regions grow into neighbouring freeform faces that lie on the
       primitive (crisp region borders at bevelled/rounded scan edges).
    5. Slivers below ``min_region_faces`` / ``min_region_area_ratio`` join the
       neighbour with the longest common boundary; changed regions are refitted.
    """
    options = options or SegmentationOptions()
    geom = as_geometry(mesh)
    report = progress or (lambda _f, _m: None)
    tol = options.fit_tolerance or auto_fit_tolerance(geom)
    classifier = _Classifier(geom, tol, options)
    total_area = float(geom.face_areas.sum())
    min_area = options.min_region_area_ratio * total_area

    report(0.05, "估计曲率")
    curvature = face_curvature(geom, options.sharp_angle_deg, options.smoothing_iterations)
    smooth = curvature.smooth_pairs
    labels = geom.components(_pairwise_curvature_mask(geom, curvature, options, smooth))

    report(0.25, "拟合区域")
    groups = _group_faces(labels)
    kinds: list[PrimitiveType] = []
    fits: list[FitResult | None] = []
    for faces in groups:
        small = len(faces) < options.min_region_faces or geom.face_areas[faces].sum() < min_area
        kind, fit = (PrimitiveType.FREEFORM, None) if small else classifier.classify(faces)
        kinds.append(kind)
        fits.append(fit)
    if options.merge_regions:
        labels, kinds, fits = _merge_compatible(geom, labels, kinds, fits, classifier, smooth)

    if options.split_freeform:
        report(0.5, "拆分自由曲面区域")
        labels, kinds, fits = _split_freeform(geom, labels, kinds, fits, classifier, options)
        if options.merge_regions:
            labels, kinds, fits = _merge_compatible(geom, labels, kinds, fits, classifier, smooth)

    report(0.7, "区域边界细化")
    labels = _grow_primitives(geom, labels, kinds, fits, classifier)
    labels, kinds, fits = _split_disconnected(geom, labels, kinds, fits)
    if options.merge_regions:  # growth can make fragments of one surface touch
        labels, kinds, fits = _merge_compatible(geom, labels, kinds, fits, classifier, smooth)
    labels, kinds, fits = _absorb_small(
        geom, labels, kinds, fits, classifier, options, min_area, smooth
    )
    if options.merge_regions:
        labels, kinds, fits = _merge_compatible(geom, labels, kinds, fits, classifier, smooth)

    report(0.9, "整理结果")
    result_regions: list[Region] = []
    groups = _group_faces(labels)
    areas = np.bincount(labels, weights=geom.face_areas, minlength=len(groups))
    final = np.empty_like(labels)
    for new_id, old in enumerate(np.argsort(-areas, kind="stable")):
        faces = groups[old]
        kind, fit = kinds[old], fits[old]
        if kind is not PrimitiveType.FREEFORM:
            fit = fit_mesh_faces(geom, faces, kind, robust_scale=tol)
        result_regions.append(Region(new_id, faces, kind, fit, float(areas[old])))
        final[faces] = new_id
    report(1.0, "完成")
    return SegmentationResult(final, result_regions, tol, options)


def _group_faces(labels: IntArray) -> list[IntArray]:
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels)
    return np.split(order, np.cumsum(counts)[:-1])


def _split_freeform(geom, labels, kinds, fits, classifier, options):
    """Peel planes/cylinders off large freeform regions with RANSAC."""
    samples = _mesh_samples(geom)
    tol = classifier.tol
    ransac = RansacOptions(
        distance_threshold=3.0 * tol,
        normal_threshold_deg=max(10.0, options.sharp_angle_deg / 2),
        min_support=options.min_region_faces,
        max_iterations=512,
        eval_samples=4000,
        seed=0,
    )
    groups = _group_faces(labels)
    new_groups, new_kinds, new_fits = [], [], []
    for faces, kind, fit in zip(groups, kinds, fits, strict=True):
        if kind is not PrimitiveType.FREEFORM or len(faces) < 4 * options.min_region_faces:
            new_groups.append(faces)
            new_kinds.append(kind)
            new_fits.append(fit)
            continue
        mask = np.zeros(geom.n_faces, dtype=bool)
        mask[faces] = True
        for found in _detect(samples, RANSAC_TYPES, ransac, max_primitives=64, available=mask):
            found_kind, found_fit = classifier.classify(found.inliers)
            if found_kind is PrimitiveType.FREEFORM:
                continue
            new_groups.append(found.inliers)
            new_kinds.append(found_kind)
            new_fits.append(found_fit)
            mask[found.inliers] = False
        if mask.any():
            new_groups.append(np.flatnonzero(mask))
            new_kinds.append(PrimitiveType.FREEFORM)
            new_fits.append(None)
    labels = np.empty(geom.n_faces, dtype=np.int64)
    for i, faces in enumerate(new_groups):
        labels[faces] = i
    return _split_disconnected(geom, labels, new_kinds, new_fits)


def _split_disconnected(geom, labels, kinds, fits):
    """Give every connected piece of a region its own label."""
    a, b = geom.face_pairs.T
    pieces = geom.components(labels[a] == labels[b])
    uniq, first = np.unique(pieces, return_index=True)
    owner = labels[first]
    remap = np.empty(uniq.max() + 1, dtype=np.int64)
    remap[uniq] = np.arange(len(uniq))
    return remap[pieces], [kinds[o] for o in owner], [fits[o] for o in owner]


def _grow_primitives(geom, labels, kinds, fits, classifier, max_rounds: int = 25):
    """Move freeform faces bordering a primitive region into it when all three
    vertices lie within tolerance and the face normal agrees with the surface."""
    labels = labels.copy()
    is_free = np.array(
        [k is PrimitiveType.FREEFORM or f is None for k, f in zip(kinds, fits, strict=True)]
    )
    if is_free.all() or not is_free.any():
        return labels
    tol = 2.5 * classifier.tol
    cos_limit = math.cos(math.radians(classifier.options.sharp_angle_deg / 2))
    a, b = geom.face_pairs.T
    for _ in range(max_rounds):
        la, lb = labels[a], labels[b]
        fa, fb = is_free[la], is_free[lb]
        face = np.concatenate([b[~fa & fb], a[fa & ~fb]])
        region = np.concatenate([la[~fa & fb], lb[fa & ~fb]])
        if not len(face):
            break
        error = np.full(len(face), np.inf)
        for rid in np.unique(region):
            sel = np.flatnonzero(region == rid)
            prim = fits[rid].primitive
            f = face[sel]
            dist = np.abs(prim.distance(geom.points[geom.faces[f]].reshape(-1, 3))).reshape(-1, 3)
            cos = np.abs(
                np.einsum("ij,ij->i", prim.normals_at(geom.face_centroids[f]), geom.face_normals[f])
            )
            worst = dist.max(axis=1)
            error[sel] = np.where((worst <= tol) & (cos >= cos_limit), worst, np.inf)
        ok = np.isfinite(error)
        if not ok.any():
            break
        face, region, error = face[ok], region[ok], error[ok]
        order = np.lexsort((error, face))  # best candidate region per face
        face, region = face[order], region[order]
        first = np.r_[True, face[1:] != face[:-1]]
        labels[face[first]] = region[first]
    return labels


def _region_adjacency(geom, labels, smooth):
    """Unique region pairs with total boundary length and smooth boundary share."""
    labels = np.asarray(labels, dtype=np.int64)
    a, b = geom.face_pairs.T
    la, lb = labels[a], labels[b]
    cross = la != lb
    lo, hi = np.minimum(la[cross], lb[cross]), np.maximum(la[cross], lb[cross])
    key = lo * (labels.max() + 1) + hi
    uniq, inv = np.unique(key, return_inverse=True)
    lengths = np.bincount(inv, weights=geom.pair_edge_lengths[cross])
    smooth_len = np.bincount(inv, weights=geom.pair_edge_lengths[cross] * smooth[cross])
    n = labels.max() + 1
    return uniq // n, uniq % n, lengths, smooth_len


def _merge_components(labels: IntArray, n: int, edges_a: IntArray, edges_b: IntArray) -> IntArray:
    """Map each of the ``n`` region ids to the id of its connected component."""
    graph = sp.coo_matrix((np.ones(len(edges_a)), (edges_a, edges_b)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def _apply_mapping(labels, mapping, kinds, fits, representative=None):
    """Relabel regions through ``mapping`` (old id -> new id); kinds/fits come from
    ``representative[new]`` (default: the first old region mapped to it)."""
    n_new = int(mapping.max()) + 1
    if representative is None:
        representative = np.full(n_new, -1)
        order = np.arange(len(mapping))[::-1]
        representative[mapping[order]] = order
    return (mapping[labels], [kinds[r] for r in representative], [fits[r] for r in representative])


def _merge_compatible(geom, labels, kinds, fits, classifier, smooth):
    """Merge neighbouring regions that describe the same surface.

    Freeform fragments joined mostly by smooth edges are merged in one vectorised
    connected-components pass; primitive neighbours of equal type are merged when
    their parameters are close and one refitted primitive explains both.
    """
    n = len(kinds)
    ra, rb, lengths, smooth_len = _region_adjacency(geom, labels, smooth)
    free = np.array([k is PrimitiveType.FREEFORM for k in kinds])
    link = free[ra] & free[rb] & (smooth_len >= 0.5 * lengths)
    mapping = _merge_components(labels, n, ra[link], rb[link])
    labels, kinds, fits = _apply_mapping(labels, mapping, kinds, fits)

    ra, rb, lengths, _ = _region_adjacency(geom, labels, smooth)
    groups = _group_faces(labels)
    parent = np.arange(len(kinds))

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    for k in np.argsort(-lengths):
        i, j = root(int(ra[k])), root(int(rb[k]))
        if i == j or kinds[i] is not kinds[j] or kinds[i] is PrimitiveType.FREEFORM:
            continue
        if not _similar(fits[i], fits[j], classifier):
            continue
        faces = np.concatenate([groups[i], groups[j]])
        kind, fit = classifier.classify(faces)
        if kind is not kinds[i] or not (
            classifier.explains(fit, groups[i]) and classifier.explains(fit, groups[j])
        ):
            continue
        parent[j] = i
        groups[i] = faces
        fits[i] = fit
    roots = np.array([root(i) for i in range(len(parent))])
    uniq, mapping = np.unique(roots, return_inverse=True)
    return _apply_mapping(labels, mapping, kinds, fits, representative=uniq)


def _similar(a: FitResult | None, b: FitResult | None, classifier) -> bool:
    """Cheap parameter pre-check before an expensive union fit."""
    if a is None or b is None:
        return False
    pa, pb = a.primitive, b.primitive
    tol = 10.0 * classifier.tol
    if isinstance(pa, PlanePrimitive) and isinstance(pb, PlanePrimitive):
        return (
            abs(pa.normal @ pb.normal) >= math.cos(math.radians(5.0))
            and abs(pa.distance([pb.center])[0]) <= tol
            and abs(pb.distance([pa.center])[0]) <= tol
        )
    if isinstance(pa, CylinderPrimitive) and isinstance(pb, CylinderPrimitive):
        return (
            pa.axis.angle_to(pb.axis) <= math.radians(5.0)
            and abs(pa.radius - pb.radius) <= max(tol, 0.05 * pa.radius)
            and pa.axis.distance([pb.axis_point])[0] <= max(tol, 0.05 * pa.radius)
        )
    if isinstance(pa, SpherePrimitive) and isinstance(pb, SpherePrimitive):
        return abs(pa.radius - pb.radius) <= max(tol, 0.05 * pa.radius) and float(
            np.linalg.norm(pa.center - pb.center)
        ) <= max(tol, 0.05 * pa.radius)
    return False


def _absorb_small(geom, labels, kinds, fits, classifier, options, min_area, smooth):
    """Merge slivers into the neighbour with the longest (preferably smooth) border."""
    areas = np.bincount(labels, weights=geom.face_areas)
    counts = np.bincount(labels)
    small = (counts < options.min_region_faces) | (areas < min_area)
    if not small.any() or small.all():
        return labels, kinds, fits
    ra, rb, lengths, smooth_len = _region_adjacency(geom, labels, smooth)
    score = lengths + 10.0 * smooth_len
    src = np.concatenate([ra, rb])
    dst = np.concatenate([rb, ra])
    score = np.concatenate([score, score])
    keep = small[src]
    src, dst, score = src[keep], dst[keep], score[keep]
    order = np.lexsort((-score, src))
    src, dst = src[order], dst[order]
    first = np.r_[True, src[1:] != src[:-1]]
    # every small region points at its best neighbour; each resulting tree is
    # rooted at one regular region (or is an isolated cluster of slivers)
    mapping = _merge_components(labels, len(kinds), src[first], dst[first])
    representative = np.full(mapping.max() + 1, -1)
    by_area = np.argsort(areas)  # the largest member of each component wins
    representative[mapping[by_area]] = by_area
    labels, kinds, fits = _apply_mapping(labels, mapping, kinds, fits, representative)
    grown = np.bincount(mapping, minlength=len(representative)) > 1
    groups = _group_faces(labels)
    for new in np.flatnonzero(grown):
        if kinds[new] is not PrimitiveType.FREEFORM:
            try:
                fits[new] = fit_mesh_faces(
                    geom, groups[new], kinds[new], robust_scale=classifier.tol
                )
            except (ValueError, np.linalg.LinAlgError):
                pass
    return labels, kinds, fits
