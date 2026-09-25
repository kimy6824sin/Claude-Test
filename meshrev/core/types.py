"""Basic geometric value types shared by all modules.

Conventions: right-handed coordinates, Z is "up" (piston axis after alignment),
lengths in millimetres unless a :class:`Units` value says otherwise.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

_EPS = 1e-12


def new_id(prefix: str = "") -> str:
    """Short unique identifier used for bodies and features."""
    return f"{prefix}{uuid.uuid4().hex[:10]}"


def as_vec3(value: ArrayLike) -> FloatArray:
    vec = np.asarray(value, dtype=np.float64).reshape(-1)
    if vec.shape != (3,):
        raise ValueError(f"expected a 3-vector, got shape {vec.shape}")
    return vec


def as_points(points: ArrayLike) -> FloatArray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected an (N, 3) point array, got shape {pts.shape}")
    return pts


def normalize(vector: ArrayLike) -> FloatArray:
    vec = as_vec3(vector)
    length = float(np.linalg.norm(vec))
    if length < _EPS:
        raise ValueError("cannot normalize a zero-length vector")
    return vec / length


def orthonormal_basis(normal: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return two unit vectors ``(u, v)`` so that ``(u, v, normal)`` is right-handed."""
    n = normalize(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(helper, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


class Units(str, Enum):
    MM = "mm"
    INCH = "in"
    M = "m"

    @property
    def to_mm(self) -> float:
        return {Units.MM: 1.0, Units.INCH: 25.4, Units.M: 1000.0}[self]


@dataclass(frozen=True, eq=False)
class Plane:
    """Infinite plane through ``origin`` with unit ``normal``."""

    origin: FloatArray
    normal: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", as_vec3(self.origin))
        object.__setattr__(self, "normal", normalize(self.normal))

    @classmethod
    def from_equation(cls, a: float, b: float, c: float, d: float) -> Plane:
        """Build the plane ``a*x + b*y + c*z + d = 0``."""
        n = np.array([a, b, c], dtype=np.float64)
        length = float(np.linalg.norm(n))
        if length < _EPS:
            raise ValueError("degenerate plane equation")
        n /= length
        return cls(origin=-(d / length) * n, normal=n)

    @property
    def d(self) -> float:
        return float(-self.normal @ self.origin)

    @property
    def equation(self) -> tuple[float, float, float, float]:
        """Normalised coefficients ``(a, b, c, d)`` with ``a² + b² + c² = 1``."""
        a, b, c = (float(x) for x in self.normal)
        return a, b, c, self.d

    def signed_distance(self, points: ArrayLike) -> FloatArray:
        return (as_points(points) - self.origin) @ self.normal

    def project(self, points: ArrayLike) -> FloatArray:
        pts = as_points(points)
        return pts - np.outer(self.signed_distance(pts), self.normal)

    def basis(self) -> tuple[FloatArray, FloatArray]:
        return orthonormal_basis(self.normal)

    def to_local(self, points: ArrayLike) -> FloatArray:
        """Express points in the plane's 2D ``(u, v)`` frame."""
        u, v = self.basis()
        rel = as_points(points) - self.origin
        return np.column_stack([rel @ u, rel @ v])

    def to_world(self, uv: ArrayLike) -> FloatArray:
        u, v = self.basis()
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        return self.origin + np.outer(uv[:, 0], u) + np.outer(uv[:, 1], v)

    def flipped(self) -> Plane:
        return Plane(self.origin, -self.normal)

    def __repr__(self) -> str:
        a, b, c, d = self.equation
        return f"Plane({a:.6g}x + {b:.6g}y + {c:.6g}z + {d:.6g} = 0)"


@dataclass(frozen=True, eq=False)
class Axis:
    """Infinite line through ``origin`` along unit ``direction``."""

    origin: FloatArray
    direction: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", as_vec3(self.origin))
        object.__setattr__(self, "direction", normalize(self.direction))

    def parameter(self, points: ArrayLike) -> FloatArray:
        """Signed position of the points' projections along the axis."""
        return (as_points(points) - self.origin) @ self.direction

    def point_at(self, t: float | ArrayLike) -> FloatArray:
        t = np.asarray(t, dtype=np.float64)
        return self.origin + np.multiply.outer(t, self.direction)

    def distance(self, points: ArrayLike) -> FloatArray:
        """Perpendicular distance of the points from the axis line."""
        rel = as_points(points) - self.origin
        return np.linalg.norm(np.cross(rel, self.direction), axis=1)

    def angle_to(self, other: Axis) -> float:
        """Angle between the two lines in radians, ignoring orientation (0..pi/2)."""
        cos = abs(float(self.direction @ other.direction))
        return float(np.arccos(min(1.0, cos)))

    def __repr__(self) -> str:
        o = ", ".join(f"{x:.6g}" for x in self.origin)
        d = ", ".join(f"{x:.6g}" for x in self.direction)
        return f"Axis(origin=({o}), direction=({d}))"


@dataclass(frozen=True, eq=False)
class BBox:
    """Axis-aligned bounding box."""

    min: FloatArray
    max: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "min", as_vec3(self.min))
        object.__setattr__(self, "max", as_vec3(self.max))

    @classmethod
    def from_points(cls, points: ArrayLike) -> BBox:
        pts = as_points(points)
        if len(pts) == 0:
            raise ValueError("cannot build a bounding box from zero points")
        return cls(pts.min(axis=0), pts.max(axis=0))

    @classmethod
    def from_bounds(cls, bounds: Sequence[float]) -> BBox:
        """From a VTK style ``(xmin, xmax, ymin, ymax, zmin, zmax)`` tuple."""
        b = [float(x) for x in bounds]
        return cls((b[0], b[2], b[4]), (b[1], b[3], b[5]))

    @property
    def center(self) -> FloatArray:
        return 0.5 * (self.min + self.max)

    @property
    def size(self) -> FloatArray:
        return self.max - self.min

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.size))

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        return (
            float(self.min[0]),
            float(self.max[0]),
            float(self.min[1]),
            float(self.max[1]),
            float(self.min[2]),
            float(self.max[2]),
        )

    def union(self, other: BBox) -> BBox:
        return BBox(np.minimum(self.min, other.min), np.maximum(self.max, other.max))

    def __repr__(self) -> str:
        return f"BBox(min={self.min.round(6).tolist()}, max={self.max.round(6).tolist()})"
