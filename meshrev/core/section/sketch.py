"""2D sketch entities living in a :class:`~meshrev.core.types.Plane`.

Sketches are the bridge between section curves (measured data) and CAD solids:
``slice_mesh -> sketch_fit.fit_section -> CadKernel.revolve/extrude``.

Structured representation (all coordinates in the plane's ``(u, v)`` frame):

* ``Line2D(p1, p2)``
* ``Arc2D(center, radius, start_angle, end_angle, ccw)`` - traversed from
  ``start_angle`` to ``end_angle``; ``ccw=True`` means counter-clockwise
  (the arc then covers the CCW sweep from start to end), ``False`` clockwise.
* ``Circle2D(center, radius)`` - a full circle (bolt holes, bores).

A :class:`Sketch` holds one closed outer loop (``entities``) and optional
inner loops (``holes``), i.e. exactly what a planar face for extrusion needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from meshrev.core.section.slicer import SectionCurve
from meshrev.core.types import FloatArray, Plane

Point2 = tuple[float, float]
TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class Line2D:
    start: Point2
    end: Point2

    @property
    def p1(self) -> Point2:
        return self.start

    @property
    def p2(self) -> Point2:
        return self.end

    @property
    def length(self) -> float:
        return float(np.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1]))

    @property
    def direction(self) -> FloatArray:
        d = np.subtract(self.end, self.start)
        return d / max(float(np.linalg.norm(d)), 1e-300)

    @property
    def angle_deg(self) -> float:
        """Direction angle in degrees, in [0, 180)."""
        d = self.direction
        return float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)

    def reversed(self) -> Line2D:
        return Line2D(self.end, self.start)

    def sample(self, n: int = 2) -> FloatArray:
        t = np.linspace(0.0, 1.0, max(2, n))[:, None]
        return (1 - t) * np.asarray(self.start) + t * np.asarray(self.end)

    def distance(self, points: FloatArray) -> FloatArray:
        """Distance of points to the segment."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        a = np.asarray(self.start)
        d = np.subtract(self.end, self.start)
        t = np.clip((p - a) @ d / max(float(d @ d), 1e-300), 0.0, 1.0)
        return np.linalg.norm(p - (a + t[:, None] * d), axis=1)

    def __repr__(self) -> str:
        (x1, y1), (x2, y2) = self.start, self.end
        return f"Line(({x1:.4f}, {y1:.4f}), ({x2:.4f}, {y2:.4f}))"


@dataclass(frozen=True)
class Arc2D:
    """Circular arc; angles in radians, see the module docstring for ``ccw``."""

    center: Point2
    radius: float
    start_angle: float
    end_angle: float
    ccw: bool = True

    @property
    def sweep(self) -> float:
        """Positive angle travelled from start to end in the arc's direction."""
        delta = (
            (self.end_angle - self.start_angle) if self.ccw else (self.start_angle - self.end_angle)
        )
        return float(delta % TWO_PI or TWO_PI)

    @property
    def length(self) -> float:
        return self.radius * self.sweep

    def point_at(self, angle: float) -> Point2:
        return (
            self.center[0] + self.radius * math.cos(angle),
            self.center[1] + self.radius * math.sin(angle),
        )

    def _angle_at(self, fraction: float) -> float:
        sign = 1.0 if self.ccw else -1.0
        return self.start_angle + sign * fraction * self.sweep

    @property
    def start(self) -> Point2:
        return self.point_at(self.start_angle)

    @property
    def end(self) -> Point2:
        return self.point_at(self._angle_at(1.0))

    @property
    def mid(self) -> Point2:
        return self.point_at(self._angle_at(0.5))

    def reversed(self) -> Arc2D:
        return Arc2D(self.center, self.radius, self.end_angle, self.start_angle, not self.ccw)

    def sample(self, n: int = 32) -> FloatArray:
        angles = np.array([self._angle_at(f) for f in np.linspace(0.0, 1.0, max(2, n))])
        return np.column_stack(
            [
                self.center[0] + self.radius * np.cos(angles),
                self.center[1] + self.radius * np.sin(angles),
            ]
        )

    def __repr__(self) -> str:
        a0, a1 = math.degrees(self.start_angle), math.degrees(self.end_angle)
        return (
            f"Arc(center=({self.center[0]:.4f}, {self.center[1]:.4f}), r={self.radius:.4f}, "
            f"start={a0:.2f}°, end={a1:.2f}°, {'ccw' if self.ccw else 'cw'})"
        )


@dataclass(frozen=True)
class Circle2D:
    center: Point2
    radius: float

    @property
    def length(self) -> float:
        return TWO_PI * self.radius

    @property
    def start(self) -> Point2:
        return (self.center[0] + self.radius, self.center[1])

    @property
    def end(self) -> Point2:
        return self.start

    def reversed(self) -> Circle2D:
        return self

    def sample(self, n: int = 64) -> FloatArray:
        angles = np.linspace(0.0, TWO_PI, max(3, n))
        cx, cy = self.center
        r = self.radius
        return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])

    def __repr__(self) -> str:
        return f"Circle(center=({self.center[0]:.4f}, {self.center[1]:.4f}), r={self.radius:.4f})"


SketchEntity = Line2D | Arc2D | Circle2D


def loop_is_closed(entities: list[SketchEntity], tol: float = 1e-6) -> bool:
    if not entities:
        return False
    if len(entities) == 1 and isinstance(entities[0], Circle2D):
        return True
    return bool(np.linalg.norm(np.subtract(entities[0].start, entities[-1].end)) <= tol)


def loop_polyline(entities: list[SketchEntity], arc_segments: int = 32) -> FloatArray:
    parts = [e.sample(arc_segments if isinstance(e, (Arc2D, Circle2D)) else 2) for e in entities]
    return np.vstack(parts) if parts else np.empty((0, 2))


@dataclass
class Sketch:
    """Closed outer loop (``entities``) plus inner loops (``holes``) on ``plane``."""

    plane: Plane
    entities: list[SketchEntity] = field(default_factory=list)
    holes: list[list[SketchEntity]] = field(default_factory=list)

    def is_closed(self, tol: float = 1e-6) -> bool:
        loops_ok = all(loop_is_closed(h, tol) for h in self.holes)
        return loop_is_closed(self.entities, tol) and loops_ok

    @property
    def loops(self) -> list[list[SketchEntity]]:
        return [self.entities, *self.holes]

    def to_polyline(self, arc_segments: int = 32) -> FloatArray:
        """World-space polyline approximation of the outer loop (for display)."""
        uv = loop_polyline(self.entities, arc_segments)
        return self.plane.to_world(uv) if len(uv) else np.empty((0, 3))


def fit_sketch(curve: SectionCurve, tolerance: float | None = None) -> list[Sketch]:
    """Fit lines/arcs/circles to a section and return one sketch per outer region.

    Thin wrapper around :func:`meshrev.core.sketch_fit.fit_section`.
    """
    from meshrev.core.sketch_fit import SketchFitOptions, fit_section

    return fit_section(curve, SketchFitOptions(tolerance=tolerance)).to_sketches()
