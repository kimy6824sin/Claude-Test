"""2D sketch entities living in a :class:`~meshrev.core.types.Plane`.

Sketches are the bridge between section curves (measured data) and CAD solids:
``slice -> fit_sketch -> CadKernel.revolve/extrude``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from meshrev.core.section.slicer import SectionCurve
from meshrev.core.types import FloatArray, Plane


@dataclass(frozen=True)
class Line2D:
    start: tuple[float, float]
    end: tuple[float, float]

    @property
    def length(self) -> float:
        return float(np.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1]))

    def sample(self, n: int = 2) -> FloatArray:
        t = np.linspace(0.0, 1.0, max(2, n))[:, None]
        return (1 - t) * np.asarray(self.start) + t * np.asarray(self.end)


@dataclass(frozen=True)
class Arc2D:
    """Circular arc from ``start_angle`` to ``end_angle`` (radians, counter-clockwise)."""

    center: tuple[float, float]
    radius: float
    start_angle: float
    end_angle: float

    @property
    def sweep(self) -> float:
        return float((self.end_angle - self.start_angle) % (2 * np.pi) or 2 * np.pi)

    @property
    def length(self) -> float:
        return self.radius * self.sweep

    def point_at(self, angle: float) -> tuple[float, float]:
        return (
            self.center[0] + self.radius * np.cos(angle),
            self.center[1] + self.radius * np.sin(angle),
        )

    @property
    def start(self) -> tuple[float, float]:
        return self.point_at(self.start_angle)

    @property
    def end(self) -> tuple[float, float]:
        return self.point_at(self.start_angle + self.sweep)

    @property
    def mid(self) -> tuple[float, float]:
        return self.point_at(self.start_angle + 0.5 * self.sweep)

    def sample(self, n: int = 32) -> FloatArray:
        angles = self.start_angle + np.linspace(0.0, self.sweep, max(2, n))
        return np.column_stack(
            [
                self.center[0] + self.radius * np.cos(angles),
                self.center[1] + self.radius * np.sin(angles),
            ]
        )


SketchEntity = Line2D | Arc2D


@dataclass
class Sketch:
    plane: Plane
    entities: list[SketchEntity] = field(default_factory=list)

    def is_closed(self, tol: float = 1e-6) -> bool:
        if not self.entities:
            return False
        first = np.asarray(self.entities[0].start)
        last = np.asarray(self.entities[-1].end)
        return bool(np.linalg.norm(first - last) <= tol)

    def to_polyline(self, arc_segments: int = 32) -> FloatArray:
        """World-space polyline approximation (for display)."""
        parts = [e.sample(arc_segments if isinstance(e, Arc2D) else 2) for e in self.entities]
        if not parts:
            return np.empty((0, 3))
        return self.plane.to_world(np.vstack(parts))


def fit_sketch(curve: SectionCurve, tolerance: float) -> Sketch:
    """Fit lines and arcs to a measured section curve.

    Planned algorithm: split each polyline at curvature breakpoints, fit a line or
    an arc to every span (whichever stays within ``tolerance``), then enforce
    tangency/coincidence constraints between neighbouring entities.
    """
    raise NotImplementedError("sketch fitting is scheduled for a later milestone")
