"""Planar sections of meshes (the raw material for sketches)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

from meshrev.core.types import FloatArray, Plane


@dataclass(eq=False)
class SectionCurve:
    """Polylines where a plane cuts a mesh. ``closed[i]`` tells whether polyline i loops."""

    plane: Plane
    polylines: list[FloatArray] = field(default_factory=list)
    closed: list[bool] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.polylines

    def to_polydata(self) -> pv.PolyData:
        if self.is_empty:
            return pv.PolyData()
        points, lines, offset = [], [], 0
        for pts, is_closed in zip(self.polylines, self.closed, strict=True):
            ids = list(range(offset, offset + len(pts)))
            if is_closed:
                ids.append(offset)
            lines.append([len(ids), *ids])
            points.append(pts)
            offset += len(pts)
        return pv.PolyData(np.vstack(points), lines=np.concatenate(lines))


def slice_mesh(mesh: pv.PolyData, plane: Plane, merge_tolerance: float = 1e-9) -> SectionCurve:
    """Cut ``mesh`` with ``plane`` and chain the segments into ordered polylines."""
    cut = mesh.slice(normal=plane.normal, origin=plane.origin)
    curve = SectionCurve(plane=plane)
    if cut.n_points == 0:
        return curve
    stripped = cut.clean(tolerance=merge_tolerance, absolute=True).strip()
    lines = stripped.lines
    i = 0
    while i < len(lines):
        n = int(lines[i])
        ids = lines[i + 1 : i + 1 + n]
        i += n + 1
        is_closed = n > 2 and ids[0] == ids[-1]
        if is_closed:
            ids = ids[:-1]
        curve.polylines.append(np.asarray(stripped.points[ids], dtype=np.float64))
        curve.closed.append(bool(is_closed))
    return curve


def slice_parallel(mesh: pv.PolyData, plane: Plane, offsets: Sequence[float]) -> list[SectionCurve]:
    """Sections at ``plane`` shifted by each offset along its normal."""
    return [
        slice_mesh(mesh, Plane(plane.origin + off * plane.normal, plane.normal)) for off in offsets
    ]
