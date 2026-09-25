"""Accuracy analysis: deviation of a measured mesh from a reference (CAD) surface.

This is the placeholder for the Design X style "Accuracy Analyzer". The current
implementation computes signed point-to-surface distances with VTK's implicit
distance; color-map display and per-feature tolerance reports come later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv

from meshrev.core.types import FloatArray


@dataclass(eq=False)
class DeviationResult:
    distances: FloatArray  # signed, one per measured mesh point (positive = outside reference)
    tolerance: float

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.distances**2))) if len(self.distances) else 0.0

    @property
    def max_abs(self) -> float:
        return float(np.max(np.abs(self.distances))) if len(self.distances) else 0.0

    @property
    def within_tolerance_ratio(self) -> float:
        if not len(self.distances):
            return 1.0
        return float(np.mean(np.abs(self.distances) <= self.tolerance))

    def summary(self) -> dict[str, float]:
        d = self.distances
        return {
            "rms": self.rms,
            "max_abs": self.max_abs,
            "mean": float(d.mean()) if len(d) else 0.0,
            "min": float(d.min()) if len(d) else 0.0,
            "max": float(d.max()) if len(d) else 0.0,
            "within_tolerance": self.within_tolerance_ratio,
        }


def compute_deviation(
    measured: pv.PolyData, reference: pv.PolyData, tolerance: float = 0.05
) -> DeviationResult:
    """Signed distance of every ``measured`` point to the closed ``reference`` surface."""
    probe = measured.compute_implicit_distance(reference, inplace=False)
    return DeviationResult(
        distances=np.asarray(probe.point_data["implicit_distance"], dtype=np.float64),
        tolerance=tolerance,
    )
