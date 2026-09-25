"""Procedural demo parts for testing and demos (no external data needed).

The demo piston is built as a signed distance field (CSG of analytic shapes) and
meshed with marching cubes, which yields a watertight, scan-like triangle mesh:
exact planes and cylinders, slightly bevelled sharp edges and uniform triangles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv

from meshrev.core.mesh.processing import clean, orient_outward


@dataclass(frozen=True)
class PistonSpec:
    """Dimensions (mm) of the demo piston. Z is the piston axis, crown at the top."""

    radius: float = 43.0  # 86 mm bore
    height: float = 70.0
    top_land: float = 6.0
    groove_depth: float = 3.5
    groove_widths: tuple[float, float, float] = (1.2, 1.5, 2.5)  # compression, compression, oil
    land_width: float = 4.0
    inner_radius: float = 36.0  # skirt/cavity inner wall
    boss_half_distance: float = 25.0  # pin boss inner faces at x = ±25
    crown_thickness: float = 12.0  # under-crown plane at height - crown_thickness
    pin_height: float = 36.0  # pin bore axis: parallel to X at z = pin_height
    pin_radius: float = 11.0
    dish_radius: float = 60.0  # spherical combustion bowl in the crown
    dish_depth: float = 4.0

    @property
    def groove_ranges(self) -> list[tuple[float, float]]:
        """(z_min, z_max) for each ring groove, top to bottom."""
        ranges, z = [], self.height - self.top_land
        for width in self.groove_widths:
            ranges.append((z - width, z))
            z -= width + self.land_width
        return ranges

    @property
    def under_crown_height(self) -> float:
        return self.height - self.crown_thickness

    @property
    def dish_center_z(self) -> float:
        return self.height + self.dish_radius - self.dish_depth


def _box2d(r: np.ndarray, z: np.ndarray, r0: float, r1: float, z0: float, z1: float) -> np.ndarray:
    dr = np.abs(r - 0.5 * (r0 + r1)) - 0.5 * (r1 - r0)
    dz = np.abs(z - 0.5 * (z0 + z1)) - 0.5 * (z1 - z0)
    outside = np.sqrt(np.maximum(dr, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2)
    return outside + np.minimum(np.maximum(dr, dz), 0.0)


def piston_sdf(x: np.ndarray, y: np.ndarray, z: np.ndarray, spec: PistonSpec) -> np.ndarray:
    """Signed distance-like field of the demo piston (negative inside the material)."""
    r = np.sqrt(x * x + y * y)
    d = _box2d(r, z, -spec.radius, spec.radius, 0.0, spec.height)
    groove_r = spec.radius - spec.groove_depth
    for z0, z1 in spec.groove_ranges:
        d = np.maximum(d, -_box2d(r, z, groove_r, spec.radius + 5.0, z0, z1))
    cavity = np.maximum(
        np.maximum(r - spec.inner_radius, np.abs(x) - spec.boss_half_distance),
        z - spec.under_crown_height,
    )
    d = np.maximum(d, -cavity)
    bore = np.sqrt(y * y + (z - spec.pin_height) ** 2) - spec.pin_radius
    d = np.maximum(d, -bore)
    dish = np.sqrt(x * x + y * y + (z - spec.dish_center_z) ** 2) - spec.dish_radius
    return np.maximum(d, -dish)


def make_demo_piston(spec: PistonSpec | None = None, voxel_size: float = 0.5) -> pv.PolyData:
    """Mesh the demo piston with marching cubes at ``voxel_size`` resolution.

    ``voxel_size=0.5`` gives ~450k triangles; 0.8 gives ~180k (good for tests).
    """
    spec = spec or PistonSpec()
    h = float(voxel_size)
    pad = 2 * h
    # an irrational-ish offset keeps grid samples off the exact zero level set
    offset = 0.3713 * h
    xs = np.arange(-spec.radius - pad, spec.radius + pad + h, h) + offset
    zs = np.arange(-pad, spec.height + pad + h, h) + 0.77 * offset
    x, y, z = np.meshgrid(
        xs.astype(np.float32), xs.astype(np.float32), zs.astype(np.float32), indexing="ij"
    )
    field = piston_sdf(x, y, z, spec)
    grid = pv.ImageData(dimensions=field.shape, spacing=(h, h, h), origin=(xs[0], xs[0], zs[0]))
    grid.point_data["sdf"] = field.ravel(order="F")
    surface = grid.contour(
        [0.0], scalars="sdf", method="flying_edges", compute_normals=False, compute_scalars=False
    )
    mesh = orient_outward(clean(surface))
    mesh.point_data.clear()
    mesh.cell_data.clear()
    return mesh
