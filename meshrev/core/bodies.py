"""Bodies: the displayable/exportable results stored in a :class:`Document`."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import pyvista as pv

from meshrev.core.cad.kernel import CadKernel, Shape
from meshrev.core.mesh.processing import MeshStats, mesh_statistics
from meshrev.core.mesh.topology import MeshGeometry
from meshrev.core.primitives import PrimitiveType, SegmentationResult
from meshrev.core.section.slicer import SectionCurve
from meshrev.core.types import Axis, BBox, Plane, Units, new_id

RGB = tuple[float, float, float]


class BodyKind(str, Enum):
    MESH = "mesh"
    CAD = "cad"
    SECTION = "section"
    REGIONS = "regions"
    DATUM_AXIS = "datum_axis"
    DATUM_PLANE = "datum_plane"
    SKETCH = "sketch"
    DEVIATION = "deviation"


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def format_vec(vec, digits: int = 4) -> str:
    return "(" + ", ".join(_fmt(float(x), digits) for x in vec) + ")"


def format_plane_equation(a: float, b: float, c: float, d: float, digits: int = 6) -> str:
    terms = [f"{a:.{digits}f}x"]
    for value, var in ((b, "y"), (c, "z"), (d, "")):
        sign = "-" if value < 0 else "+"
        terms.append(f"{sign} {abs(value):.{digits}f}{var}")
    return " ".join(terms) + " = 0"


class Body(ABC):
    """Base class of everything a feature can produce."""

    kind: ClassVar[BodyKind]
    default_color: ClassVar[RGB] = (0.72, 0.76, 0.82)

    def __init__(
        self,
        name: str,
        *,
        body_id: str | None = None,
        color: RGB | None = None,
        visible: bool = True,
    ) -> None:
        self.id = body_id or new_id("B")
        self.name = name
        self.color: RGB = color or self.default_color
        self.visible = visible
        self.source_feature: str | None = None
        # bodies this one replaces on screen (boolean operands, the mesh under a
        # region set); the GUI hides them while this body exists
        self.consumes: list[str] = []

    @abstractmethod
    def to_polydata(self) -> pv.PolyData:
        """Geometry used for display, bounds and mesh export."""

    @property
    def tag(self) -> str:
        """Short ASCII label drawn in the 3D view (VTK's default font has no CJK glyphs)."""
        match = re.search(r"(\d+)\s*$", self.name)
        return f"{self.tag_prefix}{match.group(1) if match else ''}"

    tag_prefix: ClassVar[str] = "B"

    def bounds(self) -> BBox | None:
        data = self.to_polydata()
        return BBox.from_bounds(data.bounds) if data.n_points else None

    def info(self) -> dict[str, str]:
        """Ordered, human readable properties for the property panel."""
        return {"名称": self.name, "类型": self.kind.value, "ID": self.id}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r}, name={self.name!r})"


class MeshBody(Body):
    kind = BodyKind.MESH

    def __init__(
        self,
        polydata: pv.PolyData,
        name: str,
        *,
        source_path: Path | None = None,
        units: Units = Units.MM,
        **kwargs,
    ) -> None:
        super().__init__(name, **kwargs)
        self.polydata = polydata
        self.source_path = Path(source_path) if source_path else None
        self.units = units

    def to_polydata(self) -> pv.PolyData:
        return self.polydata

    @cached_property
    def stats(self) -> MeshStats:
        return mesh_statistics(self.polydata)

    def info(self) -> dict[str, str]:
        s = self.stats
        info = super().info()
        info.update(
            {
                "文件": str(self.source_path) if self.source_path else "-",
                "单位": self.units.value,
                "顶点数": f"{s.n_points:,}",
                "三角面数": f"{s.n_faces:,}",
                "包围盒最小": format_vec(s.bbox.min, 3),
                "包围盒最大": format_vec(s.bbox.max, 3),
                "尺寸": format_vec(s.bbox.size, 3),
                "表面积": f"{s.area:.2f}",
                "体积": f"{s.volume:.2f}" if s.volume is not None else "- (非封闭)",
                "边界边": str(s.n_boundary_edges),
                "非流形边": str(s.n_non_manifold_edges),
            }
        )
        return info


class CadBody(Body):
    kind = BodyKind.CAD
    default_color = (0.80, 0.70, 0.45)

    def __init__(
        self,
        shape: Shape,
        name: str,
        *,
        kernel: CadKernel | None = None,
        tessellation_tolerance: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(name, **kwargs)
        self.shape = shape
        self._kernel = kernel
        self.tessellation_tolerance = tessellation_tolerance
        self._mesh: pv.PolyData | None = None

    @property
    def kernel(self) -> CadKernel:
        if self._kernel is None:
            from meshrev.core.cad.kernel import get_kernel

            self._kernel = get_kernel()
        return self._kernel

    def to_polydata(self) -> pv.PolyData:
        if self._mesh is None:
            self._mesh = self.kernel.tessellate(self.shape, self.tessellation_tolerance)
        return self._mesh

    def info(self) -> dict[str, str]:
        info = super().info()
        info["内核"] = self.kernel.name
        try:
            info["体积"] = f"{self.kernel.volume(self.shape):.3f}"
            counts = self.kernel.topology_counts(self.shape)
            info["实体 / 面 / 边"] = f"{counts['solids']} / {counts['faces']} / {counts['edges']}"
            info["B-Rep 有效"] = "是" if self.kernel.is_valid(self.shape) else "否"
        except Exception:  # noqa: BLE001 - informative only
            info["体积"] = "-"
        box = self.bounds()
        if box is not None:
            info["尺寸"] = format_vec(box.size, 3)
        return info


class SectionBody(Body):
    kind = BodyKind.SECTION
    default_color = (0.95, 0.35, 0.20)

    def __init__(self, curve: SectionCurve, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.curve = curve

    def to_polydata(self) -> pv.PolyData:
        return self.curve.to_polydata()

    def info(self) -> dict[str, str]:
        info = super().info()
        info["截面平面"] = format_plane_equation(*self.curve.plane.equation, digits=4)
        info["多段线数"] = str(len(self.curve.polylines))
        return info


ColorScheme = Literal["region", "type"]


class RegionSetBody(Body):
    """A mesh partitioned into primitive regions (auto segmentation / RANSAC result).

    ``polydata`` is the triangle mesh the labels refer to (one label per cell).
    """

    kind = BodyKind.REGIONS

    def __init__(
        self,
        polydata: pv.PolyData,
        segmentation: SegmentationResult,
        name: str,
        *,
        mesh_id: str | None = None,
        color_scheme: ColorScheme = "region",
        **kwargs,
    ) -> None:
        super().__init__(name, **kwargs)
        if len(segmentation.labels) != polydata.n_cells:
            raise ValueError("segmentation labels do not match the mesh cells")
        self.polydata = polydata
        self.segmentation = segmentation
        self.mesh_id = mesh_id
        if mesh_id:
            self.consumes = [mesh_id]
        self.color_scheme: ColorScheme = color_scheme

    @cached_property
    def geometry(self) -> MeshGeometry:
        return MeshGeometry.from_polydata(self.polydata)

    def to_polydata(self) -> pv.PolyData:
        return self.polydata

    def face_colors(self) -> np.ndarray:
        return self.segmentation.face_colors(self.color_scheme)

    # -- region queries used by the tree, the property panel and picking --------------
    def region_of_face(self, face_id: int) -> int:
        return int(self.segmentation.labels[face_id])

    def faces_of_regions(self, region_ids: Sequence[int]) -> np.ndarray:
        return self.segmentation.faces_of(list(region_ids))

    def region_ids_of_type(self, type_key: str) -> list[int]:
        return [r.id for r in self.segmentation.regions_of_type(PrimitiveType(type_key))]

    def region_groups(self) -> list[tuple[str, str, list[tuple[int, str, tuple[int, int, int]]]]]:
        """``(type key, label, [(region id, text, rgb)])`` for every non-empty type."""
        colors = self.segmentation.region_colors(self.color_scheme)
        groups = []
        for kind in PrimitiveType:
            rows = []
            for region in self.segmentation.regions_of_type(kind):
                text = f"R{region.id}  {kind.label}  ({len(region.face_ids):,} 面)"
                prim = region.primitive
                if prim is not None and hasattr(prim, "radius"):
                    text += f"  r={prim.radius:.3f}"
                rows.append((region.id, text, tuple(int(c) for c in colors[region.id])))
            if rows:
                groups.append((kind.value, kind.label, rows))
        return groups

    def regions_info(self, region_ids: Sequence[int]) -> dict[str, str]:
        regions = [self.segmentation.region(i) for i in region_ids]
        if len(regions) == 1:
            return regions[0].describe()
        kinds = sorted({r.type.label for r in regions})
        return {
            "选中区域": ", ".join(f"R{r.id}" for r in regions),
            "区域数": str(len(regions)),
            "类型": " / ".join(kinds),
            "面片数": f"{sum(len(r.face_ids) for r in regions):,}",
            "面积": f"{sum(r.area for r in regions):.3f}",
        }

    def info(self) -> dict[str, str]:
        info = super().info()
        info["源网格"] = self.mesh_id or "-"
        info.update(self.segmentation.summary())
        return info


class DatumAxisBody(Body):
    """Reference axis (e.g. the pin bore axis extracted from a fitted cylinder)."""

    kind = BodyKind.DATUM_AXIS
    default_color = (0.95, 0.45, 0.05)
    tag_prefix = "A"

    def __init__(
        self,
        axis: Axis,
        length: float,
        name: str,
        *,
        radius: float | None = None,
        rms: float | None = None,
        concave: bool | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name, **kwargs)
        self.axis = axis
        self.length = float(length)
        self.radius = radius
        self.rms = rms
        self.concave = concave

    def endpoints(self, margin_ratio: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
        half = 0.5 * self.length * (1.0 + 2.0 * margin_ratio)
        half = max(half, 1e-3)
        return self.axis.point_at(-half), self.axis.point_at(half)

    def to_polydata(self) -> pv.PolyData:
        start, end = self.endpoints()
        return pv.Line(start, end)

    def info(self) -> dict[str, str]:
        info = super().info()
        info["轴线点"] = format_vec(self.axis.origin, 5)
        info["轴线方向"] = format_vec(self.axis.direction, 6)
        info["长度"] = f"{self.length:.3f}"
        if self.radius is not None:
            info["源圆柱半径"] = f"{self.radius:.5f}"
            info["源圆柱直径"] = f"{2 * self.radius:.5f}"
        if self.concave is not None:
            info["源圆柱类型"] = "孔" if self.concave else "轴"
        if self.rms is not None:
            info["拟合 RMS"] = f"{self.rms:.5f}"
        return info


class DatumPlaneBody(Body):
    """Reference plane (e.g. the cylinder head gasket face)."""

    kind = BodyKind.DATUM_PLANE
    default_color = (0.30, 0.55, 0.95)
    tag_prefix = "P"

    def __init__(
        self, plane: Plane, size: float, name: str, *, rms: float | None = None, **kwargs
    ) -> None:
        super().__init__(name, **kwargs)
        self.plane = plane
        self.size = float(size)
        self.rms = rms

    def to_polydata(self) -> pv.PolyData:
        return pv.Plane(
            center=self.plane.origin,
            direction=self.plane.normal,
            i_size=self.size,
            j_size=self.size,
            i_resolution=1,
            j_resolution=1,
        )

    def info(self) -> dict[str, str]:
        info = super().info()
        info["平面方程"] = format_plane_equation(*self.plane.equation)
        info["法向"] = format_vec(self.plane.normal, 6)
        info["中心点"] = format_vec(self.plane.origin, 4)
        if self.rms is not None:
            info["拟合 RMS"] = f"{self.rms:.5f}"
        return info


ENTITY_CODES = {"line": 0, "arc": 1, "circle": 2}


class SketchBody(Body):
    """Fitted mesh sketch: structured lines/arcs/circles on a plane."""

    kind = BodyKind.SKETCH
    default_color = (0.10, 0.45, 0.95)
    tag_prefix = "S"

    def __init__(self, result, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.result = result  # meshrev.core.sketch_fit.SketchFitResult

    def to_polydata(self, arc_segments: int = 48) -> pv.PolyData:
        """One polyline cell per entity with ``entity_type`` (0 line, 1 arc, 2 circle)."""
        from meshrev.core.section.sketch import Arc2D, Circle2D

        points, lines, codes, offset = [], [], [], 0
        plane = self.result.plane
        for loop in self.result.loops:
            for entity in loop.entities:
                uv = entity.sample(arc_segments if isinstance(entity, (Arc2D, Circle2D)) else 2)
                pts = plane.to_world(uv)
                points.append(pts)
                lines.append(np.r_[len(pts), np.arange(offset, offset + len(pts))])
                offset += len(pts)
                kind = (
                    "circle"
                    if isinstance(entity, Circle2D)
                    else ("arc" if isinstance(entity, Arc2D) else "line")
                )
                codes.append(ENTITY_CODES[kind])
        if not points:
            return pv.PolyData()
        poly = pv.PolyData(np.vstack(points), lines=np.concatenate(lines))
        poly.cell_data["entity_type"] = np.array(codes, dtype=np.int8)
        return poly

    def info(self) -> dict[str, str]:
        info = super().info()
        result = self.result
        counts = result.counts()
        info["草图平面"] = format_plane_equation(*result.plane.equation, digits=4)
        info["轮廓数"] = str(len(result.loops))
        info["外轮廓 / 孔 / 开放"] = " / ".join(
            str(n)
            for n in (
                sum(lp.closed and not lp.is_hole for lp in result.loops),
                sum(lp.is_hole for lp in result.loops),
                sum(not lp.closed for lp in result.loops),
            )
        )
        info["直线 / 圆弧 / 整圆"] = f"{counts['line']} / {counts['arc']} / {counts['circle']}"
        info["拟合公差"] = f"{result.tolerance:.4f}"
        info["最大偏差"] = f"{result.max_deviation:.4f}"
        for k, text in enumerate(result.describe(limit=80)):
            info[f"#{k:02d}"] = text.strip()
        return info


class DeviationBody(Body):
    """Colour-mapped accuracy analysis result (scan mesh or CAD samples)."""

    kind = BodyKind.DEVIATION
    tag_prefix = "D"

    def __init__(self, geometry: pv.PolyData, result, name: str, bands: int = 15, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.geometry = geometry
        self.result = result  # meshrev.core.deviation.DeviationResult
        self.bands = bands

    def to_polydata(self) -> pv.PolyData:
        data = self.geometry.copy(deep=False)
        data.point_data["deviation"] = self.result.distances
        return data

    def info(self) -> dict[str, str]:
        info = super().info()
        info.update(self.result.report())
        return info
