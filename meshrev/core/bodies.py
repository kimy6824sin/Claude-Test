"""Bodies: the displayable/exportable results stored in a :class:`Document`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import pyvista as pv

from meshrev.core.cad.kernel import CadKernel, Shape
from meshrev.core.mesh.processing import MeshStats, mesh_statistics
from meshrev.core.section.slicer import SectionCurve
from meshrev.core.types import BBox, Units, new_id

RGB = tuple[float, float, float]


class BodyKind(str, Enum):
    MESH = "mesh"
    CAD = "cad"
    SECTION = "section"


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def format_vec(vec, digits: int = 4) -> str:
    return "(" + ", ".join(_fmt(float(x), digits) for x in vec) + ")"


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

    @abstractmethod
    def to_polydata(self) -> pv.PolyData:
        """Geometry used for display, bounds and mesh export."""

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
            info["体积"] = f"{self.kernel.volume(self.shape):.2f}"
        except Exception:  # noqa: BLE001 - informative only
            info["体积"] = "-"
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
        a, b, c, d = self.curve.plane.equation
        info["截面平面"] = f"{a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0"
        info["多段线数"] = str(len(self.curve.polylines))
        return info
