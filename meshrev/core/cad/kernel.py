"""Abstract CAD kernel interface.

Everything outside ``core/cad`` talks to solids only through :class:`CadKernel`,
so the OpenCASCADE dependency stays optional and swappable. Shapes are opaque
handles owned by the kernel implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyvista as pv

from meshrev.core.types import Axis, BBox, FloatArray

if TYPE_CHECKING:
    from meshrev.core.primitives import Primitive
    from meshrev.core.section.sketch import Sketch

Shape = Any


class KernelUnavailableError(RuntimeError):
    """Raised when the optional CAD kernel (OCP) is not installed."""


class BooleanOp(str, Enum):
    UNION = "union"
    CUT = "cut"
    INTERSECT = "intersect"


class CadKernel(ABC):
    name: str = "abstract"

    @abstractmethod
    def revolve(self, sketch: Sketch, axis: Axis, angle_deg: float = 360.0) -> Shape:
        """Revolve a closed sketch around ``axis`` (piston bodies are mostly revolves)."""

    @abstractmethod
    def extrude(self, sketch: Sketch, direction: FloatArray, distance: float) -> Shape: ...

    @abstractmethod
    def boolean(self, a: Shape, b: Shape, op: BooleanOp) -> Shape: ...

    @abstractmethod
    def from_primitive(self, primitive: Primitive, bounds: BBox) -> Shape:
        """Bounded solid/face for a fitted primitive, clipped to ``bounds``."""

    @abstractmethod
    def tessellate(self, shape: Shape, tolerance: float = 0.05) -> pv.PolyData: ...

    @abstractmethod
    def volume(self, shape: Shape) -> float: ...

    @abstractmethod
    def cylinder(self, axis: Axis, radius: float, length: float) -> Shape:
        """Solid cylinder centred on ``axis.origin`` (tool for pin bores / holes)."""

    @abstractmethod
    def fuse_all(self, shapes: Sequence[Shape]) -> Shape: ...

    @abstractmethod
    def is_valid(self, shape: Shape) -> bool:
        """Topological/geometric validity check (BRepCheck)."""

    @abstractmethod
    def topology_counts(self, shape: Shape) -> dict[str, int]:
        """Number of solids / faces / edges (for the property panel)."""

    @abstractmethod
    def import_iges(self, path: Path) -> list[Shape]: ...

    @abstractmethod
    def export_iges(self, shapes: Sequence[Shape], path: Path) -> None: ...

    @abstractmethod
    def import_step(self, path: Path) -> list[Shape]: ...

    @abstractmethod
    def export_step(self, shapes: Sequence[Shape], path: Path) -> None: ...


_kernel: CadKernel | None = None


def is_available() -> bool:
    try:
        get_kernel()
    except KernelUnavailableError:
        return False
    return True


def get_kernel() -> CadKernel:
    """Return the process-wide kernel, importing OCP lazily."""
    global _kernel
    if _kernel is None:
        try:
            from meshrev.core.cad.occ_kernel import OccKernel
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise KernelUnavailableError(
                "未安装 OpenCASCADE 内核：pip install -r requirements-cad.txt"
            ) from exc
        _kernel = OccKernel()
    return _kernel
