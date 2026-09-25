"""STEP / IGES exchange through the optional CAD kernel."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from meshrev.core.bodies import Body, CadBody
from meshrev.core.cad.kernel import get_kernel


class StepReader:
    name = "STEP 实体"
    extensions = (".step", ".stp")

    def read(self, path: Path) -> list[Body]:
        kernel = get_kernel()
        shapes = kernel.import_step(path)
        if len(shapes) == 1:
            return [CadBody(shapes[0], path.stem, kernel=kernel)]
        return [
            CadBody(shape, f"{path.stem}_{i + 1}", kernel=kernel) for i, shape in enumerate(shapes)
        ]


class StepWriter:
    name = "STEP 实体"
    extensions = (".step", ".stp")

    def write(self, bodies: Sequence[Body], path: Path, **options: Any) -> None:
        cad = [b for b in bodies if isinstance(b, CadBody)]
        if not cad:
            raise ValueError("STEP 只能导出 CAD 实体；网格请导出为 STL/OBJ/PLY")
        get_kernel().export_step([b.shape for b in cad], path)


class IgesReader:
    name = "IGES 实体"
    extensions = (".iges", ".igs")

    def read(self, path: Path) -> list[Body]:
        kernel = get_kernel()
        shapes = kernel.import_iges(path)
        return [
            CadBody(shape, path.stem if len(shapes) == 1 else f"{path.stem}_{i + 1}", kernel=kernel)
            for i, shape in enumerate(shapes)
        ]


class IgesWriter:
    name = "IGES 实体"
    extensions = (".iges", ".igs")

    def write(self, bodies: Sequence[Body], path: Path, **options: Any) -> None:
        cad = [b for b in bodies if isinstance(b, CadBody)]
        if not cad:
            raise ValueError("IGES 只能导出 CAD 实体；网格请导出为 STL/OBJ/PLY")
        get_kernel().export_iges([b.shape for b in cad], path)
