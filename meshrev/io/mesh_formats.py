"""STL / OBJ / PLY triangle mesh exchange (via VTK readers and writers)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyvista as pv

from meshrev.core.bodies import Body, MeshBody
from meshrev.core.mesh.processing import ensure_triangles, prepare_mesh
from meshrev.core.types import Units


class MeshReader:
    name = "三角网格"

    def __init__(self, extensions: tuple[str, ...], name: str) -> None:
        self.extensions = extensions
        self.name = name

    def read(self, path: Path) -> list[Body]:
        data = pv.read(str(path))
        if isinstance(data, pv.MultiBlock):
            data = data.combine()
        mesh = prepare_mesh(data)
        if mesh.n_cells == 0:
            raise ValueError(f"文件中没有三角面: {path}")
        return [MeshBody(mesh, path.stem, source_path=path, units=Units.MM)]


class MeshWriter:
    """Writes all given bodies merged into one mesh file (CAD bodies are tessellated)."""

    def __init__(self, extensions: tuple[str, ...], name: str) -> None:
        self.extensions = extensions
        self.name = name

    def write(self, bodies: Sequence[Body], path: Path, **options: Any) -> None:
        meshes = [ensure_triangles(b.to_polydata()) for b in bodies]
        meshes = [m for m in meshes if m.n_cells]
        if not meshes:
            raise ValueError("所选实体不包含三角面")
        merged = meshes[0] if len(meshes) == 1 else pv.merge(meshes)
        merged = pv.PolyData(merged.points, faces=merged.faces)  # geometry only
        binary = bool(options.get("binary", True))
        if path.suffix.lower() == ".obj":
            merged.save(str(path))
        else:
            merged.save(str(path), binary=binary)


STL_READER = MeshReader((".stl",), "STL 网格")
OBJ_READER = MeshReader((".obj",), "OBJ 网格")
PLY_READER = MeshReader((".ply",), "PLY 网格")
STL_WRITER = MeshWriter((".stl",), "STL 网格")
OBJ_WRITER = MeshWriter((".obj",), "OBJ 网格")
PLY_WRITER = MeshWriter((".ply",), "PLY 网格")
