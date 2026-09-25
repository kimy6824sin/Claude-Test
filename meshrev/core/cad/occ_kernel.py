"""OpenCASCADE implementation of :class:`CadKernel` (via the ``OCP`` bindings).

Importing this module raises ``ImportError`` when OCP is missing; use
:func:`meshrev.core.cad.kernel.get_kernel` instead of importing it directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
)
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakePrism,
    BRepPrimAPI_MakeRevol,
    BRepPrimAPI_MakeSphere,
)
from OCP.GC import GC_MakeArcOfCircle
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pln, gp_Pnt, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Compound

from meshrev.core.cad.kernel import BooleanOp, CadKernel, Shape
from meshrev.core.section.sketch import Arc2D, Line2D, Sketch
from meshrev.core.types import Axis, BBox, FloatArray, as_vec3

if TYPE_CHECKING:
    from meshrev.core.primitives import Primitive


def _static(cls: Any, name: str) -> Any:
    """OCP < 8 suffixes static methods with ``_s``; OCP 8 dropped it for some classes."""
    return getattr(cls, f"{name}_s", None) or getattr(cls, name)


def _pnt(p: FloatArray) -> gp_Pnt:
    return gp_Pnt(float(p[0]), float(p[1]), float(p[2]))


def _dir(d: FloatArray) -> gp_Dir:
    return gp_Dir(float(d[0]), float(d[1]), float(d[2]))


class OccKernel(CadKernel):
    name = "OpenCASCADE"

    # -- modelling --------------------------------------------------------------------
    def _sketch_face(self, sketch: Sketch) -> Any:
        if not sketch.is_closed(tol=1e-6):
            raise ValueError("sketch profile must be closed")
        wire = BRepBuilderAPI_MakeWire()
        for entity in sketch.entities:
            if isinstance(entity, Line2D):
                a, b = sketch.plane.to_world([entity.start, entity.end])
                edge = BRepBuilderAPI_MakeEdge(_pnt(a), _pnt(b)).Edge()
            elif isinstance(entity, Arc2D):
                a, m, b = sketch.plane.to_world([entity.start, entity.mid, entity.end])
                curve = GC_MakeArcOfCircle(_pnt(a), _pnt(m), _pnt(b)).Value()
                edge = BRepBuilderAPI_MakeEdge(curve).Edge()
            else:  # pragma: no cover - exhaustive over SketchEntity
                raise TypeError(f"unsupported sketch entity {entity!r}")
            wire.Add(edge)
        return BRepBuilderAPI_MakeFace(wire.Wire()).Face()

    def revolve(self, sketch: Sketch, axis: Axis, angle_deg: float = 360.0) -> Shape:
        face = self._sketch_face(sketch)
        ax = gp_Ax1(_pnt(axis.origin), _dir(axis.direction))
        return BRepPrimAPI_MakeRevol(face, ax, np.radians(angle_deg)).Shape()

    def extrude(self, sketch: Sketch, direction: FloatArray, distance: float) -> Shape:
        d = as_vec3(direction)
        d = d / np.linalg.norm(d) * distance
        face = self._sketch_face(sketch)
        return BRepPrimAPI_MakePrism(face, gp_Vec(*map(float, d))).Shape()

    def boolean(self, a: Shape, b: Shape, op: BooleanOp) -> Shape:
        algo = {
            BooleanOp.UNION: BRepAlgoAPI_Fuse,
            BooleanOp.CUT: BRepAlgoAPI_Cut,
            BooleanOp.INTERSECT: BRepAlgoAPI_Common,
        }[BooleanOp(op)]
        result = algo(a, b)
        if not result.IsDone():
            raise RuntimeError(f"boolean {op} failed")
        return result.Shape()

    def from_primitive(self, primitive: Primitive, bounds: BBox) -> Shape:
        from meshrev.core.primitives import CylinderPrimitive, PlanePrimitive, SpherePrimitive

        if isinstance(primitive, CylinderPrimitive):
            base = primitive.axis_point - 0.5 * primitive.length * primitive.axis_direction
            ax2 = gp_Ax2(_pnt(base), _dir(primitive.axis_direction))
            return BRepPrimAPI_MakeCylinder(ax2, primitive.radius, primitive.length).Shape()
        if isinstance(primitive, SpherePrimitive):
            return BRepPrimAPI_MakeSphere(_pnt(primitive.center), primitive.radius).Shape()
        if isinstance(primitive, PlanePrimitive):
            plane = primitive.plane
            half = 0.5 * bounds.diagonal
            pln = gp_Pln(_pnt(plane.origin), _dir(plane.normal))
            return BRepBuilderAPI_MakeFace(pln, -half, half, -half, half).Face()
        raise TypeError(f"no solid representation for {type(primitive).__name__}")

    # -- queries ----------------------------------------------------------------------
    def tessellate(self, shape: Shape, tolerance: float = 0.05) -> pv.PolyData:
        BRepMesh_IncrementalMesh(shape, tolerance, False, 0.3, True)
        triangulation = _static(BRep_Tool, "Triangulation")
        as_face = _static(TopoDS, "Face")
        points: list[np.ndarray] = []
        faces: list[np.ndarray] = []
        offset = 0
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = as_face(explorer.Current())
            explorer.Next()
            location = TopLoc_Location()
            tri = triangulation(face, location)
            if tri is None:
                continue
            trsf = location.Transformation()
            nodes = np.array(
                [tri.Node(i).Transformed(trsf).Coord() for i in range(1, tri.NbNodes() + 1)]
            )
            triangles = (
                np.array(
                    [tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)], dtype=np.int64
                )
                - 1
            )
            if face.Orientation() == TopAbs_REVERSED:
                triangles = triangles[:, ::-1]
            points.append(nodes)
            faces.append(triangles + offset)
            offset += len(nodes)
        if not points:
            return pv.PolyData()
        tris = np.vstack(faces)
        cells = np.column_stack([np.full(len(tris), 3), tris]).ravel()
        return pv.PolyData(np.vstack(points), faces=cells).clean(tolerance=1e-9)

    def volume(self, shape: Shape) -> float:
        props = GProp_GProps()
        _static(BRepGProp, "VolumeProperties")(shape, props)
        return float(props.Mass())

    # -- exchange ---------------------------------------------------------------------
    def import_step(self, path: Path) -> list[Shape]:
        reader = STEPControl_Reader()
        if reader.ReadFile(str(path)) != IFSelect_RetDone:
            raise OSError(f"无法读取 STEP 文件: {path}")
        reader.TransferRoots()
        return [reader.Shape(i) for i in range(1, reader.NbShapes() + 1)]

    def export_step(self, shapes: Sequence[Shape], path: Path) -> None:
        if not shapes:
            raise ValueError("nothing to export")
        if len(shapes) == 1:
            shape = shapes[0]
        else:
            builder = BRep_Builder()
            shape = TopoDS_Compound()
            builder.MakeCompound(shape)
            for s in shapes:
                builder.Add(shape, s)
        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)
        if writer.Write(str(path)) != IFSelect_RetDone:
            raise OSError(f"无法写入 STEP 文件: {path}")
