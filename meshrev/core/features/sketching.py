"""Mesh sketch feature: plane section of a mesh + 2D line/arc/circle fitting."""

from __future__ import annotations

import math

import numpy as np

from meshrev.core.bodies import (
    Body,
    DatumAxisBody,
    DatumPlaneBody,
    MeshBody,
    SectionBody,
    SketchBody,
)
from meshrev.core.features.base import (
    Feature,
    FeatureContext,
    FeatureInputError,
    ParamSpec,
    register_feature,
)
from meshrev.core.section.slicer import slice_mesh
from meshrev.core.sketch_fit import SketchFitOptions, fit_section
from meshrev.core.types import Axis, BBox, Plane

STANDARD_PLANES = {
    # normal, sketch x axis
    "xy": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    "yz": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "zx": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
}


def plane_through_axis(axis: Axis, angle_deg: float = 0.0) -> Plane:
    """Plane containing ``axis`` (sketch x = axis direction), rotated by ``angle_deg``
    around it - the natural sketch plane for revolved parts (piston profile)."""
    d = axis.direction
    ref = np.eye(3)[int(np.argmin(np.abs(d)))]  # world axis least aligned with d
    n0 = np.cross(d, ref)
    n0 /= np.linalg.norm(n0)
    m0 = np.cross(d, n0)
    a = math.radians(angle_deg)
    normal = math.cos(a) * n0 + math.sin(a) * m0
    return Plane(axis.origin, normal, x_axis=d)


def resolve_plane(
    mode: str, mesh_bounds: BBox, datum: Body | None, offset: float, angle_deg: float
) -> Plane:
    if mode == "datum":
        if isinstance(datum, DatumPlaneBody):
            base = datum.plane
        elif isinstance(datum, DatumAxisBody):
            base = plane_through_axis(datum.axis, angle_deg)
        else:
            raise FeatureInputError("网格草图需要一个基准平面或基准轴")
    elif mode in STANDARD_PLANES:
        normal, x_axis = STANDARD_PLANES[mode]
        base = Plane(mesh_bounds.center, normal, x_axis=x_axis)
    else:
        raise FeatureInputError(f"未知的草图平面: {mode}")
    return base.offset(offset)


@register_feature
class MeshSketchFeature(Feature):
    """Design X style mesh sketch.

    Inputs: ``[mesh_id]`` or ``[mesh_id, datum_id]``. Outputs the raw section
    (:class:`SectionBody`) and the fitted sketch (:class:`SketchBody`).
    """

    type_name = "MeshSketch"
    label = "网格草图"
    params_spec = (
        ParamSpec(
            "plane",
            "草图平面",
            "choice",
            "xy",
            choices=(
                ("xy", "XY 平面"),
                ("yz", "YZ 平面"),
                ("zx", "ZX 平面"),
                ("datum", "所选基准面 / 基准轴"),
            ),
        ),
        ParamSpec("offset", "偏移", "float", 0.0, -1e4, 1e4, 0.5, 3, suffix=" mm"),
        ParamSpec("angle_deg", "绕基准轴旋转", "float", 0.0, -360.0, 360.0, 5.0, 1, suffix=" °"),
        ParamSpec(
            "tolerance",
            "拟合公差",
            "float",
            0.0,
            0.0,
            10.0,
            0.01,
            4,
            suffix=" mm",
            tooltip="0 = 自动（4σ 噪声与截面尺寸的 0.1% 取大）",
        ),
        ParamSpec("corner_angle_deg", "尖角阈值", "float", 35.0, 5.0, 170.0, 5.0, 1, suffix=" °"),
        ParamSpec("snap_angle_deg", "水平/垂直吸附", "float", 0.5, 0.0, 5.0, 0.1, 2, suffix=" °"),
        ParamSpec(
            "hole_circularity",
            "孔圆度判定",
            "float",
            0.03,
            0.0,
            0.2,
            0.01,
            3,
            tooltip="闭合轮廓 RMS/半径 小于该值时识别为整圆（螺栓孔、销孔）",
        ),
    )

    def execute(self, ctx: FeatureContext) -> list[Body]:
        mesh = ctx.body(self.inputs[0], MeshBody)
        datum = ctx.body(self.inputs[1]) if len(self.inputs) > 1 else None
        p = self.params
        bounds = BBox.from_bounds(mesh.polydata.bounds)
        plane = resolve_plane(p["plane"], bounds, datum, float(p["offset"]), float(p["angle_deg"]))
        ctx.report(0.1, "网格截面")
        curve = slice_mesh(mesh.polydata, plane)
        if curve.is_empty:
            raise FeatureInputError("草图平面与网格没有交线")
        ctx.report(0.4, "拟合直线/圆弧")
        options = SketchFitOptions(
            tolerance=float(p["tolerance"]) or None,
            corner_angle_deg=float(p["corner_angle_deg"]),
            snap_angle_deg=float(p["snap_angle_deg"]),
            hole_circularity=float(p["hole_circularity"]),
        )
        result = fit_section(curve, options)
        return [
            SectionBody(curve, f"{self.name} 截面"),
            SketchBody(result, self.name),
        ]
