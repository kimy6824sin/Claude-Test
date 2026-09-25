"""Solid modelling features producing B-Rep bodies (OpenCASCADE via ``CadKernel``).

Every feature is parametric: editing a parameter (extrude height, pin bore
radius, ...) regenerates the feature and everything downstream, e.g.

    MeshSketch --> Revolve --+
    DatumAxis  --> Cylinder -+-> Boolean(cut) --> STEP / IGES
"""

from __future__ import annotations

from meshrev.core.bodies import Body, CadBody, DatumAxisBody, SketchBody
from meshrev.core.cad.kernel import BooleanOp, get_kernel
from meshrev.core.features.base import (
    Feature,
    FeatureContext,
    FeatureInputError,
    ParamSpec,
    register_feature,
)
from meshrev.core.sketch_fit import SketchFitResult, half_profile
from meshrev.core.types import Axis


def _regions(result: SketchFitResult, region: int):
    sketches = result.to_sketches()
    if not sketches:
        raise FeatureInputError("草图中没有闭合轮廓，无法建模")
    if region >= 0:
        if region >= len(sketches):
            raise FeatureInputError(f"草图只有 {len(sketches)} 个区域")
        return [sketches[region]]
    return sketches


@register_feature
class ExtrudeFeature(Feature):
    """Linear extrusion of a sketch (outer loops with their holes) along its normal."""

    type_name = "Extrude"
    label = "拉伸"
    params_spec = (
        ParamSpec("distance", "拉伸高度", "float", 10.0, 1e-4, 1e5, 0.5, 3, suffix=" mm"),
        ParamSpec(
            "direction",
            "方向",
            "choice",
            "normal",
            choices=(("normal", "沿法向"), ("reverse", "反向"), ("symmetric", "对称")),
        ),
        ParamSpec(
            "region",
            "轮廓区域",
            "int",
            -1,
            -1,
            1000,
            tooltip="-1 = 全部闭合区域；否则只拉伸第 N 个区域",
        ),
    )

    def execute(self, ctx: FeatureContext) -> list[Body]:
        sketch_body = ctx.body(self.inputs[0], SketchBody)
        kernel = get_kernel()
        distance = float(self.params["distance"])
        direction = self.params["direction"]
        shapes = []
        for sketch in _regions(sketch_body.result, int(self.params["region"])):
            normal = sketch.plane.normal
            if direction == "reverse":
                normal = -normal
            elif direction == "symmetric":
                sketch.plane = sketch.plane.offset(-0.5 * distance)
            shapes.append(kernel.extrude(sketch, normal, distance))
        return [CadBody(kernel.fuse_all(shapes), self.name, kernel=kernel)]


@register_feature
class RevolveFeature(Feature):
    """Revolve the half section of a sketch around the sketch's ``u`` axis.

    Made for sketches on a plane through a datum axis (mesh sketch in datum-axis
    mode): the ``u`` axis *is* the rotation axis. The full section is split at
    the axis and the chosen half is refitted (see ``sketch_fit.half_profile``).
    """

    type_name = "Revolve"
    label = "旋转"
    params_spec = (
        ParamSpec("angle_deg", "旋转角度", "float", 360.0, 0.1, 360.0, 15.0, 2, suffix=" °"),
        ParamSpec(
            "side",
            "半截面",
            "choice",
            "positive",
            choices=(("positive", "v > 0 一侧"), ("negative", "v < 0 一侧")),
        ),
    )

    def execute(self, ctx: FeatureContext) -> list[Body]:
        sketch_body = ctx.body(self.inputs[0], SketchBody)
        result = sketch_body.result
        side = 1.0 if self.params["side"] == "positive" else -1.0
        half = half_profile(result, side)
        sketches = half.to_sketches()
        if not sketches:
            raise FeatureInputError("草图在旋转轴该侧没有闭合区域")
        plane = result.plane
        u_axis, _ = plane.basis()
        axis = Axis(plane.origin, u_axis)
        kernel = get_kernel()
        angle = float(self.params["angle_deg"])
        solid = kernel.fuse_all([kernel.revolve(s, axis, angle) for s in sketches])
        return [
            CadBody(solid, self.name, kernel=kernel),
            SketchBody(half, f"{self.name} 半截面"),
        ]


@register_feature
class CylinderFeature(Feature):
    """Cylindrical tool body on a datum axis (pin bore, bolt hole, boss)."""

    type_name = "Cylinder"
    label = "圆柱"
    params_spec = (
        ParamSpec(
            "radius",
            "半径",
            "float",
            0.0,
            0.0,
            1e4,
            0.01,
            4,
            suffix=" mm",
            tooltip="0 = 使用基准轴的拟合半径",
        ),
        ParamSpec(
            "length",
            "长度",
            "float",
            0.0,
            0.0,
            1e5,
            1.0,
            3,
            suffix=" mm",
            tooltip="0 = 贯穿（基准轴长度的 1.5 倍）",
        ),
        ParamSpec("offset", "沿轴偏移", "float", 0.0, -1e4, 1e4, 0.5, 3, suffix=" mm"),
    )

    def execute(self, ctx: FeatureContext) -> list[Body]:
        datum = ctx.body(self.inputs[0], DatumAxisBody)
        radius = float(self.params["radius"]) or float(datum.radius or 0.0)
        if radius <= 0:
            raise FeatureInputError("请指定圆柱半径（基准轴没有拟合半径）")
        length = float(self.params["length"])
        if length <= 0:  # through all: longer than anything in the document
            bounds = ctx.document.bounds(visible_only=False)
            length = 2.0 * max(bounds.diagonal if bounds is not None else 0.0, datum.length, 1.0)
        origin = datum.axis.origin + float(self.params["offset"]) * datum.axis.direction
        kernel = get_kernel()
        shape = kernel.cylinder(Axis(origin, datum.axis.direction), radius, length)
        body = CadBody(shape, self.name, kernel=kernel, color=(0.55, 0.75, 0.95))
        return [body]


@register_feature
class BooleanFeature(Feature):
    """Union / cut / intersection of two B-Rep bodies (inputs: target, tool)."""

    type_name = "Boolean"
    label = "布尔运算"
    params_spec = (
        ParamSpec(
            "operation",
            "运算",
            "choice",
            "cut",
            choices=(("cut", "求差（目标 − 工具）"), ("union", "求并"), ("intersect", "求交")),
        ),
    )

    def execute(self, ctx: FeatureContext) -> list[Body]:
        target = ctx.body(self.inputs[0], CadBody)
        tool = ctx.body(self.inputs[1], CadBody)
        kernel = get_kernel()
        op = BooleanOp(self.params["operation"])
        shape = kernel.boolean(target.shape, tool.shape, op)
        if kernel.volume(shape) <= 0 and op is not BooleanOp.CUT:
            raise FeatureInputError("布尔运算结果为空（两实体不相交？）")
        body = CadBody(shape, self.name, kernel=kernel, color=target.color)
        body.consumes = [target.id, tool.id]
        return [body]


@register_feature
class AccuracyAnalysisFeature(Feature):
    """Design X style accuracy analyzer: signed deviation between a scan mesh and a
    CAD body, shown as a banded blue-green-red heat map with statistics."""

    type_name = "AccuracyAnalysis"
    label = "精度分析"
    params_spec = (
        ParamSpec("tolerance", "公差 ±", "float", 0.1, 1e-4, 100.0, 0.01, 4, suffix=" mm"),
        ParamSpec(
            "max_range",
            "最大偏差（显示/统计范围）",
            "float",
            1.0,
            1e-3,
            1e3,
            0.1,
            3,
            suffix=" mm",
            tooltip="超出此范围的点显示为灰色，不计入统计",
        ),
        ParamSpec(
            "direction",
            "方向",
            "choice",
            "mesh_to_cad",
            choices=(("mesh_to_cad", "网格 → CAD"), ("cad_to_mesh", "CAD → 网格")),
        ),
        ParamSpec("bands", "色带数", "int", 15, 3, 41),
    )

    def execute(self, ctx: FeatureContext) -> list[Body]:
        from meshrev.core.bodies import DeviationBody, MeshBody
        from meshrev.core.deviation import compute_deviation

        mesh = ctx.body(self.inputs[0], MeshBody)
        cad = ctx.body(self.inputs[1], CadBody)
        p = self.params
        ctx.report(0.1, "计算距离场")
        result = compute_deviation(
            mesh.polydata,
            cad,
            tolerance=float(p["tolerance"]),
            max_range=float(p["max_range"]),
            direction=p["direction"],
        )
        if p["direction"] == "mesh_to_cad":
            geometry = mesh.polydata
        else:
            import pyvista as pv

            geometry = pv.PolyData(result.points)
        body = DeviationBody(geometry, result, self.name, bands=int(p["bands"]))
        body.consumes = [mesh.id, cad.id]
        return [body]
