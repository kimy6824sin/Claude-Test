"""Primitive recognition features: auto segmentation, RANSAC extraction, datums."""

from __future__ import annotations

import numpy as np

from meshrev.core.bodies import (
    Body,
    DatumAxisBody,
    DatumPlaneBody,
    MeshBody,
    RegionSetBody,
)
from meshrev.core.features.base import (
    Feature,
    FeatureContext,
    FeatureInputError,
    ParamSpec,
    register_feature,
)
from meshrev.core.mesh.processing import ensure_triangles
from meshrev.core.mesh.topology import MeshGeometry
from meshrev.core.primitives import (
    CylinderPrimitive,
    PlanePrimitive,
    PrimitiveType,
    RansacOptions,
    Region,
    SegmentationOptions,
    SegmentationResult,
    auto_segment,
    detect_mesh_primitives,
    fit_mesh_faces,
)
from meshrev.core.types import Axis, BBox, Plane


def _source_mesh(ctx: FeatureContext, body_id: str) -> tuple[MeshBody, object]:
    mesh = ctx.body(body_id, MeshBody)
    return mesh, ensure_triangles(mesh.polydata)


@register_feature
class AutoSegmentFeature(Feature):
    """Design X style auto segmentation of a mesh into primitive regions."""

    type_name = "AutoSegment"
    label = "自动分割"
    params_spec = (
        ParamSpec(
            "sharp_angle_deg",
            "锐边角度",
            "float",
            30.0,
            5.0,
            89.0,
            1.0,
            1,
            suffix=" °",
            tooltip="二面角大于该值的边总是区域边界",
        ),
        ParamSpec(
            "curvature_tolerance",
            "曲率容差",
            "float",
            0.3,
            0.05,
            2.0,
            0.05,
            2,
            tooltip="相邻面片曲率允许的相对变化",
        ),
        ParamSpec(
            "fit_tolerance",
            "拟合公差",
            "float",
            0.0,
            0.0,
            10.0,
            0.005,
            4,
            suffix=" mm",
            tooltip="基元拟合 RMS 公差，0 = 自动",
        ),
        ParamSpec("min_region_faces", "最小区域面片数", "int", 30, 3, 1_000_000),
        ParamSpec("detect_spheres", "识别球面", "bool", True),
        ParamSpec("split_freeform", "RANSAC 拆分自由曲面", "bool", True),
    )

    def execute(self, ctx: FeatureContext) -> list[Body]:
        mesh, polydata = _source_mesh(ctx, self.inputs[0])
        p = self.params
        options = SegmentationOptions(
            sharp_angle_deg=float(p["sharp_angle_deg"]),
            curvature_tolerance=float(p["curvature_tolerance"]),
            fit_tolerance=float(p["fit_tolerance"]) or None,
            min_region_faces=int(p["min_region_faces"]),
            detect_spheres=bool(p["detect_spheres"]),
            split_freeform=bool(p["split_freeform"]),
        )
        segmentation = auto_segment(MeshGeometry.from_polydata(polydata), options, ctx.report)
        return [RegionSetBody(polydata, segmentation, f"{mesh.name} 区域", mesh_id=mesh.id)]


@register_feature
class PrimitiveDetectFeature(Feature):
    """RANSAC extraction of planes and/or cylinders; cylinders become datum axes."""

    type_name = "PrimitiveDetect"
    label = "RANSAC 基元提取"
    params_spec = (
        ParamSpec(
            "types",
            "基元类型",
            "choice",
            "both",
            choices=(("both", "平面 + 圆柱"), ("plane", "仅平面"), ("cylinder", "仅圆柱")),
        ),
        ParamSpec(
            "distance_threshold",
            "距离阈值",
            "float",
            0.0,
            0.0,
            10.0,
            0.01,
            4,
            suffix=" mm",
            tooltip="0 = 包围盒对角线的 0.4%",
        ),
        ParamSpec(
            "normal_threshold_deg", "法向阈值", "float", 20.0, 1.0, 60.0, 1.0, 1, suffix=" °"
        ),
        ParamSpec("min_support", "最小支持面片数", "int", 200, 10, 10_000_000),
        ParamSpec("max_primitives", "最多基元数", "int", 12, 1, 200),
        ParamSpec("merge_coaxial", "合并同轴圆柱", "bool", True),
        ParamSpec(
            "datums",
            "生成基准",
            "choice",
            "axes",
            choices=(("axes", "圆柱轴线"), ("all", "轴线 + 平面"), ("none", "不生成")),
        ),
    )

    def execute(self, ctx: FeatureContext) -> list[Body]:
        mesh, polydata = _source_mesh(ctx, self.inputs[0])
        p = self.params
        types = {
            "both": (PrimitiveType.PLANE, PrimitiveType.CYLINDER),
            "plane": (PrimitiveType.PLANE,),
            "cylinder": (PrimitiveType.CYLINDER,),
        }[p["types"]]
        options = RansacOptions(
            distance_threshold=float(p["distance_threshold"]) or None,
            normal_threshold_deg=float(p["normal_threshold_deg"]),
            min_support=int(p["min_support"]),
        )
        geom = MeshGeometry.from_polydata(polydata)
        ctx.report(0.1, "RANSAC 检测")
        found = detect_mesh_primitives(
            geom, types, options, int(p["max_primitives"]), bool(p["merge_coaxial"])
        )
        labels = np.full(geom.n_faces, len(found), dtype=np.int64)
        regions = []
        for i, item in enumerate(found):
            labels[item.face_ids] = i
            area = float(geom.face_areas[item.face_ids].sum())
            regions.append(Region(i, item.face_ids, item.type, item.fit, area))
        rest = np.flatnonzero(labels == len(found))
        if len(rest):
            regions.append(
                Region(
                    len(found),
                    rest,
                    PrimitiveType.FREEFORM,
                    None,
                    float(geom.face_areas[rest].sum()),
                )
            )
        tolerance = options.distance_threshold or 0.004 * geom.diagonal
        result = SegmentationResult(labels, regions, tolerance)
        outputs: list[Body] = [
            RegionSetBody(polydata, result, f"{mesh.name} RANSAC", mesh_id=mesh.id)
        ]
        for i, item in enumerate(found):
            prim = item.primitive
            if isinstance(prim, CylinderPrimitive) and p["datums"] in ("axes", "all"):
                name = f"轴线 r={prim.radius:.3f} R{i}"
                outputs.append(_axis_body(prim, name, item.fit.rms))
            elif isinstance(prim, PlanePrimitive) and p["datums"] == "all":
                outputs.append(_plane_body(geom, item.face_ids, prim, f"平面 R{i}", item.fit.rms))
        return outputs


def _axis_body(cylinder: CylinderPrimitive, name: str, rms: float) -> DatumAxisBody:
    return DatumAxisBody(
        Axis(cylinder.axis_point, cylinder.axis_direction),
        cylinder.length,
        name,
        radius=cylinder.radius,
        rms=rms,
        concave=cylinder.concave,
    )


def _plane_body(
    geom: MeshGeometry, faces, plane: PlanePrimitive, name: str, rms: float
) -> DatumPlaneBody:
    vertices = geom.points[geom.face_vertices(faces)]
    size = max(BBox.from_points(vertices).diagonal * 1.1, 1e-3)
    return DatumPlaneBody(Plane(plane.center, plane.normal), size, name, rms=rms)


class _RegionDatumFeature(Feature):
    """Fits a primitive to the union of selected regions of a region set."""

    kind: PrimitiveType

    def _fit(self, ctx: FeatureContext):
        regions_body = ctx.body(self.inputs[0], RegionSetBody)
        ids = [int(i) for i in self.params["region_ids"]]
        n = len(regions_body.segmentation.regions)
        if not ids or any(i < 0 or i >= n for i in ids):
            raise FeatureInputError(f"区域编号无效: {ids}（共 {n} 个区域）")
        faces = regions_body.faces_of_regions(ids)
        geom = regions_body.geometry
        # robust (soft-L1) fit: bevelled/rounded border faces must not bias the datum
        tolerance = regions_body.segmentation.fit_tolerance
        return geom, faces, fit_mesh_faces(geom, faces, self.kind, robust_scale=tolerance)


@register_feature
class DatumAxisFeature(_RegionDatumFeature):
    """Datum axis from a high precision cylinder fit (e.g. both pin-boss bores)."""

    type_name = "DatumAxis"
    label = "基准轴"
    kind = PrimitiveType.CYLINDER
    params_spec = (
        ParamSpec("region_ids", "区域", "ints", (), tooltip="逗号分隔的区域编号，可选多个同轴区域"),
    )

    def execute(self, ctx: FeatureContext) -> list[Body]:
        _geom, _faces, fit = self._fit(ctx)
        return [_axis_body(fit.primitive, self.name, fit.rms)]


@register_feature
class DatumPlaneFeature(_RegionDatumFeature):
    """Datum plane from a least-squares plane fit (e.g. the gasket/crown face)."""

    type_name = "DatumPlane"
    label = "基准平面"
    kind = PrimitiveType.PLANE
    params_spec = (ParamSpec("region_ids", "区域", "ints", (), tooltip="逗号分隔的区域编号"),)

    def execute(self, ctx: FeatureContext) -> list[Body]:
        geom, faces, fit = self._fit(ctx)
        return [_plane_body(geom, faces, fit.primitive, self.name, fit.rms)]
