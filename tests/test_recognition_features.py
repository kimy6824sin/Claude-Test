"""Recognition features inside the parametric history."""

import numpy as np
import pytest

from meshrev.core.bodies import DatumAxisBody, DatumPlaneBody, MeshBody, RegionSetBody
from meshrev.core.document import Document
from meshrev.core.features import (
    AutoSegmentFeature,
    DatumAxisFeature,
    DatumPlaneFeature,
    FeatureState,
    ImportFeature,
    PrimitiveDetectFeature,
)
from meshrev.core.primitives import PrimitiveType
from meshrev.core.samples import PistonSpec


@pytest.fixture(scope="module")
def segmented(demo_piston):
    doc = Document()
    source = ImportFeature("piston.stl", bodies=[MeshBody(demo_piston, "piston")])
    doc.history.append(source)
    segment = AutoSegmentFeature(inputs=[source.output_ids[0]])
    doc.history.append(segment)
    return doc, source, segment


def _bore_regions(regions: RegionSetBody) -> list[int]:
    spec = PistonSpec()
    return [
        r.id
        for r in regions.segmentation.regions_of_type(PrimitiveType.CYLINDER)
        if abs(r.primitive.radius - spec.pin_radius) < 0.05
    ]


def test_auto_segment_feature_outputs_region_set(segmented):
    doc, source, segment = segmented
    assert segment.state is FeatureState.OK
    regions = doc.get(segment.output_ids[0])
    assert isinstance(regions, RegionSetBody)
    assert regions.mesh_id == source.output_ids[0]
    assert len(regions.segmentation.labels) == regions.polydata.n_cells
    groups = regions.region_groups()
    assert [g[0] for g in groups][:2] == ["plane", "cylinder"]
    face = int(regions.faces_of_regions([3])[0])
    assert regions.region_of_face(face) == 3


def test_datum_axis_through_both_pin_bosses(segmented):
    doc, _, segment = segmented
    regions = doc.get(segment.output_ids[0])
    bores = _bore_regions(regions)
    assert len(bores) == 2
    feature = DatumAxisFeature(
        inputs=[regions.id], params={"region_ids": tuple(bores)}, name="销孔轴线 1"
    )
    doc.history.append(feature)
    assert feature.state is FeatureState.OK, feature.error
    axis = doc.get(feature.output_ids[0])
    assert isinstance(axis, DatumAxisBody) and axis.tag == "A1"
    spec = PistonSpec()
    angle = np.degrees(np.arccos(abs(axis.axis.direction @ (1, 0, 0))))
    assert angle < 0.01
    assert axis.axis.distance([[0, 0, spec.pin_height]])[0] < 0.005
    assert axis.radius == pytest.approx(spec.pin_radius, abs=0.005)
    assert axis.concave is True
    assert axis.length > 2 * (spec.radius - spec.boss_half_distance)  # spans both bosses
    doc.history.remove(feature.id)


def test_datum_plane_on_crown(segmented):
    doc, _, segment = segmented
    regions = doc.get(segment.output_ids[0])
    crown = next(
        r.id
        for r in regions.segmentation.regions
        if r.type is PrimitiveType.PLANE
        and r.primitive.normal[2] > 0.999
        and abs(r.primitive.d + PistonSpec().height) < 0.01
    )
    feature = DatumPlaneFeature(inputs=[regions.id], params={"region_ids": (crown,)})
    doc.history.append(feature)
    plane = doc.get(feature.output_ids[0])
    assert isinstance(plane, DatumPlaneBody)
    a, b, c, d = plane.plane.equation
    assert (a, b, c, d) == pytest.approx((0, 0, 1, -PistonSpec().height), abs=2e-3)
    assert "平面方程" in plane.info()
    doc.history.remove(feature.id)


def test_invalid_region_ids_mark_feature_as_failed(segmented):
    doc, _, segment = segmented
    feature = DatumAxisFeature(inputs=[segment.output_ids[0]], params={"region_ids": (9999,)})
    doc.history.append(feature)
    assert feature.state is FeatureState.ERROR and "9999" in feature.error
    doc.history.remove(feature.id)


def test_regeneration_after_parameter_change(demo_piston):
    doc = Document()
    source = ImportFeature("piston.stl", bodies=[MeshBody(demo_piston, "piston")])
    doc.history.append(source)
    segment = AutoSegmentFeature(
        inputs=[source.output_ids[0]], params={"split_freeform": False, "detect_spheres": False}
    )
    doc.history.append(segment)
    regions = doc.get(segment.output_ids[0])
    assert regions.segmentation.type_counts()[PrimitiveType.SPHERE] == 0
    doc.history.update_params(segment.id, {"detect_spheres": True})
    regions = doc.get(segment.output_ids[0])
    assert regions.segmentation.type_counts()[PrimitiveType.SPHERE] >= 1


def test_ransac_feature_creates_region_set_and_axes(demo_piston):
    doc = Document()
    source = ImportFeature("piston.stl", bodies=[MeshBody(demo_piston, "piston")])
    doc.history.append(source)
    feature = PrimitiveDetectFeature(
        inputs=[source.output_ids[0]], params={"max_primitives": 8, "datums": "all"}
    )
    doc.history.append(feature)
    assert feature.state is FeatureState.OK, feature.error
    outputs = [doc.get(i) for i in feature.output_ids]
    assert isinstance(outputs[0], RegionSetBody)
    axes = [b for b in outputs if isinstance(b, DatumAxisBody)]
    planes = [b for b in outputs if isinstance(b, DatumPlaneBody)]
    assert axes and planes
    assert any(abs(a.radius - PistonSpec().radius) < 0.02 for a in axes)  # the skirt
    labels = outputs[0].segmentation.labels
    assert labels.min() == 0 and len(labels) == demo_piston.n_cells
