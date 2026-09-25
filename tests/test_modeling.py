"""B-Rep modelling features: extrude, revolve (half profile), cylinder, boolean, exchange."""

import math

import numpy as np
import pytest

from meshrev import io as mio
from meshrev.core.bodies import CadBody, DatumAxisBody, MeshBody, SketchBody
from meshrev.core.document import Document
from meshrev.core.features import (
    BooleanFeature,
    CylinderFeature,
    ExtrudeFeature,
    FeatureState,
    ImportFeature,
    MeshSketchFeature,
    RevolveFeature,
)
from meshrev.core.samples import PistonSpec, make_capped_cylinder
from meshrev.core.sketch_fit import polygon_area, split_loop_at_axis
from meshrev.core.types import Axis

# -- half profile (no CAD kernel needed) -----------------------------------------------------


def test_split_rectangle_across_axis():
    rect = np.array([[0, -1], [4, -1], [4, 1], [0, 1]], float)
    (upper,) = split_loop_at_axis(rect, 1.0)
    (lower,) = split_loop_at_axis(rect, -1.0)
    assert polygon_area(upper) == pytest.approx(4.0) and polygon_area(lower) == pytest.approx(4.0)
    assert upper[:, 1].min() == pytest.approx(0.0)


def test_split_u_shape_gives_two_regions():
    # a "U" crossing the axis four times: two separate pieces on the upper side
    shape = np.array([[0, -2], [6, -2], [6, 2], [4, 2], [4, -1], [2, -1], [2, 2], [0, 2]], float)
    pieces = split_loop_at_axis(shape, 1.0)
    assert sorted(round(polygon_area(p), 6) for p in pieces) == [4.0, 4.0]


def test_loop_entirely_on_one_side():
    square = np.array([[0, 1], [1, 1], [1, 2], [0, 2]], float)
    assert len(split_loop_at_axis(square, 1.0)) == 1
    assert split_loop_at_axis(square, -1.0) == []


def _document_with(mesh) -> tuple[Document, str]:
    doc = Document()
    source = ImportFeature("part.stl", bodies=[MeshBody(mesh, "part")])
    doc.history.append(source)
    return doc, source.output_ids[0]


def _datum_axis(doc: Document, axis: Axis, radius: float, length: float) -> DatumAxisBody:
    body = DatumAxisBody(axis, length, "axis", radius=radius)
    doc.add_body(body)
    return body


def _add(doc: Document, feature):
    doc.history.append(feature)
    assert feature.state is FeatureState.OK, feature.error
    return [doc.get(i) for i in feature.output_ids]


@pytest.mark.cad
def test_extrude_sketch_with_holes():
    from meshrev.core.cad import get_kernel

    mesh = make_capped_cylinder(radius=10.0, height=20.0, n_theta=128)
    doc, mesh_id = _document_with(mesh)
    _, sketch = _add(doc, MeshSketchFeature(inputs=[mesh_id], params={"plane": "xy"}))
    (solid,) = _add(doc, ExtrudeFeature(inputs=[sketch.id], params={"distance": 5.0}))
    kernel = get_kernel()
    assert isinstance(solid, CadBody) and kernel.is_valid(solid.shape)
    area = abs(polygon_area(sketch.result.loops[0].points))
    assert kernel.volume(solid.shape) == pytest.approx(5.0 * math.pi * 100, rel=2e-3)
    assert kernel.volume(solid.shape) == pytest.approx(5.0 * area, rel=2e-3)
    bounds = solid.bounds()
    assert bounds.min[2] == pytest.approx(0.0, abs=1e-6) and bounds.max[2] == pytest.approx(5.0)
    feature = doc.history.features[-1]
    doc.history.update_params(feature.id, {"direction": "symmetric", "distance": 8.0})
    bounds = doc.get(feature.output_ids[0]).bounds()
    assert (bounds.min[2], bounds.max[2]) == pytest.approx((-4.0, 4.0), abs=1e-6)


@pytest.fixture(scope="module")
def revolved_piston(demo_piston):
    doc, mesh_id = _document_with(demo_piston)
    axis = _datum_axis(doc, Axis((0, 0, 30), (0, 0, 1)), 43.0, 70.0)
    _, sketch = _add(
        doc,
        MeshSketchFeature(inputs=[mesh_id, axis.id], params={"plane": "datum", "angle_deg": 90.0}),
    )
    solid, half = _add(doc, RevolveFeature(inputs=[sketch.id]))
    return doc, solid, half


def analytic_cup_volume(spec: PistonSpec) -> float:
    """Revolved demo piston without bosses: cylinder - grooves - cavity - bowl."""
    outer = math.pi * spec.radius**2 * spec.height
    groove_r = spec.radius - spec.groove_depth
    grooves = sum(
        math.pi * (spec.radius**2 - groove_r**2) * (z1 - z0) for z0, z1 in spec.groove_ranges
    )
    cavity = math.pi * spec.inner_radius**2 * spec.under_crown_height
    h, r = spec.dish_depth, spec.dish_radius
    bowl = math.pi * h * h * (3 * r - h) / 3
    return outer - grooves - cavity - bowl


@pytest.mark.cad
def test_revolve_half_profile_matches_design(revolved_piston):
    from meshrev.core.cad import get_kernel

    _, solid, half = revolved_piston
    kernel = get_kernel()
    assert kernel.is_valid(solid.shape)
    assert kernel.topology_counts(solid.shape)["solids"] == 1
    assert kernel.volume(solid.shape) == pytest.approx(analytic_cup_volume(PistonSpec()), rel=0.01)
    assert isinstance(half, SketchBody)
    # the half profile lies on one side of the axis and is closed along it
    for loop in half.result.loops:
        assert loop.points[:, 1].min() >= -1e-9
    size = solid.bounds().size
    assert size == pytest.approx([86.0, 86.0, 70.0], abs=0.3)


@pytest.mark.cad
def test_pin_bore_boolean_and_parametric_radius(revolved_piston):
    from meshrev.core.cad import get_kernel

    doc, solid, _ = revolved_piston
    spec = PistonSpec()
    kernel = get_kernel()
    pin = _datum_axis(doc, Axis((0, 0, spec.pin_height), (1, 0, 0)), spec.pin_radius, 86.0)
    cylinder_feature = CylinderFeature(inputs=[pin.id])
    (tool,) = _add(doc, cylinder_feature)
    boolean = BooleanFeature(inputs=[solid.id, tool.id], params={"operation": "cut"})
    (result,) = _add(doc, boolean)
    assert result.consumes == [solid.id, tool.id]
    before = kernel.volume(solid.shape)
    removed = before - kernel.volume(result.shape)
    # the bore cuts both skirt walls (inner radius 36, outer 43): ~2 * pi r^2 * 7
    assert removed == pytest.approx(2 * math.pi * spec.pin_radius**2 * 7.0, rel=0.08)
    assert kernel.is_valid(result.shape)
    # edit the pin radius -> cylinder and boolean regenerate
    doc.history.update_params(cylinder_feature.id, {"radius": 12.0})
    result = doc.get(boolean.output_ids[0])
    removed_12 = before - kernel.volume(result.shape)
    assert removed_12 / removed == pytest.approx((12.0 / 11.0) ** 2, rel=0.03)


@pytest.mark.cad
@pytest.mark.parametrize(("operation", "expected"), [("union", 1.75), ("intersect", 0.25)])
def test_boolean_union_and_intersection(operation, expected):
    from meshrev.core.cad import get_kernel

    kernel = get_kernel()
    doc = Document()
    a = CadBody(kernel.cylinder(Axis((0, 0, 0), (0, 0, 1)), 1.0, 4.0), "a", kernel=kernel)
    b = CadBody(kernel.cylinder(Axis((0, 0, 3), (0, 0, 1)), 1.0, 4.0), "b", kernel=kernel)
    doc.add_body(a)
    doc.add_body(b)
    (result,) = _add(doc, BooleanFeature(inputs=[a.id, b.id], params={"operation": operation}))
    assert kernel.volume(result.shape) == pytest.approx(expected * math.pi * 4.0, rel=1e-6)


@pytest.mark.cad
@pytest.mark.parametrize("ext", [".step", ".iges"])
def test_exchange_roundtrip_of_modelled_part(revolved_piston, tmp_path, ext):
    from meshrev.core.cad import get_kernel

    _, solid, _ = revolved_piston
    kernel = get_kernel()
    path = tmp_path / f"piston{ext}"
    mio.save([solid], path)
    (loaded,) = mio.load(path)
    assert isinstance(loaded, CadBody)
    assert kernel.volume(loaded.shape) == pytest.approx(kernel.volume(solid.shape), rel=1e-6)
    assert kernel.topology_counts(loaded.shape)["solids"] == 1


@pytest.mark.cad
def test_revolve_requires_profile_on_axis_side():
    mesh = make_capped_cylinder(radius=5.0, height=10.0, center=(0, 0, 50))
    doc, mesh_id = _document_with(mesh)
    _, sketch = _add(
        doc, MeshSketchFeature(inputs=[mesh_id], params={"plane": "xy", "offset": 0.0})
    )
    revolve = RevolveFeature(inputs=[sketch.id])
    doc.history.append(revolve)
    # an XY sketch of a circle centred on the u axis: the upper half revolves fine
    assert revolve.state is FeatureState.OK
