"""Mesh sketch on real sample parts, as a parametric feature and into CAD."""

import math
from pathlib import Path

import numpy as np
import pytest

from meshrev import io as mio
from meshrev.core.bodies import DatumAxisBody, MeshBody, SectionBody, SketchBody
from meshrev.core.document import Document
from meshrev.core.features import (
    FeatureState,
    ImportFeature,
    MeshSketchFeature,
    PrimitiveDetectFeature,
)
from meshrev.core.features.sketching import plane_through_axis
from meshrev.core.samples import PistonSpec, make_capped_cylinder
from meshrev.core.section import Arc2D, Circle2D, Line2D, slice_mesh
from meshrev.core.sketch_fit import fit_section, polygon_area
from meshrev.core.types import Axis, Plane

ROOT = Path(__file__).resolve().parents[1]


def sample(name: str):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"sample model {name} not available")
    return mio.load(path)[0].polydata


def tangent_at(entity, at_start: bool) -> np.ndarray:
    if isinstance(entity, Line2D):
        return entity.direction
    p = np.asarray(entity.start if at_start else entity.end) - np.asarray(entity.center)
    t = np.array([-p[1], p[0]]) if entity.ccw else np.array([p[1], -p[0]])
    return t / np.linalg.norm(t)


@pytest.fixture(scope="module")
def flange_section():
    mesh = sample("3.stl")
    center = mesh.center
    return slice_mesh(mesh, Plane((center[0], center[1], mesh.bounds[4] + 3.0), (0, 0, 1)))


def test_flange_outline_is_four_tangent_lines_and_four_arcs(flange_section):
    result = fit_section(flange_section)
    outer = [loop for loop in result.loops if loop.depth == 0]
    assert len(outer) == 1
    entities = outer[0].entities
    assert sum(isinstance(e, Line2D) for e in entities) == 4
    assert sum(isinstance(e, Arc2D) for e in entities) == 4
    # every joint is G1 (the lines are tangent to the boss / body arcs)
    for a, b in zip(entities, entities[1:] + entities[:1], strict=True):
        assert np.linalg.norm(np.subtract(a.end, b.start)) < 1e-9
        cos = float(tangent_at(a, False) @ tangent_at(b, True))
        assert math.degrees(math.acos(min(1.0, cos))) < 1.0
    assert outer[0].max_deviation <= 1.5 * result.tolerance


def test_flange_holes_are_circles(flange_section):
    result = fit_section(flange_section)
    holes = [loop for loop in result.loops if loop.is_hole]
    assert len(holes) == 3
    assert all(len(h.entities) == 1 and isinstance(h.entities[0], Circle2D) for h in holes)
    radii = sorted(h.entities[0].radius for h in holes)
    assert radii[2] == pytest.approx(13.5, abs=0.05)  # Ø27 port
    assert radii[0] == pytest.approx(3.05, abs=0.1) and radii[1] == pytest.approx(3.1, abs=0.1)
    (sketch,) = result.to_sketches()
    assert len(sketch.holes) == 3 and sketch.is_closed(1e-9)


@pytest.mark.cad
def test_fitted_flange_sketch_extrudes(flange_section):
    from meshrev.core.cad import get_kernel

    result = fit_section(flange_section)
    (sketch,) = result.to_sketches()
    kernel = get_kernel()
    solid = kernel.extrude(sketch, sketch.plane.normal, 5.0)
    areas = sorted(abs(polygon_area(p)) for p in flange_section.to_2d())
    expected = 5.0 * (areas[-1] - sum(areas[:-1]))
    assert kernel.volume(solid) == pytest.approx(expected, rel=5e-3)


def test_demo_piston_axial_profile(demo_piston):
    spec = PistonSpec()
    plane = plane_through_axis(Axis((0, 0, 0), (0, 0, 1)), angle_deg=90.0)
    result = fit_section(slice_mesh(demo_piston, plane))
    assert all(loop.closed for loop in result.loops)
    entities = result.entities
    lines = [e for e in entities if isinstance(e, Line2D)]
    arcs = [e for e in entities if isinstance(e, Arc2D)]
    # sketch x runs along the piston axis: the crown is a line at u = height
    assert any(
        abs(e.start[0] - spec.height) < 0.05
        and abs(e.end[0] - spec.height) < 0.05
        and e.length > 15
        for e in lines
    )
    # skirt: lines at v = ±radius along the axis
    for side in (spec.radius, -spec.radius):
        assert any(
            abs(e.start[1] - side) < 0.05 and abs(e.end[1] - side) < 0.05 and e.length > 30
            for e in lines
        )
    # the spherical combustion bowl appears as an arc of the dish radius
    assert any(abs(a.radius - spec.dish_radius) < 0.2 for a in arcs)
    assert result.max_deviation <= 2.5 * result.tolerance


def test_real_piston_sections_are_closed_and_accurate():
    mesh = sample("6.stl")
    center = mesh.center
    for normal in ((1, 0, 0), (0, 0, 1)):
        result = fit_section(slice_mesh(mesh, Plane(center, normal, x_axis=(0, 1, 0))))
        assert result.loops and all(loop.closed for loop in result.loops)
        for sketch in result.to_sketches():
            assert sketch.is_closed(1e-9)
        assert result.max_deviation <= 2.5 * result.tolerance


def test_plane_through_axis():
    axis = Axis((1, 2, 3), (0, 1, 0))
    plane = plane_through_axis(axis, 30.0)
    np.testing.assert_allclose(plane.x_axis, (0, 1, 0), atol=1e-12)
    assert abs(plane.normal @ axis.direction) < 1e-12
    assert abs(plane.signed_distance([axis.origin + 5 * axis.direction])[0]) < 1e-12
    other = plane_through_axis(axis, 120.0)
    assert math.degrees(math.acos(abs(plane.normal @ other.normal))) == pytest.approx(90.0)


def test_mesh_sketch_feature_on_datum_axis():
    doc = Document()
    mesh = make_capped_cylinder(radius=8.0, height=30.0, n_theta=128, n_z=30)
    source = ImportFeature("cyl.stl", bodies=[MeshBody(mesh, "cyl")])
    doc.history.append(source)
    ransac = PrimitiveDetectFeature(
        inputs=[source.output_ids[0]], params={"types": "cylinder", "max_primitives": 1}
    )
    doc.history.append(ransac)
    (axis,) = [doc.get(i) for i in ransac.output_ids if isinstance(doc.get(i), DatumAxisBody)]
    sketch_feature = MeshSketchFeature(
        inputs=[source.output_ids[0], axis.id], params={"plane": "datum"}, name="网格草图 1"
    )
    doc.history.append(sketch_feature)
    assert sketch_feature.state is FeatureState.OK, sketch_feature.error
    section, sketch = (doc.get(i) for i in sketch_feature.output_ids)
    assert isinstance(section, SectionBody) and isinstance(sketch, SketchBody)
    (loop,) = sketch.result.loops
    assert len(loop.entities) == 4 and all(isinstance(e, Line2D) for e in loop.entities)
    lengths = sorted(round(e.length, 3) for e in loop.entities)
    assert lengths == pytest.approx([16.0, 16.0, 30.0, 30.0], abs=0.01)
    assert sketch.info()["直线 / 圆弧 / 整圆"] == "4 / 0 / 0"
    assert sketch.to_polydata().n_cells == 4

    # parameter edit -> regeneration: a section 1 mm off the axis is still a rectangle
    doc.history.update_params(sketch_feature.id, {"offset": 1.0})
    sketch = doc.get(sketch_feature.output_ids[1])
    widths = sorted(e.length for e in sketch.result.loops[0].entities)
    assert widths[0] == pytest.approx(2 * math.sqrt(64 - 1), abs=0.02)


def test_mesh_sketch_feature_reports_missing_intersection():
    doc = Document()
    source = ImportFeature("cyl.stl", bodies=[MeshBody(make_capped_cylinder(), "cyl")])
    doc.history.append(source)
    feature = MeshSketchFeature(
        inputs=[source.output_ids[0]], params={"plane": "xy", "offset": 500.0}
    )
    doc.history.append(feature)
    assert feature.state is FeatureState.ERROR and "交线" in feature.error
