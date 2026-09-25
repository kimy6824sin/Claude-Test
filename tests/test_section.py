import numpy as np
import pytest
import pyvista as pv

from meshrev.core.section import Arc2D, Line2D, Sketch, slice_mesh, slice_parallel
from meshrev.core.types import Plane


def test_slice_cylinder_gives_closed_circle():
    mesh = pv.Cylinder(radius=10.0, height=20.0, direction=(0, 0, 1), resolution=256).triangulate()
    curve = slice_mesh(mesh, Plane((0, 0, 1.0), (0, 0, 1)))
    assert len(curve.polylines) == 1 and curve.closed[0]
    radii = np.linalg.norm(curve.polylines[0][:, :2], axis=1)
    assert radii.max() == pytest.approx(10.0, abs=1e-6)
    assert radii.min() > 10.0 * np.cos(np.pi / 256) - 1e-6
    np.testing.assert_allclose(curve.polylines[0][:, 2], 1.0, atol=1e-9)


def test_piston_section_has_inner_and_outer_loops(demo_piston):
    curve = slice_mesh(demo_piston, Plane((0, 0, 10.0), (0, 0, 1)))
    assert len(curve.polylines) == 2 and all(curve.closed)


def test_slice_miss_is_empty():
    curve = slice_mesh(pv.Sphere(), Plane((0, 0, 5.0), (0, 0, 1)))
    assert curve.is_empty and curve.to_polydata().n_points == 0


def test_slice_parallel():
    curves = slice_parallel(pv.Sphere(radius=1.0), Plane((0, 0, 0), (0, 0, 1)), [-0.5, 0, 0.5])
    assert [len(c.polylines) for c in curves] == [1, 1, 1]


def test_sketch_entities():
    arc = Arc2D((0.0, 0.0), 2.0, 0.0, np.pi)
    assert arc.length == pytest.approx(2 * np.pi)
    np.testing.assert_allclose(arc.end, (-2.0, 0.0), atol=1e-12)
    sketch = Sketch(Plane((0, 0, 0), (0, 0, 1)), [arc, Line2D(arc.end, arc.start)])
    assert sketch.is_closed()
    assert sketch.to_polyline().shape[1] == 3
