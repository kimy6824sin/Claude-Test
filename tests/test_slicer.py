"""Exact edge-keyed mesh slicing."""

import numpy as np
import pytest
import pyvista as pv

from meshrev.core.samples import make_capped_cylinder
from meshrev.core.section import bridge_gaps, slice_mesh
from meshrev.core.types import Plane


def test_cylinder_section_is_one_closed_circle():
    mesh = make_capped_cylinder(radius=10.0, height=20.0, n_theta=128)
    curve = slice_mesh(mesh, Plane((0, 0, 1.234), (0, 0, 1)))
    assert curve.closed == [True]
    radii = np.linalg.norm(curve.polylines[0][:, :2], axis=1)
    assert radii.max() == pytest.approx(10.0, abs=1e-9)  # vertices lie on the facet edges
    np.testing.assert_allclose(curve.polylines[0][:, 2], 1.234, atol=1e-12)
    assert curve.total_length == pytest.approx(2 * np.pi * 10, rel=1e-3)


def test_plane_through_vertices_is_handled():
    # the plane z = 0 passes exactly through a ring of vertices and edges
    box = pv.Box(bounds=(-1, 1, -1, 1, -1, 1), level=3).triangulate().clean()
    assert np.any(np.isclose(box.points[:, 2], 0.0))
    curve = slice_mesh(box, Plane((0, 0, 0), (0, 0, 1)))
    assert curve.closed == [True]
    square = curve.to_2d()[0]
    assert np.abs(square).max() == pytest.approx(1.0, abs=1e-9)
    assert curve.total_length == pytest.approx(8.0, abs=1e-9)


def test_axial_section_of_capped_cylinder_is_rectangle():
    mesh = make_capped_cylinder(radius=5.0, height=12.0, n_theta=96)
    curve = slice_mesh(mesh, Plane((0, 0, 0), (1, 0, 0), x_axis=(0, 0, 1)))
    assert curve.closed == [True]
    uv = curve.to_2d()[0]
    np.testing.assert_allclose(uv.min(axis=0), (-6, -5), atol=1e-9)
    np.testing.assert_allclose(uv.max(axis=0), (6, 5), atol=1e-9)


def test_open_mesh_gives_open_chain_and_gap_bridging():
    mesh = make_capped_cylinder(radius=5.0, height=20.0, n_theta=64)
    centers = mesh.cell_centers().points
    cut_away = (centers[:, 0] > 4.0) & (np.abs(centers[:, 1]) < 0.8)  # a narrow crack
    cracked = mesh.extract_cells(np.flatnonzero(~cut_away)).extract_surface(
        algorithm="dataset_surface"
    )
    plane = Plane((0, 0, 3.0), (0, 0, 1))
    raw = slice_mesh(cracked, plane, gap_tolerance=0)
    assert raw.closed == [False]
    bridged = slice_mesh(cracked, plane, gap_tolerance=2.0)
    assert bridged.closed == [True]


def test_bridge_gaps_joins_two_chains():
    a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    b = np.array([[2.05, 0.0], [3.0, 1.0], [0.0, 1.0], [0.0, 0.05]])
    polys, closed = bridge_gaps([a, b], [False, False], 0.1)
    assert len(polys) == 1 and closed == [True] and len(polys[0]) == 7


def test_miss_returns_empty_curve():
    curve = slice_mesh(pv.Sphere(), Plane((0, 0, 3.0), (0, 0, 1)))
    assert curve.is_empty
