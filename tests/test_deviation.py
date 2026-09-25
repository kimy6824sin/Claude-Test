"""Accuracy analyzer: signed distance, statistics, colour map, workflows."""

import math
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from meshrev.core.deviation import (
    compute_deviation,
    compute_stats,
    deviation_colors,
    deviation_lut,
    signed_distances,
    vertex_areas,
)
from meshrev.core.samples import make_capped_cylinder
from meshrev.core.types import Axis

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sphere():
    return pv.Sphere(radius=10.0, theta_resolution=180, phi_resolution=180)


# -- signed distance -------------------------------------------------------------------------


def test_sign_outside_positive_inside_negative(sphere):
    rng = np.random.default_rng(1)
    dirs = rng.normal(size=(50, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    outside = signed_distances(dirs * 10.5, sphere)
    inside = signed_distances(dirs * 9.6, sphere)
    assert outside == pytest.approx(0.5, abs=0.01)
    assert inside == pytest.approx(-0.4, abs=0.01)


def test_compute_deviation_of_offset_scan_against_surface(sphere):
    # a "scan" 0.2 larger than the reference: excess material, positive everywhere
    scan = pv.Sphere(radius=10.2, theta_resolution=60, phi_resolution=60)
    result = compute_deviation(scan, sphere, tolerance=0.1, max_range=1.0)
    s = result.stats
    assert s.out_of_range == 0
    assert s.mean == pytest.approx(0.2, abs=0.01)
    assert s.std < 0.01 and s.rms == pytest.approx(math.hypot(s.mean, s.std), rel=1e-6)
    assert s.max_negative == 0.0 and s.max_positive == pytest.approx(0.2, abs=0.01)
    assert s.above_tolerance == pytest.approx(1.0) and s.within_tolerance == 0.0
    report = result.report()
    assert report["公差内"] == "0.00%" and "面积加权" in report["统计点数"]


# -- statistics ------------------------------------------------------------------------------


def test_stats_exclude_out_of_range_points():
    d = np.array([0.05, -0.05, 0.3, -0.2, 5.0])
    s = compute_stats(d, tolerance=0.1, max_range=1.0)
    assert s.count == 4 and s.out_of_range == 1
    assert s.max_positive == pytest.approx(0.3) and s.max_negative == pytest.approx(-0.2)
    assert s.mean == pytest.approx(0.025)
    assert s.rms == pytest.approx(math.sqrt(np.mean(d[:4] ** 2)))
    assert s.std == pytest.approx(np.std(d[:4]))
    assert (s.within_tolerance, s.above_tolerance, s.below_tolerance) == (0.5, 0.25, 0.25)


def test_stats_area_weighting():
    # many tiny-area points (a finely tessellated thread) must not dominate
    d = np.r_[np.full(1000, 0.5), 0.0]
    w = np.r_[np.full(1000, 1e-3), 99.0]
    s = compute_stats(d, tolerance=0.1, max_range=1.0, weights=w)
    assert s.within_tolerance == pytest.approx(0.99)
    assert s.mean == pytest.approx(0.005)
    assert compute_stats(d, 0.1, 1.0).within_tolerance == pytest.approx(1 / 1001)


def test_vertex_areas_sum_to_surface_area(sphere):
    assert vertex_areas(sphere).sum() == pytest.approx(sphere.area, rel=1e-6)


# -- colour map ------------------------------------------------------------------------------


def test_banded_colours_blue_green_red():
    tol, rng = 0.1, 1.0
    zero, plus, minus, out = deviation_colors([0.0, 0.95, -0.95, 3.0], tol, rng).astype(int)
    assert zero[1] > 150 and zero[0] < 60 and zero[2] < 100  # green
    assert plus[0] > 200 and plus[1] < 80 and plus[2] < 40  # red
    assert minus[2] > 200 and minus[0] < 40  # blue
    assert max(out) - min(out) < 20  # grey
    lut = deviation_lut(tol, rng, bands=15)
    assert lut.scalar_range == (-rng, rng)
    assert len(lut.values) == 15


# -- against a B-Rep -------------------------------------------------------------------------


@pytest.mark.cad
@pytest.mark.parametrize("direction", ["mesh_to_cad", "cad_to_mesh"])
def test_deviation_scan_vs_cad_cylinder(direction):
    from meshrev.core.bodies import CadBody
    from meshrev.core.cad import get_kernel

    kernel = get_kernel()
    cad = CadBody(kernel.cylinder(Axis((0, 0, 0), (0, 0, 1)), 10.0, 20.0), "cad", kernel=kernel)
    # scan radius 9.8: material missing on the wall (blue), ends exact
    scan = make_capped_cylinder(9.8, 20.0, n_theta=256)  # centred, like the CAD cylinder
    result = compute_deviation(scan, cad, tolerance=0.1, max_range=1.0, direction=direction)
    s = result.stats
    sign = -1.0 if direction == "mesh_to_cad" else 1.0  # CAD outside the scan: positive
    assert s.out_of_range == 0
    peak = s.max_negative if sign < 0 else s.max_positive
    assert peak == pytest.approx(sign * 0.2, abs=0.01)
    assert abs(s.mean) > 0.05 and np.sign(s.mean) == sign


# -- workflows (geometry helpers need no CAD kernel) ----------------------------------------


def test_revolution_score_and_axis_origin():
    from meshrev.core.workflows import fit_axis_origin, revolution_score

    mesh = make_capped_cylinder(5.0, 10.0, center=(3.0, -2.0, 0.0), n_theta=128)
    true_axis = Axis((3.0, -2.0, 0.0), (0, 0, 1))
    assert revolution_score(mesh, true_axis) == pytest.approx(1.0)
    assert revolution_score(mesh, Axis((3.0, -2.0, 0.0), (1, 0, 0))) < 0.8
    fitted = fit_axis_origin(mesh, np.array([0.0, 0.0, 1.0]))
    offset = np.asarray(fitted.origin) - np.array([3.0, -2.0, 0.0])
    assert np.linalg.norm(offset - (offset @ [0, 0, 1]) * np.array([0, 0, 1])) < 1e-6


def test_grid_region_loops_with_hole():
    from meshrev.core.sketch_fit import polygon_area
    from meshrev.core.workflows import grid_region_loops

    mask = np.ones((3, 3), bool)
    mask[1, 1] = False
    loops = grid_region_loops(mask, np.arange(4.0), np.array([0.0, 1.0, 3.0, 4.0]))
    areas = sorted(polygon_area(lp) for lp in loops)
    assert areas == pytest.approx([-2.0, 12.0])  # CW hole, CCW outer
    assert all(len(lp) == 4 for lp in loops)  # collinear vertices removed


@pytest.mark.cad
def test_revolved_workflow_on_demo_piston(demo_piston):
    from meshrev.core.workflows import reconstruct_revolved

    result = reconstruct_revolved(demo_piston)
    assert result.valid and result.method == "revolve+holes"
    assert any("r=11.0" in step for step in result.steps)  # pin bore cut
    s = result.deviation.stats
    assert s.within_tolerance > 0.9 and s.rms < 0.15
    assert result.score() > 0.8  # the (unmodelled) bosses are out of range


@pytest.mark.cad
@pytest.mark.skipif(not (ROOT / "3.stl").exists(), reason="3.stl sample not present")
def test_prismatic_workflow_on_flange():
    from meshrev import io as mio
    from meshrev.core.workflows import reconstruct_prismatic

    (body,) = mio.load(ROOT / "3.stl")
    result = reconstruct_prismatic(body.polydata)
    assert result.valid
    assert result.body.kernel.topology_counts(result.body.shape)["solids"] == 1
    s = result.deviation.stats
    assert s.within_tolerance > 0.9 and s.rms < 0.1 and s.out_of_range == 0


def test_grid_regions_touching_diagonally_stay_separate():
    from meshrev.core.sketch_fit import polygon_area
    from meshrev.core.workflows import grid_region_loops

    mask = np.array([[True, False], [False, True]])
    loops = grid_region_loops(mask, np.arange(3.0), np.arange(3.0))
    assert sorted(polygon_area(lp) for lp in loops) == pytest.approx([1.0, 1.0])
