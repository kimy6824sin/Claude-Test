"""2D profile fitting: primitives, segmentation, topology and hierarchy."""

import math

import numpy as np
import pytest

from meshrev.core.section import Arc2D, Circle2D, Line2D, loop_is_closed
from meshrev.core.sketch_fit import (
    SketchFitOptions,
    circle_from_3_points,
    detect_corners,
    estimate_noise,
    fit_circle_lsq,
    fit_line_lsq,
    fit_polyline,
    fit_profiles_2d,
    loop_deviation,
    polygon_area,
    ransac_circle_2d,
    ransac_line_2d,
)
from meshrev.core.types import Plane

PLANE = Plane((0, 0, 0), (0, 0, 1))


def rounded_rectangle(w, h, r, step=0.2):
    """CCW rounded rectangle: 4 lines tangent to 4 quarter arcs."""
    pts = []
    edges = [
        ((-w / 2 + r, -h / 2), (w / 2 - r, -h / 2)),
        ((w / 2, -h / 2 + r), (w / 2, h / 2 - r)),
        ((w / 2 - r, h / 2), (-w / 2 + r, h / 2)),
        ((-w / 2, h / 2 - r), (-w / 2, -h / 2 + r)),
    ]
    centers = [
        (w / 2 - r, -h / 2 + r),
        (w / 2 - r, h / 2 - r),
        (-w / 2 + r, h / 2 - r),
        (-w / 2 + r, -h / 2 + r),
    ]
    for k, ((a, b), c) in enumerate(zip(edges, centers, strict=True)):
        a, b = np.array(a), np.array(b)
        n = max(2, int(np.linalg.norm(b - a) / step))
        pts.extend(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
        a0 = -math.pi / 2 + k * math.pi / 2
        th = np.linspace(a0, a0 + math.pi / 2, max(3, int(r * math.pi / 2 / step)), endpoint=False)
        pts.extend(np.array(c) + r * np.column_stack([np.cos(th), np.sin(th)]))
    return np.array(pts)


def circle_points(center, r, n=90, start=0.0, sweep=2 * math.pi):
    th = start + np.linspace(0, sweep, n, endpoint=sweep < 2 * math.pi)
    return np.asarray(center) + r * np.column_stack([np.cos(th), np.sin(th)])


# -- primitive fits ---------------------------------------------------------------------
def test_total_least_squares_line_handles_vertical():
    rng = np.random.default_rng(0)
    y = np.linspace(0, 10, 50)
    pts = np.column_stack([3.0 + rng.normal(scale=0.01, size=50), y])
    fit = fit_line_lsq(pts)
    assert abs(fit.direction[1]) == pytest.approx(1.0, abs=1e-4)
    assert fit.point[0] == pytest.approx(3.0, abs=0.005)


def test_geometric_circle_fit_on_short_arc():
    rng = np.random.default_rng(1)
    pts = circle_points((5, -2), 20.0, n=60, start=0.3, sweep=math.radians(40))
    pts = pts + rng.normal(scale=0.005, size=pts.shape)
    fit = fit_circle_lsq(pts)
    assert fit.radius == pytest.approx(20.0, abs=0.05)
    np.testing.assert_allclose(fit.center, (5, -2), atol=0.05)


def test_circle_from_three_points():
    c = circle_from_3_points(np.array([1.0, 0]), np.array([0.0, 1]), np.array([-1.0, 0]))
    assert c.radius == pytest.approx(1.0) and np.allclose(c.center, 0)
    assert circle_from_3_points(np.zeros(2), np.ones(2), 2 * np.ones(2)) is None


def test_ransac_line_and_circle_reject_outliers():
    rng = np.random.default_rng(2)
    line = np.column_stack([np.linspace(0, 10, 80), 0.5 * np.linspace(0, 10, 80) + 1])
    noisy = np.vstack([line, rng.uniform(0, 10, (30, 2))])
    fit, inliers = ransac_line_2d(noisy, 0.02)
    assert inliers[:80].all() and inliers[80:].mean() < 0.3
    assert abs(fit.direction @ np.array([1, 0.5]) / np.hypot(1, 0.5)) == pytest.approx(1, abs=1e-9)
    ring = np.vstack([circle_points((0, 0), 4.0, 70), rng.uniform(-3, 3, (25, 2))])
    circle, inl = ransac_circle_2d(ring, 0.02)
    assert circle.radius == pytest.approx(4.0, abs=1e-6) and inl[:70].all()


def test_noise_estimate():
    rng = np.random.default_rng(3)
    pts = rounded_rectangle(40, 20, 5, step=0.1)
    assert estimate_noise(pts, True) < 0.005
    noisy = pts + rng.normal(scale=0.02, size=pts.shape)
    assert estimate_noise(noisy, True) == pytest.approx(0.02, rel=0.35)


def test_corner_detection():
    square = np.vstack(
        [
            np.column_stack([np.linspace(0, 1, 20, endpoint=False), np.zeros(20)]),
            np.column_stack([np.ones(20), np.linspace(0, 1, 20, endpoint=False)]),
            np.column_stack([np.linspace(1, 0, 20, endpoint=False), np.ones(20)]),
            np.column_stack([np.zeros(20), np.linspace(1, 0, 20, endpoint=False)]),
        ]
    )
    assert detect_corners(square, True, 35.0) == [0, 20, 40, 60]


# -- segmentation & topology ----------------------------------------------------------------
@pytest.mark.parametrize("noise", [0.0, 0.01])
def test_rounded_rectangle_gives_tangent_lines_and_arcs(noise):
    rng = np.random.default_rng(4)
    pts = rounded_rectangle(40, 20, 5) + rng.normal(scale=noise, size=(1, 2)) * 0
    pts = pts + rng.normal(scale=noise, size=pts.shape)
    entities, _, dev = fit_polyline(pts, True, 0.05)
    lines = [e for e in entities if isinstance(e, Line2D)]
    arcs = [e for e in entities if isinstance(e, Arc2D)]
    assert len(lines) == 4 and len(arcs) == 4
    assert loop_is_closed(entities, 1e-9)
    for arc in arcs:
        assert arc.radius == pytest.approx(5.0, abs=0.05 + 3 * noise)
        assert math.degrees(arc.sweep) == pytest.approx(90.0, abs=2.0)
        assert arc.ccw
    for line in lines:  # axis snapping makes them exactly horizontal / vertical
        assert line.angle_deg in (0.0, 90.0)
    lengths = sorted(round(line.length, 1) for line in lines)
    assert lengths == pytest.approx([10.0, 10.0, 30.0, 30.0], abs=0.15)
    assert dev <= 0.05


def test_open_profile_keeps_ends():
    # a step: line, 90° corner, line, 180° half-circle
    a = np.column_stack([np.linspace(0, 10, 51), np.zeros(51)])
    b = np.column_stack([np.full(26, 10.0), np.linspace(0, 5, 26)])[1:]
    arc = circle_points((13, 5), 3.0, n=60, start=math.pi, sweep=-math.pi)[1:]
    entities, _, dev = fit_polyline(np.vstack([a, b, arc]), False, 0.02)
    kinds = [type(e).__name__ for e in entities]
    assert kinds == ["Line2D", "Line2D", "Arc2D"]
    assert entities[0].start == pytest.approx((0.0, 0.0), abs=1e-6)
    assert entities[1].start == pytest.approx((10.0, 0.0), abs=1e-6)  # exact corner
    assert entities[2].radius == pytest.approx(3.0, abs=1e-3) and not entities[2].ccw
    assert dev < 0.02


def test_spikes_are_removed():
    pts = circle_points((0, 0), 10.0, 200)
    pts[50] += (0.8, 0.8)  # a single outlier (scan spike)
    entities, _, dev = fit_polyline(pts, True, 0.02)
    assert len(entities) == 1 and isinstance(entities[0], Circle2D)
    assert entities[0].radius == pytest.approx(10.0, abs=1e-3)


def test_small_bevels_collapse_to_sharp_corners():
    # a 20 x 10 rectangle whose corners are cut by 0.1 mm bevels (like voxel/scan data)
    c = 0.1
    corners = [
        (c, 0),
        (20 - c, 0),
        (20, c),
        (20, 10 - c),
        (20 - c, 10),
        (c, 10),
        (0, 10 - c),
        (0, c),
    ]
    pts = []
    for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
        a, b = np.array(a), np.array(b)
        n = max(2, int(np.linalg.norm(b - a) / 0.05))
        pts.extend(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    entities, _, _ = fit_polyline(np.array(pts), True, 0.03)
    assert len(entities) == 4 and all(isinstance(e, Line2D) for e in entities)
    vertices = sorted((round(e.start[0], 6), round(e.start[1], 6)) for e in entities)
    assert vertices == [(0.0, 0.0), (0.0, 10.0), (20.0, 0.0), (20.0, 10.0)]


# -- sections: hierarchy, holes, intent -------------------------------------------------------
def test_hierarchy_orientation_and_holes():
    outer = rounded_rectangle(60, 40, 6)[::-1]  # given clockwise on purpose
    bore = circle_points((0, 0), 10.0, 120)
    bolt = circle_points((22, 12), 2.5, 40)
    island = circle_points((0, 0), 4.0, 60)  # a boss inside the bore (depth 2)
    speck = circle_points((25, -15), 0.05, 12)  # noise loop
    result = fit_profiles_2d(
        [bore, outer, bolt, island, speck], [True] * 5, PLANE, SketchFitOptions(tolerance=0.02)
    )
    depths = sorted(loop.depth for loop in result.loops)
    assert depths == [0, 1, 1, 2]  # speck dropped; the island in the bore is material again
    for loop in result.loops:
        area = polygon_area(loop.points)
        assert (area > 0) == (loop.depth % 2 == 0)  # outer CCW, holes CW
    sketches = result.to_sketches()
    assert len(sketches) == 2
    main = max(sketches, key=lambda s: len(s.entities))
    assert len(main.holes) == 2 and main.is_closed(1e-9)
    radii = sorted(h[0].radius for h in main.holes)
    assert radii == pytest.approx([2.5, 10.0], abs=1e-3)


def test_tapped_hole_is_recognised_as_circle():
    th = np.linspace(0, 2 * math.pi, 180, endpoint=False)
    r = 3.0 + 0.08 * np.sign(np.sin(24 * th))  # shallow thread flanks in the section
    thread = np.column_stack([r * np.cos(th), r * np.sin(th)])
    entities, _, _ = fit_polyline(thread, True, 0.03)
    assert len(entities) == 1 and isinstance(entities[0], Circle2D)
    assert entities[0].radius == pytest.approx(3.0, abs=0.05)


def test_hexagon_is_not_a_circle():
    hexagon = circle_points((0, 0), 5.0, 6)
    pts = np.vstack(
        [
            a + (b - a) * np.linspace(0, 1, 20, endpoint=False)[:, None]
            for a, b in zip(hexagon, np.roll(hexagon, -1, axis=0), strict=True)
        ]
    )
    entities, _, _ = fit_polyline(pts, True, 0.02)
    assert len(entities) == 6 and all(isinstance(e, Line2D) for e in entities)


def test_loop_deviation_measures_distance_to_entities():
    entities = [Line2D((0, 0), (10, 0))]
    assert loop_deviation(entities, np.array([[5, 0.3], [12, 0]])) == pytest.approx(2.0)
    arc = Arc2D((0, 0), 1.0, 0.0, math.pi / 2)
    assert loop_deviation([arc], np.array([[0.0, 1.2]])) == pytest.approx(0.2)
    assert arc.end == pytest.approx((0.0, 1.0))
    assert Arc2D((0, 0), 1.0, math.pi / 2, 0.0, ccw=False).sweep == pytest.approx(math.pi / 2)
