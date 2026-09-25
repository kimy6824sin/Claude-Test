"""Least-squares fitting, normal estimation and RANSAC extraction."""

import numpy as np
import pytest
import pyvista as pv

from meshrev.core.primitives import (
    CylinderPrimitive,
    PlanePrimitive,
    PrimitiveType,
    RansacOptions,
    axis_from_normals,
    detect_primitives,
    estimate_normals,
    extract_cylinders,
    extract_planes,
    fit_cylinder,
    fit_plane,
    fit_sphere,
    ransac_cylinder,
    ransac_plane,
)
from meshrev.core.samples import PistonSpec, make_capped_cylinder
from meshrev.core.types import orthonormal_basis


def angle_deg(a, b) -> float:
    return float(
        np.degrees(np.arccos(min(1.0, abs(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)))))
    )


def cylinder_points(rng, center, axis, radius, arc_deg=360.0, length=20.0, n=3000, noise=0.0):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    u, v = orthonormal_basis(axis)
    theta = rng.uniform(0, np.radians(arc_deg), n)
    t = rng.uniform(-length / 2, length / 2, n)
    radial = np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v)
    pts = np.asarray(center) + np.outer(t, axis) + radius * radial
    return pts + rng.normal(scale=noise, size=pts.shape), radial


def plane_points(rng, normal, d, size=20.0, n=2000, noise=0.0):
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    u, v = orthonormal_basis(normal)
    pts = (
        -d * normal
        + np.outer(rng.uniform(-size / 2, size / 2, n), u)
        + np.outer(rng.uniform(-size / 2, size / 2, n), v)
    )
    return pts + rng.normal(scale=noise, size=pts.shape), np.tile(normal, (n, 1))


def random_unit_normals(rng, n):
    vec = rng.normal(size=(n, 3))
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)


# -- least squares -------------------------------------------------------------------------
def test_fit_plane_equation():
    rng = np.random.default_rng(0)
    normal = np.array([0.2, -0.1, 1.0])
    pts, nrm = plane_points(rng, normal, d=-7.5, noise=0.01)
    fit = fit_plane(pts, normals=nrm)
    a, b, c, d = fit.primitive.equation
    assert a * a + b * b + c * c == pytest.approx(1.0)
    assert angle_deg((a, b, c), normal) < 0.01
    assert np.dot((a, b, c), normal) > 0  # oriented like the given normals
    assert d == pytest.approx(-7.5, abs=2e-3)  # plane_points uses the unit normal
    assert fit.rms == pytest.approx(0.01, rel=0.1)
    np.testing.assert_allclose(fit.primitive.distance(pts), fit.residuals)


@pytest.mark.parametrize("arc_deg", [60.0, 120.0, 360.0])
@pytest.mark.parametrize("use_normals", [True, False])
def test_fit_cylinder_high_precision(arc_deg, use_normals):
    rng = np.random.default_rng(1)
    center, axis, radius = np.array([10.0, -5.0, 3.0]), np.array([0.3, -0.5, 0.8]), 11.0
    pts, radial = cylinder_points(rng, center, axis, radius, arc_deg, noise=0.005)
    fit = fit_cylinder(pts, normals=radial if use_normals else None)
    cyl = fit.primitive
    assert isinstance(cyl, CylinderPrimitive)
    assert angle_deg(cyl.axis_direction, axis) < 0.02
    assert cyl.radius == pytest.approx(radius, abs=2e-3)
    assert cyl.axis.distance([center])[0] < 2e-3
    assert cyl.length == pytest.approx(20.0, abs=0.1)
    assert fit.rms == pytest.approx(0.005, rel=0.15)


def test_fit_cylinder_detects_holes():
    rng = np.random.default_rng(2)
    pts, radial = cylinder_points(rng, (0, 0, 0), (1, 0, 0), 5.0)
    assert fit_cylinder(pts, normals=radial).primitive.concave is False  # boss / shaft
    assert fit_cylinder(pts, normals=-radial).primitive.concave is True  # bore / hole


def test_fit_cylinder_robust_to_outliers():
    rng = np.random.default_rng(3)
    pts, radial = cylinder_points(rng, (0, 0, 0), (0, 0, 1), 8.0, arc_deg=180, noise=0.002)
    outliers = rng.uniform(-8, 8, (150, 3))
    fit = fit_cylinder(
        np.vstack([pts, outliers]), normals=np.vstack([radial, radial[:150]]), robust_scale=0.01
    )
    assert fit.primitive.radius == pytest.approx(8.0, abs=5e-3)
    assert angle_deg(fit.primitive.axis_direction, (0, 0, 1)) < 0.05


def test_fit_sphere():
    rng = np.random.default_rng(4)
    normals = random_unit_normals(rng, 2000)
    normals = normals[normals[:, 2] > 0.5]  # a spherical cap, like a piston bowl
    pts = np.array([1.0, 2.0, 3.0]) + 60.0 * normals + rng.normal(scale=0.005, size=normals.shape)
    fit = fit_sphere(pts, normals=-normals)
    assert fit.primitive.radius == pytest.approx(60.0, abs=0.01)
    np.testing.assert_allclose(fit.primitive.center, (1, 2, 3), atol=0.02)
    assert fit.primitive.concave is True


def test_axis_from_normals_conditioning():
    rng = np.random.default_rng(5)
    _, radial = cylinder_points(rng, (0, 0, 0), (0, 1, 0), 3.0)
    axis, spread = axis_from_normals(radial)
    assert angle_deg(axis, (0, 1, 0)) < 1e-6 and spread == pytest.approx(0.5, abs=0.05)
    _, spread_plane = axis_from_normals(np.tile((0, 0, 1.0), (50, 1)))
    assert spread_plane < 1e-12


def test_estimate_normals_on_sphere():
    sphere = pv.Sphere(radius=5.0, theta_resolution=60, phi_resolution=60)
    normals = estimate_normals(sphere.points, k=12)
    expected = sphere.points / np.linalg.norm(sphere.points, axis=1, keepdims=True)
    cos = np.einsum("ij,ij->i", normals, expected)
    assert np.all(cos > 0)  # oriented outward
    assert np.median(cos) > 0.999


# -- RANSAC ----------------------------------------------------------------------------------------
def test_ransac_plane_with_50_percent_outliers():
    rng = np.random.default_rng(6)
    pts, nrm = plane_points(rng, (0, 0, 1), d=-2.0, n=2000, noise=0.01)
    noise_pts = rng.uniform(-10, 10, (2000, 3))
    fit = ransac_plane(
        np.vstack([pts, noise_pts]),
        np.vstack([nrm, random_unit_normals(rng, 2000)]),
        RansacOptions(distance_threshold=0.05, min_support=100),
    )
    assert fit is not None
    assert angle_deg(fit.primitive.normal, (0, 0, 1)) < 0.05
    assert abs(fit.primitive.d) == pytest.approx(2.0, abs=5e-3)
    assert np.mean(fit.inliers < 2000) > 0.98  # nearly all inliers are plane samples
    assert len(fit.inliers) > 1900


def test_ransac_cylinder_among_plane_and_outliers():
    rng = np.random.default_rng(7)
    cyl, radial = cylinder_points(rng, (2, 1, 0), (0, 1, 0), 6.0, arc_deg=200, noise=0.01)
    pl, pn = plane_points(rng, (0, 0, 1), d=15.0, noise=0.01)
    junk = rng.uniform(-15, 15, (1500, 3))
    pts = np.vstack([cyl, pl, junk])
    nrm = np.vstack([radial, pn, random_unit_normals(rng, 1500)])
    fit = ransac_cylinder(pts, nrm, RansacOptions(distance_threshold=0.05))
    assert fit is not None
    assert fit.primitive.radius == pytest.approx(6.0, abs=0.01)
    assert angle_deg(fit.primitive.axis_direction, (0, 1, 0)) < 0.1
    assert np.all(fit.inliers < len(cyl) + 50)


def test_detect_primitives_scene():
    rng = np.random.default_rng(8)
    cyl, radial = cylinder_points(rng, (0, 0, 0), (0, 0, 1), 5.0, n=3000, noise=0.005)
    p1, n1 = plane_points(rng, (0, 0, 1), d=-15.0, n=2000, noise=0.005)
    p2, n2 = plane_points(rng, (1, 0, 0), d=-20.0, n=1500, noise=0.005)
    results = detect_primitives(
        np.vstack([cyl, p1, p2]),
        np.vstack([radial, n1, n2]),
        options=RansacOptions(distance_threshold=0.03, min_support=200),
    )
    kinds = sorted(r.type.value for r in results)
    assert kinds == ["cylinder", "plane", "plane"]
    assert [len(r.inliers) for r in results] == sorted(
        (len(r.inliers) for r in results), reverse=True
    )


def test_ransac_finds_nothing_in_noise():
    rng = np.random.default_rng(9)
    pts = rng.uniform(-10, 10, (1500, 3))
    options = RansacOptions(distance_threshold=0.01, min_support=200, max_iterations=512)
    assert ransac_plane(pts, random_unit_normals(rng, 1500), options) is None


# -- mesh level ----------------------------------------------------------------------------------
def test_extract_from_capped_cylinder_mesh():
    mesh = make_capped_cylinder(radius=10.0, height=30.0, n_theta=128, n_z=30)
    cylinders = extract_cylinders(mesh)
    assert len(cylinders) == 1
    cyl = cylinders[0].primitive
    assert cyl.radius == pytest.approx(10.0, abs=1e-6)  # vertices lie on the surface
    assert angle_deg(cyl.axis_direction, (0, 0, 1)) < 1e-4
    assert cyl.concave is False
    planes = extract_planes(mesh)
    assert len(planes) == 2
    offsets = sorted(p.primitive.d for p in planes)
    assert offsets == pytest.approx([-15.0, -15.0], abs=1e-9)  # outward normals: n·x = 15
    for item in planes:
        assert isinstance(item.primitive, PlanePrimitive)
        assert abs(item.primitive.normal[2]) == pytest.approx(1.0)


def test_pin_bore_axis_from_demo_piston(demo_piston):
    spec = PistonSpec()
    cylinders = extract_cylinders(demo_piston, max_count=10)
    bores = [c for c in cylinders if abs(c.primitive.radius - spec.pin_radius) < 0.05]
    assert len(bores) == 1, "the two boss bores must be merged into one coaxial cylinder"
    bore = bores[0].primitive
    assert bore.concave is True
    assert angle_deg(bore.axis_direction, (1, 0, 0)) < 0.02
    assert bore.axis.distance([[0.0, 0.0, spec.pin_height]])[0] < 0.01
    assert bore.radius == pytest.approx(spec.pin_radius, abs=0.01)
    centroids = demo_piston.cell_centers().points[bores[0].face_ids]
    assert (centroids[:, 0] > 0).any() and (centroids[:, 0] < 0).any()  # both bosses
    skirt = max(cylinders, key=lambda c: len(c.face_ids)).primitive
    assert skirt.radius == pytest.approx(spec.radius, abs=0.01)
    assert angle_deg(skirt.axis_direction, (0, 0, 1)) < 0.02


def test_primitive_type_labels():
    assert PrimitiveType.PLANE.label == "平面"
    assert PrimitiveType("cylinder") is PrimitiveType.CYLINDER
