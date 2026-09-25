import numpy as np
import pytest

from meshrev.core.types import Axis, BBox, Plane, orthonormal_basis


def test_plane_equation_roundtrip():
    plane = Plane.from_equation(0.0, 0.0, 2.0, -10.0)  # z = 5
    a, b, c, d = plane.equation
    assert (a, b, c) == pytest.approx((0.0, 0.0, 1.0))
    assert d == pytest.approx(-5.0)
    assert plane.signed_distance([[0, 0, 7]])[0] == pytest.approx(2.0)


def test_plane_local_world_roundtrip():
    plane = Plane((1, 2, 3), (1, 1, 1))
    pts = np.array([[4.0, -1.0, 0.5], [0.0, 0.0, 0.0]])
    projected = plane.project(pts)
    np.testing.assert_allclose(plane.to_world(plane.to_local(pts)), projected, atol=1e-12)
    np.testing.assert_allclose(plane.signed_distance(projected), 0.0, atol=1e-12)


def test_axis_distance_and_angle():
    axis = Axis((0, 0, 0), (0, 0, 5))
    assert axis.distance([[3, 4, 10]])[0] == pytest.approx(5.0)
    assert axis.parameter([[3, 4, 10]])[0] == pytest.approx(10.0)
    assert axis.angle_to(Axis((1, 1, 1), (0, 0, -1))) == pytest.approx(0.0)
    assert axis.angle_to(Axis((0, 0, 0), (1, 0, 0))) == pytest.approx(np.pi / 2)


def test_bbox():
    box = BBox.from_points([[0, 0, 0], [2, 4, 4]])
    assert box.diagonal == pytest.approx(6.0)
    np.testing.assert_allclose(box.center, [1, 2, 2])
    assert BBox.from_bounds(box.bounds).diagonal == pytest.approx(6.0)


@pytest.mark.parametrize("normal", [(0, 0, 1), (1, 0, 0), (0.3, -0.2, 0.9)])
def test_orthonormal_basis(normal):
    u, v = orthonormal_basis(normal)
    n = np.asarray(normal) / np.linalg.norm(normal)
    assert abs(u @ v) < 1e-12 and abs(u @ n) < 1e-12
    np.testing.assert_allclose(np.cross(u, v), n, atol=1e-12)


def test_zero_normal_rejected():
    with pytest.raises(ValueError):
        Plane((0, 0, 0), (0, 0, 0))
