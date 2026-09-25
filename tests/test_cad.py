import numpy as np
import pytest

from meshrev.core.cad import BooleanOp
from meshrev.core.section import Line2D, Sketch
from meshrev.core.types import Axis, Plane

pytestmark = pytest.mark.cad


def rectangle_sketch(plane: Plane, u0, u1, v0, v1) -> Sketch:
    corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    lines = [Line2D(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    return Sketch(plane, lines)


def revolved_cylinder(kernel):
    """Radius 4, height 10 cylinder standing on z = 0, built by revolving a rectangle."""
    plane = Plane((0, 0, 0), (0, -1, 0))  # local u = -Z, v = +X
    u, v = plane.basis()
    assert np.allclose(u, (0, 0, -1)) and np.allclose(v, (1, 0, 0))
    profile = rectangle_sketch(plane, -10.0, 0.0, 0.0, 4.0)
    return kernel.revolve(profile, Axis((0, 0, 0), (0, 0, 1)))


def test_revolve_extrude_boolean():
    from meshrev.core.cad import get_kernel

    kernel = get_kernel()
    cylinder = revolved_cylinder(kernel)
    assert kernel.volume(cylinder) == pytest.approx(np.pi * 16 * 10, rel=1e-6)

    box = kernel.extrude(
        rectangle_sketch(Plane((0, 0, 0), (0, 0, 1)), -1, 1, -1, 1), (0, 0, 1), 20.0
    )
    assert kernel.volume(box) == pytest.approx(4 * 20, rel=1e-9)
    cut = kernel.boolean(cylinder, box, BooleanOp.CUT)
    assert kernel.volume(cut) == pytest.approx(np.pi * 160 - 40, rel=1e-6)
    mesh = kernel.tessellate(cut, tolerance=0.01)
    assert mesh.n_cells > 0
