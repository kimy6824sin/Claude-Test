import numpy as np
import pytest

from meshrev.core.mesh import mesh_statistics, signed_volume
from meshrev.core.samples import PistonSpec


def test_demo_piston_is_watertight_and_outward(demo_piston):
    stats = mesh_statistics(demo_piston)
    assert stats.is_closed
    assert signed_volume(demo_piston) > 0
    spec = PistonSpec()
    np.testing.assert_allclose(
        stats.bbox.size, [2 * spec.radius, 2 * spec.radius, spec.height], atol=0.8
    )


def test_demo_piston_volume_close_to_analytic(demo_piston):
    spec = PistonSpec()
    # crude analytic volume: outer cylinder minus grooves, cavity, bores and dish
    outer = np.pi * spec.radius**2 * spec.height
    grooves = sum(
        np.pi * (spec.radius**2 - (spec.radius - spec.groove_depth) ** 2) * (z1 - z0)
        for z0, z1 in spec.groove_ranges
    )
    assert signed_volume(demo_piston) < outer - grooves
    assert signed_volume(demo_piston) == pytest.approx(195_000, rel=0.02)
