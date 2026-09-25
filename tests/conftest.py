from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_API", "pyside6")


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def pytest_collection_modifyitems(config, items):
    skip_gui = pytest.mark.skip(reason="no display; run under xvfb-run to enable GUI tests")
    from meshrev.core.cad import is_available

    skip_cad = pytest.mark.skip(reason="OCP (cadquery-ocp) not installed")
    cad_ok = is_available()
    for item in items:
        if "gui" in item.keywords and not _has_display():
            item.add_marker(skip_gui)
        if "cad" in item.keywords and not cad_ok:
            item.add_marker(skip_cad)


@pytest.fixture(scope="session")
def demo_piston():
    """Coarse (0.8 mm voxels, ~180k faces) demo piston shared by the tests."""
    from meshrev.core.samples import make_demo_piston

    return make_demo_piston(voxel_size=0.8)
