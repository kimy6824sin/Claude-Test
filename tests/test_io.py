import numpy as np
import pytest
import pyvista as pv

from meshrev import io as mio
from meshrev.core.bodies import CadBody, MeshBody


@pytest.mark.parametrize("ext", [".stl", ".obj", ".ply"])
def test_mesh_roundtrip(tmp_path, ext):
    source = MeshBody(pv.Cylinder(resolution=48).triangulate().clean(), "cyl")
    path = tmp_path / f"part{ext}"
    mio.save([source], path)
    (loaded,) = mio.load(path)
    assert isinstance(loaded, MeshBody)
    assert loaded.name == "part"
    assert loaded.polydata.n_cells == source.polydata.n_cells
    # duplicated STL/OBJ vertices are merged on import, so the mesh is closed again
    assert loaded.stats.is_closed
    np.testing.assert_allclose(loaded.polydata.bounds, source.polydata.bounds, atol=1e-5)


def test_ascii_stl(tmp_path):
    path = tmp_path / "ascii.stl"
    mio.save([MeshBody(pv.Sphere(), "s")], path, binary=False)
    assert path.read_text(errors="ignore").lstrip().startswith("solid")
    assert mio.load(path)[0].polydata.n_cells == pv.Sphere().n_cells


def test_multiple_bodies_are_merged(tmp_path):
    a = MeshBody(pv.Sphere(center=(0, 0, 0)), "a")
    b = MeshBody(pv.Sphere(center=(3, 0, 0)), "b")
    path = tmp_path / "both.stl"
    mio.save([a, b], path)
    assert mio.load(path)[0].polydata.n_cells == a.polydata.n_cells + b.polydata.n_cells


def test_unsupported_extension(tmp_path):
    bad = tmp_path / "x.xyz"
    bad.write_text("1 2 3")
    with pytest.raises(mio.UnsupportedFormatError):
        mio.load(bad)
    with pytest.raises(mio.UnsupportedFormatError):
        mio.save([MeshBody(pv.Sphere(), "s")], tmp_path / "x.dwg")


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        mio.load(tmp_path / "missing.stl")


def test_dialog_filter_lists_formats():
    read_filter = mio.dialog_filter("read")
    for ext in ("*.stl", "*.obj", "*.ply", "*.step"):
        assert ext in read_filter
    assert "*.stl" in mio.dialog_filter("write")


def test_step_rejects_meshes(tmp_path):
    with pytest.raises(ValueError):
        mio.save([MeshBody(pv.Sphere(), "s")], tmp_path / "mesh.step")


@pytest.mark.cad
def test_step_roundtrip(tmp_path):
    from meshrev.core.cad import get_kernel
    from tests.test_cad import revolved_cylinder

    kernel = get_kernel()
    path = tmp_path / "cyl.step"
    mio.save([CadBody(revolved_cylinder(kernel), "cyl", kernel=kernel)], path)
    (loaded,) = mio.load(path)
    assert isinstance(loaded, CadBody)
    assert kernel.volume(loaded.shape) == pytest.approx(np.pi * 16 * 10, rel=1e-6)
    mesh = loaded.to_polydata()
    np.testing.assert_allclose(mesh.bounds, (-4, 4, -4, 4, 0, 10), atol=0.05)  # chordal tol
