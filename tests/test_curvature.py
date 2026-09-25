import numpy as np
import pytest
import pyvista as pv

from meshrev.core.mesh.curvature import face_curvature
from meshrev.core.mesh.topology import MeshGeometry
from meshrev.core.samples import make_capped_cylinder


def test_sphere_curvature():
    geom = MeshGeometry.from_polydata(pv.Icosphere(radius=10.0, nsub=4))
    curv = face_curvature(geom)
    assert np.median(curv.k_max) == pytest.approx(0.1, rel=0.01)
    assert np.median(curv.k_min) == pytest.approx(0.1, rel=0.02)
    assert np.median(curv.gaussian) == pytest.approx(0.01, rel=0.03)


def test_cylinder_curvature_and_sign():
    mesh = make_capped_cylinder(radius=5.0, height=30.0, n_theta=128, n_z=40)
    geom = MeshGeometry.from_polydata(mesh)
    side = np.abs(geom.face_normals[:, 2]) < 0.1
    middle = side & (np.abs(geom.face_centroids[:, 2]) < 10)
    curv = face_curvature(geom)
    assert np.median(curv.k_max[middle]) == pytest.approx(0.2, rel=0.01)
    assert np.median(np.abs(curv.k_min[middle])) < 1e-6
    # caps are planar, and the sharp rim does not bleed into them
    cap = ~side & (np.hypot(*geom.face_centroids[:, :2].T) < 3.0)
    assert np.max(curv.abs_max[cap]) < 1e-9

    flipped = MeshGeometry.from_polydata(mesh.flip_faces())  # inward normals = a bore
    curv_hole = face_curvature(flipped)
    assert np.median(curv_hole.k_min[middle]) == pytest.approx(-0.2, rel=0.01)


def test_sharp_edges_are_excluded():
    geom = MeshGeometry.from_polydata(pv.Box(level=4).triangulate())
    curv = face_curvature(geom, sharp_angle_deg=30.0)
    assert np.max(curv.abs_max) < 1e-9
    assert not curv.smooth_pairs.all()  # the 12 box edges are sharp


def test_topology_adjacency():
    geom = MeshGeometry.from_polydata(pv.Box(level=2).triangulate().clean())
    assert len(geom.face_pairs) == geom.n_faces * 3 // 2  # closed manifold
    assert np.all(geom.pair_edge_lengths > 0)
    assert geom.components().max() == 0
    sharp = geom.pair_normal_cos < 0.5
    assert geom.components(~sharp).max() + 1 == 6
