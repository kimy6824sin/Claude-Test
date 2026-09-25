import numpy as np
import pytest
import pyvista as pv

from meshrev.core.mesh.processing import ensure_triangles
from meshrev.core.primitives import (
    CylinderPrimitive,
    PlanePrimitive,
    PrimitiveType,
    SegmentationOptions,
    SpherePrimitive,
    auto_segment,
)
from meshrev.core.samples import PistonSpec, make_capped_cylinder


def assert_partition(result, n_faces):
    assert result.labels.shape == (n_faces,)
    all_faces = np.concatenate([r.face_ids for r in result.regions])
    assert len(all_faces) == n_faces and len(np.unique(all_faces)) == n_faces
    for region in result.regions:
        assert np.all(result.labels[region.face_ids] == region.id)
    areas = [r.area for r in result.regions]
    assert areas == sorted(areas, reverse=True)


def test_box_gives_six_planes():
    mesh = pv.Box(bounds=(0, 40, 0, 30, 0, 20), level=6).triangulate().clean()
    result = auto_segment(mesh)
    assert_partition(result, mesh.n_cells)
    assert [r.type for r in result.regions] == [PrimitiveType.PLANE] * 6
    normals = np.array([r.primitive.normal for r in result.regions])
    np.testing.assert_allclose(np.sort(np.abs(normals).max(axis=1)), 1.0, atol=1e-9)


@pytest.mark.parametrize("noise", [0.0, 0.01])
def test_capped_cylinder_segments(noise):
    mesh = make_capped_cylinder(radius=10.0, height=30.0, n_theta=160, n_z=40)
    if noise:
        rng = np.random.default_rng(0)
        mesh.points = mesh.points + rng.normal(scale=noise, size=mesh.points.shape)
    result = auto_segment(mesh)
    assert_partition(result, mesh.n_cells)
    kinds = [r.type for r in result.regions]
    assert kinds == [PrimitiveType.CYLINDER, PrimitiveType.PLANE, PrimitiveType.PLANE]
    side = result.regions[0].primitive
    assert side.radius == pytest.approx(10.0, abs=5 * noise + 1e-4)
    assert abs(side.axis_direction[2]) == pytest.approx(1.0, abs=1e-5)


def test_crease_between_two_planes():
    # a 20° crease: split by the sharp-edge rule (10°) or by curvature continuity (45°)
    x = np.linspace(-10, 10, 41)
    y = np.linspace(0, 10, 21)
    xx, yy = np.meshgrid(x, y)
    zz = np.abs(xx) * np.tan(np.radians(10))
    grid = ensure_triangles(pv.StructuredGrid(xx, yy, zz))
    for sharp in (10.0, 45.0):
        result = auto_segment(grid, SegmentationOptions(sharp_angle_deg=sharp))
        planes = [r.primitive for r in result.regions if r.type is PrimitiveType.PLANE]
        assert len(planes) == 2
        angle = np.degrees(np.arccos(abs(planes[0].normal @ planes[1].normal)))
        assert angle == pytest.approx(20.0, abs=1e-6)
        assert sum(len(r.face_ids) for r in result.regions[:2]) >= 0.9 * grid.n_cells


@pytest.fixture(scope="module")
def piston_segmentation(demo_piston):
    return auto_segment(demo_piston)


def test_demo_piston_partition_and_colors(demo_piston, piston_segmentation):
    result = piston_segmentation
    assert_partition(result, demo_piston.n_cells)
    colors = result.face_colors("type")
    assert colors.shape == (demo_piston.n_cells, 3) and colors.dtype == np.uint8
    counts = result.type_counts()
    assert counts[PrimitiveType.PLANE] >= 10
    assert counts[PrimitiveType.CYLINDER] >= 6
    assert counts[PrimitiveType.SPHERE] >= 1


def test_demo_piston_key_features(piston_segmentation):
    spec = PistonSpec()
    regions = piston_segmentation.regions
    planes = [r.primitive for r in regions if isinstance(r.primitive, PlanePrimitive)]
    cylinders = [r.primitive for r in regions if isinstance(r.primitive, CylinderPrimitive)]
    spheres = [r.primitive for r in regions if isinstance(r.primitive, SpherePrimitive)]

    def has_plane(normal, offset, tol=0.01):
        return any(
            abs(abs(p.normal @ normal) - 1) < 1e-4 and abs(abs(p.d) - offset) < tol for p in planes
        )

    assert has_plane((0, 0, 1), spec.height)  # crown / gasket side face
    assert has_plane((0, 0, 1), spec.under_crown_height)
    assert has_plane((1, 0, 0), spec.boss_half_distance)
    for z0, z1 in spec.groove_ranges:  # ring groove flanks (1.2 mm groove ~ 1.5 voxels)
        assert has_plane((0, 0, 1), z0, 0.03) and has_plane((0, 0, 1), z1, 0.03)

    skirt = regions[0].primitive  # the largest region
    assert isinstance(skirt, CylinderPrimitive) and not skirt.concave
    assert skirt.radius == pytest.approx(spec.radius, abs=0.01)

    bores = [c for c in cylinders if abs(c.radius - spec.pin_radius) < 0.02]
    assert len(bores) == 2  # one region per pin boss
    for bore in bores:
        assert bore.concave
        assert abs(bore.axis_direction[0]) == pytest.approx(1.0, abs=1e-5)
        assert bore.axis.distance([[0, 0, spec.pin_height]])[0] < 0.02  # 0.8 mm voxels

    dish = [s for s in spheres if abs(s.radius - spec.dish_radius) < 0.1]
    assert dish and dish[0].concave
    np.testing.assert_allclose(dish[0].center, (0, 0, spec.dish_center_z), atol=0.05)


def test_region_queries(piston_segmentation):
    result = piston_segmentation
    cyl_ids = [r.id for r in result.regions_of_type(PrimitiveType.CYLINDER)]
    faces = result.faces_of(cyl_ids[:2])
    assert len(faces) == sum(len(result.regions[i].face_ids) for i in cyl_ids[:2])
    info = result.regions[0].describe()
    assert info["基元类型"] == "圆柱面" and "半径" in info
