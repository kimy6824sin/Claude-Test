import numpy as np
import pytest
import pyvista as pv

from meshrev.core.bodies import MeshBody
from meshrev.core.types import BBox
from meshrev.gui.camera import StandardView, camera_for_view, view_vectors
from meshrev.gui.display import DisplayMode
from meshrev.gui.scene import SceneManager


@pytest.mark.parametrize(
    ("view", "direction", "up"),
    [
        (StandardView.FRONT, (0, -1, 0), (0, 0, 1)),
        (StandardView.TOP, (0, 0, 1), (0, 1, 0)),
        (StandardView.RIGHT, (1, 0, 0), (0, 0, 1)),
        (StandardView.ISOMETRIC, np.array([1, -1, 1]) / np.sqrt(3), None),
    ],
)
def test_view_vectors(view, direction, up):
    d, u = view_vectors(view)
    np.testing.assert_allclose(d, direction, atol=1e-12)
    assert abs(d @ u) < 1e-12
    if up is not None:
        np.testing.assert_allclose(u, up, atol=1e-12)
    else:
        assert u[2] > 0.8  # iso view keeps Z pointing up on screen


def test_camera_looks_at_bounds_center():
    box = BBox((0, 0, 0), (10, 20, 30))
    position, focal, _ = camera_for_view(StandardView.FRONT, box)
    np.testing.assert_allclose(focal, box.center)
    assert position[1] < box.min[1] and position[0] == pytest.approx(box.center[0])


@pytest.fixture
def scene():
    plotter = pv.Plotter(off_screen=True, window_size=(320, 240))
    yield SceneManager(plotter)
    plotter.close()


def test_scene_display_modes(scene):
    body = MeshBody(pv.Sphere(), "s")
    scene.add_body(body)
    prop = scene.visual(body.id).surface.GetProperty()
    assert prop.GetInterpolationAsString() == "Phong"  # smooth is the default
    scene.set_display_mode(DisplayMode.SHADED)
    assert prop.GetInterpolationAsString() == "Flat"
    assert prop.GetRepresentationAsString() == "Surface"
    scene.set_display_mode(DisplayMode.WIREFRAME)
    assert prop.GetRepresentationAsString() == "Wireframe"
    scene.set_display_mode(DisplayMode.SMOOTH)
    scene.set_show_edges(True)
    assert prop.GetEdgeVisibility()


def test_scene_add_remove_visibility_and_highlight(scene):
    body = MeshBody(pv.Sphere(), "s")
    scene.add_body(body)
    actor = scene.visual(body.id).surface
    assert scene.body_for_actor(actor) == body.id
    assert scene.source_cell(body.id, 5) == 5
    scene.highlight_cells(body.id, [0, 1, 2])
    assert "__highlight__" in scene.plotter.actors
    scene.set_visible(body.id, False)
    assert not actor.GetVisibility()
    assert scene.visible_bounds() is None
    scene.remove_body(body.id)
    assert scene.body_ids() == [] and "__highlight__" not in scene.plotter.actors


def test_scene_standard_view_sets_camera(scene):
    scene.add_body(MeshBody(pv.Box(bounds=(0, 10, 0, 10, 0, 50)), "box"))
    scene.set_standard_view(StandardView.TOP)
    camera = scene.plotter.camera
    direction = np.subtract(camera.position, camera.focal_point)
    direction /= np.linalg.norm(direction)
    np.testing.assert_allclose(direction, (0, 0, 1), atol=1e-9)
    np.testing.assert_allclose(camera.up, (0, 1, 0), atol=1e-9)
    np.testing.assert_allclose(camera.focal_point, (5, 5, 25), atol=1e-6)
