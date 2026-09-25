import pytest
import pyvista as pv

from meshrev.core.bodies import MeshBody

pytestmark = pytest.mark.gui


@pytest.fixture
def window(qtbot):
    from meshrev.gui.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win.show()
    yield win
    win.close()


def test_import_file_populates_tree_and_viewport(window, tmp_path, qtbot):
    from meshrev import io as mio

    path = tmp_path / "part.stl"
    mio.save([MeshBody(pv.Cylinder().triangulate(), "c")], path)
    window.import_files([str(path)])
    assert window.controller.wait_for_tasks()
    assert len(window.controller.document) == 1
    assert window.feature_tree.topLevelItemCount() == 1
    assert len(window.viewport.scene.body_ids()) == 1


def test_display_mode_actions(window):
    from meshrev.gui.display import DisplayMode

    window.controller.document.history.append(
        __import__("meshrev.core.features", fromlist=["ImportFeature"]).ImportFeature(
            "s.stl", bodies=[MeshBody(pv.Sphere(), "s")]
        )
    )
    window.mode_actions[DisplayMode.WIREFRAME].trigger()
    assert window.viewport.scene.display_mode is DisplayMode.WIREFRAME
    window.act_edges.trigger()
    assert window.viewport.scene.show_edges


def test_click_picks_body(window, qtbot):
    from meshrev.core.features import ImportFeature
    from meshrev.gui.camera import StandardView

    window.controller.document.history.append(
        ImportFeature("s.stl", bodies=[MeshBody(pv.Sphere(radius=5.0), "s")])
    )
    window.viewport.set_standard_view(StandardView.FRONT)
    window.viewport.render()
    width, height = window.viewport.render_window.GetSize()
    with qtbot.waitSignal(window.viewport.cellPicked, timeout=2000) as blocker:
        window.viewport.pick_at(width // 2, height // 2)
    body_id, cell, _additive = blocker.args
    assert body_id == window.controller.document.bodies()[0].id and cell >= 0
    assert window.controller.selection.body_id == body_id


def test_segmentation_workflow(window, qtbot):
    """Auto segment -> region highlight -> datum axis, through the real window."""
    from meshrev.core.bodies import DatumAxisBody, RegionSetBody
    from meshrev.core.features import ImportFeature
    from meshrev.core.samples import make_capped_cylinder
    from meshrev.gui.controller import Selection

    controller = window.controller
    controller.document.history.append(
        ImportFeature("cyl.stl", bodies=[MeshBody(make_capped_cylinder(), "cyl")])
    )
    mesh = controller.document.bodies()[0]
    window.act_segment.trigger()
    assert controller.wait_for_tasks(60000)
    (regions,) = controller.document.bodies_of_type(RegionSetBody)
    assert not mesh.visible  # the coloured region set replaces the plain mesh
    visual = window.viewport.scene.visual(regions.id)
    assert visual is not None and "region_rgb" in visual.mesh.cell_data

    side = next(r.id for r in regions.segmentation.regions if r.type.value == "cylinder")
    controller.select(Selection(regions.source_feature, regions.id, (side,)))
    assert "__highlight__" in window.viewport.scene.plotter.actors
    assert window.act_datum_axis.isEnabled()
    window.act_datum_axis.trigger()
    (axis,) = controller.document.bodies_of_type(DatumAxisBody)
    assert abs(axis.axis.direction[2]) == pytest.approx(1.0, abs=1e-6)
    assert window.viewport.scene.visual(axis.id).datum is not None

    window.scheme_actions["type"].trigger()
    assert regions.color_scheme == "type"

    controller.undo()  # datum axis
    controller.undo()  # segmentation
    assert not controller.document.bodies_of_type(RegionSetBody)
    assert mesh.visible
