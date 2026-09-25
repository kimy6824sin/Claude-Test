import numpy as np
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


def test_mesh_sketch_workflow(window):
    from meshrev.core.bodies import SketchBody
    from meshrev.core.features import ImportFeature
    from meshrev.core.samples import make_capped_cylinder

    controller = window.controller
    controller.document.history.append(
        ImportFeature("cyl.stl", bodies=[MeshBody(make_capped_cylinder(), "cyl")])
    )
    assert controller.create_mesh_sketch({"plane": "yz"})
    assert controller.wait_for_tasks(60000)
    (sketch,) = controller.document.bodies_of_type(SketchBody)
    visual = window.viewport.scene.visual(sketch.id)
    assert visual is not None and visual.overlay  # drawn on top of the mesh
    camera = window.viewport.camera
    view = np.subtract(camera.position, camera.focal_point)
    assert abs(view[0]) / np.linalg.norm(view) == pytest.approx(1.0, abs=1e-6)  # normal to YZ
    controller.undo()
    assert not controller.document.bodies_of_type(SketchBody)
    assert window.viewport.scene.visual(sketch.id) is None


@pytest.mark.cad
def test_modeling_workflow_with_live_parameter_edit(window, qtbot):
    from meshrev.core.bodies import CadBody, DatumAxisBody, SketchBody
    from meshrev.core.features import ImportFeature
    from meshrev.core.samples import make_capped_cylinder
    from meshrev.core.types import Axis
    from meshrev.gui.controller import Selection

    c = window.controller
    c.document.history.append(
        ImportFeature("cyl.stl", bodies=[MeshBody(make_capped_cylinder(10.0, 20.0), "cyl")])
    )
    assert c.create_mesh_sketch({"plane": "xy"}, background=False)
    (sketch,) = c.document.bodies_of_type(SketchBody)
    c.select(Selection(sketch.source_feature, sketch.id))
    assert c.extrude_sketch(distance=8.0)
    (block,) = c.document.bodies_of_type(CadBody)
    axis = DatumAxisBody(Axis((0, 0, 4), (1, 0, 0)), 20.0, "hole", radius=2.0)
    c.document.add_body(axis)
    assert c.pin_bore(block.id, axis.id)
    solids = c.document.bodies_of_type(CadBody)
    result = next(b for b in solids if b.consumes)
    assert not block.visible and result.visible  # operands hidden behind the result
    volume = result.kernel.volume(result.shape)

    cylinder = next(f for f in c.document.history if f.type_name == "Cylinder")
    c.select(Selection(feature_id=cylinder.id))
    editor = window.property_panel.editor
    editor.live.setChecked(True)
    editor._widgets["radius"][1].setValue(3.0)
    qtbot.waitUntil(lambda: cylinder.params["radius"] == 3.0, timeout=3000)
    result = c.document.get(result.id)
    assert result.kernel.volume(result.shape) < volume

    c.undo()  # radius edit
    c.undo()  # boolean
    assert block.visible


def test_accuracy_analysis_heat_map(window):
    from meshrev.core.bodies import CadBody, DeviationBody
    from meshrev.core.cad import get_kernel, is_available
    from meshrev.core.features import ImportFeature
    from meshrev.core.samples import make_capped_cylinder
    from meshrev.core.types import Axis
    from meshrev.gui.scene import DEVIATION_BAR

    if not is_available():
        pytest.skip("OCP (cadquery-ocp) not installed")
    c = window.controller
    scan = MeshBody(make_capped_cylinder(9.9, 20.0, n_theta=128), "scan")
    c.document.history.append(ImportFeature("scan.stl", bodies=[scan]))
    kernel = get_kernel()
    cad = CadBody(kernel.cylinder(Axis((0, 0, 0), (0, 0, 1)), 10.0, 20.0), "cad", kernel=kernel)
    c.document.add_body(cad)
    assert c.accuracy_analysis(scan.id, cad.id, background=False, tolerance=0.05)
    (heat,) = c.document.bodies_of_type(DeviationBody)
    stats = heat.result.stats
    assert stats.max_negative == pytest.approx(-0.1, abs=0.01)  # material missing
    assert "deviation" in heat.to_polydata().point_data
    assert not scan.visible and not cad.visible and heat.visible
    plotter = window.viewport.scene.plotter
    assert DEVIATION_BAR in plotter.scalar_bars
    c.undo()
    assert scan.visible and cad.visible
    assert DEVIATION_BAR not in plotter.scalar_bars
