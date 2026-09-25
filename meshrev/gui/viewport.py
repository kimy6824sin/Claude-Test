"""Interactive 3D viewport (pyvistaqt ``QtInteractor``) with CAD view controls."""

from __future__ import annotations

import vtk
from PySide6.QtCore import Signal
from pyvistaqt import QtInteractor

from meshrev.core.bodies import Body
from meshrev.gui.camera import DesignXInteractorStyle, InteractionPreset, StandardView
from meshrev.gui.display import BACKGROUND_BOTTOM, BACKGROUND_TOP, DisplayMode
from meshrev.gui.scene import SceneManager

CLICK_TOLERANCE_PX = 3


class Viewport3D(QtInteractor):
    """3D view of the document.

    Mouse (VTK default preset): left drag rotates, middle drag (or Shift+left) pans,
    right drag / wheel zooms, Ctrl+left spins. A left *click* (no drag) picks.
    """

    cellPicked = Signal(str, int, bool)  # body id, source cell id (-1 if n/a), additive
    backgroundClicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent, auto_update=False)
        self.scene = SceneManager(self)
        self.set_background(BACKGROUND_BOTTOM, top=BACKGROUND_TOP)
        self.add_axes(interactive=False, line_width=2, color="black")
        self.camera.SetParallelProjection(True)
        self.interaction_preset = InteractionPreset.VTK_DEFAULT
        self._press_position: tuple[int, int] | None = None
        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.004)
        self.iren.add_observer("LeftButtonPressEvent", self._on_left_press)
        self.iren.add_observer("LeftButtonReleaseEvent", self._on_left_release)
        self.set_interaction_preset(InteractionPreset.VTK_DEFAULT)

    # -- body management (delegates to the Qt-free scene) -------------------------------
    def add_body(self, body: Body) -> None:
        self.scene.add_body(body)

    def update_body(self, body: Body) -> None:
        self.scene.update_body(body)

    def remove_body(self, body_id: str) -> None:
        self.scene.remove_body(body_id)

    def set_body_visible(self, body_id: str, visible: bool) -> None:
        self.scene.set_visible(body_id, visible)

    # -- view controls ----------------------------------------------------------------
    def set_display_mode(self, mode: DisplayMode) -> None:
        self.scene.set_display_mode(mode)

    def set_show_edges(self, show: bool) -> None:
        self.scene.set_show_edges(show)

    def set_standard_view(self, view: StandardView) -> None:
        self.scene.set_standard_view(view)

    def fit_all(self) -> None:
        self.scene.fit_all()

    def set_parallel_projection(self, enabled: bool) -> None:
        self.scene.set_parallel_projection(enabled)

    def set_interaction_preset(self, preset: InteractionPreset) -> None:
        self.interaction_preset = InteractionPreset(preset)
        if self.interaction_preset is InteractionPreset.DESIGN_X:
            style = DesignXInteractorStyle()
            style.SetDefaultRenderer(self.renderer)
            self.iren.interactor.SetInteractorStyle(style)
            self._style = style  # keep a Python reference alive
        else:
            self.enable_trackball_style()

    # -- picking ----------------------------------------------------------------------
    def _on_left_press(self, _obj, _event) -> None:
        self._press_position = self.iren.interactor.GetEventPosition()

    def _on_left_release(self, _obj, _event) -> None:
        if self._press_position is None:
            return
        x0, y0 = self._press_position
        x, y = self.iren.interactor.GetEventPosition()
        self._press_position = None
        if abs(x - x0) > CLICK_TOLERANCE_PX or abs(y - y0) > CLICK_TOLERANCE_PX:
            return  # it was a drag (camera move), not a click
        additive = bool(self.iren.interactor.GetControlKey())
        self.pick_at(x, y, additive)

    def pick_at(self, x: int, y: int, additive: bool = False) -> str | None:
        picker = self._picker
        picker.InitializePickList()
        picker.RemoveAllLocators()
        for body_id in self.scene.body_ids():
            visual = self.scene.visual(body_id)
            if visual is not None and visual.locator is not None:
                picker.AddLocator(visual.locator)
        picker.Pick(x, y, 0, self.renderer)
        body_id = self.scene.body_for_actor(picker.GetActor())
        if body_id is None:
            if not additive:
                self.backgroundClicked.emit()
            return None
        self.cellPicked.emit(body_id, self.scene.source_cell(body_id, picker.GetCellId()), additive)
        return body_id
