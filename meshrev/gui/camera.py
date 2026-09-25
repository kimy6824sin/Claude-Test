"""Standard CAD views and mouse interaction presets (no Qt dependency)."""

from __future__ import annotations

from enum import Enum

import numpy as np
import vtk

from meshrev.core.types import BBox, FloatArray


class StandardView(str, Enum):
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    ISOMETRIC = "isometric"

    @property
    def label(self) -> str:
        return {
            StandardView.FRONT: "前视",
            StandardView.BACK: "后视",
            StandardView.LEFT: "左视",
            StandardView.RIGHT: "右视",
            StandardView.TOP: "俯视",
            StandardView.BOTTOM: "仰视",
            StandardView.ISOMETRIC: "等轴测",
        }[self]


# direction from the focal point towards the camera, and the view-up vector (Z-up world)
_VIEW_VECTORS: dict[StandardView, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    StandardView.FRONT: ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    StandardView.BACK: ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    StandardView.LEFT: ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    StandardView.RIGHT: ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    StandardView.TOP: ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    StandardView.BOTTOM: ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
    StandardView.ISOMETRIC: ((1.0, -1.0, 1.0), (0.0, 0.0, 1.0)),
}


def view_vectors(view: StandardView) -> tuple[FloatArray, FloatArray]:
    """Unit ``(direction_to_camera, view_up)`` for a standard view."""
    direction, up = (np.asarray(v, dtype=np.float64) for v in _VIEW_VECTORS[view])
    direction /= np.linalg.norm(direction)
    # make "up" exactly orthogonal to the viewing direction
    up = up - (up @ direction) * direction
    return direction, up / np.linalg.norm(up)


def camera_for_view(
    view: StandardView, bounds: BBox | None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(position, focal_point, view_up)`` looking at ``bounds`` from ``view``."""
    center = bounds.center if bounds is not None else np.zeros(3)
    distance = max(bounds.diagonal if bounds is not None else 1.0, 1e-6) * 2.0
    direction, up = view_vectors(view)
    return center + distance * direction, center, up


class InteractionPreset(str, Enum):
    VTK_DEFAULT = "vtk"
    DESIGN_X = "design_x"

    @property
    def label(self) -> str:
        return {
            InteractionPreset.VTK_DEFAULT: "VTK 默认（左键旋转 / 中键平移 / 右键或滚轮缩放）",
            InteractionPreset.DESIGN_X: "类 Design X（右键旋转 / Ctrl+右键平移 / 滚轮缩放）",
        }[self]


class DesignXInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Right button rotates, Ctrl+right pans, Shift+right zooms; left is kept for picking.

    Registering an observer on an interactor style replaces its default handler for
    that event, so the left button no longer rotates the camera.
    """

    def __init__(self) -> None:
        super().__init__()
        self.AddObserver("LeftButtonPressEvent", self._ignore)
        self.AddObserver("LeftButtonReleaseEvent", self._ignore)
        self.AddObserver("RightButtonPressEvent", self._right_down)
        self.AddObserver("RightButtonReleaseEvent", self._right_up)

    def _ignore(self, _obj, _event) -> None:
        pass

    def _right_down(self, _obj, _event) -> None:
        interactor = self.GetInteractor()
        x, y = interactor.GetEventPosition()
        self.FindPokedRenderer(x, y)
        if self.GetCurrentRenderer() is None:
            return
        if interactor.GetControlKey():
            self.StartPan()
        elif interactor.GetShiftKey():
            self.StartDolly()
        else:
            self.StartRotate()

    def _right_up(self, _obj, _event) -> None:
        state = self.GetState()
        if state == 1:  # VTKIS_ROTATE
            self.EndRotate()
        elif state == 2:  # VTKIS_PAN
            self.EndPan()
        elif state == 4:  # VTKIS_DOLLY
            self.EndDolly()
