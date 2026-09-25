"""Actor bookkeeping for a pyvista plotter (Qt-free, so it is testable off-screen)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv
import vtk

from meshrev.core.bodies import Body, BodyKind
from meshrev.core.mesh.processing import ORIGINAL_CELL_ID, display_mesh
from meshrev.core.types import BBox
from meshrev.gui.camera import StandardView, camera_for_view
from meshrev.gui.display import HIGHLIGHT_COLOR, DisplayMode, apply_display_mode

HIGHLIGHT_NAME = "__highlight__"
REGION_RGB = "region_rgb"
SKETCH_RGB = "sketch_rgb"
# line: blue, arc: magenta, circle: orange (indexed by SketchBody entity_type)
SKETCH_COLORS = np.array([[25, 115, 240], [210, 40, 170], [245, 140, 20]], dtype=np.uint8)


@dataclass(eq=False)
class BodyVisual:
    body_id: str
    kind: BodyKind
    actors: list[vtk.vtkProp] = field(default_factory=list)
    surface: vtk.vtkActor | None = None  # the actor display modes apply to
    mesh: pv.PolyData | None = None  # rendered dataset (carries orig_cell_id)
    datum: vtk.vtkActor | None = None  # datum geometry actor (selection highlight)
    overlay: list[vtk.vtkProp] = field(default_factory=list)  # drawn on top of everything
    base_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    _locator: vtk.vtkStaticCellLocator | None = None

    @property
    def locator(self) -> vtk.vtkStaticCellLocator | None:
        if self._locator is None and self.mesh is not None:
            self._locator = vtk.vtkStaticCellLocator()
            self._locator.SetDataSet(self.mesh)
            self._locator.BuildLocator()
        return self._locator


Builder = Callable[[Body], BodyVisual]


class SceneManager:
    """Maps document bodies to VTK actors and applies view settings."""

    def __init__(self, plotter: pv.BasePlotter) -> None:
        self.plotter = plotter
        self.display_mode = DisplayMode.SMOOTH
        self.show_edges = False
        self._visuals: dict[str, BodyVisual] = {}
        self._actor_owner: dict[int, str] = {}
        self._highlight: tuple[str, np.ndarray] | None = None
        self._selected_body: str | None = None
        self._builders: dict[BodyKind, Builder] = {
            BodyKind.MESH: self._build_surface,
            BodyKind.CAD: self._build_surface,
            BodyKind.SECTION: self._build_lines,
            BodyKind.REGIONS: self._build_regions,
            BodyKind.DATUM_AXIS: self._build_datum_axis,
            BodyKind.DATUM_PLANE: self._build_datum_plane,
            BodyKind.SKETCH: self._build_sketch,
        }

    # -- registration -----------------------------------------------------------------
    def register_builder(self, kind: BodyKind, builder: Builder) -> None:
        self._builders[kind] = builder

    def add_body(self, body: Body) -> None:
        if body.id in self._visuals:
            self.remove_body(body.id, render=False)
        builder = self._builders.get(body.kind)
        if builder is None:
            return
        visual = builder(body)
        self._visuals[body.id] = visual
        for actor in visual.actors:
            self._actor_owner[id(actor)] = body.id
            actor.SetVisibility(body.visible)
        if visual.surface is not None:
            apply_display_mode(visual.surface.GetProperty(), self.display_mode, self.show_edges)
        if self._highlight and self._highlight[0] == body.id:
            self.highlight_cells(body.id, self._highlight[1], render=False)
        if self._selected_body == body.id:
            self._style_datum(visual, selected=True)
        self.render()

    def update_body(self, body: Body) -> None:
        self.add_body(body)

    def remove_body(self, body_id: str, render: bool = True) -> None:
        visual = self._visuals.pop(body_id, None)
        if visual is None:
            return
        for actor in visual.actors:
            self._actor_owner.pop(id(actor), None)
            if actor in visual.overlay:
                self._overlay_renderer().RemoveActor(actor)
            else:
                self.plotter.remove_actor(actor, render=False)
        if self._highlight and self._highlight[0] == body_id:
            self._remove_highlight_actor()
        if render:
            self.render()

    def set_visible(self, body_id: str, visible: bool) -> None:
        visual = self._visuals.get(body_id)
        if visual is None:
            return
        for actor in visual.actors:
            actor.SetVisibility(visible)
        highlight = self.plotter.actors.get(HIGHLIGHT_NAME)
        if highlight is not None and self._highlight and self._highlight[0] == body_id:
            highlight.SetVisibility(visible)
        self.render()

    def clear(self) -> None:
        for body_id in list(self._visuals):
            self.remove_body(body_id, render=False)
        self.clear_highlight(render=False)
        self.render()

    # -- lookup -----------------------------------------------------------------------
    def visual(self, body_id: str) -> BodyVisual | None:
        return self._visuals.get(body_id)

    def body_ids(self) -> list[str]:
        return list(self._visuals)

    def body_for_actor(self, actor: vtk.vtkProp | None) -> str | None:
        return None if actor is None else self._actor_owner.get(id(actor))

    def source_cell(self, body_id: str, rendered_cell: int) -> int:
        """Map a rendered cell index back to the body's source mesh cell."""
        visual = self._visuals.get(body_id)
        if visual is None or visual.mesh is None or rendered_cell < 0:
            return -1
        ids = visual.mesh.cell_data.get(ORIGINAL_CELL_ID)
        return int(ids[rendered_cell]) if ids is not None else int(rendered_cell)

    # -- display settings -------------------------------------------------------------
    def set_display_mode(self, mode: DisplayMode) -> None:
        self.display_mode = DisplayMode(mode)
        self._restyle()

    def set_show_edges(self, show: bool) -> None:
        self.show_edges = bool(show)
        self._restyle()

    def _restyle(self) -> None:
        for visual in self._visuals.values():
            if visual.surface is not None:
                apply_display_mode(visual.surface.GetProperty(), self.display_mode, self.show_edges)
        self.render()

    # -- highlight --------------------------------------------------------------------
    def highlight_cells(
        self, body_id: str, cell_ids: Iterable[int] | np.ndarray, render: bool = True
    ) -> None:
        """Overlay the given *source* cells of ``body_id`` in the highlight color."""
        self._remove_highlight_actor()
        cells = np.unique(
            np.asarray(
                list(cell_ids) if not isinstance(cell_ids, np.ndarray) else cell_ids, dtype=np.int64
            )
        )
        self._highlight = (body_id, cells)
        visual = self._visuals.get(body_id)
        if visual is None or visual.mesh is None or cells.size == 0:
            if render:
                self.render()
            return
        source_ids = visual.mesh.cell_data.get(ORIGINAL_CELL_ID)
        rendered = np.flatnonzero(np.isin(source_ids, cells)) if source_ids is not None else cells
        faces = visual.mesh.faces.reshape(-1, 4)[rendered]
        overlay = pv.PolyData(visual.mesh.points, faces=faces.ravel())
        if "Normals" in visual.mesh.point_data:
            overlay.point_data["Normals"] = visual.mesh.point_data["Normals"]
        actor = self.plotter.add_mesh(
            overlay,
            color=HIGHLIGHT_COLOR,
            name=HIGHLIGHT_NAME,
            reset_camera=False,
            pickable=False,
            show_scalar_bar=False,
            render=False,
        )
        mapper = actor.GetMapper()
        mapper.ScalarVisibilityOff()
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-4.0, -4.0)
        prop = actor.GetProperty()
        prop.SetInterpolationToFlat()
        prop.SetAmbient(0.35)
        if visual.surface is not None:
            actor.SetVisibility(visual.surface.GetVisibility())
        if render:
            self.render()

    def set_selected_body(self, body_id: str | None) -> None:
        """Emphasise a selected datum (surfaces are highlighted via cells instead)."""
        previous, self._selected_body = self._selected_body, body_id
        for bid in {previous, body_id} - {None}:
            visual = self._visuals.get(bid)
            if visual is not None:
                self._style_datum(visual, selected=bid == body_id)
        self.render()

    @staticmethod
    def _style_datum(visual: BodyVisual, selected: bool) -> None:
        if visual.datum is None:
            return
        prop = visual.datum.GetProperty()
        prop.SetColor(*(HIGHLIGHT_COLOR if selected else visual.base_color))
        prop.SetLineWidth(7.0 if selected else 4.0)

    def clear_highlight(self, render: bool = True) -> None:
        self._highlight = None
        self._remove_highlight_actor()
        if render:
            self.render()

    def _remove_highlight_actor(self) -> None:
        if HIGHLIGHT_NAME in self.plotter.actors:
            self.plotter.remove_actor(HIGHLIGHT_NAME, render=False)

    # -- camera -----------------------------------------------------------------------
    def visible_bounds(self) -> BBox | None:
        box: BBox | None = None
        for visual in self._visuals.values():
            for actor in visual.actors:
                if not actor.GetVisibility() or not isinstance(actor, vtk.vtkActor):
                    continue
                b = BBox.from_bounds(actor.GetBounds())
                box = b if box is None else box.union(b)
        return box

    def set_standard_view(self, view: StandardView) -> None:
        position, focal, up = camera_for_view(StandardView(view), self.visible_bounds())
        self.plotter.camera_position = [tuple(position), tuple(focal), tuple(up)]
        self.fit_all()

    def fit_all(self) -> None:
        bounds = self.visible_bounds()
        if bounds is not None:
            self.plotter.reset_camera(bounds=bounds.bounds, render=False)
        else:
            self.plotter.reset_camera(render=False)
        self.render()

    def set_parallel_projection(self, enabled: bool) -> None:
        camera = self.plotter.camera
        camera.SetParallelProjection(bool(enabled))
        self.fit_all()

    def render(self) -> None:
        self.plotter.render()

    # -- builders ---------------------------------------------------------------------
    def _build_surface(self, body: Body) -> BodyVisual:
        mesh = display_mesh(body.to_polydata())
        actor = self.plotter.add_mesh(
            mesh,
            color=body.color,
            name=f"body:{body.id}",
            reset_camera=False,
            show_scalar_bar=False,
            render=False,
        )
        actor.GetMapper().ScalarVisibilityOff()
        return BodyVisual(body.id, body.kind, [actor], surface=actor, mesh=mesh)

    def _build_regions(self, body: Body) -> BodyVisual:
        """Mesh coloured per region (Design X style segment colours)."""
        mesh = display_mesh(body.to_polydata())
        mesh.cell_data[REGION_RGB] = body.face_colors()[mesh.cell_data[ORIGINAL_CELL_ID]]
        actor = self.plotter.add_mesh(
            mesh,
            scalars=REGION_RGB,
            rgb=True,
            preference="cell",
            name=f"body:{body.id}",
            reset_camera=False,
            show_scalar_bar=False,
            render=False,
        )
        mapper = actor.GetMapper()
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1.0, -1.0)
        return BodyVisual(body.id, body.kind, [actor], surface=actor, mesh=mesh)

    def _label(self, body: Body, position: np.ndarray) -> vtk.vtkActor2D:
        return self.plotter.add_point_labels(
            [position],
            [body.tag],
            font_size=12,
            text_color="black",
            shape_color=body.color,
            shape_opacity=0.55,
            show_points=False,
            always_visible=True,
            name=f"label:{body.id}",
            reset_camera=False,
            render=False,
        )

    def _build_datum_axis(self, body: Body) -> BodyVisual:
        line = body.to_polydata()
        actor = self.plotter.add_mesh(
            line,
            color=body.color,
            line_width=4,
            render_lines_as_tubes=True,
            name=f"body:{body.id}",
            reset_camera=False,
            render=False,
        )
        label = self._label(body, line.points[-1])
        return BodyVisual(
            body.id, body.kind, [actor, label], datum=actor, base_color=tuple(body.color)
        )

    def _build_datum_plane(self, body: Body) -> BodyVisual:
        plane = body.to_polydata()
        actor = self.plotter.add_mesh(
            plane,
            color=body.color,
            opacity=0.3,
            show_edges=True,
            edge_color=body.color,
            line_width=2,
            name=f"body:{body.id}",
            reset_camera=False,
            render=False,
        )
        label = self._label(body, plane.points[np.argmax(plane.points @ np.ones(3))])
        return BodyVisual(
            body.id, body.kind, [actor, label], datum=actor, base_color=tuple(body.color)
        )

    def _build_sketch(self, body: Body) -> BodyVisual:
        """Fitted sketch: entities coloured by type, vertices as dots, ASCII tag."""
        poly = body.to_polydata()
        if poly.n_points == 0:
            return BodyVisual(body.id, body.kind)
        poly.cell_data[SKETCH_RGB] = SKETCH_COLORS[poly.cell_data["entity_type"]]
        actor = self.plotter.add_mesh(
            poly,
            scalars=SKETCH_RGB,
            rgb=True,
            preference="cell",
            line_width=4,
            render_lines_as_tubes=True,
            name=f"body:{body.id}",
            reset_camera=False,
            show_scalar_bar=False,
            render=False,
        )
        ends = []
        for loop in body.result.loops:
            for entity in loop.entities:
                ends.extend([entity.start, entity.end])
        dots = self.plotter.add_mesh(
            pv.PolyData(body.result.plane.to_world(np.asarray(ends))),
            color="black",
            point_size=7,
            render_points_as_spheres=True,
            name=f"verts:{body.id}",
            reset_camera=False,
            pickable=False,
            render=False,
        )
        label = self._label(body, poly.points[int(np.argmax(poly.points @ np.ones(3)))])
        visual = BodyVisual(body.id, body.kind, [actor, dots, label], datum=None)
        self._to_overlay(visual, [actor, dots])
        return visual

    def _overlay_renderer(self) -> vtk.vtkRenderer:
        """Second render layer sharing the camera: its props are never hidden by
        the mesh (sketch curves cut through the middle of a part stay visible)."""
        if getattr(self, "_overlay", None) is None:
            window = self.plotter.render_window
            overlay = vtk.vtkRenderer()
            overlay.SetLayer(1)
            overlay.InteractiveOff()
            overlay.SetActiveCamera(self.plotter.renderer.GetActiveCamera())
            window.SetNumberOfLayers(max(2, window.GetNumberOfLayers()))
            window.AddRenderer(overlay)
            self._overlay = overlay
        return self._overlay

    def _to_overlay(self, visual: BodyVisual, actors: list[vtk.vtkProp]) -> None:
        if self.plotter.render_window is None:
            return  # closed/off-screen-less plotter: keep them in the main layer
        overlay = self._overlay_renderer()
        for actor in actors:
            self.plotter.renderer.RemoveActor(actor)
            overlay.AddActor(actor)
            visual.overlay.append(actor)

    def look_at_plane(self, plane, bounds: BBox | None = None) -> None:
        """Camera normal to a sketch plane, sketch x axis to the right."""
        u, v = plane.basis()
        center = bounds.center if bounds is not None else plane.origin
        center = center - ((center - plane.origin) @ plane.normal) * plane.normal
        distance = 2.0 * (bounds.diagonal if bounds is not None else 100.0)
        self.plotter.camera_position = [
            tuple(center + distance * plane.normal),
            tuple(center),
            tuple(v),
        ]
        if bounds is not None:
            self.plotter.reset_camera(bounds=bounds.bounds, render=False)
        self.render()

    def _build_lines(self, body: Body) -> BodyVisual:
        data = body.to_polydata()
        if data.n_points == 0:
            return BodyVisual(body.id, body.kind)
        actor = self.plotter.add_mesh(
            data,
            color=body.color,
            name=f"body:{body.id}",
            line_width=1.5,
            reset_camera=False,
            pickable=False,
            render=False,
        )
        return BodyVisual(body.id, body.kind, [actor])
