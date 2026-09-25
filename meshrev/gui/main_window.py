"""Main application window (single document)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QStyle,
    QToolBar,
)

from meshrev import __version__
from meshrev import io as mio
from meshrev.core.bodies import BodyKind, DatumAxisBody, SketchBody
from meshrev.gui.camera import InteractionPreset, StandardView
from meshrev.gui.controller import DocumentController, Selection
from meshrev.gui.display import DisplayMode
from meshrev.gui.feature_tree import FeatureTree
from meshrev.gui.modeling_dialogs import BooleanDialog, PinBoreDialog
from meshrev.gui.property_panel import PropertyPanel
from meshrev.gui.sketch_dialog import MeshSketchDialog
from meshrev.gui.viewport import Viewport3D

MAX_RECENT = 10

VIEW_SHORTCUTS = {
    StandardView.FRONT: "Ctrl+1",
    StandardView.BACK: "Ctrl+2",
    StandardView.LEFT: "Ctrl+3",
    StandardView.RIGHT: "Ctrl+4",
    StandardView.TOP: "Ctrl+5",
    StandardView.BOTTOM: "Ctrl+6",
    StandardView.ISOMETRIC: "Ctrl+0",
}

MOUSE_HELP = """<b>鼠标操作（当前预设可在 视图 → 鼠标操作 中切换）</b>
<table cellpadding=3>
<tr><th align=left>操作</th><th align=left>VTK 默认</th><th align=left>类 Design X</th></tr>
<tr><td>旋转</td><td>左键拖动</td><td>右键拖动</td></tr>
<tr><td>平移</td><td>中键拖动 / Shift+左键</td><td>Ctrl+右键 / 中键拖动</td></tr>
<tr><td>缩放</td><td>滚轮 / 右键拖动</td><td>滚轮 / Shift+右键</td></tr>
<tr><td>选择</td><td>左键单击（Ctrl 追加）</td><td>左键单击（Ctrl 追加）</td></tr>
</table>"""


class MainWindow(QMainWindow):
    def __init__(self, controller: DocumentController | None = None) -> None:
        super().__init__()
        self.controller = controller or DocumentController(self)
        self.settings = QSettings("meshrev", "meshrev")
        self.viewport = Viewport3D(self)
        self.setCentralWidget(self.viewport)
        self.feature_tree = FeatureTree(self.controller)
        self.property_panel = PropertyPanel(self.controller)
        self.tree_dock = self._make_dock(
            "特征树", self.feature_tree, Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.property_dock = self._make_dock(
            "属性", self.property_panel, Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._build_actions()
        self._build_menus()
        self._build_toolbars()
        self._build_statusbar()
        self._connect()
        self.setAcceptDrops(True)
        self.resize(1440, 900)
        self._update_title()
        self._update_undo_actions()
        self._on_selection_changed(self.controller.selection)

    # -- construction -----------------------------------------------------------------
    def _make_dock(self, title: str, widget, area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{title}")
        dock.setWidget(widget)
        dock.setMinimumWidth(260)
        self.addDockWidget(area, dock)
        return dock

    def _action(
        self,
        text: str,
        slot=None,
        shortcut: str | None = None,
        icon=None,
        checkable: bool = False,
        tip: str = "",
    ) -> QAction:
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(self.style().standardIcon(icon))
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.setCheckable(checkable)
        if tip:
            action.setStatusTip(tip)
            action.setToolTip(tip)
        if slot is not None:
            action.triggered.connect(slot)
        return action

    def _build_actions(self) -> None:
        sp = QStyle.StandardPixmap
        c = self.controller
        self.act_new = self._action("新建(&N)", self._new_document, "Ctrl+N", sp.SP_FileIcon)
        self.act_import = self._action(
            "导入(&I)…",
            self._import_dialog,
            "Ctrl+O",
            sp.SP_DialogOpenButton,
            tip="导入 STL/OBJ/PLY/STEP",
        )
        self.act_demo = self._action("打开示例活塞", lambda: c.load_demo_piston())
        self.act_export = self._action(
            "导出(&E)…", self._export_dialog, "Ctrl+E", sp.SP_DialogSaveButton, tip="导出可见实体"
        )
        self.act_quit = self._action("退出(&Q)", self.close, "Ctrl+Q")
        self.act_undo = self._action("撤销", c.undo, "Ctrl+Z", sp.SP_ArrowBack)
        self.act_redo = self._action("重做", c.redo, "Ctrl+Y", sp.SP_ArrowForward)
        self.act_delete = self._action(
            "删除特征", self._delete_selected_feature, "Delete", sp.SP_TrashIcon
        )
        self.act_suppress = self._action("抑制 / 取消抑制", self._toggle_suppress)
        self.act_toggle_visible = self._action("显示 / 隐藏", self._toggle_selected_visibility, "H")
        self.act_export_body = self._action("导出所选实体…", self._export_selected)

        self.mode_group = QActionGroup(self)
        self.mode_actions: dict[DisplayMode, QAction] = {}
        for mode, key in (
            (DisplayMode.WIREFRAME, "F5"),
            (DisplayMode.SHADED, "F6"),
            (DisplayMode.SMOOTH, "F7"),
        ):
            act = self._action(
                mode.label,
                lambda _=False, m=mode: self.set_display_mode(m),
                key,
                checkable=True,
                tip=f"{mode.label}显示",
            )
            self.mode_group.addAction(act)
            self.mode_actions[mode] = act
        self.mode_actions[DisplayMode.SMOOTH].setChecked(True)
        self.act_edges = self._action(
            "显示网格边", self.viewport.set_show_edges, "F8", checkable=True
        )
        self.view_actions: dict[StandardView, QAction] = {
            view: self._action(
                view.label,
                lambda _=False, v=view: self.viewport.set_standard_view(v),
                VIEW_SHORTCUTS[view],
                tip=f"{view.label} ({VIEW_SHORTCUTS[view]})",
            )
            for view in StandardView
        }
        self.act_fit = self._action(
            "适应窗口", self.viewport.fit_all, "F", tip="缩放到全部可见实体"
        )
        self.act_parallel = self._action(
            "正交投影", self.viewport.set_parallel_projection, checkable=True
        )
        self.act_parallel.setChecked(True)
        self.preset_group = QActionGroup(self)
        self.preset_actions: dict[InteractionPreset, QAction] = {}
        for preset in InteractionPreset:
            act = self._action(
                preset.label,
                lambda _=False, p=preset: self._set_interaction_preset(p),
                checkable=True,
            )
            self.preset_group.addAction(act)
            self.preset_actions[preset] = act
        self.act_segment = self._action(
            "自动分割",
            lambda: c.auto_segment(),
            "Ctrl+Shift+A",
            tip="按法向连续性与曲率变化把网格分割为平面/圆柱/球面/自由曲面区域",
        )
        self.act_ransac = self._action(
            "RANSAC 基元提取",
            lambda: c.detect_primitives(),
            "Ctrl+Shift+R",
            tip="基于法向与 RANSAC 提取平面和圆柱，圆柱轴线生成基准轴",
        )
        self.act_datum_axis = self._action(
            "创建基准轴",
            lambda: c.create_datum("axis"),
            "Ctrl+Shift+X",
            tip="用所选区域（可多选同轴区域，如两侧销孔）拟合圆柱并生成基准轴",
        )
        self.act_datum_plane = self._action(
            "创建基准平面",
            lambda: c.create_datum("plane"),
            "Ctrl+Shift+P",
            tip="用所选区域拟合平面并生成基准平面",
        )
        self.act_mesh_sketch = self._action(
            "网格草图…",
            self._mesh_sketch_dialog,
            "Ctrl+Shift+K",
            tip="平面截取网格并自动拟合直线/圆弧/圆，可在所选基准面或基准轴上创建",
        )
        self.act_look_sketch = self._action("正视于草图", self._look_at_selected_sketch)
        self.act_extrude = self._action(
            "拉伸",
            lambda: c.extrude_sketch(),
            "Ctrl+Shift+E",
            tip="将所选草图的闭合区域沿法向拉伸成实体",
        )
        self.act_revolve = self._action(
            "旋转",
            lambda: c.revolve_sketch(),
            "Ctrl+Shift+V",
            tip="取所选（过基准轴的）草图的半截面，绕草图 u 轴旋转 360°",
        )
        self.act_cylinder = self._action("圆柱（由基准轴）", lambda: c.cylinder_from_axis())
        self.act_boolean = self._action("布尔运算…", self._boolean_dialog, "Ctrl+Shift+B")
        self.act_pin_bore = self._action(
            "由基准轴挖孔…",
            self._pin_bore_dialog,
            tip="圆柱 + 求差，孔半径可在圆柱特征中修改并自动重建",
        )
        self.act_export_cad = self._action("导出 CAD（STEP / IGES）…", self._export_cad_dialog)
        self.scheme_group = QActionGroup(self)
        self.scheme_actions: dict[str, QAction] = {}
        for scheme, text in (("region", "按区域着色"), ("type", "按基元类型着色")):
            act = self._action(
                text, lambda _=False, sc=scheme: c.set_region_color_scheme(sc), checkable=True
            )
            self.scheme_group.addAction(act)
            self.scheme_actions[scheme] = act
        self.scheme_actions["region"].setChecked(True)
        self.act_mouse_help = self._action("鼠标操作说明", self._show_mouse_help)
        self.act_about = self._action("关于", self._show_about)

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("文件(&F)")
        file_menu.addActions([self.act_new, self.act_import, self.act_demo])
        self.recent_menu = file_menu.addMenu("最近打开")
        self._refresh_recent_menu()
        file_menu.addSeparator()
        file_menu.addAction(self.act_export)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        edit_menu = bar.addMenu("编辑(&E)")
        edit_menu.addActions([self.act_undo, self.act_redo])
        edit_menu.addSeparator()
        edit_menu.addActions([self.act_delete, self.act_suppress, self.act_toggle_visible])

        view_menu = bar.addMenu("视图(&V)")
        view_menu.addActions(list(self.mode_actions.values()))
        view_menu.addAction(self.act_edges)
        view_menu.addSeparator()
        views = view_menu.addMenu("标准视角")
        views.addActions(list(self.view_actions.values()))
        view_menu.addActions([self.act_fit, self.act_parallel])
        view_menu.addSeparator()
        mouse = view_menu.addMenu("鼠标操作")
        mouse.addActions(list(self.preset_actions.values()))
        view_menu.addSeparator()
        view_menu.addActions(
            [self.tree_dock.toggleViewAction(), self.property_dock.toggleViewAction()]
        )
        self.view_menu = view_menu

        self.tools_menu = bar.addMenu("识别(&R)")
        self.tools_menu.addActions([self.act_segment, self.act_ransac])
        self.tools_menu.addSeparator()
        self.tools_menu.addActions([self.act_datum_axis, self.act_datum_plane])
        self.tools_menu.addSeparator()
        self.tools_menu.addActions([self.act_mesh_sketch, self.act_look_sketch])
        self.tools_menu.addSeparator()
        self.tools_menu.addActions(list(self.scheme_actions.values()))

        model_menu = bar.addMenu("建模(&M)")
        model_menu.addActions([self.act_extrude, self.act_revolve])
        model_menu.addSeparator()
        model_menu.addActions([self.act_cylinder, self.act_pin_bore, self.act_boolean])
        model_menu.addSeparator()
        model_menu.addAction(self.act_export_cad)
        self.model_menu = model_menu

        help_menu = bar.addMenu("帮助(&H)")
        help_menu.addActions([self.act_mouse_help, self.act_about])

        self.feature_tree.set_context_actions("feature", [self.act_suppress, self.act_delete])
        self.feature_tree.set_context_actions(
            "body",
            [
                self.act_toggle_visible,
                self.act_export_body,
                self.act_segment,
                self.act_ransac,
                self.act_mesh_sketch,
                self.act_look_sketch,
            ],
        )
        self.feature_tree.set_context_actions("region", [self.act_datum_axis, self.act_datum_plane])

    def _build_toolbars(self) -> None:
        file_bar = QToolBar("文件", self)
        file_bar.setObjectName("toolbar_file")
        file_bar.addActions([self.act_import, self.act_export])
        file_bar.addSeparator()
        file_bar.addActions([self.act_undo, self.act_redo])
        self.addToolBar(file_bar)

        view_bar = QToolBar("视图", self)
        view_bar.setObjectName("toolbar_view")
        view_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        view_bar.addActions(list(self.mode_actions.values()))
        view_bar.addAction(self.act_edges)
        view_bar.addSeparator()
        view_bar.addActions(list(self.view_actions.values()))
        view_bar.addAction(self.act_fit)
        self.addToolBar(view_bar)
        self.view_toolbar = view_bar

        tools_bar = QToolBar("识别", self)
        tools_bar.setObjectName("toolbar_recognition")
        tools_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        tools_bar.addActions(
            [
                self.act_segment,
                self.act_ransac,
                self.act_datum_axis,
                self.act_datum_plane,
                self.act_mesh_sketch,
            ]
        )
        self.addToolBar(tools_bar)
        self.tools_toolbar = tools_bar

        model_bar = QToolBar("建模", self)
        model_bar.setObjectName("toolbar_modeling")
        model_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        model_bar.addActions(
            [self.act_extrude, self.act_revolve, self.act_pin_bore, self.act_boolean]
        )
        self.addToolBar(model_bar)

    def _build_statusbar(self) -> None:
        status = self.statusBar()
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(160)
        self.progress.setVisible(False)
        self.stats_label = QLabel()
        status.addPermanentWidget(self.progress)
        status.addPermanentWidget(self.stats_label)

    def _connect(self) -> None:
        c = self.controller
        c.bodyAdded.connect(self._on_body_added)
        c.bodyRemoved.connect(self._on_body_removed)
        c.bodyChanged.connect(self._on_body_changed)
        c.selectionChanged.connect(self._on_selection_changed)
        c.busyChanged.connect(self._on_busy)
        c.progressChanged.connect(self._on_progress)
        c.statusMessage.connect(lambda text: self.statusBar().showMessage(text, 8000))
        c.errorOccurred.connect(lambda title, text: QMessageBox.warning(self, title, text))
        c.undoStateChanged.connect(self._update_undo_actions)
        c.historyChanged.connect(self._update_undo_actions)
        self.viewport.cellPicked.connect(self._on_cell_picked)
        self.viewport.backgroundClicked.connect(lambda: c.select(Selection()))
        preset = InteractionPreset(
            self.settings.value("interaction", InteractionPreset.VTK_DEFAULT.value)
        )
        self._set_interaction_preset(preset)

    # -- document -> view ---------------------------------------------------------------------
    def _on_body_added(self, body_id: str) -> None:
        body = self.controller.document.get(body_id)
        first = len(self.viewport.scene.body_ids()) == 0
        self.viewport.add_body(body)
        if first:
            self.viewport.set_standard_view(StandardView.ISOMETRIC)
        elif isinstance(body, SketchBody):
            self._look_at_sketch(body)
        self._update_stats()

    def _mesh_sketch_dialog(self) -> None:
        datum = self.controller.selected_datum()
        dialog = MeshSketchDialog(
            datum.name if datum else None, isinstance(datum, DatumAxisBody), self
        )
        if dialog.exec():
            self.controller.create_mesh_sketch(dialog.params())

    def _cad_bodies(self) -> list:
        return [b for b in self.controller.document.bodies() if b.kind is BodyKind.CAD]

    def _boolean_dialog(self) -> None:
        solids = self._cad_bodies()
        if len(solids) < 2:
            QMessageBox.information(self, "布尔运算", "需要至少两个 CAD 实体")
            return
        dialog = BooleanDialog(solids, self.controller.selection.body_id, self)
        if dialog.exec():
            self.controller.boolean(*dialog.values())

    def _pin_bore_dialog(self) -> None:
        solids = self._cad_bodies()
        axes = self.controller.document.bodies_of_type(DatumAxisBody)
        if not solids or not axes:
            QMessageBox.information(self, "挖孔", "需要一个 CAD 实体和一个基准轴")
            return
        dialog = PinBoreDialog(solids, axes, self.controller.selection.body_id, self)
        if dialog.exec():
            target, axis, radius = dialog.values()
            self.controller.pin_bore(target, axis, radius)

    def _export_cad_dialog(self) -> None:
        solids = [b for b in self._cad_bodies() if b.visible] or self._cad_bodies()
        if not solids:
            QMessageBox.information(self, "导出 CAD", "文档中没有 CAD 实体")
            return
        start = self.settings.value("last_dir", str(Path.home()))
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 CAD", start, "STEP (*.step *.stp);;IGES (*.iges *.igs)"
        )
        if path:
            try:
                self.controller.export_bodies(path, [b.id for b in solids])
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "导出失败", str(exc))

    def _look_at_sketch(self, body: SketchBody) -> None:
        box = body.bounds()
        self.viewport.scene.look_at_plane(body.result.plane, box)

    def _look_at_selected_sketch(self) -> None:
        document = self.controller.document
        body = document.find(self.controller.selection.body_id or "")
        sketches = document.bodies_of_type(SketchBody)
        target = body if isinstance(body, SketchBody) else (sketches[-1] if sketches else None)
        if target is not None:
            self._look_at_sketch(target)

    def _on_body_removed(self, body_id: str) -> None:
        self.viewport.remove_body(body_id)
        self._update_stats()

    def _on_body_changed(self, body_id: str, attribute: str) -> None:
        body = self.controller.document.find(body_id)
        if body is None:
            return
        if attribute == "visible":
            self.viewport.set_body_visible(body_id, body.visible)
        elif attribute != "name":
            self.viewport.update_body(body)
        self._update_stats()

    def _on_selection_changed(self, selection: Selection) -> None:
        scene = self.viewport.scene
        body = self.controller.document.find(selection.body_id) if selection.body_id else None
        cells = None
        if body is not None and selection.region_ids and hasattr(body, "faces_of_regions"):
            cells = body.faces_of_regions(selection.region_ids)
        if cells is not None and len(cells):
            scene.highlight_cells(body.id, cells)
        else:
            scene.clear_highlight()
        scene.set_selected_body(selection.body_id)
        has_regions = bool(selection.region_ids)
        self.act_datum_axis.setEnabled(has_regions)
        self.act_datum_plane.setEnabled(has_regions)

    def _on_cell_picked(self, body_id: str, cell_id: int, additive: bool) -> None:
        document = self.controller.document
        body = document.find(body_id)
        if body is None:
            return
        region_of_face = getattr(body, "region_of_face", None)
        if region_of_face is not None and cell_id >= 0:
            region = region_of_face(cell_id)
            current = self.controller.selection
            regions = (region,)
            if additive and current.body_id == body_id and current.region_type is None:
                regions = tuple(r for r in current.region_ids if r != region)
                if region not in current.region_ids:
                    regions += (region,)
            self.controller.select(Selection(body.source_feature, body_id, regions))
            return
        self.controller.select(Selection(feature_id=body.source_feature, body_id=body_id))

    def _update_stats(self) -> None:
        bodies = self.controller.document.bodies()
        faces = sum(
            b.to_polydata().n_cells
            for b in bodies
            if b.visible and b.kind in (BodyKind.MESH, BodyKind.CAD, BodyKind.REGIONS)
        )
        self.stats_label.setText(f"实体 {len(bodies)} · 可见三角面 {faces:,}")

    def _on_busy(self, busy: bool, message: str) -> None:
        self.progress.setRange(0, 0)
        self.progress.setVisible(busy)
        if busy:
            self.statusBar().showMessage(message)
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        else:
            QApplication.restoreOverrideCursor()
            self.statusBar().clearMessage()

    def _on_progress(self, fraction: float, message: str) -> None:
        if not self.progress.isVisible():
            return
        self.progress.setRange(0, 100)
        self.progress.setValue(int(round(100 * fraction)))
        if message:
            self.statusBar().showMessage(message)

    def _update_undo_actions(self) -> None:
        stack = self.controller.undo_stack
        self.act_undo.setEnabled(stack.can_undo)
        self.act_redo.setEnabled(stack.can_redo)
        self.act_undo.setText(f"撤销 {stack.undo_text}" if stack.can_undo else "撤销")
        self.act_redo.setText(f"重做 {stack.redo_text}" if stack.can_redo else "重做")

    def _update_title(self) -> None:
        self.setWindowTitle(f"{self.controller.document.name} - MeshRev {__version__}")

    def set_display_mode(self, mode: DisplayMode) -> None:
        self.viewport.set_display_mode(mode)
        self.mode_actions[DisplayMode(mode)].setChecked(True)

    # -- commands ---------------------------------------------------------------------
    def _new_document(self) -> None:
        if (
            len(self.controller.document.history)
            and QMessageBox.question(self, "新建文档", "放弃当前文档的全部内容？")
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.controller.new_document()
        self.viewport.scene.clear()

    def _import_dialog(self) -> None:
        start = self.settings.value("last_dir", str(Path.home()))
        paths, _ = QFileDialog.getOpenFileNames(self, "导入", start, mio.dialog_filter("read"))
        if paths:
            self.import_files(paths)

    def import_files(self, paths: list[str]) -> None:
        self.settings.setValue("last_dir", str(Path(paths[0]).parent))
        for path in paths:
            self._add_recent(path)
        self.controller.import_files(paths)

    def _export_dialog(self, body_ids: list[str] | None = None) -> None:
        start = self.settings.value("last_dir", str(Path.home()))
        path, _ = QFileDialog.getSaveFileName(self, "导出", start, mio.dialog_filter("write"))
        if not path:
            return
        try:
            self.controller.export_bodies(path, body_ids)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "导出失败", str(exc))

    def _export_selected(self) -> None:
        body_id = self.controller.selection.body_id
        if body_id:
            self._export_dialog([body_id])

    def _delete_selected_feature(self) -> None:
        feature_id = self.controller.selection.feature_id
        if feature_id:
            self.controller.remove_feature(feature_id)

    def _toggle_suppress(self) -> None:
        feature_id = self.controller.selection.feature_id
        if feature_id:
            feature = self.controller.document.history.get(feature_id)
            self.controller.set_feature_suppressed(feature_id, not feature.suppressed)

    def _toggle_selected_visibility(self) -> None:
        body_id = self.controller.selection.body_id
        body = self.controller.document.find(body_id) if body_id else None
        if body is not None:
            self.controller.set_body_visible(body.id, not body.visible)

    def _set_interaction_preset(self, preset: InteractionPreset) -> None:
        self.viewport.set_interaction_preset(preset)
        self.preset_actions[preset].setChecked(True)
        self.settings.setValue("interaction", preset.value)

    def _show_mouse_help(self) -> None:
        QMessageBox.information(self, "鼠标操作", MOUSE_HELP)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "关于 MeshRev",
            f"<b>MeshRev {__version__}</b><br>活塞发动机零件网格逆向建模工具<br>"
            "PySide6 + PyVista/VTK (+ OpenCASCADE)",
        )

    # -- recent files -----------------------------------------------------------------
    def _recent_files(self) -> list[str]:
        value = self.settings.value("recent_files", [])
        if isinstance(value, str):
            value = [value]
        return [str(v) for v in value or []]

    def _add_recent(self, path: str) -> None:
        files = [p for p in self._recent_files() if p != path]
        files.insert(0, path)
        self.settings.setValue("recent_files", files[:MAX_RECENT])
        self._refresh_recent_menu()

    def _refresh_recent_menu(self) -> None:
        menu: QMenu = self.recent_menu
        menu.clear()
        files = self._recent_files()
        for path in files:
            menu.addAction(path, lambda p=path: self.import_files([p]))
        menu.setEnabled(bool(files))

    # -- drag & drop ------------------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt API
        if any(self._droppable(url.toLocalFile()) for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt API
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        paths = [p for p in paths if self._droppable(p)]
        if paths:
            self.import_files(paths)
            event.acceptProposedAction()

    @staticmethod
    def _droppable(path: str) -> bool:
        try:
            mio.reader_for(path)
        except mio.UnsupportedFormatError:
            return False
        return True

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self.viewport.close()
        super().closeEvent(event)
