"""Property panel: read-only info of the selection + editable feature parameters."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from meshrev.core.features import Feature, ParamSpec
from meshrev.gui.controller import DocumentController, Selection


def _format_sequence(values: Any) -> str:
    return ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in values)


class ParamEditor(QGroupBox):
    """Builds one widget per :class:`ParamSpec` and reports changed values."""

    def __init__(self, parent=None) -> None:
        super().__init__("参数", parent)
        self._form = QFormLayout()
        self._apply = QPushButton("应用并重新生成")
        self.live = QCheckBox("实时重建")
        self.live.setToolTip("参数一改动（停顿 0.4 s 后）就自动重建模型")
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self._apply.click)
        buttons = QHBoxLayout()
        buttons.addWidget(self.live)
        buttons.addStretch(1)
        buttons.addWidget(self._apply)
        layout = QVBoxLayout(self)
        layout.addLayout(self._form)
        layout.addLayout(buttons)
        self._widgets: dict[str, tuple[ParamSpec, QWidget]] = {}
        self.feature: Feature | None = None
        self.apply_button = self._apply

    def set_feature(self, feature: Feature | None) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)
        self._widgets.clear()
        self.feature = feature
        specs = feature.params_spec if feature is not None else ()
        editable = False
        for spec in specs:
            widget = self._make_widget(spec, feature.params.get(spec.name, spec.default))
            widget.setToolTip(spec.tooltip)
            widget.setEnabled(not spec.readonly)
            self._watch(widget)
            editable |= not spec.readonly
            self._form.addRow(spec.label, widget)
            self._widgets[spec.name] = (spec, widget)
        self.setVisible(bool(specs))
        self._apply.setVisible(editable)
        self.live.setVisible(editable)

    def _changed(self, *_args) -> None:
        if self.live.isChecked():
            self._timer.start()

    def _watch(self, widget: QWidget) -> None:
        for signal in ("valueChanged", "toggled", "currentIndexChanged", "editingFinished"):
            if hasattr(widget, signal):
                getattr(widget, signal).connect(self._changed)
                return

    @staticmethod
    def _make_widget(spec: ParamSpec, value: Any) -> QWidget:
        if spec.kind == "float":
            box = QDoubleSpinBox()
            box.setDecimals(spec.decimals)
            box.setRange(
                spec.minimum if spec.minimum is not None else -1e9,
                spec.maximum if spec.maximum is not None else 1e9,
            )
            box.setSingleStep(spec.step or 1.0)
            box.setSuffix(spec.suffix)
            box.setValue(float(value))
            return box
        if spec.kind == "int":
            box = QSpinBox()
            box.setRange(
                int(spec.minimum) if spec.minimum is not None else -(10**9),
                int(spec.maximum) if spec.maximum is not None else 10**9,
            )
            box.setSingleStep(int(spec.step or 1))
            box.setSuffix(spec.suffix)
            box.setValue(int(value))
            return box
        if spec.kind == "bool":
            check = QCheckBox()
            check.setChecked(bool(value))
            return check
        if spec.kind == "choice":
            combo = QComboBox()
            for data, label in spec.choices:
                combo.addItem(label, data)
            index = combo.findData(value)
            combo.setCurrentIndex(max(index, 0))
            return combo
        edit = QLineEdit(_format_sequence(value) if spec.kind in ("ints", "floats") else str(value))
        return edit

    def values(self) -> dict[str, Any]:
        """Current widget values of the editable parameters."""
        result: dict[str, Any] = {}
        for name, (spec, widget) in self._widgets.items():
            if spec.readonly:
                continue
            if isinstance(widget, QDoubleSpinBox):
                result[name] = widget.value()
            elif isinstance(widget, QSpinBox):
                result[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                result[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                result[name] = widget.currentData()
            elif isinstance(widget, QLineEdit):
                text = widget.text()
                if spec.kind in ("ints", "floats"):
                    cast = int if spec.kind == "ints" else float
                    result[name] = tuple(
                        cast(x) for x in text.replace(";", ",").split(",") if x.strip()
                    )
                else:
                    result[name] = text
        return result

    def changed_values(self) -> dict[str, Any]:
        if self.feature is None:
            return {}
        return {k: v for k, v in self.values().items() if self.feature.params.get(k) != v}


class PropertyPanel(QWidget):
    def __init__(self, controller: DocumentController, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.table = QTreeWidget()
        self.table.setHeaderLabels(["属性", "值"])
        self.table.setRootIsDecorated(False)
        self.table.setAlternatingRowColors(True)
        self.table.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.editor = ParamEditor()
        self.editor.apply_button.clicked.connect(self._apply)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.table, stretch=1)
        layout.addWidget(self.editor)
        controller.selectionChanged.connect(self.show_selection)
        controller.historyChanged.connect(self.refresh)
        controller.bodyChanged.connect(lambda *_: self.refresh())
        self.show_selection(controller.selection)

    def refresh(self) -> None:
        self.show_selection(self.controller.selection)

    def show_selection(self, selection: Selection) -> None:
        document = self.controller.document
        sections: list[tuple[str, dict[str, str]]] = []
        body = document.find(selection.body_id) if selection.body_id else None
        feature = None
        if selection.feature_id:
            try:
                feature = document.history.get(selection.feature_id)
            except KeyError:
                feature = None
        if body is not None and selection.region_ids and hasattr(body, "regions_info"):
            sections.append(("区域", body.regions_info(selection.region_ids)))
        if body is not None:
            sections.append(("实体", body.info()))
        if feature is not None:
            sections.append(("特征", feature.info()))
        self._fill(sections)
        self.editor.set_feature(feature)

    def _fill(self, sections: list[tuple[str, dict[str, str]]]) -> None:
        self.table.clear()
        for title, info in sections:
            header = QTreeWidgetItem([title, ""])
            font = header.font(0)
            font.setBold(True)
            header.setFont(0, font)
            header.setFirstColumnSpanned(True)
            self.table.addTopLevelItem(header)
            header.setFirstColumnSpanned(True)
            for key, value in info.items():
                row = QTreeWidgetItem([key, value])
                row.setToolTip(1, value)
                self.table.addTopLevelItem(row)

    def _apply(self) -> None:
        feature = self.editor.feature
        if feature is None:
            return
        try:
            changed = self.editor.changed_values()
        except ValueError as exc:
            self.controller.errorOccurred.emit("参数无效", str(exc))
            return
        if changed:
            self.controller.update_feature_params(feature.id, changed)
