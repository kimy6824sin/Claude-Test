"""Dialog collecting the parameters of a mesh sketch."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
)


class MeshSketchDialog(QDialog):
    def __init__(self, datum_label: str | None, datum_is_axis: bool, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("网格草图")
        form = QFormLayout(self)
        self.plane = QComboBox()
        if datum_label:
            self.plane.addItem(f"所选基准：{datum_label}", "datum")
        for value, text in (("xy", "XY 平面（过网格中心）"), ("yz", "YZ 平面"), ("zx", "ZX 平面")):
            self.plane.addItem(text, value)
        self.offset = QDoubleSpinBox(minimum=-1e4, maximum=1e4, decimals=3, singleStep=0.5)
        self.offset.setSuffix(" mm")
        self.angle = QDoubleSpinBox(minimum=-360, maximum=360, decimals=1, singleStep=15)
        self.angle.setSuffix(" °")
        self.angle.setEnabled(datum_is_axis)
        self.tolerance = QDoubleSpinBox(minimum=0, maximum=10, decimals=4, singleStep=0.01)
        self.tolerance.setSuffix(" mm")
        self.tolerance.setSpecialValueText("自动")
        form.addRow("草图平面", self.plane)
        form.addRow("沿法向偏移", self.offset)
        form.addRow("绕基准轴旋转", self.angle)
        form.addRow("拟合公差", self.tolerance)
        form.addRow(QLabel("提示：先选中基准平面或基准轴，可在其上创建草图。"))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def params(self) -> dict[str, Any]:
        return {
            "plane": self.plane.currentData(),
            "offset": self.offset.value(),
            "angle_deg": self.angle.value(),
            "tolerance": self.tolerance.value(),
        }
