"""Dialogs for solid modelling commands."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
)

from meshrev.core.bodies import Body


def _combo(bodies: list[Body], preferred: str | None) -> QComboBox:
    combo = QComboBox()
    for body in bodies:
        combo.addItem(body.name, body.id)
    if preferred is not None and combo.findData(preferred) >= 0:
        combo.setCurrentIndex(combo.findData(preferred))
    return combo


class _Dialog(QDialog):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.form = QFormLayout(self)

    def finish(self) -> None:
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.form.addRow(buttons)


class BooleanDialog(_Dialog):
    OPERATIONS = (("cut", "求差（目标 − 工具）"), ("union", "求并"), ("intersect", "求交"))

    def __init__(self, solids: list[Body], preferred: str | None, parent=None) -> None:
        super().__init__("布尔运算", parent)
        self.target = _combo(solids, preferred)
        self.tool = _combo(solids, None)
        if self.tool.count() > 1 and self.tool.currentData() == self.target.currentData():
            self.tool.setCurrentIndex(1)
        self.operation = QComboBox()
        for value, text in self.OPERATIONS:
            self.operation.addItem(text, value)
        self.form.addRow("目标实体", self.target)
        self.form.addRow("工具实体", self.tool)
        self.form.addRow("运算", self.operation)
        self.finish()

    def values(self) -> tuple[str, str, str]:
        return self.target.currentData(), self.tool.currentData(), self.operation.currentData()


class PinBoreDialog(_Dialog):
    def __init__(
        self, solids: list[Body], axes: list[Body], preferred_axis: str | None, parent=None
    ) -> None:
        super().__init__("由基准轴挖孔（销孔）", parent)
        self.target = _combo(solids, None)
        self.axis = _combo(axes, preferred_axis)
        self.radius = QDoubleSpinBox(minimum=0.0, maximum=1e4, decimals=4, singleStep=0.01)
        self.radius.setSuffix(" mm")
        self.radius.setSpecialValueText("使用拟合半径")
        self.form.addRow("目标实体", self.target)
        self.form.addRow("基准轴", self.axis)
        self.form.addRow("孔半径", self.radius)
        self.finish()

    def values(self) -> tuple[str, str, float]:
        return self.target.currentData(), self.axis.currentData(), self.radius.value()
