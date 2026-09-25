"""Feature history tree: features, their output bodies and (for region sets) regions."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QAction, QBrush, QColor
from PySide6.QtWidgets import QAbstractItemView, QMenu, QTreeWidget, QTreeWidgetItem

from meshrev.core.bodies import Body
from meshrev.core.features import Feature, FeatureState
from meshrev.gui.controller import DocumentController, Selection

ROLE_KEY = Qt.ItemDataRole.UserRole
# item keys: ("feature", fid) | ("body", bid) | ("region", bid, rid) | ("rtype", bid, type)

_STATE_COLORS = {
    FeatureState.ERROR: QColor(200, 40, 40),
    FeatureState.SUPPRESSED: QColor(140, 140, 140),
    FeatureState.ROLLED_BACK: QColor(170, 170, 170),
}


class FeatureTree(QTreeWidget):
    def __init__(self, controller: DocumentController, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setHeaderLabels(["特征树"])
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.setUniformRowHeights(True)
        self._items: dict[tuple, QTreeWidgetItem] = {}
        self._syncing = False
        self._context_actions: dict[str, Sequence[QAction]] = {}

        controller.historyChanged.connect(self.rebuild)
        controller.documentCleared.connect(self.rebuild)
        controller.bodyChanged.connect(self._on_body_changed)
        controller.selectionChanged.connect(self._sync_selection)
        self.itemSelectionChanged.connect(self._on_item_selection)
        self.itemChanged.connect(self._on_item_changed)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def set_context_actions(self, kind: str, actions: Sequence[QAction]) -> None:
        """Actions shown in the context menu for ``feature`` / ``body`` / ``region`` items."""
        self._context_actions[kind] = actions

    # -- building ---------------------------------------------------------------------
    def rebuild(self) -> None:
        expanded = {key for key, item in self._items.items() if item.isExpanded()}
        collapsed = {
            key for key, item in self._items.items() if item.childCount() and not item.isExpanded()
        }
        self._syncing = True
        self.clear()
        self._items.clear()
        document = self.controller.document
        for feature in document.history:
            f_item = self._make_item(("feature", feature.id), self._feature_text(feature))
            self._style_feature(f_item, feature)
            self.addTopLevelItem(f_item)
            for body_id in feature.output_ids:
                body = document.find(body_id)
                if body is None:
                    continue
                b_item = self._make_item(("body", body.id), body.name)
                b_item.setFlags(b_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                b_item.setCheckState(
                    0, Qt.CheckState.Checked if body.visible else Qt.CheckState.Unchecked
                )
                f_item.addChild(b_item)
                self.add_body_children(b_item, body)
            f_item.setExpanded(("feature", feature.id) not in collapsed)
        for key in expanded:
            if key in self._items:
                self._items[key].setExpanded(True)
        self._syncing = False
        self._sync_selection(self.controller.selection)

    def add_body_children(self, item: QTreeWidgetItem, body: Body) -> None:
        """Hook for bodies with sub-items (region sets add their regions here)."""
        groups = getattr(body, "region_groups", None)
        if groups is None:
            return
        for type_key, label, region_rows in groups():
            g_item = self._make_item(("rtype", body.id, type_key), f"{label} ({len(region_rows)})")
            item.addChild(g_item)
            for region_id, text, color in region_rows:
                r_item = self._make_item(("region", body.id, region_id), text)
                r_item.setForeground(0, QBrush(QColor(*color).darker(135)))
                g_item.addChild(r_item)

    def _make_item(self, key: tuple, text: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([text])
        item.setData(0, ROLE_KEY, key)
        self._items[key] = item
        return item

    @staticmethod
    def _feature_text(feature: Feature) -> str:
        suffix = {
            FeatureState.ERROR: "  ⚠",
            FeatureState.SUPPRESSED: "  (抑制)",
            FeatureState.ROLLED_BACK: "  (回滚)",
        }.get(feature.state, "")
        return feature.name + suffix

    @staticmethod
    def _style_feature(item: QTreeWidgetItem, feature: Feature) -> None:
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        if feature.state in _STATE_COLORS:
            item.setForeground(0, QBrush(_STATE_COLORS[feature.state]))
        item.setToolTip(0, feature.error or feature.label)

    # -- updates ----------------------------------------------------------------------
    def _on_body_changed(self, body_id: str, attribute: str) -> None:
        item = self._items.get(("body", body_id))
        body = self.controller.document.find(body_id)
        if item is None or body is None:
            return
        self._syncing = True
        if attribute == "visible":
            item.setCheckState(
                0, Qt.CheckState.Checked if body.visible else Qt.CheckState.Unchecked
            )
        elif attribute == "name":
            item.setText(0, body.name)
        self._syncing = False

    def _on_item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        if self._syncing:
            return
        key = item.data(0, ROLE_KEY)
        if key and key[0] == "body":
            visible = item.checkState(0) == Qt.CheckState.Checked
            self.controller.set_body_visible(key[1], visible)

    # -- selection --------------------------------------------------------------------
    def _on_item_selection(self) -> None:
        if self._syncing:
            return
        keys = [item.data(0, ROLE_KEY) for item in self.selectedItems()]
        self.controller.select(self.selection_from_keys(keys))

    def selection_from_keys(self, keys: list[tuple]) -> Selection:
        if not keys:
            return Selection()
        document = self.controller.document
        kind = keys[0][0]
        if kind == "feature":
            return Selection(feature_id=keys[0][1])
        body_id = keys[0][1]
        body = document.find(body_id)
        feature_id = body.source_feature if body is not None else None
        if kind == "body":
            return Selection(feature_id=feature_id, body_id=body_id)
        if kind == "rtype":
            ids = tuple(body.region_ids_of_type(keys[0][2])) if body is not None else ()
            return Selection(feature_id, body_id, ids, region_type=keys[0][2])
        region_ids = tuple(k[2] for k in keys if k[0] == "region" and k[1] == body_id)
        return Selection(feature_id=feature_id, body_id=body_id, region_ids=region_ids)

    def _sync_selection(self, selection: Selection) -> None:
        self._syncing = True
        self.clearSelection()
        keys: list[tuple] = []
        if selection.region_type is not None:
            keys = [("rtype", selection.body_id, selection.region_type)]
        elif selection.region_ids:
            keys = [("region", selection.body_id, rid) for rid in selection.region_ids]
        elif selection.body_id is not None:
            keys = [("body", selection.body_id)]
        elif selection.feature_id is not None:
            keys = [("feature", selection.feature_id)]
        for key in keys:
            item = self._items.get(key)
            if item is not None:
                item.setSelected(True)
                self.scrollToItem(item)
        self._syncing = False

    # -- context menu -----------------------------------------------------------------
    def _show_context_menu(self, pos: QPoint) -> None:
        item = self.itemAt(pos)
        if item is None:
            return
        key = item.data(0, ROLE_KEY)
        kind = {"feature": "feature", "body": "body"}.get(key[0], "region")
        actions = [a for a in self._context_actions.get(kind, ()) if a.isVisible()]
        if not actions:
            return
        menu = QMenu(self)
        menu.addActions(list(actions))
        menu.exec(self.viewport().mapToGlobal(pos))
