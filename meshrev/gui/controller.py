"""Bridges the Qt-free document model to the widgets."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThreadPool, Signal, Slot

from meshrev import io as mio
from meshrev.core.bodies import Body, MeshBody
from meshrev.core.document import (
    BODY_ADDED,
    BODY_CHANGED,
    BODY_REMOVED,
    DOCUMENT_CLEARED,
    HISTORY_CHANGED,
    Document,
)
from meshrev.core.features import (
    AddFeatureCommand,
    Command,
    EditParamsCommand,
    Feature,
    FeatureContext,
    ImportFeature,
    RemoveFeatureCommand,
    SuppressFeatureCommand,
    UndoStack,
)
from meshrev.gui.workers import FunctionWorker

log = logging.getLogger(__name__)

DEMO_PISTON_LABEL = "示例活塞（程序生成）"


@dataclass(frozen=True)
class Selection:
    """What the user selected: a feature, a body, and optionally regions of a body."""

    feature_id: str | None = None
    body_id: str | None = None
    region_ids: tuple[int, ...] = ()
    region_type: str | None = None  # a whole primitive-type group of a region set

    @property
    def is_empty(self) -> bool:
        return self.feature_id is None and self.body_id is None


class DocumentController(QObject):
    bodyAdded = Signal(str)
    bodyRemoved = Signal(str)
    bodyChanged = Signal(str, str)  # body id, attribute
    historyChanged = Signal()
    documentCleared = Signal()
    selectionChanged = Signal(object)  # Selection
    busyChanged = Signal(bool, str)
    statusMessage = Signal(str)
    errorOccurred = Signal(str, str)  # title, message
    undoStateChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.document = Document()
        self.undo_stack = UndoStack()
        self.undo_stack.on_changed.append(self.undoStateChanged.emit)
        self.document.history.context_factory = self.make_context
        events = self.document.events
        events.subscribe(BODY_ADDED, lambda body_id: self.bodyAdded.emit(body_id))
        events.subscribe(BODY_REMOVED, lambda body_id: self._on_body_removed(body_id))
        events.subscribe(
            BODY_CHANGED, lambda body_id, attribute: self.bodyChanged.emit(body_id, attribute)
        )
        events.subscribe(HISTORY_CHANGED, lambda: self.historyChanged.emit())
        events.subscribe(DOCUMENT_CLEARED, lambda: self.documentCleared.emit())
        self._selection = Selection()
        self._pool = QThreadPool.globalInstance()
        self._tasks: dict[int, tuple[FunctionWorker, Callable[[Any], None], str]] = {}
        self._next_task_id = 0

    def make_context(self) -> FeatureContext:
        return FeatureContext(self.document, load_file=mio.load)

    # -- state ------------------------------------------------------------------------
    @property
    def selection(self) -> Selection:
        return self._selection

    @property
    def busy(self) -> bool:
        return bool(self._tasks)

    def select(self, selection: Selection | None) -> None:
        self._selection = selection or Selection()
        self.selectionChanged.emit(self._selection)

    def _on_body_removed(self, body_id: str) -> None:
        self.bodyRemoved.emit(body_id)
        if self._selection.body_id == body_id:
            self.select(Selection())

    # -- background tasks -------------------------------------------------------------
    def run_async(
        self,
        message: str,
        fn: Callable[[], Any],
        on_done: Callable[[Any], None],
        error_title: str = "操作失败",
    ) -> None:
        """Run ``fn`` in the thread pool, then ``on_done(result)`` on the GUI thread."""
        self._next_task_id += 1
        worker = FunctionWorker(self._next_task_id, fn)
        self._tasks[worker.task_id] = (worker, on_done, error_title)
        worker.signals.finished.connect(self._on_task_finished)
        worker.signals.failed.connect(self._on_task_failed)
        self.busyChanged.emit(True, message)
        self._pool.start(worker)

    @Slot(int, object)
    def _on_task_finished(self, task_id: int, result: Any) -> None:
        _worker, on_done, error_title = self._pop_task(task_id)
        try:
            on_done(result)
        except Exception as exc:  # noqa: BLE001
            log.exception("applying background result failed")
            self.errorOccurred.emit(error_title, str(exc))

    @Slot(int, str)
    def _on_task_failed(self, task_id: int, message: str) -> None:
        _worker, _on_done, error_title = self._pop_task(task_id)
        self.errorOccurred.emit(error_title, message)

    def _pop_task(self, task_id: int) -> tuple[FunctionWorker, Callable[[Any], None], str]:
        task = self._tasks.pop(task_id)
        if not self._tasks:
            self.busyChanged.emit(False, "")
        return task

    def wait_for_tasks(self, timeout_ms: int = 60000) -> bool:
        """Block until background tasks finished and their results were applied."""
        from PySide6.QtCore import QCoreApplication, QDeadlineTimer

        deadline = QDeadlineTimer(timeout_ms)
        while self._tasks and not deadline.hasExpired():
            self._pool.waitForDone(50)
            QCoreApplication.processEvents()
        return not self._tasks

    # -- document level ---------------------------------------------------------------
    def new_document(self) -> None:
        self.select(Selection())
        self.document.clear()
        self.undo_stack.clear()
        self.statusMessage.emit("已新建文档")

    def import_files(self, paths: Iterable[str | Path]) -> None:
        for raw in paths:
            path = Path(raw)
            self.run_async(
                f"正在读取 {path.name} …",
                lambda path=path: mio.load(path),
                lambda bodies, path=path: self._add_import(path, bodies),
                error_title=f"无法导入 {path.name}",
            )

    def _add_import(self, path: Path, bodies: list[Body]) -> None:
        feature = ImportFeature(path, bodies=bodies)
        self.push(AddFeatureCommand(self.document.history, feature))
        faces = sum(b.to_polydata().n_cells for b in bodies)
        self.statusMessage.emit(f"已导入 {path.name}：{faces:,} 个三角面")

    def load_demo_piston(self, voxel_size: float = 0.5) -> None:
        from meshrev.core.samples import make_demo_piston

        def build() -> list[Body]:
            return [MeshBody(make_demo_piston(voxel_size=voxel_size), "demo_piston")]

        self.run_async(
            "正在生成示例活塞 …",
            build,
            lambda bodies: self.push(
                AddFeatureCommand(
                    self.document.history,
                    ImportFeature(DEMO_PISTON_LABEL, bodies=bodies, name="导入 示例活塞"),
                )
            ),
        )

    def export_bodies(self, path: str | Path, body_ids: Iterable[str] | None = None) -> None:
        if body_ids is None:
            bodies = [b for b in self.document.bodies() if b.visible]
        else:
            bodies = [self.document.get(i) for i in body_ids]
        mio.save(bodies, path)
        self.statusMessage.emit(f"已导出 {len(bodies)} 个实体到 {Path(path).name}")

    # -- features ---------------------------------------------------------------------
    def push(self, command: Command) -> bool:
        try:
            self.undo_stack.push(command)
        except Exception as exc:  # noqa: BLE001
            log.exception("command failed")
            self.errorOccurred.emit(command.text or "操作失败", str(exc))
            return False
        return True

    def add_feature(self, feature: Feature, background: bool = False) -> None:
        history = self.document.history
        if not background:
            self.push(AddFeatureCommand(history, feature))
            return
        ctx = self.make_context()
        self.run_async(
            f"正在计算 {feature.name} …",
            lambda: history.evaluate(feature, ctx),
            lambda outputs: self.push(AddFeatureCommand(history, feature, outputs)),
            error_title=f"{feature.name} 失败",
        )

    def remove_feature(self, feature_id: str) -> None:
        if self._selection.feature_id == feature_id:
            self.select(Selection())
        self.push(RemoveFeatureCommand(self.document.history, feature_id))

    def set_feature_suppressed(self, feature_id: str, suppressed: bool) -> None:
        self.push(SuppressFeatureCommand(self.document.history, feature_id, suppressed))

    def update_feature_params(self, feature_id: str, params: dict[str, Any]) -> None:
        self.push(EditParamsCommand(self.document.history, feature_id, params))

    def set_body_visible(self, body_id: str, visible: bool) -> None:
        self.document.set_visible(body_id, visible)

    def undo(self) -> None:
        self._guarded(self.undo_stack.undo, "撤销失败")

    def redo(self) -> None:
        self._guarded(self.undo_stack.redo, "重做失败")

    def _guarded(self, fn: Callable[[], None], title: str) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            log.exception(title)
            self.errorOccurred.emit(title, str(exc))
