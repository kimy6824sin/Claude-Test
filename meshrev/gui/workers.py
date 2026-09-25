"""Background execution helpers (keep the UI responsive while reading big scans)."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal

log = logging.getLogger(__name__)


class WorkerSignals(QObject):
    finished = Signal(int, object)  # task id, result
    failed = Signal(int, str)  # task id, message


class FunctionWorker(QRunnable):
    """Runs ``fn()`` on a thread pool; results arrive via queued Qt signals.

    ``fn`` must not touch Qt widgets or mutate the document: compute and return,
    then apply the result in the ``finished`` slot on the GUI thread. Connect the
    signals to slots of a QObject living in the GUI thread so they are queued.
    """

    def __init__(self, task_id: int, fn: Callable[[], Any]) -> None:
        super().__init__()
        self.task_id = task_id
        self.fn = fn
        self.signals = WorkerSignals()
        self.setAutoDelete(False)

    def run(self) -> None:  # executed in a pool thread
        try:
            result = self.fn()
        except Exception as exc:  # noqa: BLE001 - reported to the UI
            log.debug("worker failed:\n%s", traceback.format_exc())
            self.signals.failed.emit(self.task_id, str(exc) or type(exc).__name__)
        else:
            self.signals.finished.emit(self.task_id, result)
