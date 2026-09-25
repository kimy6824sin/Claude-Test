"""Undoable commands operating on a :class:`FeatureHistory`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from meshrev.core.features.base import Feature

if TYPE_CHECKING:
    from meshrev.core.bodies import Body
    from meshrev.core.features.history import FeatureHistory


class Command(ABC):
    text: str = ""

    @abstractmethod
    def do(self) -> None: ...

    @abstractmethod
    def undo(self) -> None: ...


class UndoStack:
    def __init__(self, limit: int = 100) -> None:
        self._done: list[Command] = []
        self._undone: list[Command] = []
        self._limit = limit
        self.on_changed: list[Callable[[], None]] = []

    def push(self, command: Command) -> None:
        command.do()
        self._done.append(command)
        del self._done[: -self._limit]
        self._undone.clear()
        self._notify()

    def undo(self) -> None:
        if self._done:
            command = self._done.pop()
            command.undo()
            self._undone.append(command)
            self._notify()

    def redo(self) -> None:
        if self._undone:
            command = self._undone.pop()
            command.do()
            self._done.append(command)
            self._notify()

    def clear(self) -> None:
        self._done.clear()
        self._undone.clear()
        self._notify()

    @property
    def can_undo(self) -> bool:
        return bool(self._done)

    @property
    def can_redo(self) -> bool:
        return bool(self._undone)

    @property
    def undo_text(self) -> str:
        return self._done[-1].text if self._done else ""

    @property
    def redo_text(self) -> str:
        return self._undone[-1].text if self._undone else ""

    def _notify(self) -> None:
        for callback in list(self.on_changed):
            callback()


class AddFeatureCommand(Command):
    def __init__(
        self, history: FeatureHistory, feature: Feature, outputs: list[Body] | None = None
    ) -> None:
        self.history = history
        self.feature = feature
        self._outputs = outputs  # reused on the first do() only; redo recomputes
        self._index: int | None = None
        self.text = f"添加 {feature.name}"

    def do(self) -> None:
        index = self.history.active_end if self._index is None else self._index
        self.history.insert(index, self.feature, self._outputs)
        self._index = self.history.index_of(self.feature.id)
        self._outputs = None

    def undo(self) -> None:
        self.history.remove(self.feature.id)


class RemoveFeatureCommand(Command):
    def __init__(self, history: FeatureHistory, feature_id: str) -> None:
        self.history = history
        self.feature = history.get(feature_id)
        self._index = history.index_of(feature_id)
        self.text = f"删除 {self.feature.name}"

    def do(self) -> None:
        self._index = self.history.index_of(self.feature.id)
        self.history.remove(self.feature.id)

    def undo(self) -> None:
        self.history.insert(self._index, self.feature)


class EditParamsCommand(Command):
    def __init__(self, history: FeatureHistory, feature_id: str, params: dict[str, Any]) -> None:
        self.history = history
        self.feature_id = feature_id
        feature = history.get(feature_id)
        self._new = dict(params)
        self._old = {key: feature.params[key] for key in params}
        self.text = f"编辑 {feature.name}"

    def do(self) -> None:
        self.history.update_params(self.feature_id, self._new)

    def undo(self) -> None:
        self.history.update_params(self.feature_id, self._old)


class SuppressFeatureCommand(Command):
    def __init__(self, history: FeatureHistory, feature_id: str, suppressed: bool) -> None:
        self.history = history
        self.feature_id = feature_id
        self._suppressed = suppressed
        self.text = ("抑制 " if suppressed else "取消抑制 ") + history.get(feature_id).name

    def do(self) -> None:
        self.history.set_suppressed(self.feature_id, self._suppressed)

    def undo(self) -> None:
        self.history.set_suppressed(self.feature_id, not self._suppressed)
