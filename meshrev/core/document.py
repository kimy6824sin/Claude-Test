"""The document model: bodies + parametric history + change events."""

from __future__ import annotations

from typing import TypeVar

from meshrev.core.bodies import Body, BodyKind
from meshrev.core.events import EventBus
from meshrev.core.features.history import HISTORY_CHANGED, FeatureHistory
from meshrev.core.types import BBox, Units

BODY_ADDED = "body_added"  # payload: body_id
BODY_REMOVED = "body_removed"  # payload: body_id
BODY_CHANGED = "body_changed"  # payload: body_id, attribute ("geometry", "visible", "name", ...)
DOCUMENT_CLEARED = "document_cleared"

__all__ = [
    "BODY_ADDED",
    "BODY_CHANGED",
    "BODY_REMOVED",
    "DOCUMENT_CLEARED",
    "HISTORY_CHANGED",
    "Document",
]

B = TypeVar("B", bound=Body)


class Document:
    def __init__(self, name: str = "未命名", units: Units = Units.MM) -> None:
        self.name = name
        self.units = units
        self.events = EventBus()
        self._bodies: dict[str, Body] = {}
        self.history = FeatureHistory(self)

    # -- bodies -----------------------------------------------------------------------
    def add_body(self, body: Body) -> str:
        """Add ``body``; a body with the same id is replaced (``body_changed``)."""
        replaced = body.id in self._bodies
        self._bodies[body.id] = body
        if replaced:
            self.events.emit(BODY_CHANGED, body_id=body.id, attribute="geometry")
        else:
            self.events.emit(BODY_ADDED, body_id=body.id)
        return body.id

    def remove_body(self, body_id: str) -> Body:
        body = self._bodies.pop(body_id)
        self.events.emit(BODY_REMOVED, body_id=body_id)
        return body

    def get(self, body_id: str) -> Body:
        return self._bodies[body_id]

    def find(self, body_id: str) -> Body | None:
        return self._bodies.get(body_id)

    def bodies(self, kind: BodyKind | None = None) -> list[Body]:
        return [b for b in self._bodies.values() if kind is None or b.kind == kind]

    def bodies_of_type(self, cls: type[B]) -> list[B]:
        return [b for b in self._bodies.values() if isinstance(b, cls)]

    def __contains__(self, body_id: object) -> bool:
        return body_id in self._bodies

    def __len__(self) -> int:
        return len(self._bodies)

    def set_visible(self, body_id: str, visible: bool) -> None:
        body = self.get(body_id)
        if body.visible != visible:
            body.visible = visible
            self.events.emit(BODY_CHANGED, body_id=body_id, attribute="visible")

    def rename_body(self, body_id: str, name: str) -> None:
        self.get(body_id).name = name
        self.events.emit(BODY_CHANGED, body_id=body_id, attribute="name")

    def notify_changed(self, body_id: str, attribute: str = "geometry") -> None:
        self.events.emit(BODY_CHANGED, body_id=body_id, attribute=attribute)

    def bounds(self, visible_only: bool = True) -> BBox | None:
        result: BBox | None = None
        for body in self._bodies.values():
            if visible_only and not body.visible:
                continue
            box = body.bounds()
            if box is not None:
                result = box if result is None else result.union(box)
        return result

    def clear(self) -> None:
        self.history.clear()
        for body_id in list(self._bodies):
            self.remove_body(body_id)
        self.events.emit(DOCUMENT_CLEARED)
