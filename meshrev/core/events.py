"""Minimal synchronous observer used by the core model.

The core must not depend on Qt, so the document publishes changes through this
bus; ``gui.controller.DocumentController`` re-emits them as Qt signals.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

Callback = Callable[..., None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callback]] = defaultdict(list)

    def subscribe(self, event: str, callback: Callback) -> Callable[[], None]:
        """Register ``callback(**payload)`` for ``event``; returns an unsubscribe function."""
        self._subscribers[event].append(callback)

        def unsubscribe() -> None:
            callbacks = self._subscribers.get(event, [])
            if callback in callbacks:
                callbacks.remove(callback)

        return unsubscribe

    def emit(self, event: str, **payload: Any) -> None:
        for callback in list(self._subscribers.get(event, ())):
            callback(**payload)
