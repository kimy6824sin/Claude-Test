"""Ordered feature list with rollback, suppression and regeneration."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from meshrev.core.features.base import Feature, FeatureContext, FeatureState

if TYPE_CHECKING:
    from meshrev.core.bodies import Body
    from meshrev.core.document import Document

log = logging.getLogger(__name__)

HISTORY_CHANGED = "history_changed"


class FeatureHistory:
    """Linear parametric history.

    Features before ``rollback_index`` are active; the rest are rolled back.
    Any change regenerates the affected feature and everything after it.
    """

    def __init__(self, document: Document) -> None:
        self._document = document
        self._features: list[Feature] = []
        self._rollback_index: int | None = None
        self.context_factory: Callable[[], FeatureContext] = lambda: FeatureContext(document)

    # -- read access ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Feature]:
        return iter(self._features)

    def __len__(self) -> int:
        return len(self._features)

    @property
    def features(self) -> tuple[Feature, ...]:
        return tuple(self._features)

    @property
    def rollback_index(self) -> int | None:
        return self._rollback_index

    @property
    def active_end(self) -> int:
        return len(self._features) if self._rollback_index is None else self._rollback_index

    def get(self, feature_id: str) -> Feature:
        for feature in self._features:
            if feature.id == feature_id:
                return feature
        raise KeyError(feature_id)

    def index_of(self, feature_id: str) -> int:
        return self._features.index(self.get(feature_id))

    def producer_of(self, body_id: str) -> Feature | None:
        for feature in self._features:
            if body_id in feature.output_ids:
                return feature
        return None

    # -- editing ----------------------------------------------------------------------
    def evaluate(self, feature: Feature, ctx: FeatureContext | None = None) -> list[Body]:
        """Run ``feature`` without committing (safe to call from a worker thread)."""
        outputs = feature.execute(ctx or self.context_factory())
        for index, body in enumerate(outputs):
            body.id = feature.output_id(index)
            body.source_feature = feature.id
        return outputs

    def insert(self, index: int, feature: Feature, outputs: list[Body] | None = None) -> None:
        """Insert at ``index`` (clamped to the active range) and regenerate from there.

        ``outputs`` may carry a result computed earlier by :meth:`evaluate`.
        """
        index = max(0, min(index, self.active_end))
        self._features.insert(index, feature)
        if self._rollback_index is not None:
            self._rollback_index += 1
        self.regenerate(index, precomputed={feature.id: outputs} if outputs is not None else None)

    def append(self, feature: Feature, outputs: list[Body] | None = None) -> None:
        self.insert(self.active_end, feature, outputs)

    def remove(self, feature_id: str) -> Feature:
        index = self.index_of(feature_id)
        feature = self._features.pop(index)
        self._drop_outputs(feature)
        if self._rollback_index is not None and index < self._rollback_index:
            self._rollback_index -= 1
        self.regenerate(index)
        return feature

    def update_params(self, feature_id: str, params: dict[str, Any]) -> None:
        feature = self.get(feature_id)
        unknown = set(params) - set(feature.params)
        if unknown:
            raise KeyError(f"unknown parameters {sorted(unknown)}")
        feature.params.update(params)
        self.regenerate(self.index_of(feature_id))

    def set_suppressed(self, feature_id: str, suppressed: bool) -> None:
        feature = self.get(feature_id)
        feature.suppressed = suppressed
        self.regenerate(self.index_of(feature_id))

    def rollback(self, index: int | None) -> None:
        """Deactivate every feature at position >= ``index`` (``None`` rolls forward)."""
        if index is not None:
            index = max(0, min(index, len(self._features)))
            if index == len(self._features):
                index = None
        previous = self.active_end
        self._rollback_index = index
        self.regenerate(min(previous, self.active_end))

    def clear(self) -> None:
        for feature in self._features:
            self._drop_outputs(feature)
        self._features.clear()
        self._rollback_index = None
        self._document.events.emit(HISTORY_CHANGED)

    # -- regeneration -----------------------------------------------------------------
    def regenerate(
        self, start: int = 0, precomputed: dict[str, list[Body] | None] | None = None
    ) -> None:
        precomputed = precomputed or {}
        ctx = self.context_factory()
        for index in range(max(0, start), len(self._features)):
            feature = self._features[index]
            if index >= self.active_end:
                self._drop_outputs(feature)
                feature.state = FeatureState.ROLLED_BACK
                continue
            if feature.suppressed:
                self._drop_outputs(feature)
                feature.state = FeatureState.SUPPRESSED
                continue
            try:
                outputs = precomputed.get(feature.id)
                if outputs is None:
                    outputs = self.evaluate(feature, ctx)
            except Exception as exc:  # noqa: BLE001 - a failing feature must not abort regen
                log.exception("feature %s failed", feature.name)
                self._drop_outputs(feature)
                feature.state = FeatureState.ERROR
                feature.error = str(exc) or type(exc).__name__
                continue
            self._commit(feature, outputs)
            feature.state = FeatureState.OK
            feature.error = None
        self._document.events.emit(HISTORY_CHANGED)

    def _commit(self, feature: Feature, outputs: list[Body]) -> None:
        new_ids = [body.id for body in outputs]
        for stale in set(feature.output_ids) - set(new_ids):
            self._document.remove_body(stale)
        for body in outputs:
            previous = self._document.find(body.id)
            if previous is not None:
                body.visible = previous.visible  # keep the user's view state
            self._document.add_body(body)
        feature.output_ids = new_ids

    def _drop_outputs(self, feature: Feature) -> None:
        for body_id in feature.output_ids:
            if body_id in self._document:
                self._document.remove_body(body_id)
        feature.output_ids = []
