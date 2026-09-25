"""Parametric feature base classes.

A feature is a pure function of its parameters and its input bodies:
``execute(ctx)`` must only *read* the document and *return* new bodies. The
:class:`~meshrev.core.features.history.FeatureHistory` commits the outputs,
which lets heavy features run in a worker thread and makes regeneration
deterministic (output ids are derived from the feature id).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from meshrev.core.types import new_id

if TYPE_CHECKING:
    from meshrev.core.bodies import Body
    from meshrev.core.document import Document

B = TypeVar("B", bound="Body")


class FeatureState(str, Enum):
    PENDING = "pending"
    OK = "ok"
    ERROR = "error"
    SUPPRESSED = "suppressed"
    ROLLED_BACK = "rolled_back"


class FeatureInputError(RuntimeError):
    """A referenced input body is missing or has the wrong type."""


@dataclass(frozen=True)
class ParamSpec:
    """Describes one editable parameter; the property panel builds widgets from it."""

    name: str
    label: str
    kind: str  # "float" | "int" | "bool" | "choice" | "str" | "ints"
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    decimals: int = 3
    choices: tuple[tuple[str, str], ...] = ()  # (value, label)
    suffix: str = ""
    readonly: bool = False
    tooltip: str = ""


@dataclass
class FeatureContext:
    document: Document
    load_file: Callable[[Path], list[Body]] | None = None
    progress: Callable[[float, str], None] | None = field(default=None, repr=False)

    def body(self, body_id: str, expected: type[B] | None = None) -> B:
        body = self.document.find(body_id)
        if body is None:
            raise FeatureInputError(f"输入实体 {body_id} 不存在")
        if expected is not None and not isinstance(body, expected):
            raise FeatureInputError(
                f"输入实体 {body.name} 类型为 {type(body).__name__}，需要 {expected.__name__}"
            )
        return body  # type: ignore[return-value]

    def report(self, fraction: float, message: str = "") -> None:
        if self.progress is not None:
            self.progress(fraction, message)


FEATURE_TYPES: dict[str, type[Feature]] = {}


def register_feature(cls: type[Feature]) -> type[Feature]:
    """Class decorator making a feature type discoverable by name (for persistence)."""
    FEATURE_TYPES[cls.type_name] = cls
    return cls


class Feature(ABC):
    type_name: ClassVar[str] = "Feature"
    label: ClassVar[str] = "特征"
    params_spec: ClassVar[tuple[ParamSpec, ...]] = ()

    def __init__(
        self,
        inputs: list[str] | tuple[str, ...] = (),
        params: dict[str, Any] | None = None,
        *,
        name: str | None = None,
        feature_id: str | None = None,
    ) -> None:
        self.id = feature_id or new_id("F")
        self.name = name or self.label
        self.inputs: list[str] = list(inputs)
        self.params: dict[str, Any] = {spec.name: spec.default for spec in self.params_spec}
        if params:
            unknown = set(params) - set(self.params)
            if unknown:
                raise KeyError(f"{self.type_name}: unknown parameters {sorted(unknown)}")
            self.params.update(params)
        self.state = FeatureState.PENDING
        self.error: str | None = None
        self.suppressed = False
        self.output_ids: list[str] = []

    @abstractmethod
    def execute(self, ctx: FeatureContext) -> list[Body]:
        """Compute and return the output bodies. Must not modify the document."""

    def output_id(self, index: int) -> str:
        return f"{self.id}.{index}"

    def spec(self, name: str) -> ParamSpec:
        for spec in self.params_spec:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def info(self) -> dict[str, str]:
        info = {"名称": self.name, "类型": self.label, "状态": self.state.value, "ID": self.id}
        if self.error:
            info["错误"] = self.error
        return info

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r}, state={self.state.value})"
