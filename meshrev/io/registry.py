"""Extension based registry of readers and writers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from meshrev.core.bodies import Body


class UnsupportedFormatError(ValueError):
    pass


@runtime_checkable
class Reader(Protocol):
    name: str
    extensions: tuple[str, ...]

    def read(self, path: Path) -> list[Body]: ...


@runtime_checkable
class Writer(Protocol):
    name: str
    extensions: tuple[str, ...]

    def write(self, bodies: Sequence[Body], path: Path, **options: Any) -> None: ...


@dataclass(frozen=True)
class FormatInfo:
    name: str
    extensions: tuple[str, ...]

    @property
    def dialog_filter(self) -> str:
        return f"{self.name} ({' '.join('*' + ext for ext in self.extensions)})"


_readers: dict[str, Reader] = {}
_writers: dict[str, Writer] = {}


def _norm_ext(ext: str) -> str:
    ext = ext.lower()
    return ext if ext.startswith(".") else "." + ext


def register_reader(reader: Reader) -> Reader:
    for ext in reader.extensions:
        _readers[_norm_ext(ext)] = reader
    return reader


def register_writer(writer: Writer) -> Writer:
    for ext in writer.extensions:
        _writers[_norm_ext(ext)] = writer
    return writer


def reader_for(path: str | Path) -> Reader:
    ext = Path(path).suffix.lower()
    try:
        return _readers[ext]
    except KeyError:
        raise UnsupportedFormatError(f"不支持读取的文件格式: {ext or path}") from None


def writer_for(path: str | Path) -> Writer:
    ext = Path(path).suffix.lower()
    try:
        return _writers[ext]
    except KeyError:
        raise UnsupportedFormatError(f"不支持导出的文件格式: {ext or path}") from None


def load(path: str | Path) -> list[Body]:
    """Read ``path`` into bodies (format chosen by extension)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return reader_for(path).read(path)


def save(bodies: Sequence[Body], path: str | Path, **options: Any) -> None:
    if not bodies:
        raise ValueError("没有可导出的实体")
    path = Path(path)
    writer_for(path).write(list(bodies), path, **options)


def supported_formats(mode: Literal["read", "write"]) -> list[FormatInfo]:
    table = _readers if mode == "read" else _writers
    grouped: dict[str, list[str]] = {}
    for ext, handler in table.items():
        grouped.setdefault(handler.name, []).append(ext)
    return [FormatInfo(name, tuple(exts)) for name, exts in grouped.items()]


def dialog_filter(mode: Literal["read", "write"]) -> str:
    """Qt file dialog filter string, e.g. ``"网格 (*.stl *.obj);;所有文件 (*)"``."""
    formats = supported_formats(mode)
    parts = [fmt.dialog_filter for fmt in formats]
    if mode == "read" and formats:
        every = " ".join("*" + ext for fmt in formats for ext in fmt.extensions)
        parts.insert(0, f"所有支持的格式 ({every})")
    parts.append("所有文件 (*)")
    return ";;".join(parts)
