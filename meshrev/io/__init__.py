"""File exchange. Always import as ``meshrev.io`` (the package name shadows the
standard library only if ``meshrev/`` itself is put on ``sys.path``; launch the
application with ``python -m meshrev``)."""

from meshrev.io import mesh_formats, step_format
from meshrev.io.registry import (
    FormatInfo,
    Reader,
    UnsupportedFormatError,
    Writer,
    dialog_filter,
    load,
    reader_for,
    register_reader,
    register_writer,
    save,
    supported_formats,
    writer_for,
)

for _reader in (
    mesh_formats.STL_READER,
    mesh_formats.OBJ_READER,
    mesh_formats.PLY_READER,
    step_format.StepReader(),
    step_format.IgesReader(),
):
    register_reader(_reader)
for _writer in (
    mesh_formats.STL_WRITER,
    mesh_formats.OBJ_WRITER,
    mesh_formats.PLY_WRITER,
    step_format.StepWriter(),
    step_format.IgesWriter(),
):
    register_writer(_writer)

__all__ = [
    "FormatInfo",
    "Reader",
    "UnsupportedFormatError",
    "Writer",
    "dialog_filter",
    "load",
    "reader_for",
    "register_reader",
    "register_writer",
    "save",
    "supported_formats",
    "writer_for",
]
