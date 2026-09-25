"""Built-in features that do not depend on recognition algorithms."""

from __future__ import annotations

from pathlib import Path

from meshrev.core.bodies import Body, MeshBody
from meshrev.core.features.base import (
    Feature,
    FeatureContext,
    FeatureInputError,
    ParamSpec,
    register_feature,
)
from meshrev.core.section.slicer import slice_mesh
from meshrev.core.types import Plane


@register_feature
class ImportFeature(Feature):
    """Root feature of imported data (meshes or STEP solids).

    Loaded bodies are cached so that regenerating the history does not re-read
    large scan files; the loader comes from the :class:`FeatureContext` so the
    core stays independent from ``meshrev.io``.
    """

    type_name = "Import"
    label = "导入"
    params_spec = (ParamSpec("path", "文件", "str", "", readonly=True),)

    def __init__(self, path: str | Path = "", bodies: list[Body] | None = None, **kwargs) -> None:
        params = kwargs.pop("params", None) or {"path": str(path)}
        super().__init__(params=params, **kwargs)
        if "name" not in kwargs and self.params["path"]:
            self.name = f"导入 {Path(self.params['path']).name}"
        self._cache = bodies

    def execute(self, ctx: FeatureContext) -> list[Body]:
        if self._cache is None:
            if ctx.load_file is None:
                raise FeatureInputError("没有可用的文件加载器")
            self._cache = ctx.load_file(Path(self.params["path"]))
        return list(self._cache)


@register_feature
class SectionFeature(Feature):
    """Planar section through a mesh body."""

    type_name = "Section"
    label = "截面"
    params_spec = (
        ParamSpec("origin", "原点", "floats", (0.0, 0.0, 0.0)),
        ParamSpec("normal", "法向", "floats", (0.0, 0.0, 1.0)),
    )

    def execute(self, ctx: FeatureContext) -> list[Body]:
        from meshrev.core.bodies import SectionBody

        mesh = ctx.body(self.inputs[0], MeshBody)
        plane = Plane(self.params["origin"], self.params["normal"])
        curve = slice_mesh(mesh.polydata, plane)
        return [SectionBody(curve, f"{mesh.name} 截面")]
