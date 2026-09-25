import pytest
import pyvista as pv

from meshrev.core.bodies import Body, MeshBody
from meshrev.core.document import BODY_ADDED, BODY_CHANGED, BODY_REMOVED, Document
from meshrev.core.features import (
    AddFeatureCommand,
    EditParamsCommand,
    Feature,
    FeatureContext,
    FeatureState,
    ImportFeature,
    ParamSpec,
    RemoveFeatureCommand,
    SectionFeature,
    SuppressFeatureCommand,
    UndoStack,
)


class ScaleFeature(Feature):
    """Test feature: scaled copy of its input mesh."""

    type_name = "TestScale"
    label = "缩放"
    params_spec = (ParamSpec("factor", "系数", "float", 2.0),)

    def execute(self, ctx: FeatureContext) -> list[Body]:
        mesh = ctx.body(self.inputs[0], MeshBody)
        if self.params["factor"] <= 0:
            raise ValueError("factor must be positive")
        return [MeshBody(mesh.polydata.scale(self.params["factor"], inplace=False), "scaled")]


def _recorder(doc):
    events = []
    for name in (BODY_ADDED, BODY_REMOVED, BODY_CHANGED):
        doc.events.subscribe(name, lambda name=name, **kw: events.append((name, kw["body_id"])))
    return events


def _import(doc):
    feature = ImportFeature("sphere.stl", bodies=[MeshBody(pv.Sphere(), "sphere")])
    doc.history.append(feature)
    return feature


def test_import_creates_stable_body_ids():
    doc = Document()
    events = _recorder(doc)
    feature = _import(doc)
    assert feature.state is FeatureState.OK
    assert feature.output_ids == [f"{feature.id}.0"]
    assert events == [(BODY_ADDED, feature.output_ids[0])]
    doc.history.regenerate()
    assert feature.output_ids == [f"{feature.id}.0"]
    assert events[-1] == (BODY_CHANGED, feature.output_ids[0])


def test_param_change_regenerates_downstream():
    doc = Document()
    source = _import(doc)
    scale = ScaleFeature(inputs=[source.output_ids[0]])
    doc.history.append(scale)
    before = doc.get(scale.output_ids[0]).polydata.bounds
    doc.history.update_params(scale.id, {"factor": 4.0})
    after = doc.get(scale.output_ids[0]).polydata.bounds
    assert after[1] == pytest.approx(2 * before[1])


def test_failing_feature_is_marked_and_outputs_removed():
    doc = Document()
    source = _import(doc)
    scale = ScaleFeature(inputs=[source.output_ids[0]])
    doc.history.append(scale)
    out = scale.output_ids[0]
    doc.history.update_params(scale.id, {"factor": -1.0})
    assert scale.state is FeatureState.ERROR
    assert "positive" in scale.error
    assert out not in doc


def test_missing_input_propagates_error():
    doc = Document()
    source = _import(doc)
    scale = ScaleFeature(inputs=[source.output_ids[0]])
    doc.history.append(scale)
    doc.history.set_suppressed(source.id, True)
    assert source.state is FeatureState.SUPPRESSED
    assert scale.state is FeatureState.ERROR
    doc.history.set_suppressed(source.id, False)
    assert scale.state is FeatureState.OK


def test_rollback_and_roll_forward():
    doc = Document()
    source = _import(doc)
    scale = ScaleFeature(inputs=[source.output_ids[0]])
    doc.history.append(scale)
    doc.history.rollback(1)
    assert scale.state is FeatureState.ROLLED_BACK and len(doc) == 1
    doc.history.rollback(None)
    assert scale.state is FeatureState.OK and len(doc) == 2


def test_visibility_survives_regeneration():
    doc = Document()
    source = _import(doc)
    body_id = source.output_ids[0]
    doc.set_visible(body_id, False)
    doc.history.regenerate()
    assert doc.get(body_id).visible is False


def test_undo_redo_commands():
    doc = Document()
    stack = UndoStack()
    source = ImportFeature("sphere.stl", bodies=[MeshBody(pv.Sphere(), "sphere")])
    stack.push(AddFeatureCommand(doc.history, source))
    scale = ScaleFeature(inputs=[source.output_ids[0]])
    stack.push(AddFeatureCommand(doc.history, scale))
    stack.push(EditParamsCommand(doc.history, scale.id, {"factor": 3.0}))
    stack.push(SuppressFeatureCommand(doc.history, scale.id, True))
    assert len(doc) == 1
    stack.undo()  # unsuppress
    assert len(doc) == 2
    stack.undo()  # factor back to 2
    assert scale.params["factor"] == 2.0
    stack.undo()  # remove scale
    assert len(doc.history) == 1
    stack.redo()
    assert len(doc.history) == 2 and scale.state is FeatureState.OK
    stack.push(RemoveFeatureCommand(doc.history, source.id))
    assert len(doc.history) == 1 and scale.state is FeatureState.ERROR
    stack.undo()
    assert doc.history.index_of(source.id) == 0 and scale.state is FeatureState.OK
    assert stack.can_redo and stack.redo_text.startswith("删除")


def test_section_feature():
    doc = Document()
    source = ImportFeature("cyl.stl", bodies=[MeshBody(pv.Cylinder(direction=(0, 0, 1)), "c")])
    doc.history.append(source)
    section = SectionFeature(inputs=[source.output_ids[0]])
    doc.history.append(section)
    body = doc.get(section.output_ids[0])
    assert len(body.curve.polylines) == 1 and body.curve.closed[0]


def test_unknown_param_rejected():
    with pytest.raises(KeyError):
        ScaleFeature(params={"nope": 1})


def test_clear_document():
    doc = Document()
    _import(doc)
    doc.clear()
    assert len(doc) == 0 and len(doc.history) == 0
