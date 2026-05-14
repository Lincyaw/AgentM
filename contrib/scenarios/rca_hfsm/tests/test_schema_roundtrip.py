"""Round-trip every L1 dataclass through ``dataclasses.asdict`` and back.

Single fail-stop test, not exhaustive field coverage. The observability JSONL
sink consumes ``asdict`` output verbatim; if a future schema change breaks
re-construction from the dict form, every replay tool downstream silently
diverges from the source-of-truth graph.
"""

from __future__ import annotations

from dataclasses import asdict

from agentm_rca_hfsm.schema import (
    CheckResult,
    Hypothesis,
    Interpretation,
    Observation,
    Prediction,
    Symptom,
    WorkerReturn,
)


def test_full_schema_roundtrip() -> None:
    obs = Observation(
        id="o1",
        text="disk usage 99%",
        source_tool_call="tc1",
        tool_signature="sig1",
        related_symptoms=["s1"],
        related_predictions=["p1"],
        ts=1.0,
    )
    interp = Interpretation(
        proposed_update="confirm H1",
        reasoning="signal aligned with prediction",
        confidence="high",
    )
    check = CheckResult(
        id="c1",
        prediction_id="p1",
        worker_session_id="w1",
        observations=[obs],
        interpretation=interp,
        verdict_proposal="supports H1",
        ts=2.0,
    )
    pred = Prediction(
        id="p1",
        hypothesis_id="h1",
        claim="if H1, then disk fills",
        polarity="positive",
        test_plan="query disk metric",
        checks=[check],
    )
    h = Hypothesis(
        id="h1",
        claim="logrotate failed",
        parent_ids=[],
        predictions=[pred],
        status="open",
        generation=0,
        rationale="recent config change",
    )
    sym = Symptom(id="s1", text="alert at 14:32", source="user_intake", ts=0.5)
    wr = WorkerReturn(observations=[obs], interpretation=interp)

    originals = [obs, interp, check, pred, h, sym, wr]
    for original in originals:
        # Reconstruct from the asdict form via the same dataclass type. The
        # constructor accepts the nested dicts directly only for the leaf
        # ``Observation`` / ``Interpretation`` / ``Symptom`` — but at the
        # composite levels the test rebuilds explicitly to keep the
        # assertion structural (equality), not a smoke check.
        cls = type(original)
        # Re-instantiate using the dataclass field map; this exercises the
        # asdict → dict → kwargs path that observability replay depends on.
        rebuilt = _rebuild(cls, asdict(original))
        assert rebuilt == original, f"roundtrip failed for {cls.__name__}"


def _rebuild(cls: type, data: dict) -> object:
    """Reconstruct a dataclass instance from its ``asdict`` form.

    Walks one level of nested dataclasses — enough for the Phase 1 schema.
    """

    from dataclasses import fields, is_dataclass

    kwargs = {}
    for f in fields(cls):
        value = data[f.name]
        annotation = f.type
        # ``Interpretation | None``-style annotations are surfaced as
        # strings under ``from __future__ import annotations``. Resolve
        # them against the schema module so isinstance-equivalent
        # reconstruction works.
        if isinstance(value, dict):
            nested_cls = _maybe_dataclass_from_annotation(annotation)
            if nested_cls is not None:
                kwargs[f.name] = _rebuild(nested_cls, value)
                continue
        if isinstance(value, list) and value and isinstance(value[0], dict):
            nested_cls = _maybe_dataclass_from_list_annotation(annotation)
            if nested_cls is not None:
                kwargs[f.name] = [_rebuild(nested_cls, item) for item in value]
                continue
        kwargs[f.name] = value
    instance = cls(**kwargs)
    assert is_dataclass(instance)
    return instance


def _maybe_dataclass_from_annotation(annotation: object) -> type | None:
    from agentm_rca_hfsm import schema as schema_mod
    from dataclasses import is_dataclass

    name = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    if " | " in name:
        name = name.split(" | ", 1)[0].strip()
    candidate = getattr(schema_mod, name, None)
    if isinstance(candidate, type) and is_dataclass(candidate):
        return candidate
    return None


def _maybe_dataclass_from_list_annotation(annotation: object) -> type | None:
    if not isinstance(annotation, str):
        return None
    # Annotation strings look like "list[CheckResult]" / "list[Observation]"
    # under ``from __future__ import annotations``.
    if not annotation.startswith("list["):
        return None
    inner = annotation[len("list["):-1]
    return _maybe_dataclass_from_annotation(inner)
