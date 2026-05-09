"""Fail-stop test (B-2): a synthetic ``tool_reflect`` output round-trips
through the B-3 ChangeSpec validators.

Why load-bearing: the reflect atom is the *only* place in the loop that
defines the prompt scaffold the tuner uses to author a ChangeSpec. If
the schema hint it returns drifts away from what
``tool_propose_change`` accepts, every reflect-produced mutation gets
rejected at the validator gate, and the loop silently stops making
progress. We pin the contract from both sides.

We deliberately do not drive the agent — the failure mode lives at the
shape-contract boundary, not in the LLM round trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contrib.extensions.changespec_validators import atom_source as v_atom_source


def _reflect_emitted_change_spec() -> dict[str, object]:
    """Return a ChangeSpec dict shaped exactly like what a tuner LLM
    would emit after consuming ``tool_reflect``'s output. Mirrors the
    template at ``contrib/scenarios/format_fix/eval/reflection_template.md``.
    """
    return {
        "kind": "atom_source",
        "path": "tool_normalize_json.py",
        "new_content": (
            "from agentm.extensions import ExtensionManifest\n\n"
            "MANIFEST = ExtensionManifest(\n"
            "    name='tool_normalize_json',\n"
            "    description='Normalize JSON output.',\n"
            "    registers=('tool:normalize_json',),\n"
            ")\n\n"
            "def install(api, config):\n"
            "    return None\n"
        ),
        "target_atom": "tool_normalize_json",
    }


def test_reflect_changespec_validates_atom_source(tmp_path: Path) -> None:
    spec = _reflect_emitted_change_spec()
    result = v_atom_source.validate(spec, tmp_path, "format_fix")
    assert result["ok"] is True, result.get("error")
    assert result.get("error") is None


@pytest.mark.parametrize(
    "drop_field",
    ["new_content", "target_atom", "path"],
)
def test_reflect_changespec_rejects_missing_field(
    tmp_path: Path, drop_field: str
) -> None:
    """Complement-sanity: each load-bearing field's absence yields a
    validator rejection. Locks the failure mode in case the reflect
    template ever drops a slot from the schema hint.
    """
    spec = _reflect_emitted_change_spec()
    spec.pop(drop_field, None)
    result = v_atom_source.validate(spec, tmp_path, "format_fix")
    assert result["ok"] is False
    assert isinstance(result.get("error"), str) and result["error"]


def test_reflect_atom_emits_template_aware_schema_hint(tmp_path: Path) -> None:
    """Quick structural test of the atom itself: when given a populated
    eval-run dir + scenario template, ``tool_reflect`` emits a JSON
    payload whose ``change_spec_schema`` matches what the validator
    accepts (kind + required fields)."""
    # Stage the scenario template so the atom does not bail with a
    # template-not-found error.
    template_dir = tmp_path / "contrib" / "scenarios" / "format_fix" / "eval"
    template_dir.mkdir(parents=True)
    (template_dir / "reflection_template.md").write_text(
        "module=<TARGET_MODULE> traces=<TRACES> source=<CURRENT_SOURCE>",
        encoding="utf-8",
    )

    # Import the atom and drive its install + tool fn directly with a
    # minimal ExtensionAPI stub.
    from agentm.extensions.builtin import tool_reflect

    captured: dict[str, object] = {}

    class _Stub:
        cwd = str(tmp_path)

        def register_tool(self, tool: object) -> None:
            captured["tool"] = tool

    api: object = _Stub()
    tool_reflect.install(api, {"default_scenario": "format_fix"})  # type: ignore[arg-type]

    tool = captured["tool"]
    fn = getattr(tool, "fn", None)
    assert callable(fn)

    import asyncio

    result = asyncio.run(
        fn(
            {
                "failures": [
                    {
                        "trace_id": "t1",
                        "task_id": "task_a",
                        "stop_reason": "grader_failed",
                    }
                ],
                "target_module": "tool_normalize_json",
                "target_scenario": "format_fix",
            }
        )
    )
    assert result.is_error is False, result.content
    payload = json.loads(result.content[0].text)
    schema = payload["change_spec_schema"]
    assert schema["required"] == ["kind", "path", "new_content"]
    assert "atom_source" in schema["properties"]["kind"]["enum"]
    # The assembled prompt must have substituted at least the trace and
    # module slots — leaving raw `<TARGET_MODULE>` in the output is a
    # silent template-failure mode.
    prompt = payload["diagnosis_prompt"]
    assert "<TARGET_MODULE>" not in prompt
    assert "<TRACES>" not in prompt
    assert "tool_normalize_json" in prompt
