"""Commit 3 — every evidence tool routes through the falsification gate.

One sub-test per tool. Each verifies that:

* The tool builds a well-formed ``UpdateProposal``.
* The gate's ``UpdateResult`` is reflected in ``ToolResult.text``
  (``status=applied|downgraded|rejected`` plus the reason on the latter
  two paths).
* The graph state mutates (or doesn't, on downgrade / rejection) as
  expected.

Tests use ``install_full_stack`` from ``_gate_fixtures`` so the wiring
mirrors what the scenario manifest will produce.
"""

from __future__ import annotations

import asyncio

from tests._gate_fixtures import install_full_stack


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _tool(api: object, name: str) -> object:
    for t in getattr(api, "tools", []):  # type: ignore[attr-defined]
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not registered")


def _text(result: object) -> str:
    chunks = [c.text for c in result.content]  # type: ignore[attr-defined]
    return "\n".join(chunks)




def test_record_symptom_rejects_empty_text() -> None:
    api, _, read = install_full_stack()
    tool = _tool(api, "record_symptom")

    result = _run(tool.execute({"text": "   "}))  # type: ignore[attr-defined]

    text = _text(result)
    assert text.startswith("status=rejected"), text
    assert read.get_symptoms() == []












def test_propose_update_rejects_unknown_op() -> None:
    api, _, _ = install_full_stack()
    tool = _tool(api, "propose_update")

    result = _run(tool.execute({"op": "explode", "target_id": "X"}))  # type: ignore[attr-defined]
    text = _text(result)
    assert text.startswith("status=rejected"), text
    assert "unknown op" in text


def test_propose_update_refuses_record_observation_alias() -> None:
    api, _, _ = install_full_stack()
    tool = _tool(api, "propose_update")

    result = _run(tool.execute({"op": "record_observation"}))  # type: ignore[attr-defined]
    text = _text(result)
    assert text.startswith("status=rejected"), text
    assert "dedicated tool" in text
