"""Fail-stop integration test for PR-A: ChangeSpec.asi + mu_f.failure_kind.

Both fields are free-text, but their *presence* in the activation
record is load-bearing for cross-episode learning. If asi.hypothesis
gets dropped between propose-time and the persisted record, the
evolution loop loses the proposer's reasoning. If failure_kind is
dropped between the grader output and the activation record, the
4-floor gate (and downstream reflection) cannot distinguish a runtime
crash from a metric regression.

Three assertions:

1. ChangeSpec carrying ``asi.hypothesis`` -> the persisted activation
   record carries the same hypothesis verbatim under
   ``record.change_spec.asi.hypothesis`` AND at the top level
   ``record.asi.hypothesis``.
2. An eval_run summary with ``failure_kind="runtime"`` -> the
   persisted activation record carries ``failure_kind="runtime"``.
3. A ChangeSpec without an ``asi`` field at all is still accepted
   (backward-compatible) and the persisted record carries an empty
   ``asi`` dict.

These mirror the stub-provider pattern from
``tests/integration/test_per_task_evolution.py`` so we don't make any
real LLM calls.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import (
    AssistantStreamEvent,
    MessageEnd,
    Model,
    TextContent,
)
from agentm.core.abi.messages import AssistantMessage
from agentm.core.abi.extension import ProviderConfig
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession


_PROVIDER_MODULE = "agentm._tests.asi_failure_kind_provider"


def _install_static_provider() -> str:
    if _PROVIDER_MODULE in sys.modules:
        return _PROVIDER_MODULE
    module = types.ModuleType(_PROVIDER_MODULE)

    async def _stream(
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del messages, model, tools, system, signal, thinking
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="ok")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "asi-test",
            ProviderConfig(
                stream_fn=_stream,
                model=Model(
                    id="asi-test",
                    provider="fake",
                    context_window=16_000,
                    max_output_tokens=2_000,
                ),
                name="asi-test",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[_PROVIDER_MODULE] = module
    return _PROVIDER_MODULE


def _git_init(path: Path) -> None:
    subprocess.run(
        ["git", "init", "-q", str(path)], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"], check=True
    )


def _tool(session: AgentSession, name: str) -> Any:
    for tool in session.tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"missing tool {name}")


def _materialize_scenario(tmp_path: Path) -> Path:
    """Materialize a tiny scenario with a tier-1 atom we can mutate."""
    scenario_dir = tmp_path / "scenarios" / "asi_test"
    scenario_dir.mkdir(parents=True)
    atom_src = (
        "from agentm.core.abi import FunctionTool, TextContent, ToolResult\n"
        "from agentm.extensions import ExtensionManifest\n"
        "MANIFEST = ExtensionManifest(name='tiny_atom',"
        " description='tiny', registers=('tool:tiny',))\n"
        "async def _exec(args):\n"
        "    return ToolResult(content=[TextContent(type='text', text='v1')])\n"
        "def install(api, config):\n"
        "    api.register_tool(FunctionTool(name='tiny', description='t',"
        " parameters={'type':'object','properties':{}}, fn=_exec))\n"
    )
    (scenario_dir / "tiny_atom.py").write_text(atom_src, encoding="utf-8")
    (scenario_dir / "manifest.yaml").write_text(
        "name: asi_test\nextensions:\n"
        "  - module: agentm.extensions.builtin.operations_local\n"
        "  - local: tiny_atom\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "-A"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "init"],
        check=True,
        capture_output=True,
    )
    return scenario_dir


def _write_eval_summary(
    eval_runs: Path,
    run_id: str,
    *,
    primary_score: float,
    failure_kind: str | None,
) -> None:
    eval_runs.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "kind": "eval_run.summary",
        "eval_run_id": run_id,
        "task_class": "asi_test",
        "primary_score": primary_score,
        "primary_score_stderr": 0.01,
        "guard_metrics": {"tool_error_rate": 0.0, "turns_mean": 1.0},
        "samples_per_task": 3,
        "task_count": 1,
    }
    if failure_kind is not None:
        payload["failure_kind"] = failure_kind
    (eval_runs / f"{run_id}.jsonl").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_asi_and_failure_kind_propagate_to_activation(
    tmp_path: Path,
) -> None:
    """End-to-end: a ChangeSpec carrying ``asi.hypothesis`` and a
    proposed eval run tagged ``failure_kind="runtime"`` produce an
    activation record that carries both verbatim.

    We use a *gate-rejected* path (proposed score below threshold) so
    we exercise the persistence without invoking reload_atom — the
    fail-stop is the record contents, not the swap.
    """

    _git_init(tmp_path)
    scenario_dir = _materialize_scenario(tmp_path)
    eval_runs = tmp_path / ".agentm" / "eval_runs"
    # Both runs have the same primary_score so the gate's threshold
    # check rejects (delta below threshold_relative). Proposed run
    # carries failure_kind="runtime".
    _write_eval_summary(eval_runs, "er_b", primary_score=0.5, failure_kind="ok")
    _write_eval_summary(
        eval_runs, "er_p", primary_score=0.5, failure_kind="runtime"
    )

    provider_module = _install_static_provider()
    from agentm.extensions.loader import load_scenario

    scenario_extensions = load_scenario(str(scenario_dir))
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(provider_module, {}),
            extensions=scenario_extensions
            + [
                (
                    "agentm.extensions.builtin.tool_propose_change",
                    {
                        "target_scenario": "asi_test",
                        "promotion": {
                            "threshold_relative": 0.05,
                            "guard_tolerance": 0.50,
                        },
                    },
                ),
            ],
        )
    )
    try:
        propose = _tool(session, "propose_change")
        new_source = (
            "from agentm.core.abi import FunctionTool, TextContent, ToolResult\n"
            "from agentm.extensions import ExtensionManifest\n"
            "MANIFEST = ExtensionManifest(name='tiny_atom',"
            " description='tiny', registers=('tool:tiny',))\n"
            "async def _exec(args):\n"
            "    return ToolResult(content=[TextContent("
            "type='text', text='v2')])\n"
            "def install(api, config):\n"
            "    api.register_tool(FunctionTool(name='tiny',"
            " description='t', parameters={'type':'object',"
            "'properties':{}}, fn=_exec))\n"
        )
        result = await propose.execute(
            {
                "target": {
                    "kind": "atom_source",
                    "path": "tiny_atom.py",
                    "new_content": new_source,
                    "target_atom": "tiny_atom",
                    "asi": {
                        "hypothesis": "swap pruning order",
                        "next_focus": "look at retry handler if this fails",
                        "learned": "",
                    },
                },
                "rationale": "test asi+failure_kind propagation",
                "eval_run_baseline": "er_b",
                "eval_run_proposed": "er_p",
                "decision": "activate",
            }
        )
        # Gate rejects (no improvement) — that's expected. We're
        # asserting on the persisted record, not the swap outcome.
        assert result.is_error is True
        assert "deployment gate failed" in result.content[0].text
    finally:
        await session.shutdown()

    # Inspect activations.jsonl for the rejected record's fields.
    decisions_path = (
        tmp_path
        / ".agentm"
        / "decisions"
        / "asi_test"
        / "activations.jsonl"
    )
    assert decisions_path.is_file()
    with decisions_path.open("r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    rejected = [r for r in records if r.get("kind") == "rejected"]
    assert len(rejected) == 1, records
    rec = rejected[0]

    # Assertion 1: asi.hypothesis preserved (top-level + inside change_spec).
    assert rec["asi"] == {
        "hypothesis": "swap pruning order",
        "next_focus": "look at retry handler if this fails",
        "learned": "",
    }
    assert rec["change_spec"]["asi"]["hypothesis"] == "swap pruning order"

    # Assertion 2: failure_kind from proposed eval-run propagates.
    assert rec["failure_kind"] == "runtime"


@pytest.mark.asyncio
async def test_changespec_without_asi_still_validates(tmp_path: Path) -> None:
    """Backward-compat: a ChangeSpec missing the ``asi`` field is
    accepted (old pre-PR-A proposals do not crash). The persisted
    record carries an empty ``asi`` dict.
    """

    _git_init(tmp_path)
    scenario_dir = _materialize_scenario(tmp_path)
    eval_runs = tmp_path / ".agentm" / "eval_runs"
    _write_eval_summary(eval_runs, "er_b", primary_score=0.5, failure_kind=None)
    _write_eval_summary(eval_runs, "er_p", primary_score=0.5, failure_kind=None)

    provider_module = _install_static_provider()
    from agentm.extensions.loader import load_scenario

    scenario_extensions = load_scenario(str(scenario_dir))
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(provider_module, {}),
            extensions=scenario_extensions
            + [
                (
                    "agentm.extensions.builtin.tool_propose_change",
                    {
                        "target_scenario": "asi_test",
                        "promotion": {
                            "threshold_relative": 0.05,
                            "guard_tolerance": 0.50,
                        },
                    },
                ),
            ],
        )
    )
    try:
        propose = _tool(session, "propose_change")
        result = await propose.execute(
            {
                "target": {
                    # No ``asi`` field — old shape.
                    "kind": "atom_source",
                    "path": "tiny_atom.py",
                    "new_content": (
                        "from agentm.core.abi import FunctionTool,"
                        " TextContent, ToolResult\n"
                        "from agentm.extensions import ExtensionManifest\n"
                        "MANIFEST = ExtensionManifest(name='tiny_atom',"
                        " description='tiny', registers=('tool:tiny',))\n"
                        "async def _exec(args):\n"
                        "    return ToolResult(content=[TextContent("
                        "type='text', text='v2')])\n"
                        "def install(api, config):\n"
                        "    api.register_tool(FunctionTool(name='tiny',"
                        " description='t', parameters={'type':'object',"
                        "'properties':{}}, fn=_exec))\n"
                    ),
                    "target_atom": "tiny_atom",
                },
                "rationale": "backward-compat probe",
                "eval_run_baseline": "er_b",
                "eval_run_proposed": "er_p",
                "decision": "activate",
            }
        )
        # Accepted by the validator — gate may still reject on metrics,
        # which is the path we exercise.
        assert result.is_error is True
        assert "deployment gate failed" in result.content[0].text
    finally:
        await session.shutdown()

    decisions_path = (
        tmp_path
        / ".agentm"
        / "decisions"
        / "asi_test"
        / "activations.jsonl"
    )
    with decisions_path.open("r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    rejected = [r for r in records if r.get("kind") == "rejected"]
    assert len(rejected) == 1
    # Empty default propagates (not missing — explicit empty dict).
    assert rejected[0]["asi"] == {}
    assert rejected[0]["change_spec"]["asi"] == {}
    # failure_kind from a summary that didn't carry it -> None.
    assert rejected[0]["failure_kind"] is None
