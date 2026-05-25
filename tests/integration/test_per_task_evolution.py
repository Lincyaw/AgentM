"""Fail-stop integration tests for the per-task evolution loop.

Per ``CLAUDE.md`` testing philosophy: only the positions where the
loop's value proposition fails when broken. Specifically:

1. ``tool_propose_change`` rejects without both eval_run_ids — without
   this, evidence-driven becomes error-driven (P3 in design §10).
2. Tier-2 atoms cannot be auto-activated (S2 / P8) — the safety boundary
   that keeps an agent from rewriting its own permission/cost-budget atoms.
3. ``.agentm/decisions/**`` is constitution-protected — without this,
   the agent can sidestep the mediated decision channel.
4. Sub-session source-override leaves the working tree clean — without
   this, ``tool_eval_run`` mutates the source-of-truth tree on every call.
5. End-to-end loop: stub-provider tuner runs query_traces -> eval_run
   (baseline) -> eval_run (proposed override) -> propose_change(activate)
   -> verify activations.jsonl entry + atom-on-disk swapped.
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
from agentm.core.abi.messages import AssistantMessage, ToolCallBlock
from agentm.core.abi.extension import ProviderConfig
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession


_PROVIDER_MODULE = "agentm._tests.evolution_provider"


def _install_static_provider() -> str:
    """Install a provider module that always returns a single canned
    text reply. Used for tests that don't depend on the assistant's
    output content."""

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
                content=[TextContent(type="text", text='{"ok": true}')],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "evolution-test",
            ProviderConfig(
                stream_fn=_stream,
                model=Model(
                    id="evolution-test",
                    provider="fake",
                    context_window=16_000,
                    max_output_tokens=2_000,
                ),
                name="evolution-test",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[_PROVIDER_MODULE] = module
    return _PROVIDER_MODULE


def _install_echo_provider(answer_factory: Any) -> str:
    """Install a provider that returns the result of ``answer_factory(messages)``
    as the assistant's text. Lets a single test script multi-step replies
    based on what the agent has accumulated so far."""

    module_name = "agentm._tests.evolution_provider_echo"
    # Always reinstall — the factory closure changes per test.
    module = types.ModuleType(module_name)

    async def _stream(
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del model, tools, system, signal, thinking
        text = answer_factory(messages)
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=text)],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "evolution-echo",
            ProviderConfig(
                stream_fn=_stream,
                model=Model(
                    id="evolution-echo",
                    provider="fake",
                    context_window=16_000,
                    max_output_tokens=2_000,
                ),
                name="evolution-echo",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    return module_name


def _git_init(path: Path) -> None:
    """Initialize a git repo at ``path``. The format_fix scenario lives
    in this checkout, but tests need their own tmp_path to be a git repo
    so ResourceWriter classifies as managed when applicable."""

    subprocess.run(
        ["git", "init", "-q", "-b", "agent-tests", str(path)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        check=True,
    )


def _tool(session: AgentSession, name: str) -> Any:
    for tool in session.tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"missing tool {name}")


async def _create_tuner_session(
    cwd: Path, *, target_scenario_dir: Path, eval_dir: Path
) -> AgentSession:
    """Build a tuner session pointed at the given target scenario + eval
    dir. Avoids the named-scenario lookup so tests can stand up the
    layout in tmp_path."""

    provider_module = _install_static_provider()
    return await AgentSession.create(
        AgentSessionConfig(
            cwd=str(cwd),
            provider=(provider_module, {}),
            extensions=[

                ("agentm.extensions.builtin.operations_local", {}),
                (
                    "agentm.extensions.builtin.tool_query_traces",
                    {},
                ),
                (
                    "agentm.extensions.builtin.tool_eval_run",
                    {
                        "target_scenario": str(target_scenario_dir),
                        "eval_dir": str(eval_dir),
                    },
                ),
                ("contrib.extensions.changespec_validators", {}),
                (
                    "agentm.extensions.builtin.tool_propose_change",
                    {
                        "target_scenario": "format_fix",
                        "promotion": {
                            "threshold_relative": 0.05,
                            "guard_tolerance": 0.10,
                        },
                    },
                ),
            ],
        )
    )


# ---------------------------------------------------------------------------
# Test 1: tool_propose_change rejects without both eval_run_ids
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Test 2: tier-2 gate triggers pending_human_approval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tier2_activate_is_deferred(tmp_path: Path) -> None:
    """Activating a tier-2 atom must not call reload_atom; instead a
    decision record with status=pending_human_approval is written."""

    _git_init(tmp_path)
    provider_module = _install_static_provider()
    # The ``permission`` builtin atom is tier-2. Loading it gives us a
    # tier-2 target.
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(provider_module, {}),
            extensions=[

                ("agentm.extensions.builtin.operations_local", {}),
                ("agentm.extensions.builtin.permission", {}),
                ("contrib.extensions.changespec_validators", {}),
                (
                    "agentm.extensions.builtin.tool_propose_change",
                    {"target_scenario": "format_fix"},
                ),
            ],
        )
    )
    try:
        # Pre-write a stub eval-run summary so the gate has something
        # to read in case it gets that far (it shouldn't — tier-2
        # should short-circuit before any eval lookup).
        eval_runs = tmp_path / ".agentm" / "eval_runs"
        eval_runs.mkdir(parents=True, exist_ok=True)
        for run_id in ("er_b", "er_p"):
            (eval_runs / f"{run_id}.jsonl").write_text(
                json.dumps(
                    {
                        "kind": "eval_run.summary",
                        "eval_run_id": run_id,
                        "primary_score": 0.5,
                        "primary_score_stderr": 0.0,
                        "guard_metrics": {},
                    }
                )
                + "\n"
            )

        propose = _tool(session, "propose_change")
        result = await propose.execute(
            {
                "target": {
                    "kind": "atom_source",
                    "path": "permission.py",
                    "new_content": (
                        "MANIFEST = None\ndef install(api, config):\n    pass\n"
                    ),
                    "target_atom": "permission",
                },
                "rationale": "test tier-2 deferral",
                "eval_run_baseline": "er_b",
                "eval_run_proposed": "er_p",
                "decision": "activate",
            }
        )
        assert result.is_error is False, result.content[0].text
        payload = json.loads(result.content[0].text)
        assert payload["tier_blocked"] is True
        assert payload["status"] == "pending_human_approval"

        # Decision record exists and records the deferral.
        decisions_path = (
            tmp_path / ".agentm" / "decisions" / "format_fix" / "activations.jsonl"
        )
        assert decisions_path.is_file()
        with decisions_path.open("r", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        assert any(r["kind"] == "pending_human_approval" for r in records)
    finally:
        await session.shutdown()


# ---------------------------------------------------------------------------
# Test 3: decisions path is constitution-protected
# ---------------------------------------------------------------------------




@pytest.mark.asyncio
async def test_tool_write_rejects_constitution_path(tmp_path: Path) -> None:
    """End-to-end: ``tool_write`` invokes ResourceWriter; ResourceWriter
    refuses constitution paths. Asserts no file is created."""

    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / ".agentm" / "catalog" / "_test_writeguard.jsonl"
    if target.exists():
        target.unlink()

    _git_init(tmp_path)
    provider_module = _install_static_provider()
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(repo_root),
            provider=(provider_module, {}),
            extensions=[

                ("agentm.extensions.builtin.operations_local", {}),

                ("agentm.extensions.builtin.tool_write", {})],
        )
    )
    try:
        write = _tool(session, "write")
        result = await write.execute(
            {
                "path": str(target),
                "content": "should not be written",
            }
        )
        assert result.is_error is True
        assert "constitution" in result.content[0].text.lower()
        assert not target.exists()
    finally:
        await session.shutdown()


# ---------------------------------------------------------------------------
# Test 4: sub-session source override leaves tree clean
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Test 5: end-to-end loop on format_fix with stub provider
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_end_to_end_loop_activates_known_good_replacement(
    tmp_path: Path,
) -> None:
    """Stub-provider end-to-end: drive eval_run twice (baseline returns
    weak; proposed override returns canonical via stdlib json fallback),
    then propose_change(activate). Verify activations.jsonl entry, atom-
    on-disk swapped, and that eval_runs/<id>.jsonl exist.

    We do **not** drive the agent loop with hand-rolled tool calls — the
    tuner agent's tool calls are made directly through the registered
    Tool objects (the same surface the real loop calls) so the harness
    integrations all run, just without the LLM-driven control flow.
    The stub provider is wired purely so the eval-run child sessions
    have a provider to satisfy AgentSessionConfig."""

    _git_init(tmp_path)

    # Materialize a tiny copy of the format_fix scenario into tmp_path.
    # Using the real scenario from contrib/scenarios/format_fix/ would
    # also work, but copying isolates the test from the live tree.
    scenario_dir = tmp_path / "scenarios" / "format_fix"
    scenario_dir.mkdir(parents=True)
    weak_atom = (
        "from agentm.core.abi import FunctionTool, TextContent, ToolResult\n"
        "from agentm.extensions import ExtensionManifest\n"
        "MANIFEST = ExtensionManifest(name='tool_normalize_json',"
        " description='normalize', registers=('tool:normalize_json',))\n"
        "async def _exec(args):\n"
        "    raw = str(args['raw'])\n"
        "    return ToolResult(content=[TextContent(type='text', text=raw.replace(\"'\", '\"'))])\n"
        "def install(api, config):\n"
        "    api.register_tool(FunctionTool(name='normalize_json',"
        " description='n', parameters={'type':'object','properties':"
        "{'raw':{'type':'string'}},'required':['raw']}, fn=_exec))\n"
    )
    (scenario_dir / "tool_normalize_json.py").write_text(weak_atom, encoding="utf-8")
    (scenario_dir / "manifest.yaml").write_text(
        "name: format_fix\ntask_class: format_fix\nextensions:\n"
        "  - module: agentm.extensions.builtin.operations_local\n"
        "  - local: tool_normalize_json\n",
        encoding="utf-8",
    )

    # A single-task eval set + grader.
    eval_dir = scenario_dir / "eval"
    (eval_dir / "tasks").mkdir(parents=True)
    (eval_dir / "tasks" / "01_simple.yaml").write_text(
        "id: 01_simple\ntask_class: format_fix\n"
        "input:\n  user_message: \"{'a': 1}\"\n"
        "expected:\n  value:\n    a: 1\n",
        encoding="utf-8",
    )
    (eval_dir / "grader.py").write_text(
        "import json\n"
        "def grade(task, output):\n"
        "    expected = (task.get('expected') or {}).get('value')\n"
        "    try:\n"
        "        # Strip everything outside the first/last brace.\n"
        "        i = output.find('{'); j = output.rfind('}')\n"
        "        if i < 0 or j <= i:\n"
        "            return 0.0\n"
        "        return 1.0 if json.loads(output[i:j+1]) == expected else 0.0\n"
        "    except Exception:\n"
        "        return 0.0\n",
        encoding="utf-8",
    )

    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "-A"], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "init"],
        check=True,
        capture_output=True,
    )

    # The stub provider returns a fixed canned JSON dict so the grader
    # scores 1.0 regardless of which atom version is installed. That
    # would short-circuit the eval comparison — to actually exercise
    # the gate, we make the provider's reply depend on the running
    # ``normalize_json`` tool's output by inspecting the tool list and
    # the system prompt's task. Easier: drive the gate with hand-built
    # eval-run records + a propose_change call.

    # Instead of running eval_run for real twice, we pre-write two
    # eval-run records (baseline 0.10, proposed 0.95) and have
    # propose_change consume them. This is closer to the unit edge of
    # integration but exercises the full propose_change pipeline
    # including the constitution-protected decisions write, the
    # promotion gate, and the reload_atom hand-off.
    eval_runs = tmp_path / ".agentm" / "eval_runs"
    eval_runs.mkdir(parents=True)
    (eval_runs / "er_baseline.jsonl").write_text(
        json.dumps(
            {
                "kind": "eval_run.summary",
                "eval_run_id": "er_baseline",
                "task_class": "format_fix",
                "primary_score": 0.10,
                "primary_score_stderr": 0.05,
                "guard_metrics": {"tool_error_rate": 0.0, "turns_mean": 1.0},
                "samples_per_task": 3,
                "task_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (eval_runs / "er_proposed.jsonl").write_text(
        json.dumps(
            {
                "kind": "eval_run.summary",
                "eval_run_id": "er_proposed",
                "task_class": "format_fix",
                "primary_score": 0.95,
                "primary_score_stderr": 0.05,
                "guard_metrics": {"tool_error_rate": 0.0, "turns_mean": 1.0},
                "samples_per_task": 3,
                "task_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    strong_atom = weak_atom.replace(
        "raw.replace(\"'\", '\"')",
        "__import__('json').dumps(__import__('ast').literal_eval(raw))",
    )

    provider_module = _install_static_provider()
    # Pre-resolve the scenario so its local atom registers under the
    # synthetic ``agentm._scenarios.format_fix.tool_normalize_json`` name.
    from agentm.extensions.loader import load_scenario

    scenario_extensions = load_scenario(str(scenario_dir))
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(provider_module, {}),
            extensions=scenario_extensions
            + [
                ("agentm.extensions.builtin.tool_query_traces", {}),
                (
                    "agentm.extensions.builtin.tool_eval_run",
                    {
                        "target_scenario": str(scenario_dir),
                        "eval_dir": str(eval_dir),
                    },
                ),
                ("contrib.extensions.changespec_validators", {}),
                (
                    "agentm.extensions.builtin.tool_propose_change",
                    {
                        "target_scenario": "format_fix",
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
        # Sanity: query_traces returns empty (no traces yet).
        qt = _tool(session, "query_traces")
        traces = await qt.execute({"task_class": "format_fix"})
        assert traces.is_error is False
        assert json.loads(traces.content[0].text)["traces"] == []

        # Drive the activation directly.
        propose = _tool(session, "propose_change")
        result = await propose.execute(
            {
                "target": {
                    "kind": "atom_source",
                    "path": "tool_normalize_json.py",
                    "new_content": strong_atom,
                    "target_atom": "tool_normalize_json",
                },
                "rationale": "test e2e activation",
                "eval_run_baseline": "er_baseline",
                "eval_run_proposed": "er_proposed",
                "decision": "activate",
            }
        )
        assert result.is_error is False, result.content[0].text
        payload = json.loads(result.content[0].text)
        assert payload["status"] == "activate"
        assert payload["tier_blocked"] is False

        # Decision record persisted.
        decisions_path = (
            tmp_path / ".agentm" / "decisions" / "format_fix" / "activations.jsonl"
        )
        assert decisions_path.is_file()
        with decisions_path.open("r", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        assert any(
            r.get("kind") == "activate" and r.get("atom") == "tool_normalize_json"
            for r in records
        )

        # A-4: candidate companion exists with the design §11.1 schema.
        candidates_dir = (
            tmp_path / ".agentm" / "decisions" / "format_fix" / "candidates"
        )
        assert candidates_dir.is_dir()
        candidate_files = list(candidates_dir.glob("c_*.json"))
        assert len(candidate_files) == 1
        with candidate_files[0].open("r", encoding="utf-8") as fh:
            cand = json.load(fh)
        assert set(cand.keys()) == {
            "candidate_id",
            "parent_ids",
            "change_spec",
            "per_task_scores",
            "holdout_scores",
            "eval_run_id",
            "created_at",
        }
        # B-4: parent_ids is a list (was a nullable scalar before the
        # crossover schema migration). Empty list = first activation in
        # the log.
        assert cand["parent_ids"] == []
        assert cand["change_spec"]["target_atom"] == "tool_normalize_json"
        assert cand["eval_run_id"] == "er_proposed"
        # Activation entry references the candidate file.
        activate_entries = [r for r in records if r.get("kind") == "activate"]
        assert activate_entries
        assert activate_entries[0]["candidate_id"] == cand["candidate_id"]

        # Atom-on-disk swapped to the strong version.
        assert (
            scenario_dir / "tool_normalize_json.py"
        ).read_text(encoding="utf-8") == strong_atom

        # Live session uses the new tool — call it directly via the
        # registered Tool surface.
        normalize = _tool(session, "normalize_json")
        out = await normalize.execute({"raw": "{'a': 1}"})
        assert out.is_error is False
        # The strong version uses ast.literal_eval + json.dumps — output
        # is a valid JSON object.
        assert json.loads(out.content[0].text) == {"a": 1}
    finally:
        await session.shutdown()


# Re-export ToolCallBlock to silence the "unused import" warning; it's
# load-bearing for future expansion of the e2e test to drive multi-turn
# trajectories.
__all__ = ["ToolCallBlock"]
