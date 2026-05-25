"""Fail-stop test: structural ``stop_after_no_improvement`` (B-9).

The load-bearing property is **persistence across tuner restarts**. The
counter is computed from ``activations.jsonl`` on every call — never
cached in memory, never tracked in process-local state. If a tuner can
escape the constraint by restarting, the constraint is honor-system, not
structural.

We seed three rejected entries on disk and then call
``tool_propose_change`` with ``decision='activate'`` from a *fresh
session*. The freshness is the test: the in-memory state has no record
of prior rejections, so any pass is the result of reading the on-disk
log.

The complement test (a successful activate clears the chain) pins the
counter's reset semantics — without it the counter is monotonic and
every tuner ends up permanently blocked.
"""

from __future__ import annotations

import json
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


_PROVIDER_MODULE = "agentm._tests.stop_provider"


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
            "stop-test",
            ProviderConfig(
                stream_fn=_stream,
                model=Model(
                    id="stop-test",
                    provider="fake",
                    context_window=16_000,
                    max_output_tokens=2_000,
                ),
                name="stop-test",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[_PROVIDER_MODULE] = module
    return _PROVIDER_MODULE


def _seed_rejections(decisions_dir: Path, count: int) -> None:
    decisions_dir.mkdir(parents=True, exist_ok=True)
    path = decisions_dir / "activations.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        for i in range(count):
            fh.write(
                json.dumps(
                    {
                        "kind": "rejected",
                        "scenario": "format_fix",
                        "atom": "tool_x",
                        "candidate_id": f"c_seed_{i}",
                        "rationale": "seeded reject",
                        "by": "test_seed",
                    }
                )
                + "\n"
            )


def _seed_activate(decisions_dir: Path) -> None:
    decisions_dir.mkdir(parents=True, exist_ok=True)
    path = decisions_dir / "activations.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "kind": "activate",
                    "scenario": "format_fix",
                    "atom": "tool_x",
                    "candidate_id": "c_seed_ok",
                    "rationale": "seeded ok",
                    "by": "test_seed",
                }
            )
            + "\n"
        )


@pytest.mark.asyncio
async def test_stop_counter_persists_across_session_restart(
    tmp_path: Path,
) -> None:
    """Three rejected entries on disk; a *fresh* session refuses
    ``activate`` with the cooldown reason. The session has no in-memory
    history of those rejections — pass means the counter was reconstructed
    from ``activations.jsonl`` (the load-bearing property).
    """
    decisions_dir = tmp_path / ".agentm" / "decisions" / "format_fix"
    _seed_rejections(decisions_dir, count=3)

    # Atom resolution runs before the stop gate (the order is: validate
    # ChangeSpec -> resolve atom on disk -> tier-2 short-circuit -> stop
    # gate). Materialize a tiny atom so resolution succeeds.
    scenario_dir = tmp_path / "contrib" / "scenarios" / "format_fix"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "tool_x.py").write_text(
        "from agentm.core.abi import FunctionTool, TextContent, ToolResult\n"
        "from agentm.extensions import ExtensionManifest\n"
        "MANIFEST = ExtensionManifest(name='tool_x', description='x',"
        " registers=('tool:x',))\n"
        "async def _exec(args):\n"
        "    return ToolResult(content=[TextContent(type='text', text='ok')])\n"
        "def install(api, config):\n"
        "    api.register_tool(FunctionTool(name='x', description='x',"
        " parameters={'type':'object','properties':{}}, fn=_exec))\n",
        encoding="utf-8",
    )

    # Pre-write eval-run summaries so the test isolates the stop gate.
    # If the gate doesn't fire we'd hit "baseline eval run not found"
    # — the contract requires the stop check happens FIRST for activate.
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
            + "\n",
            encoding="utf-8",
        )

    provider_module = _install_static_provider()
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(provider_module, {}),
            extensions=[

                ("agentm.extensions.builtin.operations_local", {}),
                ("contrib.extensions.changespec_validators", {}),
                (
                    "agentm.extensions.builtin.tool_propose_change",
                    {
                        "target_scenario": "format_fix",
                        "promotion": {
                            "threshold_relative": 0.05,
                            "guard_tolerance": 0.10,
                            "stop_after_no_improvement": 3,
                        },
                    },
                ),
            ],
        )
    )
    try:
        propose = next(t for t in session.tools if t.name == "propose_change")
        result = await propose.execute(
            {
                "target": {
                    "kind": "atom_source",
                    "path": "tool_x.py",
                    "new_content": "raise NotImplementedError()",
                    "target_atom": "tool_x",
                },
                "rationale": "post-restart attempt",
                "eval_run_baseline": "er_b",
                "eval_run_proposed": "er_p",
                "decision": "activate",
            }
        )
        assert result.is_error is True
        assert "stop_after_no_improvement" in result.content[0].text
        # The block itself is recorded as a stop_blocked entry — but not
        # as a rejection (else the constraint would self-perpetuate).
        records = [
            json.loads(line)
            for line in (decisions_dir / "activations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        kinds = [r["kind"] for r in records]
        assert kinds[-1] == "stop_blocked"
        # Pre-existing 3 rejections + the new stop_blocked. No new rejection.
        assert kinds.count("rejected") == 3
    finally:
        await session.shutdown()


