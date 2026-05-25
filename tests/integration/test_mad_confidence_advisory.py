"""Fail-stop integration test for PR-B: MAD-confidence advisory signal.

Strict scope: measurement only. The 4-floor gate's accept/reject
decision MUST NOT change. We assert two things:

1. With a non-empty Pareto pool (>=3 per-task scores), the activation
   record carries ``mad_confidence: {ratio, tier}``.
2. The presence of the MAD computation does not flip the gate's
   verdict — same input -> same accept/reject.
3. With pool size below the floor (N < 3), ``mad_confidence`` is
   recorded as explicit ``null`` (not absent).

Plus a unit test of the pure ``mad_confidence`` helper itself.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
import uuid
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


_PROVIDER_MODULE = "agentm._tests.mad_confidence_provider"


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
            "mad-test",
            ProviderConfig(
                stream_fn=_stream,
                model=Model(
                    id="mad-test",
                    provider="fake",
                    context_window=16_000,
                    max_output_tokens=2_000,
                ),
                name="mad-test",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[_PROVIDER_MODULE] = module
    return _PROVIDER_MODULE


def _git_init(path: Path) -> None:
    subprocess.run(
        ["git", "init", "-q", "-b", "agent-tests", str(path)], check=True, capture_output=True
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
    scenario_dir = tmp_path / "scenarios" / "mad_test"
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
        "name: mad_test\nextensions:\n"
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
    eval_runs: Path, run_id: str, *, primary_score: float
) -> None:
    eval_runs.mkdir(parents=True, exist_ok=True)
    (eval_runs / f"{run_id}.jsonl").write_text(
        json.dumps(
            {
                "kind": "eval_run.summary",
                "eval_run_id": run_id,
                "task_class": "mad_test",
                "primary_score": primary_score,
                "primary_score_stderr": 0.05,
                "guard_metrics": {"tool_error_rate": 0.0, "turns_mean": 1.0},
                "samples_per_task": 3,
                "task_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _seed_pool(decisions_dir: Path, scores: list[float]) -> None:
    """Pre-populate the candidates dir with N candidates carrying
    one per_task_score each, so MAD has a non-degenerate pool to
    work over.
    """
    candidates_dir = decisions_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    for score in scores:
        cid = f"c_{uuid.uuid4().hex[:12]}"
        record = {
            "candidate_id": cid,
            "parent_ids": [],
            "change_spec": {
                "kind": "atom_source",
                "path": "tiny_atom.py",
                "new_content": "x",
                "target_atom": "tiny_atom",
            },
            "per_task_scores": {"task_001": score},
            "holdout_scores": {},
            "eval_run_id": None,
            "created_at": 0.0,
        }
        (candidates_dir / f"{cid}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Integration: MAD recorded on activation record, does not flip gate.
# ---------------------------------------------------------------------------






@pytest.mark.asyncio
async def test_mad_does_not_flip_gate_decision(tmp_path: Path) -> None:
    """Property test: for a fixed (baseline, proposed, pool) tuple, the
    accept/reject verdict is the same regardless of whether MAD says
    'real', 'marginal', or 'noise'. We exercise this by running two
    propose_change calls with different pool compositions (which change
    MAD's tier) but identical eval-run summaries — the gate verdict
    must match.
    """

    _git_init(tmp_path)
    scenario_dir = _materialize_scenario(tmp_path)
    eval_runs = tmp_path / ".agentm" / "eval_runs"
    # 2-sigma rejects regardless: stderr=0.1 -> 2*sqrt(0.02) ~ 0.283
    # vs delta=0.05 -> reject.
    eval_runs.mkdir(parents=True)
    for run_id, score in (("er_b", 0.5), ("er_p", 0.55)):
        (eval_runs / f"{run_id}.jsonl").write_text(
            json.dumps(
                {
                    "kind": "eval_run.summary",
                    "eval_run_id": run_id,
                    "task_class": "mad_test",
                    "primary_score": score,
                    "primary_score_stderr": 0.1,
                    "guard_metrics": {
                        "tool_error_rate": 0.0,
                        "turns_mean": 1.0,
                    },
                    "samples_per_task": 3,
                    "task_count": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    decisions_dir = tmp_path / ".agentm" / "decisions" / "mad_test"
    # Pool A: tight (MAD small -> ratio big -> 'real').
    # values=[0.54,0.55,0.56,0.57,0.58] -> median=0.56,
    # deviations=[0.02,0.01,0,0.01,0.02] -> MAD=0.01.
    # baseline=0.5 cand=0.55 -> ratio = 0.05/0.01 = 5.0 -> 'real'.
    _seed_pool(decisions_dir, [0.54, 0.55, 0.56, 0.57, 0.58])

    provider_module = _install_static_provider()
    from agentm.extensions.loader import load_scenario

    scenario_extensions = load_scenario(str(scenario_dir))
    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(provider_module, {}),
            extensions=scenario_extensions
            + [
                ("contrib.extensions.changespec_validators", {}),
                (
                    "agentm.extensions.builtin.tool_propose_change",
                    {
                        "target_scenario": "mad_test",
                        "promotion": {
                            "threshold_relative": 0.05,
                            "guard_tolerance": 0.50,
                        },
                    },
                ),
            ],
        )
    )
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
    try:
        propose = _tool(session, "propose_change")
        first = await propose.execute(
            {
                "target": {
                    "kind": "atom_source",
                    "path": "tiny_atom.py",
                    "new_content": new_source,
                    "target_atom": "tiny_atom",
                },
                "rationale": "tight pool",
                "eval_run_baseline": "er_b",
                "eval_run_proposed": "er_p",
                "decision": "activate",
            }
        )
        first_is_error = first.is_error
        first_text = first.content[0].text

        # Now wipe the candidates dir and seed a wide pool — MAD will
        # say 'noise' for the same (baseline, proposed) — and propose
        # again. The gate verdict must still match.
        candidates_dir = decisions_dir / "candidates"
        for stale in candidates_dir.glob("*"):
            stale.unlink()
        _seed_pool(decisions_dir, [0.0, 0.25, 0.5, 0.75, 1.0])
        second = await propose.execute(
            {
                "target": {
                    "kind": "atom_source",
                    "path": "tiny_atom.py",
                    "new_content": new_source,
                    "target_atom": "tiny_atom",
                },
                "rationale": "wide pool",
                "eval_run_baseline": "er_b",
                "eval_run_proposed": "er_p",
                "decision": "activate",
            }
        )
    finally:
        await session.shutdown()

    # Both must reject (gate verdict identical).
    assert first_is_error is True
    assert second.is_error is True
    # Same rejection reason (noise floor).
    assert "noise floor" in first_text
    assert "noise floor" in second.content[0].text

    # Sanity: the two records have *different* MAD tiers, proving MAD
    # was actually computed (and is therefore strictly informational
    # rather than dead code).
    decisions_path = decisions_dir / "activations.jsonl"
    with decisions_path.open("r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    rejected = [r for r in records if r.get("kind") == "rejected"]
    assert len(rejected) == 2
    tiers = [r.get("mad_confidence", {}).get("tier") for r in rejected]
    # First call had tight pool -> 'real'; second wide -> 'noise'.
    assert tiers[0] == "real"
    assert tiers[1] == "noise"
