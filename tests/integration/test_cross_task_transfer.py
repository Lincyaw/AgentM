"""Integration test: B-8 cross-task transfer.

Opt-in behavior: when ``tool_propose_change`` is configured with
``transfer_to: [<sibling>]``, a successful ``activate`` on a kind=atom_source
change plants a candidate (NOT an activation) in each sibling scenario's
``.agentm/decisions/<sibling>/candidates/`` directory, marked
``transferred_from=<source>``. The sibling's tuner sees it via
``tool_query_candidates`` and decides independently whether to eval +
activate.

We assert:
- A candidate file appears under the destination scenario.
- The record carries ``transferred_from`` and ``source_*`` traceability.
- ``per_task_scores`` is empty (different task class, scores don't carry).
- No activation ever appears in the destination's ``activations.jsonl``
  (transfer is non-activating by contract).

Per the plan, no fail-stop test — opt-in feature; this is integration only.
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

_PROVIDER_MODULE = "agentm._tests.xfer_provider"


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
            "xfer-test",
            ProviderConfig(
                stream_fn=_stream,
                model=Model(
                    id="xfer-test",
                    provider="fake",
                    context_window=16_000,
                    max_output_tokens=2_000,
                ),
                name="xfer-test",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[_PROVIDER_MODULE] = module
    return _PROVIDER_MODULE


_ATOM_V1 = (
    "from agentm.core.abi import FunctionTool, TextContent, ToolResult\n"
    "from agentm.extensions import ExtensionManifest\n"
    "MANIFEST = ExtensionManifest(name='tool_x', description='x',"
    " registers=('tool:x',))\n"
    "async def _exec(args):\n"
    "    return ToolResult(content=[TextContent(type='text', text='v1')])\n"
    "def install(api, config):\n"
    "    api.register_tool(FunctionTool(name='x', description='x',"
    " parameters={'type':'object','properties':{}}, fn=_exec))\n"
)


_ATOM_V2 = _ATOM_V1.replace("'v1'", "'v2'")


@pytest.mark.asyncio
async def test_transfer_plants_candidate_in_sibling_scenario(
    tmp_path: Path,
) -> None:
    """A successful atom_source activate in ``rca`` writes a candidate to
    ``review_mode``'s candidates/ directory carrying ``transferred_from='rca'``.
    The sibling's activations.jsonl is not touched — transfer never deploys.
    """
    # Source scenario: real on-disk atom so cross-session resolution works.
    source_dir = tmp_path / "contrib" / "scenarios" / "rca"
    source_dir.mkdir(parents=True)
    (source_dir / "tool_x.py").write_text(_ATOM_V1, encoding="utf-8")

    # Eval-runs that pass the deployment gate (huge delta, low stderr).
    eval_runs = tmp_path / ".agentm" / "eval_runs"
    eval_runs.mkdir(parents=True, exist_ok=True)
    (eval_runs / "er_b.jsonl").write_text(
        json.dumps(
            {
                "kind": "eval_run.summary",
                "eval_run_id": "er_b",
                "primary_score": 0.10,
                "primary_score_stderr": 0.05,
                "guard_metrics": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (eval_runs / "er_p.jsonl").write_text(
        json.dumps(
            {
                "kind": "eval_run.summary",
                "eval_run_id": "er_p",
                "primary_score": 0.95,
                "primary_score_stderr": 0.05,
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
                (
                    "agentm.extensions.builtin.tool_propose_change",
                    {
                        "target_scenario": "rca",
                        "transfer_to": ["review_mode", "general_purpose"],
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
                    "new_content": _ATOM_V2,
                    "target_atom": "tool_x",
                },
                "rationale": "rca tuner activate",
                "eval_run_baseline": "er_b",
                "eval_run_proposed": "er_p",
                "decision": "activate",
            }
        )
        assert result.is_error is False, result.content[0].text
        payload = json.loads(result.content[0].text)
        assert payload["status"] == "activate"
        # The atom message reports which siblings got transferred candidates.
        transferred = payload.get("transferred") or []
        assert any(s.startswith("review_mode:") for s in transferred), transferred
        assert any(s.startswith("general_purpose:") for s in transferred), transferred

        # Destination candidates exist and carry transfer metadata.
        for sibling in ("review_mode", "general_purpose"):
            cand_dir = (
                tmp_path / ".agentm" / "decisions" / sibling / "candidates"
            )
            files = list(cand_dir.glob("c_*.json"))
            assert len(files) == 1, f"sibling {sibling}: {files}"
            rec = json.loads(files[0].read_text(encoding="utf-8"))
            assert rec["transferred_from"] == "rca"
            assert rec["source_eval_run_id"] == "er_p"
            assert rec["source_candidate_id"]  # non-empty traceability
            # Per-task scores do not carry — destination must re-eval to
            # claim a frontier slot.
            assert rec["per_task_scores"] == {}
            assert rec["holdout_scores"] == {}
            # Roots in the destination subtree (no parents inherited).
            assert rec["parent_ids"] == []

            # No activation occurred in the sibling — transfer is non-activating.
            sibling_log = (
                tmp_path
                / ".agentm"
                / "decisions"
                / sibling
                / "activations.jsonl"
            )
            assert not sibling_log.is_file(), (
                f"transfer must never deploy in sibling {sibling}"
            )

            # No tree.jsonl in the destination — transferred candidates are
            # roots of new subtrees, not edges in the source's lineage graph.
            assert not (
                tmp_path / ".agentm" / "decisions" / sibling / "tree.jsonl"
            ).is_file()

        # Source candidate retained the original (non-transferred) lineage.
        src_candidates = (
            tmp_path / ".agentm" / "decisions" / "rca" / "candidates"
        )
        src_files = list(src_candidates.glob("c_*.json"))
        assert len(src_files) == 1
        src_rec = json.loads(src_files[0].read_text(encoding="utf-8"))
        assert "transferred_from" not in src_rec
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_transfer_skipped_for_non_atom_source(tmp_path: Path) -> None:
    """Only ``kind='atom_source'`` transfers. A successful manifest_field
    activation does NOT plant transfer candidates — system_prompt /
    manifest changes are scenario-shape and don't carry cleanly.
    """
    # Source scenario with a manifest.yaml the manifest_field validator
    # can resolve.
    source_dir = tmp_path / "contrib" / "scenarios" / "rca"
    source_dir.mkdir(parents=True)
    (source_dir / "manifest.yaml").write_text(
        "name: rca\nextensions: []\n", encoding="utf-8"
    )

    eval_runs = tmp_path / ".agentm" / "eval_runs"
    eval_runs.mkdir(parents=True, exist_ok=True)
    for run_id, score in (("er_b", 0.10), ("er_p", 0.95)):
        (eval_runs / f"{run_id}.jsonl").write_text(
            json.dumps(
                {
                    "kind": "eval_run.summary",
                    "eval_run_id": run_id,
                    "primary_score": score,
                    "primary_score_stderr": 0.05,
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
                (
                    "agentm.extensions.builtin.tool_propose_change",
                    {
                        "target_scenario": "rca",
                        "transfer_to": ["review_mode"],
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
                    "kind": "system_prompt",
                    "path": "system_prompt.md",
                    "new_content": "new prompt body\n",
                    "target_atom": None,
                },
                "rationale": "system prompt rewrite",
                "eval_run_baseline": "er_b",
                "eval_run_proposed": "er_p",
                "decision": "activate",
            }
        )
        # Whether the gate passes is incidental; what matters is no transfer
        # was planted regardless of outcome.
        sibling_dir = tmp_path / ".agentm" / "decisions" / "review_mode"
        if result.is_error is False:
            payload = json.loads(result.content[0].text)
            assert payload.get("transferred") in ([], None), payload
        # Sibling candidates dir either absent or empty.
        if sibling_dir.is_dir():
            cand_files = list((sibling_dir / "candidates").glob("c_*.json")) \
                if (sibling_dir / "candidates").is_dir() else []
            assert cand_files == []
    finally:
        await session.shutdown()
