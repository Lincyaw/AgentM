"""Standalone auditor session runner for offline evaluation."""

from __future__ import annotations

from typing import Any

from llmharness.agents.auditor.tools import SUBMIT_VERDICT_TOOL_NAME
from llmharness.offline import flatten_assistant_blocks, run_phase_standalone
from llmharness.schema import Verdict


class StandaloneChildRunner:
    """Spawns top-level auditor sessions for offline audit."""

    def __init__(
        self,
        cwd: str,
        *,
        parent_session_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        self._cwd = cwd
        self._parent_session_id = parent_session_id
        self._trace_id = trace_id

    async def run_auditor(
        self,
        *,
        prompt_text: str,
        tools_config: dict[str, Any],
        provider: tuple[str, dict[str, Any]] | None = None,
        model: str | None = None,
        context_index: dict[str, Any] | None = None,
        recent_verdicts: list[dict[str, Any]] | None = None,
        continuation_notes_from_prior_firing: list[str] | None = None,
        trajectory: list[dict[str, Any]] | None = None,
        symbols: list[dict[str, Any]] | None = None,
        references: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        _auditor_tools = "llmharness.agents.auditor.tools"
        _index_tools = "llmharness.agents.auditor.index_tools"
        _observability = "agentm.extensions.builtin.observability"
        _operations = "agentm.extensions.builtin.operations"
        _system_prompt = "agentm.extensions.builtin.system_prompt"
        extensions: list[tuple[str, dict[str, Any]]] = [
            (_observability, {}),
            (_operations, {}),
            (_auditor_tools, dict(tools_config)),
            (
                _index_tools,
                {
                    "trajectory": trajectory or [],
                    "symbols": symbols or [],
                    "references": references or [],
                    "context_index": context_index or {},
                },
            ),
            (_system_prompt, {"prompt": prompt_text}),
        ]
        payload: dict[str, Any] = {
            "context_index": context_index,
            "recent_verdicts": list(recent_verdicts or []),
            "continuation_notes_from_prior_firing": list(
                continuation_notes_from_prior_firing or []
            ),
        }
        result = await run_phase_standalone(
            cwd=self._cwd,
            extensions=extensions,
            provider=provider,
            model=model,
            payload=payload,
            terminal_tool=SUBMIT_VERDICT_TOOL_NAME,
            purpose="cognitive_audit_auditor_offline",
            parent_session_id=self._parent_session_id,
            trace_id=self._trace_id,
        )
        raw_blocks = flatten_assistant_blocks(result.messages)
        if result.status != "ok" or result.output is None:
            return {
                "verdict": None,
                "raw_blocks": raw_blocks,
                "error": result.error,
                "latency_ms": result.latency_ms,
            }
        try:
            verdict_raw = result.output.get("verdict") or result.output
            if isinstance(verdict_raw, dict):
                verdict = Verdict.from_dict(verdict_raw)
                return {
                    "verdict": verdict,
                    "raw_blocks": raw_blocks,
                    "error": None,
                    "latency_ms": result.latency_ms,
                }
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "verdict": None,
                "raw_blocks": raw_blocks,
                "error": f"malformed: {exc}",
                "latency_ms": result.latency_ms,
            }
        return {
            "verdict": None,
            "raw_blocks": raw_blocks,
            "error": "no verdict in output",
            "latency_ms": result.latency_ms,
        }
