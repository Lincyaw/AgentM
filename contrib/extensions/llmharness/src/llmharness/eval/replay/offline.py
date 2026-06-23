"""Offline seams for replay drivers.

* :class:`StandaloneChildRunner` — spawns a top-level audit child via
  :func:`llmharness.eval.replay.engine.run_phase_standalone` (no
  ``api.spawn_child_session``).

* :class:`InMemorySink` — drops every persisted entry into a Python list
  instead of the AgentM session log.
"""

from __future__ import annotations

import json
from typing import Any

from llmharness.agents.auditor.tools import SUBMIT_VERDICT_TOOL_NAME
from llmharness.agents.extractor.tools import (
    FINALIZE_EXTRACTION_TOOL_NAME,
    ExtractionState,
    IndexOp,
)
from llmharness.schema import Event, Verdict

from .engine import run_phase_standalone


def _flatten_assistant_blocks(messages: list[Any]) -> list[dict[str, Any]]:
    """Extract serialized content blocks from AssistantMessages."""
    from agentm.core.abi import AssistantMessage, ToolCallBlock

    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolCallBlock):
                out.append({
                    "type": "tool_call",
                    "name": block.name,
                    "arguments": dict(block.arguments),
                })
            elif hasattr(block, "text"):
                btype = getattr(block, "type", "text")
                out.append({"type": btype, "text": block.text})
    return out

# --- child runner -----------------------------------------------------------

class ExtractorSpawnError(RuntimeError):
    """Raised by the child runner on spawn/prompt failure."""

class StandaloneChildRunner:
    """Spawns top-level sessions per phase for offline replay."""

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

    async def run_extractor(
        self,
        *,
        state: ExtractionState,
        prompt_text: str,
        provider: tuple[str, dict[str, Any]] | None,
        payload: dict[str, Any],
        turn_window: list[int],
        tool_call_budget: int | None = None,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Run one extractor firing as a top-level session."""
        del turn_window
        _EXT_TOOLS = "llmharness.agents.extractor.tools"
        _OBS = "agentm.extensions.builtin.observability"
        _OPS = "agentm.extensions.builtin.operations"
        _SYS = "agentm.extensions.builtin.system_prompt"
        extensions: list[tuple[str, dict[str, Any]]] = [
            (_OBS, {}), (_OPS, {}),
            (_EXT_TOOLS, {"state": state, "llmharness.extractor_state": state}),
            (_SYS, {"prompt": prompt_text}),
        ]
        if tool_call_budget is not None:
            budget = int(tool_call_budget)
            extensions.extend([
                ("agentm.extensions.builtin.loop_budget", {"max_tool_calls": budget}),
                ("agentm.extensions.builtin.turn_reminder", {"warn_within": budget}),
            ])

        payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        result = await run_phase_standalone(
            cwd=self._cwd,
            extensions=extensions,
            provider=provider,
            payload=payload_json,
            terminal_tool=FINALIZE_EXTRACTION_TOOL_NAME,
            purpose="cognitive_audit_extractor_offline",
            parent_session_id=self._parent_session_id,
            trace_id=self._trace_id,
        )
        if result.status in ("spawn_error", "prompt_error"):
            raise ExtractorSpawnError(result.error or result.status)
        raw_blocks = _flatten_assistant_blocks(result.messages)
        return (result.status == "ok", raw_blocks)

    async def run_auditor(
        self,
        *,
        prompt_text: str,
        tools_config: dict[str, Any],
        provider: tuple[str, dict[str, Any]] | None,
        records: list[Event],
        links: list[dict[str, Any]],
        recent_verdicts: list[dict[str, Any]],
        continuation_notes_from_prior_firing: list[str],
    ) -> dict[str, Any]:
        """Run one auditor firing as a top-level session.

        Returns a dict with keys: verdict, raw_blocks, error, latency_ms.
        """
        _AUD_TOOLS = "llmharness.agents.auditor.tools"
        _OBS = "agentm.extensions.builtin.observability"
        _OPS = "agentm.extensions.builtin.operations"
        _SYS = "agentm.extensions.builtin.system_prompt"
        extensions: list[tuple[str, dict[str, Any]]] = [
            (_OBS, {}), (_OPS, {}),
            (_AUD_TOOLS, dict(tools_config)),
            (_SYS, {"prompt": prompt_text}),
        ]

        payload: dict[str, Any] = {
            "records": [e.to_dict() for e in records],
            "links": list(links),
            "recent_verdicts": list(recent_verdicts),
            "continuation_notes_from_prior_firing": list(continuation_notes_from_prior_firing),
        }
        result = await run_phase_standalone(
            cwd=self._cwd,
            extensions=extensions,
            provider=provider,
            payload=payload,
            terminal_tool=SUBMIT_VERDICT_TOOL_NAME,
            purpose="cognitive_audit_auditor_offline",
            parent_session_id=self._parent_session_id,
            trace_id=self._trace_id,
        )
        raw_blocks = _flatten_assistant_blocks(result.messages)
        latency_ms = result.latency_ms

        if result.status != "ok" or result.output is None:
            return {
                "verdict": None,
                "raw_blocks": raw_blocks,
                "error": result.error,
                "latency_ms": latency_ms,
            }

        # Parse verdict from raw output
        try:
            verdict_raw = result.output.get("verdict") or result.output
            if isinstance(verdict_raw, dict):
                verdict = Verdict.from_dict(verdict_raw)
                return {
                    "verdict": verdict,
                    "raw_blocks": raw_blocks,
                    "error": None,
                    "latency_ms": latency_ms,
                }
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "verdict": None,
                "raw_blocks": raw_blocks,
                "error": f"malformed: {exc}",
                "latency_ms": latency_ms,
            }

        return {
            "verdict": None,
            "raw_blocks": raw_blocks,
            "error": "no verdict in output",
            "latency_ms": latency_ms,
        }

# --- op sink ----------------------------------------------------------------

class InMemorySink:
    """Captures every entry in local lists — for offline drivers and tests."""

    def __init__(self) -> None:
        self.ops: list[tuple[IndexOp, int, int, list[int]]] = []
        self.cursors: list[int] = []
        self.verdicts: list[dict[str, Any]] = []
        self.failures: list[tuple[str, dict[str, Any]]] = []
        self.partials: list[dict[str, Any]] = []

    def append_op(
        self,
        op: IndexOp,
        *,
        firing_id: int,
        op_index: int,
        turn_window: list[int],
    ) -> None:
        self.ops.append((op, firing_id, op_index, list(turn_window)))

    def append_cursor(self, *, last_turn_index: int) -> None:
        self.cursors.append(last_turn_index)

    def append_verdict(self, verdict: dict[str, Any]) -> None:
        self.verdicts.append(dict(verdict))

    def append_failure(self, entry_type: str, payload: dict[str, Any]) -> None:
        self.failures.append((entry_type, dict(payload)))

    def append_partial(self, payload: dict[str, Any]) -> None:
        self.partials.append(dict(payload))

__all__ = ["ExtractorSpawnError", "InMemorySink", "StandaloneChildRunner"]
