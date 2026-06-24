"""Offline seams for replay drivers.

:class:`StandaloneChildRunner` — spawns a top-level audit child via
:func:`llmharness.eval.replay.engine.run_phase_standalone` (no
``api.spawn_child_session``).
"""

from __future__ import annotations

from typing import Any

from llmharness.agents.auditor.tools import SUBMIT_VERDICT_TOOL_NAME
from llmharness.schema import Verdict

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

    async def run_auditor(
        self,
        *,
        prompt_text: str,
        tools_config: dict[str, Any],
        provider: tuple[str, dict[str, Any]] | None,
        context_index: dict[str, Any] | None = None,
        recent_verdicts: list[dict[str, Any]] | None = None,
        continuation_notes_from_prior_firing: list[str] | None = None,
        trajectory: list[dict[str, Any]] | None = None,
        symbols: list[dict[str, Any]] | None = None,
        references: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run one auditor firing as a top-level session.

        Returns a dict with keys: verdict, raw_blocks, error, latency_ms.
        """
        _AUD_TOOLS = "llmharness.agents.auditor.tools"
        _IDX_TOOLS = "llmharness.agents.auditor.index_tools"
        _OBS = "agentm.extensions.builtin.observability"
        _OPS = "agentm.extensions.builtin.operations"
        _SYS = "agentm.extensions.builtin.system_prompt"
        extensions: list[tuple[str, dict[str, Any]]] = [
            (_OBS, {}), (_OPS, {}),
            (_AUD_TOOLS, dict(tools_config)),
            (_IDX_TOOLS, {
                "trajectory": trajectory or [],
                "symbols": symbols or [],
                "references": references or [],
            }),
            (_SYS, {"prompt": prompt_text}),
        ]

        payload: dict[str, Any] = {
            "context_index": context_index,
            "recent_verdicts": list(recent_verdicts or []),
            "continuation_notes_from_prior_firing": list(continuation_notes_from_prior_firing or []),
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

__all__ = ["StandaloneChildRunner"]
