"""Convert TELBench instances to cognitive-audit pipeline inputs.

TELBench (arxiv 2606.02060) is a benchmark for span-level error
localization in agent trajectories. Each instance carries a list of
spans (trajectory segments) plus gold-standard error annotations.

This module parses the JSONL dataset and converts spans into typed
messages. Spans arrive as bare ``{id, raw}`` with no provenance — the
conversion reconstructs it by content classification (ingestion is the
one place benchmark-specific heuristics belong):

- **observation** spans (fetched pages ``URL Source:``, command output
  ``CommandResult(`` / ``[ERROR]``) → ``ToolResultMessage`` — they form
  the grounded evidence universe for the index's dataflow layer. Without
  this, the grounding analysis is silently degenerate (zero grounded
  defs; every entity looks fabricated).
- **plan** spans (``{'subtask': ...}`` planner instructions) →
  ``UserMessage``.
- everything else (agent reasoning, queries, reports) →
  ``AssistantMessage``.

Message index still equals span index — gold ``error_span_ids`` map
unchanged (gold can legitimately sit on observation spans: TELBench
attributes a stage's error to that stage's span).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolResultBlock,
    ToolResultMessage,
    UserMessage,
)

# Content markers of tool/environment output. Checked against the span
# head — fetched pages and command results carry these structurally.
_OBS_MARKERS = (
    re.compile(r"^CommandResult\("),
    re.compile(r"^\[ERROR\]"),
    re.compile(r"URL Source:"),
    re.compile(r"Markdown Content:"),
    re.compile(r"^\s*Title:.*\bURL\b"),
    re.compile(r"stdout='"),
)
# Search-result compound spans: `query\n---\nfetched page(s)` — the `---`
# separates result items. Anchored at the span head or right after the
# separator: agent reports also contain `---` (markdown rules) and URLs in
# prose, so an unanchored content match would misclassify final reports —
# the most commitment-heavy agent spans — as observations.
_SEP_RE = re.compile(r"\n---\n")
_WEB_HEAD_RE = re.compile(r"^\s*(?:Title:|https?://)")
_PLAN_RE = re.compile(r"^\{['\"]subtask")
_OBS_SCAN_CHARS = 2000  # markers appear structurally near the head


def _compound_observation(r: str) -> bool:
    """Fetched-content span shape: web-content head, or `query\\n---\\npage`
    where the segment after the first separator has a web-content head.
    Markdown-heading spans are agent reports, never observations."""
    if r.startswith("#"):
        return False
    if _WEB_HEAD_RE.match(r):
        return True
    m = _SEP_RE.search(r)
    if m:
        after = r[m.end():m.end() + 200].lstrip()
        return bool(_WEB_HEAD_RE.match(after))
    return False


def classify_span(raw: str) -> str:
    """Reconstruct a span's provenance: ``plan`` / ``observation`` / ``agent``.

    Unclassified content stays ``agent`` — the honest fallback: an agent
    utterance mislabeled as observation would grant fabricated content
    grounded status, the reverse only loses one grounded def.
    """
    r = raw.strip()
    if _PLAN_RE.match(r):
        return "plan"
    if any(p.search(r[:_OBS_SCAN_CHARS]) for p in _OBS_MARKERS):
        return "observation"
    if _compound_observation(r):
        return "observation"
    return "agent"


@dataclass(frozen=True, slots=True)
class TelBenchInstance:
    """One TELBench evaluation instance."""

    id: str
    source_id: str
    question: str
    spans: list[dict[str, str]]
    gold_error_span_ids: list[str]
    annotations: dict[str, Any]
    meta: dict[str, str]

    @property
    def span_ids(self) -> list[str]:
        return [s["id"] for s in self.spans]

    @property
    def gold_error_indices(self) -> set[int]:
        """0-based indices of error spans."""
        id_to_idx = {s["id"]: i for i, s in enumerate(self.spans)}
        return {id_to_idx[sid] for sid in self.gold_error_span_ids if sid in id_to_idx}


def load_telbench(path: Path) -> list[TelBenchInstance]:
    """Parse a TELBench JSONL file into typed instances."""
    instances: list[TelBenchInstance] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gold = obj.get("gold", {})
            instances.append(
                TelBenchInstance(
                    id=str(obj["id"]),
                    source_id=str(obj.get("source_id", "")),
                    question=str(obj.get("question", "")),
                    spans=list(obj.get("spans", [])),
                    gold_error_span_ids=list(gold.get("error_span_ids", [])),
                    annotations=dict(obj.get("annotations", {})),
                    meta=dict(obj.get("meta", {})),
                )
            )
    return instances


def spans_to_messages(spans: list[dict[str, str]]) -> list[AgentMessage]:
    """Convert TELBench spans to typed messages with reconstructed provenance.

    One message per span, in order — the message index equals the span
    index, so gold ``error_span_ids`` and extracted-event ``source_turns``
    map directly back to span indices. Roles come from :func:`classify_span`.
    """
    messages: list[AgentMessage] = []
    for idx, span in enumerate(spans):
        raw = span["raw"]
        kind = classify_span(raw)
        msg: AgentMessage
        if kind == "observation":
            msg = ToolResultMessage(
                role="tool_result",
                content=[ToolResultBlock(
                    type="tool_result",
                    tool_call_id="",
                    content=[TextContent(type="text", text=raw)],
                    is_error=False,
                )],
                timestamp=float(idx),
            )
        elif kind == "plan":
            msg = UserMessage(
                role="user",
                content=[TextContent(type="text", text=raw)],
                timestamp=float(idx),
            )
        else:
            msg = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=raw)],
                timestamp=float(idx),
            )
        messages.append(msg)
    return messages


__all__ = [
    "TelBenchInstance",
    "load_telbench",
    "spans_to_messages",
]
