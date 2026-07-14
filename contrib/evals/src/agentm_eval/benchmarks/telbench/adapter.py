"""Convert TELBench instances to cognitive-audit pipeline inputs.

TELBench (arxiv 2606.02060) is a benchmark for span-level error
localization in agent trajectories. Each instance carries a list of
spans (trajectory segments) plus gold-standard error annotations.

Spans arrive as bare ``{id, raw}`` — the source trajectory's role
structure was flattened to opaque text before export. Provenance is
therefore recovered, not attested, with ONE exception: when a span's
text is the framework's own serialization of a known kind (a ``subtask``
delegation dict, an ``[ERROR]`` tool failure, a ``Title:/URL Source:``
fetch result), the class is decidable by parsing that format — not by
keyword-guessing free prose. :func:`classify_span` does exactly that, per
framework, and abstains (``unknown``) on everything else. This is the
same discipline as ``data.extract_structural_symbols`` (parse the grammar,
never guess semantics); the deleted keyword classifier that silently
missed 56% of the hard set is what abstention avoids.

Recognized observations (``error`` / ``tool``) become attested
``tool_result`` messages; ``subtask`` (an agent delegation) and every
unrecognized span stay assistant, deferring observation-labeling to
Pass 1. Format alone tops out around a quarter of miroflow spans: the
rest are agent actions — search queries and report prose — that no
marker distinguishes from result text (queries even share a span with
the ``Title:`` result they fetched), so abstention hands them to Pass 1
by design, not by omission. ``oagent`` carries none of these markers
(100% ``unknown`` on the dataset) and abstains entirely until its format
is characterized.

Message index equals span index — gold ``error_span_ids`` map unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    tool_result,
)


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

    @property
    def framework(self) -> str:
        """Agent framework that produced this trajectory (selects the span parser)."""
        return self.meta.get("framework", "")


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


# --- Sound per-framework span classification -------------------------------

SpanClass = Literal["subtask", "error", "tool", "unknown"]

# miroflow non-error tool-output markers — each is a framework repr or a
# system notice, never agent prose (verified by sampling the dataset).
# ``LLM extracted final answer:`` is deliberately ABSENT: despite the
# tool-looking prefix, its body is the model's own answer reasoning, not an
# external observation, so attesting it would be exactly the mis-label the
# abstain discipline exists to prevent.
_MIROFLOW_TOOL_PREFIXES = (
    "Title:",             # web fetch result (paired with a URL Source: line)
    "Page Title:",        # web fetch result (page-scoped variant)
    "Execution(",         # code/sandbox execution result repr
    "CommandResult(",     # shell command result repr
    "File downloaded",    # sandbox file-write environment feedback
    "[NOTE]:",            # system tool-routing notice
)


def _classify_miroflow(raw: str) -> SpanClass:
    """miroflow's serialization markers (decidable by format, not semantics)."""
    s = raw.lstrip()
    if s.startswith(("{'subtask'", '{"subtask"')):
        return "subtask"           # the framework's delegation dict — an agent action
    if s.startswith(("[ERROR]", "[error]")):
        return "error"             # a tool failure — environment feedback
    # Tool-output templates. Sound only when the WHOLE span is the result: a
    # trailing subtask dict means the span bundles an agent action too, so
    # attesting the whole span as observation would mis-label that agent text.
    # Abstain there and let Pass 1 carve the obs region. The ``Title:`` fetch
    # template additionally requires its ``URL Source:`` line to be present.
    if "{'subtask'" in s or '{"subtask"' in s:
        return "unknown"
    if s.startswith("Title:") and "URL Source:" not in s[:200]:
        return "unknown"           # a bare "Title:" line is not a fetch result
    if s.startswith(_MIROFLOW_TOOL_PREFIXES):
        return "tool"
    return "unknown"


# Per-framework classifiers. Add one to onboard a framework; absent → abstain.
# oagent carries none of these markers (100% unknown on the dataset).
_CLASSIFIERS: dict[str, Callable[[str], SpanClass]] = {
    "miroflow": _classify_miroflow,
}


def classify_span(raw: str, framework: str) -> SpanClass:
    """Deterministic span class from the framework's own serialization markers.

    Parsing, not keyword-guessing over free prose: a positive class is
    returned only when the marker proves it, otherwise ``unknown`` (abstain →
    Pass 1 decides). High precision by construction — it never produces a
    wrong label, which is what the deleted keyword classifier could not
    guarantee. An unknown framework abstains on every span.
    """
    fn = _CLASSIFIERS.get(framework)
    return fn(raw) if fn else "unknown"


def spans_to_messages(
    spans: list[dict[str, str]], *, framework: str = "",
) -> list[AgentMessage]:
    """Convert TELBench spans to typed messages, one per span, in order.

    Sound provenance recovery: spans the framework's serialization proves to
    be tool output (``error`` / ``tool``) become attested ``tool_result``
    messages (the whole span counts as observation, no model labeling
    needed); ``subtask`` (an agent delegation) and every unrecognized span
    stay assistant, deferring observation-labeling to Pass 1. ``framework``
    selects the parser — omit it (or pass an unrecognized framework) and every
    span is assistant, the previous whole-dataset-recovered behavior. The
    message index equals the span index, so gold ``error_span_ids`` map
    directly back.
    """
    out: list[AgentMessage] = []
    for idx, span in enumerate(spans):
        raw = span["raw"]
        cls = classify_span(raw, framework)
        if cls in ("error", "tool"):
            out.append(
                tool_result(
                    f"telbench-{idx}", raw,
                    is_error=(cls == "error"), timestamp=float(idx),
                )
            )
        else:
            out.append(
                AssistantMessage(
                    role="assistant",
                    content=[TextContent(type="text", text=raw)],
                    timestamp=float(idx),
                )
            )
    return out


__all__ = [
    "SpanClass",
    "TelBenchInstance",
    "classify_span",
    "load_telbench",
    "spans_to_messages",
]
