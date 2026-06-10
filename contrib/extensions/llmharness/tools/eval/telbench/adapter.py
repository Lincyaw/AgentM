"""Convert TELBench instances to cognitive-audit pipeline inputs.

TELBench (arxiv 2606.02060) is a benchmark for span-level error
localization in agent trajectories. Each instance carries a list of
spans (trajectory segments) plus gold-standard error annotations.

This module parses the JSONL dataset and converts spans into synthetic
:class:`AssistantMessage` objects so the extractor/auditor pipeline can
process them identically to a real agent trajectory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentm.core.abi.messages import AgentMessage, AssistantMessage, TextContent


@dataclass(frozen=True)
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
    """Convert TELBench spans to synthetic AssistantMessage list.

    Each span becomes an AssistantMessage with ``role="assistant"``,
    content as a single TextContent block, and ``timestamp=float(index)``.
    The message index equals the span index, so ``source_turns`` from
    extracted events maps directly back to span indices.
    """
    messages: list[AgentMessage] = []
    for idx, span in enumerate(spans):
        msg = AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text=span["raw"])],
            timestamp=float(idx),
        )
        messages.append(msg)
    return messages


__all__ = [
    "TelBenchInstance",
    "load_telbench",
    "spans_to_messages",
]
