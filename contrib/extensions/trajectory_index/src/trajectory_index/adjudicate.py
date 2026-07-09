"""Model-judgment side of the pipeline — the semantic decisions code can't make.

The rest of ``trajectory_index`` is deterministic (Pass 1 tagging, Pass 3 dataflow).
This module is the one place a model earns its keep, and it only ever answers a
*local* question — it never traverses the graph. Code does the traversal and
propagation ("the model gives a point, code propagates it"; SCHEMA-readwrite).

* ``resolve_aliases`` — **Pass 2, name resolution.** "Are these two surface forms
  the same entity?" Code blocks candidate pairs (``alias_candidates``), the model
  judges each pair here, and code clusters the yes-pairs into merge groups for
  ``TrajectoryIndex.apply_alias_merges``. Runs **before** ``build_dependencies``.

The deferred fabricated-value check (does a tool's value confirm or contradict the
value the model acted on?) would live here too, as a local compare at the tail of
Pass 3's dataflow — see SCHEMA-readwrite "Deferred". Not built.

The model is swappable (~8B is enough); each call takes a ``model`` name and speaks
plain JSON.
"""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from .data import extract_json

if TYPE_CHECKING:
    from .index import AliasCandidate, TrajectoryIndex


# ---------------------------------------------------------------------------
# Alias resolution: decide same-entity pairs -> cluster -> merge groups
# ---------------------------------------------------------------------------

_SAME_ENTITY_INSTRUCTIONS = """\
You are an entity-resolution judge for an agent-trajectory index. Each numbered
pair below is two surface forms that a lexical blocker flagged as *possibly* the
same underlying entity. Decide, for each pair independently, whether the two forms
denote the SAME concrete entity in this trajectory.

Same entity (merge) — examples of the reasoning, not rules:
  - a data file and the table/view registered from it (``X.parquet`` and ``X``);
  - a full identifier and an unambiguous short form of it;
  - the same endpoint written two ways.

Different entity (do NOT merge), even when the strings look close:
  - opposites or paired variants (``abnormal_*`` vs ``normal_*``);
  - singular vs plural of different resources (``.../travel`` vs ``.../travels``);
  - sibling metrics/paths sharing a prefix but naming different things
    (``cpu.usage`` vs ``memory.usage``; ``page_faults`` vs ``major_page_faults``).

Judge on identity, not string similarity. When unsure, answer false.

Return ONLY a JSON object of this exact shape, one verdict per pair id:
{"verdicts": [{"id": 0, "same": true, "confidence": 0.9, "reason": "..."}, ...]}
"""


@dataclass(frozen=True, slots=True)
class SameEntityVerdict:
    """One model decision over an ``AliasCandidate`` (aligned by list position)."""

    same: bool
    confidence: float
    reason: str


def _format_candidates(candidates: list[AliasCandidate]) -> str:
    rows: list[dict[str, Any]] = []
    for i, c in enumerate(candidates):
        rows.append(
            {
                "id": i,
                "a": {"name": c.name_a, "kind": c.kind_a, "uses": list(c.context_a)},
                "b": {"name": c.name_b, "kind": c.kind_b, "uses": list(c.context_b)},
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2)


def cluster_merges(
    candidates: list[AliasCandidate],
    verdicts: list[SameEntityVerdict],
    min_confidence: float = 0.0,
) -> list[list[str]]:
    """Union-find over the yes-pairs -> connected merge groups (code, not model).

    Transitivity is code's job: if the model says A~B and B~C, the group {A,B,C}
    emerges here even though no A~C pair was ever judged. This is the "code does
    the traversal" half of the divide-and-conquer (SCHEMA §7).
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for cand, v in zip(candidates, verdicts, strict=True):
        if v.same and v.confidence >= min_confidence:
            union(cand.symbol_a_id, cand.symbol_b_id)

    groups: dict[str, list[str]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return [g for g in groups.values() if len(g) > 1]


async def _ask_model(instructions: str, payload: str, model: str | None) -> list[Any] | None:
    """One plain-JSON model call; returns the ``verdicts`` list or None."""
    from pathlib import Path

    from agentm.core.abi import AssistantMessage, LoopConfig, TextContent
    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.runtime.session import AgentSession

    config = AgentSessionConfig(
        cwd=str(Path.cwd()),
        model=model,
        scenario="minimal",
        purpose="alias_resolution",
        loop_config=LoopConfig(max_turns=1),
        log_trace_command=False,
    )
    session = await AgentSession.create(config)
    try:
        messages = await session.prompt(f"{instructions}\n\n{payload}")
    finally:
        with contextlib.suppress(Exception):
            await session.shutdown()

    text = "".join(
        block.text
        for msg in messages
        if isinstance(msg, AssistantMessage)
        for block in msg.content
        if isinstance(block, TextContent)
    )
    obj = extract_json(text)
    if obj is None:
        logger.warning("alias resolution: model returned no parseable JSON")
        return None
    verdicts = obj.get("verdicts")
    if not isinstance(verdicts, list):
        logger.warning("alias resolution: JSON missing 'verdicts' list")
        return None
    return verdicts


def _align_verdicts(
    candidates: list[AliasCandidate], raw: list[Any]
) -> list[SameEntityVerdict]:
    """Map id-keyed model verdicts back onto the candidate list (missing -> false)."""
    by_id: dict[int, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        with contextlib.suppress(TypeError, ValueError, KeyError):
            by_id[int(item["id"])] = item
    out: list[SameEntityVerdict] = []
    for i in range(len(candidates)):
        item = by_id.get(i)
        if item is None:
            out.append(SameEntityVerdict(False, 0.0, "no verdict returned"))
            continue
        out.append(
            SameEntityVerdict(
                same=bool(item.get("same", False)),
                confidence=float(item.get("confidence", 0.0)),
                reason=str(item.get("reason", "")),
            )
        )
    return out


async def resolve_aliases(
    index: TrajectoryIndex,
    model: str | None = None,
    min_ratio: float = 0.82,
    min_confidence: float = 0.5,
    apply: bool = True,
) -> list[list[str]]:
    """Block -> decide same-entity -> cluster -> (optionally) merge. Returns the groups.

    ``apply=True`` folds the groups into the index via ``apply_alias_merges`` and
    rebuilds the ledger. ``apply=False`` returns the groups without mutating (for
    inspection / eval).

    The returned groups are the *pre-merge* decision record: each is a list of the
    symbol ids that were fused. With ``apply=True`` the folded-away ids no longer
    exist in ``index.symbols`` (only the surviving canonical does), so resolve names
    from the index *before* applying, or call with ``apply=False`` to inspect first.
    """
    candidates = index.alias_candidates(min_ratio=min_ratio)
    if not candidates:
        return []

    raw = await _ask_model(_SAME_ENTITY_INSTRUCTIONS, _format_candidates(candidates), model)
    if raw is None:
        return []
    verdicts = _align_verdicts(candidates, raw)
    groups = cluster_merges(candidates, verdicts, min_confidence=min_confidence)

    if apply and groups:
        index.apply_alias_merges(groups)
        index.build_dependencies()
    return groups
