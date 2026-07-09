"""Model-judgment side of the pipeline — the semantic decisions code can't make.

The rest of ``trajectory_index`` is deterministic (Pass 1 tagging, Pass 3 dataflow).
This module is the one place a model earns its keep, and it only ever answers a
*local* question — it never traverses the graph. Code does the traversal and
propagation ("the model gives a point, code propagates it"; SCHEMA).

* ``resolve_aliases`` — **Pass 2a, name resolution.** "Are these two surface forms
  the same entity?" Code blocks candidate pairs (``alias_candidates``), the model
  judges each pair, code clusters into merge groups. Runs **before** the dataflow.
* ``resolve_references`` — **Pass 2b, coreference.** "Which earlier entity does this
  anaphor (``this`` / ``it`` / ``the previous result``) denote?" Code detects
  anaphors + proposes recent in-scope candidates, the model picks the referent, code
  adds a resolved reference so the dataflow links it.
* ``compare_values`` — **Pass 3.5, value fidelity.** "Does the tool's grounded value
  confirm or contradict the value the agent acted on?" Fires only on value-class
  edges; flags ``contradicted``.

Each pass is an *independent local judgment* that fires only where it is needed
(the divide-and-conquer "B" shape): the model never traverses the graph — code does
that. The model is swappable (~8B is enough); each call speaks plain JSON.
"""
from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from loguru import logger

from .data import extract_json

if TYPE_CHECKING:
    from .index import AliasCandidate, Step, TrajectoryIndex


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
    the traversal" half of the divide-and-conquer.
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

    from agentm.core.abi import (
        AgentSessionConfig,
        AssistantMessage,
        LoopConfig,
        TextContent,
    )
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


# ---------------------------------------------------------------------------
# Value comparison (Pass 3.5): did the agent act on a value a tool contradicts?
# ---------------------------------------------------------------------------

_VALUE_COMPARE_INSTRUCTIONS = """\
You check whether a value an agent acted on matches what a tool actually produced.
Each item names a value-bearing entity (a metric, status, price, computed answer),
the text where a tool GROUNDED its value, and the text where the agent USED a value
for it. Decide, per item independently:
  - confirm:    the used value matches the grounded value.
  - contradict: the used value differs from the grounded value (acted on a wrong value).
  - unclear:    the texts don't pin down comparable values.
Judge the value, not the wording. Return ONLY:
{"verdicts": [{"id": 0, "outcome": "confirm|contradict|unclear", "confidence": 0.9, "reason": "..."}]}
"""


def _step_text(index: TrajectoryIndex, run_id: str, step_id: str, cap: int = 500) -> str:
    st = index.steps.get((run_id, step_id))
    return (st.content or "")[:cap] if st else ""


def _align_outcomes(n: int, raw: list[Any]) -> list[tuple[str, float, str]]:
    """Map id-keyed {outcome,confidence,reason} back by position (missing -> unclear)."""
    by_id: dict[int, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        with contextlib.suppress(TypeError, ValueError, KeyError):
            by_id[int(item["id"])] = item
    out: list[tuple[str, float, str]] = []
    for i in range(n):
        item = by_id.get(i)
        if item is None:
            out.append(("unclear", 0.0, "no verdict"))
            continue
        oc = str(item.get("outcome", "unclear"))
        oc = oc if oc in ("confirm", "contradict", "unclear") else "unclear"
        out.append((oc, float(item.get("confidence", 0.0)), str(item.get("reason", ""))))
    return out


async def compare_values(
    index: TrajectoryIndex, model: str | None = None, apply: bool = True,
) -> list[tuple[str, str]]:
    """For value-class edges with a grounded binding to compare against, ask the model
    confirm/contradict and flag ``contradicted``. Local: fires only on value entities.

    Returns (dependency_id, outcome) for every edge judged.
    """
    targets: list[tuple[Any, str]] = []
    for d in index.get_dependencies():
        sym = index.symbols.get(d.symbol_id)
        if not sym or sym.entity_class != "value":
            continue
        grounded_step = (
            d.def_step_id if d.risk == "grounded"
            else d.grounded_by_step_id if d.risk == "premature" else None
        )
        if grounded_step:
            targets.append((d, grounded_step))
    if not targets:
        return []

    rows = [
        {
            "id": i,
            "entity": index.symbols[d.symbol_id].canonical_name,
            "grounded_text": _step_text(index, d.run_id, gs),
            "used_text": _step_text(index, d.run_id, d.use_step_id),
        }
        for i, (d, gs) in enumerate(targets)
    ]
    raw = await _ask_model(
        _VALUE_COMPARE_INSTRUCTIONS, json.dumps(rows, ensure_ascii=False, indent=2), model,
    )
    if raw is None:
        return []
    outcomes = _align_outcomes(len(targets), raw)

    result: list[tuple[str, str]] = []
    for (d, _), (outcome, _conf, _why) in zip(targets, outcomes, strict=True):
        result.append((d.id, outcome))
        if apply and outcome == "contradict":
            index.dependencies[d.id] = replace(d, risk="contradicted")
    return result


# ---------------------------------------------------------------------------
# Coreference resolution (Pass 2b): bind an anaphor to its antecedent entity.
# ---------------------------------------------------------------------------

# Closed-class deictic/anaphoric surfaces a small deterministic proposer can spot.
_ANAPHOR = re.compile(
    r"\b(?:this|that|these|those|it|its|the (?:previous|above|prior|last|former|same)"
    r"\s+\w+|the (?:result|output|value|answer|response))\b",
    re.IGNORECASE,
)

_COREF_INSTRUCTIONS = """\
An agent's text used an anaphor (a pronoun or back-reference like "this", "it",
"the previous result"). Decide which earlier entity it refers to. You are given the
anaphor in its sentence and a numbered list of candidate entities recently in scope.
Pick the single entity the anaphor denotes, or -1 if none fits (it refers to
something not listed, or to an idea/action rather than a tracked entity).
Return ONLY: {"verdicts": [{"id": 0, "entity": <candidate-index or -1>, "confidence": 0.9}]}
"""


async def resolve_references(
    index: TrajectoryIndex, model: str | None = None,
    window: int = 8, max_candidates: int = 8, apply: bool = True,
) -> int:
    """Detect anaphors in agent text, bind each to an antecedent entity (LLM), and add
    a resolved reference so the def-use layer links it. Returns #anaphors resolved.

    Divide-and-conquer: code detects anaphors + proposes recent in-scope candidates,
    the model picks the referent (local), code adds the reference and rebuilds Pass 3.
    """
    # steps in run/index order, with their entities-in-scope running set
    steps = sorted(index.steps.values(), key=lambda s: (s.run_id, s.index))
    # entity last-seen step index per run, for recency-ranked candidates
    seen: dict[str, list[tuple[int, str]]] = {}  # run_id -> [(step_index, symbol_id)]
    for r in index.references.values():
        st = index.steps.get((r.run_id, r.step_id))
        sym = index.symbols.get(r.symbol_id)
        if st and sym and sym.entity_class in ("identifier", "value"):
            seen.setdefault(r.run_id, []).append((st.index, r.symbol_id))
    for lst in seen.values():
        lst.sort()

    items: list[tuple[Step, str, list[str]]] = []  # (step, anaphor phrase, candidate sym_ids)
    for st in steps:
        if st.role not in ("assistant",) or not st.content:
            continue
        m = _ANAPHOR.search(st.content)
        if not m:
            continue
        # candidates: distinct entities seen strictly earlier, most-recent first
        cands: list[str] = []
        for idx, sid in reversed(seen.get(st.run_id, [])):
            if idx < st.index and sid not in cands:
                cands.append(sid)
            if len(cands) >= max_candidates:
                break
        if not cands:
            continue
        items.append((st, m.group(0), cands))

    if not items:
        return 0

    rows = []
    for i, (st, phrase, cands) in enumerate(items):
        sentence = st.content[max(0, st.content.lower().find(phrase.lower()) - 80):][:200]
        rows.append({
            "id": i,
            "anaphor": phrase,
            "sentence": sentence,
            "candidates": [
                {"index": j, "name": index.symbols[sid].canonical_name,
                 "kind": index.symbols[sid].kind}
                for j, sid in enumerate(cands)
            ],
        })
    raw = await _ask_model(_COREF_INSTRUCTIONS, json.dumps(rows, ensure_ascii=False, indent=2), model)
    if raw is None:
        return 0

    by_id: dict[int, dict[str, Any]] = {}
    for it in raw:
        if isinstance(it, dict):
            with contextlib.suppress(TypeError, ValueError, KeyError):
                by_id[int(it["id"])] = it

    resolved = 0
    for i, (st, phrase, cands) in enumerate(items):
        v = by_id.get(i)
        if not v:
            continue
        with contextlib.suppress(TypeError, ValueError):
            pick = int(v.get("entity", -1))
        if pick < 0 or pick >= len(cands):
            continue
        sym = index.symbols[cands[pick]]
        pos = st.content.lower().find(phrase.lower())
        # the added reference records form="anaphor" and text=phrase (the surface it
        # was resolved from); Pass 3 then links it as a use of `sym`.
        index.add_reference(
            symbol=sym, step=st, text=phrase, kind="mention",
            start=max(0, pos), form="anaphor",
        )
        resolved += 1

    if apply and resolved:
        index.build_dependencies()
    return resolved
