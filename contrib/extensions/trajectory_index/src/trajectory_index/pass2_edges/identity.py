"""Pass 2a — identity: alias resolution + coreference.

One entity, many surfaces. The division of labor is code-blocks →
model-judges-locally → code-folds:

    alias_candidates   →  resolve_aliases       →  apply_alias_merges
    (block same-entity     (model: same entity?)     (union-find cluster +
     pairs by name)                                   re-point refs/relations)

    anaphor detection  →  resolve_references     →  add_reference
    (regex closed-class    (model: which antecedent?)  (materialize the link
     deictic surfaces)                                  so dataflow sees it)

Code never decides identity and the model never traverses the graph. Runs
before the Pass 3 dataflow so the def-use layer sees merged entities.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ..ir.models import AliasCandidate, normalize_name
from ..oracle import SessionFactory, _ask_model, _index_by_id, _safe_float
from ..pass3_folds.grounding import drives_defuse

if TYPE_CHECKING:
    from ..ir.index import TrajectoryIndex
    from ..ir.models import Step

_MIN_BLOCK_SUBSTR = 3         # alias blocking: shortest name a substring match may involve
_SNIPPET_CHARS = 120          # per-reference context snippet cap for the merge adjudicator


# ---------------------------------------------------------------------------
# Blocking + merge mechanism (code)
# ---------------------------------------------------------------------------


def rebuild_symbol_name_index(index: TrajectoryIndex) -> None:
    from collections import defaultdict

    index._symbol_ids_by_norm = defaultdict(set)
    for sid, sym in index.symbols.items():
        ns = str(sym.metadata.get("namespace", ""))
        for name in sym.all_names:
            index._index_symbol_name(sid, name, ns)


def rebuild_relation_index(index: TrajectoryIndex) -> None:
    from collections import defaultdict

    index._relation_ids_by_symbol = defaultdict(list)
    for rid, rel in index.relations.items():
        index._relation_ids_by_symbol[rel.from_symbol_id].append(rid)
        index._relation_ids_by_symbol[rel.to_symbol_id].append(rid)


def _choose_canonical(index: TrajectoryIndex, symbol_ids: list[str]) -> str:
    """Representative of a merge group: most-referenced, tie broken by stable id.
    Which id represents the merged entity is cosmetic; the choice is deterministic."""
    return sorted(
        symbol_ids, key=lambda sid: (-len(index._ref_ids_by_symbol.get(sid, [])), sid),
    )[0]


def _ref_snippets(index: TrajectoryIndex, symbol_id: str, limit: int = 3) -> tuple[str, ...]:
    """A few reference texts for a symbol — context for the merge adjudicator."""
    out: list[str] = []
    for rid in index._ref_ids_by_symbol.get(symbol_id, [])[:limit]:
        text = index.references[rid].text.strip()
        if text:
            out.append(text[:_SNIPPET_CHARS])
    return tuple(out)


def _tokenize_name(name: str) -> set[str]:
    """Split a symbol name into tokens on common delimiters."""
    return {
        t for t in re.split(r"[-_./\s]+", normalize_name(name))
        if len(t) >= 2
    }


def _token_jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def alias_candidates(index: TrajectoryIndex, min_jaccard: float = 0.5) -> list[AliasCandidate]:
    """Deterministically block structured-symbol pairs that MIGHT be one entity.

    Uses token-level Jaccard similarity on names split by common
    delimiters (``-_./``). This avoids false matches between names
    that share long prefixes/suffixes but differ in the distinctive
    token (e.g. ``ts-cancel-service`` vs ``ts-config-service``
    shares only ``{ts, service}`` → jaccard 0.33, below threshold).

    Substring containment is also checked. Pairs above threshold are
    proposed for LLM judgment via :func:`resolve_aliases`.
    """
    items = [
        (sid, sym, normalize_name(sym.canonical_name),
         _tokenize_name(sym.canonical_name), _ref_snippets(index, sid))
        for sid, sym in index.symbols.items()
        if drives_defuse(sym.entity_class)
    ]
    out: list[AliasCandidate] = []
    for i in range(len(items)):
        aid, asym, anorm, atokens, asnip = items[i]
        for j in range(i + 1, len(items)):
            bid, bsym, bnorm, btokens, bsnip = items[j]
            if anorm == bnorm:
                continue
            substring = (
                (anorm in bnorm or bnorm in anorm)
                and min(len(anorm), len(bnorm)) >= _MIN_BLOCK_SUBSTR
            )
            jaccard = _token_jaccard(atokens, btokens)
            if not substring and jaccard < min_jaccard:
                continue
            out.append(AliasCandidate(
                symbol_a_id=aid, symbol_b_id=bid,
                name_a=asym.canonical_name, name_b=bsym.canonical_name,
                kind_a=asym.kind, kind_b=bsym.kind,
                signal="substring" if substring else "similar",
                score=round(jaccard, 3),
                context_a=asnip, context_b=bsnip,
            ))
    return sorted(out, key=lambda c: (-c.score, c.symbol_a_id, c.symbol_b_id))


def apply_alias_merges(index: TrajectoryIndex, groups: list[list[str]]) -> None:
    """Fold each decided group of symbol ids into one canonical symbol.

    The mechanism (deterministic, idempotent); *which* symbols form a group is
    decided upstream — a name-resolution judgment over :func:`alias_candidates`, not
    a rule here. References and relations are re-pointed; folded names become
    aliases.
    """
    from collections import defaultdict

    merged = False
    for sids in groups:
        live = [s for s in dict.fromkeys(sids) if s in index.symbols]
        if len(live) < 2:
            continue
        canonical_id = _choose_canonical(index, live)
        canon = index.symbols[canonical_id]
        for other_id in live:
            if other_id == canonical_id:
                continue
            other = index.symbols[other_id]
            canon.aliases.update(other.all_names)
            canon.aliases.discard(canon.canonical_name)
            for rid in index._ref_ids_by_symbol.get(other_id, []):
                index.references[rid] = replace(index.references[rid], symbol_id=canonical_id)
                index._ref_ids_by_symbol[canonical_id].append(rid)
            index._ref_ids_by_symbol.pop(other_id, None)
            # only relations touching other_id need re-pointing (a folded symbol
            # is never a canonical target, so this index entry is complete).
            for rel_id in dict.fromkeys(index._relation_ids_by_symbol.get(other_id, [])):
                rel = index.relations.get(rel_id)
                if rel is None:
                    continue
                nf = canonical_id if rel.from_symbol_id == other_id else rel.from_symbol_id
                nt = canonical_id if rel.to_symbol_id == other_id else rel.to_symbol_id
                if nf == nt:  # self-loop after the merge — drop
                    index.relations.pop(rel_id, None)
                else:
                    index.relations[rel_id] = replace(rel, from_symbol_id=nf, to_symbol_id=nt)
            index.symbols.pop(other_id, None)
            merged = True

    if merged:
        rebuild_symbol_name_index(index)
        rebuild_relation_index(index)
        # dependencies are derived from symbols; a merge invalidates them.
        index.dependencies = {}
        index._dep_ids_by_symbol = defaultdict(list)
        # constraint findings name candidate symbols; a merge invalidates
        # them wholesale (Pass E/J/L rebuild from facts + transcript).
        index.constraint_findings = []


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


def _align_verdicts(
    candidates: list[AliasCandidate], raw: list[Any]
) -> list[SameEntityVerdict]:
    """Map id-keyed model verdicts back onto the candidate list (missing -> false)."""
    by_id = _index_by_id(raw)
    out: list[SameEntityVerdict] = []
    for i in range(len(candidates)):
        item = by_id.get(i)
        if item is None:
            out.append(SameEntityVerdict(False, 0.0, "no verdict returned"))
            continue
        out.append(
            SameEntityVerdict(
                same=bool(item.get("same", False)),
                confidence=_safe_float(item, "confidence"),
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
    session_factory: SessionFactory | None = None,
) -> list[list[str]]:
    """Block -> decide same-entity -> cluster -> (optionally) merge. Returns the groups.

    ``session_factory`` is required when called from the atom (§11). Offline
    callers may pass ``AgentSession.create`` from their own import.

    ``apply=True`` folds the groups into the index via ``apply_alias_merges`` and
    rebuilds the ledger. ``apply=False`` returns the groups without mutating (for
    inspection / eval).

    The returned groups are the *pre-merge* decision record: each is a list of the
    symbol ids that were fused. With ``apply=True`` the folded-away ids no longer
    exist in ``index.symbols`` (only the surviving canonical does), so resolve names
    from the index *before* applying, or call with ``apply=False`` to inspect first.
    """
    candidates = index.alias_candidates(min_jaccard=min_ratio)
    if not candidates:
        return []
    if session_factory is None:
        raise ValueError("session_factory is required (pass AgentSession.create for offline use)")

    raw = await _ask_model(
        _SAME_ENTITY_INSTRUCTIONS, _format_candidates(candidates), model,
        session_factory=session_factory,
    )
    if raw is None:
        return []
    verdicts = _align_verdicts(candidates, raw)
    groups = cluster_merges(candidates, verdicts, min_confidence=min_confidence)

    if apply and groups:
        index.apply_alias_merges(groups)
        index.build_dependencies()
    return groups


# ---------------------------------------------------------------------------
# Coreference resolution: bind an anaphor to its antecedent entity.
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
    session_factory: SessionFactory | None = None,
) -> int:
    """Detect anaphors in agent text, bind each to an antecedent entity (LLM), and add
    a resolved reference so the def-use layer links it. Returns #anaphors resolved.

    Divide-and-conquer: code detects anaphors + proposes recent in-scope candidates,
    the model picks the referent (local), code adds the reference and rebuilds Pass 3.
    """
    import contextlib

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
    if session_factory is None:
        raise ValueError("session_factory is required (pass AgentSession.create for offline use)")

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
    raw = await _ask_model(
        _COREF_INSTRUCTIONS, json.dumps(rows, ensure_ascii=False, indent=2), model,
        session_factory=session_factory,
    )
    if raw is None:
        return 0

    by_id = _index_by_id(raw)
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
        index.add_reference(
            symbol=sym, step=st, text=sym.canonical_name, kind="mention",
            start=max(0, pos), form="anaphor", resolved_from=phrase,
        )
        resolved += 1

    if apply and resolved:
        index.build_dependencies()
    return resolved
