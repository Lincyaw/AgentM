"""Pass 2a — identity: alias resolution.

One entity, many surfaces. The division of labor is code-blocks →
model-judges-locally → code-folds:

    alias_candidates   →  resolve_aliases       →  apply_alias_merges
    (block same-entity     (model: same entity?)     (union-find cluster +
     pairs by name)                                   re-point refs/relations)

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
        "alias_resolution", _format_candidates(candidates), model,
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

