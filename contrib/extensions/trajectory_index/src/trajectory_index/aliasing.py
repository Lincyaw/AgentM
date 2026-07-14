"""Pass 2a — alias blocking + merge mechanism.

Code proposes candidate same-entity pairs deterministically
(:func:`alias_candidates`) and applies a decided set of merge groups
(:func:`apply_alias_merges`). The merge DECISION is a name-resolution model
judgment (``adjudicate.resolve_aliases``), not a rule here — this module only
blocks candidates and folds the groups the model returns.

Functions take a ``TrajectoryIndex``; ``index.py`` exposes them as thin methods
so the public API is unchanged.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

from .grounding import drives_defuse
from .models import AliasCandidate, normalize_name

if TYPE_CHECKING:
    from .index import TrajectoryIndex

_MIN_BLOCK_SUBSTR = 3         # alias blocking: shortest name a substring match may involve
_SNIPPET_CHARS = 120          # per-reference context snippet cap for the merge adjudicator


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
    proposed for LLM judgment via :func:`apply_alias_merges`.
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
