"""Pass 3 — def-use / grounding dataflow over the symbol-reference graph.

Deterministic, no model. Chains each structured entity's references within a
run into def-use edges, reads a grounding risk off the reaching binding, and
folds the result into per-symbol warnings. See SCHEMA.md for the def/use
classification, reaching-def selection with forward grounding propagation,
and the grounded/premature/ungrounded risk derivation.

The functions take a ``TrajectoryIndex`` and mutate/read it; ``index.py``
exposes them as thin ``TrajectoryIndex`` methods so the public API is
unchanged. The per-reference kind helpers (``drives_defuse`` /
``grounded_from_kind`` / ``_provenance_kind``) live here too because they are
the grounding layer's vocabulary, and are re-exported from ``index.py`` for
the reference write path and the loaders.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ..ir.models import Dependency, Reference, Risk, Step, stable_id
from ..oracle import SessionFactory, _ask_model, _index_by_id, _safe_float

if TYPE_CHECKING:
    from ..ir.index import TrajectoryIndex

# Reference kinds that produce/introduce a resource value (a "write"). The
# extractor only emits tool_output (data.py); define/write/observe are inert for
# freshly-built indexes and only matter for loaded/external ones. This is the
# file's single producing-kind vocabulary — DEFINITION_PREFERRED_KINDS aliases it.
_PRODUCING_KINDS: frozenset[str] = frozenset({"tool_output", "define", "write", "observe"})


def drives_defuse(entity_class: str) -> bool:
    """Does an entity of this class drive the def-use layer?

    The name-vs-value-vs-vague judgment is the LLM's (Pass 1, ``entity_class``) —
    it knows a rigid identifier from an anaphor and a proper noun from a vague
    concept, which a regex cannot. Only ``unknown`` (vague, unresolved) is excluded;
    once Pass 2 coreference ties a vague surface to a concrete entity it inherits a
    real class. Replaces the old ``looks_structured`` regex gate.
    """
    return entity_class in ("identifier", "value")


def grounded_from_kind(kind: str) -> bool:
    """Base per-reference grounding from block-derived kind (no look-back).

    A tool-produced value is grounded; anything the model wrote (``tool_input``,
    ``mention``) starts ungrounded. A ``tool_input`` may still be upgraded to
    grounded during dependency build if it copies a prior grounded value.
    """
    return kind in _PRODUCING_KINDS


def _provenance_kind(step: Step, kind: str, start: int, end: int) -> str:
    """Upgrade a block-derived reference kind with Pass 1 provenance.

    On role-degraded records everything arrives as text blocks, so every
    occurrence is kind="mention" even inside a ``⟦obs⟧``-labeled region —
    which made the grounding layer call Pass-1-attested observation
    content "fabricated". An occurrence fully inside the step's
    observation spans IS tool output; the upgrade only ADDs observation
    status (attested roles keep their block-derived kind, including the
    deliberate non-deterministic downgrade).
    """
    if kind != "mention" or not step.obs_regions:
        return kind
    for a, b in step.obs_regions:
        if start >= a and end <= b:
            return "tool_output"
    return kind


# ---------------------------------------------------------------------------
# def-use build (Pass 3: dataflow, deterministic)
# ---------------------------------------------------------------------------


def build_dependencies(index: TrajectoryIndex) -> None:
    """Build the def-use layer over structured entities (Pass 3: dataflow).

    Idempotent: clears and rebuilds every edge. Deterministic (no model),
    global traversal. See SCHEMA.md for the def/use classification,
    reaching-def selection with forward grounding propagation, and the
    grounded/premature/ungrounded risk derivation.

    Name resolution (Pass 2) is a separate, upstream step: run apply_alias_merges()
    with the model-decided groups from alias_candidates() *before* this, if
    desired. This method does no merging on its own.
    """
    index.dependencies = {}
    index._dep_ids_by_symbol = defaultdict(list)

    # Reset tool_input grounding upgrades from prior builds so the rebuild
    # is truly idempotent — without this, a re-run sees already-upgraded
    # refs and may flip risk labels (premature -> grounded).
    for ref_id, ref in index.references.items():
        if ref.grounds_ref_id is not None:
            index.references[ref_id] = replace(
                ref, grounded=grounded_from_kind(ref.kind), grounds_ref_id=None,
            )

    for symbol_id in index.symbols:
        refs = index.get_references(symbol_id)  # sorted (run_id, step index, start)
        if not refs or not any(r.structured for r in refs):
            continue  # only structured entities drive the def-use layer
        by_run: dict[str, list[Reference]] = defaultdict(list)
        for r in refs:
            by_run[r.run_id].append(r)
        for run_id, run_refs in by_run.items():
            _build_run_dependencies(index, symbol_id, run_id, run_refs)


def _build_run_dependencies(
    index: TrajectoryIndex, symbol_id: str, run_id: str, refs: list[Reference],
) -> None:
    """Chain one symbol's references within a single run into def-use edges.

    The reaching def for a use is the most-recent def at a *strictly earlier*
    step (most-recent at a strictly earlier step). Defs in the use's own step do not
    count — a same-step def/use is not cross-step reliance, so it produces no
    edge. Defs are therefore committed only when we cross a step boundary;
    within a step they buffer in ``pending``.
    """
    # (step_index, step_id) of every grounded def, for the look-ahead that tells
    # "used before grounded, but grounded later" (premature) from "never grounded".
    grounded_defs: list[tuple[int, str]] = sorted(
        (index._ref_step_index(r), r.step_id) for r in refs if r.grounded
    )

    reaching: Reference | None = None          # committed from a strictly earlier step
    grounded_reaching: Reference | None = None
    pending: Reference | None = None           # latest def in the current step
    pending_grounded: Reference | None = None
    cur_step: int | None = None
    reaching_version = -1                       # SSA version of `reaching`

    for ref in refs:
        use_idx = index._ref_step_index(ref)
        if use_idx != cur_step:
            # crossed a step boundary: the prior step's defs become reaching
            if pending is not None:
                reaching = pending
                reaching_version += 1
            if pending_grounded is not None:
                grounded_reaching = pending_grounded
            pending = pending_grounded = None
            cur_step = use_idx

        # A tool_input that copies a prior grounded def is itself grounded.
        if (
            ref.kind == "tool_input"
            and not ref.grounded
            and grounded_reaching is not None
        ):
            ref = replace(ref, grounded=True, grounds_ref_id=grounded_reaching.id)
            index.references[ref.id] = ref

        is_def = ref.kind in _PRODUCING_KINDS
        if is_def:
            pending = ref
            if ref.grounded:
                pending_grounded = ref
            continue

        # use → link to the most-recent def at a strictly earlier step
        if reaching is None:
            continue
        grounded_by: str | None = None
        if reaching.grounded:
            risk: Risk = "grounded"
            grounded_by = reaching.step_id
        else:
            grounded_by = next(
                (sid for idx, sid in grounded_defs if idx > use_idx), None,
            )
            risk = "premature" if grounded_by else "ungrounded"

        # ref ids discriminate multiple uses of one reaching def in a step
        dep_id = stable_id("dep", run_id, reaching.id, ref.id, symbol_id)
        dep = Dependency(
            id=dep_id,
            symbol_id=symbol_id,
            run_id=run_id,
            def_step_id=reaching.step_id,
            def_ref_id=reaching.id,
            def_version=max(0, reaching_version),
            use_step_id=ref.step_id,
            use_ref_id=ref.id,
            risk=risk,
            grounded_by_step_id=grounded_by,
            def_value=reaching.value,
            use_value=ref.value,
            confidence=min(reaching.confidence, ref.confidence),
        )
        index.dependencies[dep.id] = dep
        index._dep_ids_by_symbol[symbol_id].append(dep.id)


# ---------------------------------------------------------------------------
# Warnings (code-only, no LLM) — runs after build_dependencies()
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Warning:
    kind: str
    symbol_id: str
    symbol_name: str
    detail: str
    step_ids: tuple[str, ...] = ()


def compute_warnings(index: TrajectoryIndex) -> list[Warning]:
    """Compute grounding warnings from the index structure.

    Pure code — no LLM. Runs after ``build_dependencies()``.
    """
    out: list[Warning] = []

    for sym_id, sym in index.symbols.items():
        refs = index._ref_ids_by_symbol.get(sym_id, [])
        if not refs:
            out.append(Warning(
                kind="orphan",
                symbol_id=sym_id,
                symbol_name=sym.canonical_name,
                detail="extracted but has 0 references in the trajectory",
            ))
            continue

        ref_objs = [index.references[rid] for rid in refs]
        ref_kinds = {r.kind for r in ref_objs}
        has_grounded_def = any(r.grounded for r in ref_objs)

        if not has_grounded_def:
            steps = tuple(dict.fromkeys(r.step_id for r in ref_objs))
            if ref_kinds == {"tool_input"} or ref_kinds <= {"tool_input", "mention"}:
                if "tool_input" in ref_kinds:
                    out.append(Warning(
                        kind="blind_query",
                        symbol_id=sym_id,
                        symbol_name=sym.canonical_name,
                        detail="used in tool calls but never returned by any tool",
                        step_ids=steps,
                    ))
                else:
                    out.append(Warning(
                        kind="fabricated_name",
                        symbol_id=sym_id,
                        symbol_name=sym.canonical_name,
                        detail="mentioned in reasoning but never returned by any tool",
                        step_ids=steps,
                    ))

    # Dependency-level warnings
    deps = index.get_dependencies()
    premature = [d for d in deps if d.risk == "premature"]
    ungrounded = [d for d in deps if d.risk == "ungrounded"]

    for d in premature:
        dsym = index.symbols.get(d.symbol_id)
        if dsym:
            out.append(Warning(
                kind="premature_use",
                symbol_id=d.symbol_id,
                symbol_name=dsym.canonical_name,
                detail=f"used at step {d.use_step_id} before grounded at step {d.grounded_by_step_id}",
                step_ids=(d.use_step_id, d.def_step_id),
            ))

    for d in ungrounded:
        dsym = index.symbols.get(d.symbol_id)
        if dsym:
            out.append(Warning(
                kind="ungrounded_use",
                symbol_id=d.symbol_id,
                symbol_name=dsym.canonical_name,
                detail=f"used at step {d.use_step_id} with ungrounded def at step {d.def_step_id}",
                step_ids=(d.use_step_id, d.def_step_id),
            ))

    return sorted(out, key=lambda w: (
        {"fabricated_name": 0, "blind_query": 1, "ungrounded_use": 2, "premature_use": 3, "orphan": 4}.get(w.kind, 5),
        w.symbol_name,
    ))


def warning_summary(index: TrajectoryIndex) -> dict[str, int]:
    """Count warnings by kind."""
    return dict(Counter(w.kind for w in compute_warnings(index)))


# ---------------------------------------------------------------------------
# Pass 3.5 — value fidelity: did the agent act on a value a tool contradicts?
# ---------------------------------------------------------------------------
#
# The only model-judged part of the grounding layer. It rides on the dataflow:
# for each dependency whose reaching binding is grounded, it asks whether the
# value the agent used matches what the tool actually produced, and flags
# ``contradicted``. Local per-edge judgment (the model never traverses).

_VALUE_COMPARE_INSTRUCTIONS = """\
You check whether a value an agent acted on matches what a tool actually produced.
Each item names an entity, the full text of the step where a tool provided
information about it (grounded), and the full text of the step where the agent
referenced it (used). Decide, per item independently:
  - confirm:    the agent's usage is consistent with what the tool provided.
  - contradict: the agent stated or used a different value than what the tool provided.
  - unclear:    the texts don't contain comparable values for this entity.
Judge the substance, not the wording. Return ONLY:
{"verdicts": [{"id": 0, "outcome": "confirm|contradict|unclear", "confidence": 0.9, "reason": "..."}]}
"""


def _step_text(index: TrajectoryIndex, run_id: str, step_id: str) -> str:
    st = index.steps.get((run_id, step_id))
    return (st.content or "") if st else ""


def _align_outcomes(n: int, raw: list[Any]) -> list[tuple[str, float, str]]:
    """Map id-keyed {outcome,confidence,reason} back by position (missing -> unclear)."""
    by_id = _index_by_id(raw)
    out: list[tuple[str, float, str]] = []
    for i in range(n):
        item = by_id.get(i)
        if item is None:
            out.append(("unclear", 0.0, "no verdict"))
            continue
        oc = str(item.get("outcome", "unclear"))
        oc = oc if oc in ("confirm", "contradict", "unclear") else "unclear"
        out.append((oc, _safe_float(item, "confidence"), str(item.get("reason", ""))))
    return out


async def compare_values(
    index: TrajectoryIndex, model: str | None = None, apply: bool = True,
    session_factory: SessionFactory | None = None,
) -> list[tuple[str, str]]:
    """For dependency edges with a grounded binding, ask the model whether the
    agent's usage is consistent with what the tool provided. Flags ``contradicted``.

    Returns (dependency_id, outcome) for every edge judged.
    """
    targets: list[tuple[Any, str]] = []
    for d in index.get_dependencies():
        sym = index.symbols.get(d.symbol_id)
        if not sym:
            continue
        grounded_step = (
            d.def_step_id if d.risk == "grounded"
            else d.grounded_by_step_id if d.risk == "premature" else None
        )
        if grounded_step:
            targets.append((d, grounded_step))
    if not targets:
        return []
    if session_factory is None:
        raise ValueError("session_factory is required (pass AgentSession.create for offline use)")

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
        session_factory=session_factory,
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
