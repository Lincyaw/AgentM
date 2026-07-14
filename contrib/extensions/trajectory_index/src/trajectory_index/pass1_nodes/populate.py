"""Pass 1 — populate the index IR from a markup extraction result.

The extractor re-emits each annotated message with ``⟦tag attrs|content⟧``
spans; this pass parses them, verifies each message by strip-and-compare
(offset-exact), and lands symbols, observation regions, claims and
constraints as first-class nodes. References are then built deterministically
by grepping symbol names over the same message content.

``populate_from_extraction`` takes a ``TrajectoryIndex`` and mutates it;
``index.py`` exposes it as a thin method so the public API is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..ir.models import (
    _ENTITY_CLASS_VALUES,
    Claim,
    Constraint,
    MetadataValue,
    Step,
    StepRole,
    normalize_name,
    stable_id,
)

if TYPE_CHECKING:
    from ..ir.index import TrajectoryIndex


def _constraint_normalized(attrs: dict[str, str]) -> dict[str, MetadataValue] | None:
    """Validate model-emitted machine-checkable attrs (code owns the types).

    Recognized shapes: kind=year_range lo=<int> hi=<int>;
    kind=number op∈{==,<=,>=,<,>} value=<number>. Anything else → None
    (the constraint stays semantic — the oracle judges it, code does not).
    """
    kind = attrs.get("kind", "")
    try:
        if kind == "year_range":
            return {"kind": "year_range", "lo": int(attrs["lo"]), "hi": int(attrs["hi"])}
        if kind == "number":
            op = attrs.get("op", "==")
            if op in {"==", "<=", ">=", "<", ">"}:
                return {"kind": "number", "op": op, "value": float(attrs["value"])}
    except (KeyError, ValueError):
        return None
    return None


def _message_step_content(msg: dict[str, Any]) -> tuple[str, str | None]:
    """Step content + tool name, from the single shared message walk.

    Derived from ``data.message_parts`` — the same pairs the extractor's
    view is built from — so the two representations that offset alignment
    depends on cannot drift apart.
    """
    from .serialize import message_parts

    pairs, tool_name = message_parts(msg)
    return "\n".join(c for c, _ in pairs if c), tool_name


def populate_from_extraction(
    index: TrajectoryIndex,
    result: Any,
    messages: list[Any],
    *,
    run_id: str = "",
    namespace_fn: Any | None = None,
    reference_confidence: float = 0.8,
    message_id_start: int = 0,
    diagnostics: Any | None = None,
) -> None:
    """Populate index from an extraction result and source messages.

    ``result`` carries the unified markup output (``.annotated`` — see
    ``markup.py``); every Pass 1 node kind is parsed out of it here.
    Verification is strip-and-compare: the annotated body, with all
    ``⟦…⟧`` removed, must reproduce the extractor's view of the message
    (whitespace-tolerant), which makes every span offset exact. A
    message that fails is rejected whole; every rejection lands in
    ``diagnostics.prune_log`` when a sink is passed (P2), and in the
    debug log always.

    ``messages`` can be typed ``AgentMessage`` objects or pre-serialized
    dicts. ``namespace_fn(run_id, sym_dict) -> str`` optionally scopes
    symbols. ``message_id_start`` is the absolute offset of the first
    message in the full trajectory.
    """
    from loguru import logger as _clog

    from .markup import GAP_TAG, MarkupError, align, align_gapped
    from .markup import parse as parse_markup
    from .serialize import _build_references, view_body_with_map

    if messages and not isinstance(messages[0], dict):
        from ..atom import _agentmsg_to_extraction_dict
        # truncate=False: steps/references are the substrate every
        # downstream pass reads — content goes in whole. Truncation is
        # legitimate only in the extractor's own prompt window.
        messages = [
            d for i, m in enumerate(messages, start=message_id_start)
            if (d := _agentmsg_to_extraction_dict(m, i, truncate=False))
        ]

    def _prune(what: str, why: str) -> None:
        if diagnostics is not None:
            diagnostics.prune("populate", what, why)
        _clog.debug("populate: {} — {}", what, why)

    role_map = {"user": StepRole.USER, "assistant": StepRole.ASSISTANT, "tool_result": StepRole.TOOL_RESULT}
    base_idx = len(index.steps)
    steps_by_id: dict[str, Step] = {}

    ann_by_mid: dict[str, str] = {
        str(am.message_id): am.text
        for am in getattr(result, "annotated", []) or []
    }

    # node collections parsed out of the markup, keyed by message id
    obs_by_mid: dict[str, tuple[tuple[int, int], ...]] = {}
    claims_by_mid: dict[str, list[tuple[int, int, str]]] = {}
    constraints_new: list[tuple[str, int, int, dict[str, str]]] = []
    syms: list[tuple[str, dict[str, str]]] = []   # (surface, attrs)

    for i, msg in enumerate(messages):
        mid = str(msg.get("id", f"s{base_idx + i}"))
        annotated = ann_by_mid.pop(mid, None)
        if annotated is None:
            continue
        try:
            plain, annotations = parse_markup(annotated)
        except MarkupError as exc:
            _prune(f"step {mid}", f"malformed markup: {exc}")
            continue
        view, vmap = view_body_with_map(msg)
        gap_offsets = [a.start for a in annotations if a.tag == GAP_TAG]
        annotations = [a for a in annotations if a.tag != GAP_TAG]
        if gap_offsets:
            amap, gap_err = align_gapped(plain, gap_offsets, view)
            if amap is None:
                _prune(f"step {mid}", f"gapped re-emission rejected: {gap_err}")
                continue
        else:
            amap = align(plain, view)
            if amap is None:
                _prune(f"step {mid}",
                       f"re-emission diverges from original ({len(plain)} vs {len(view)} chars)")
                continue

        def _to_content(
            plain_off: int,
            _amap: list[int] = amap,
            _vmap: list[int | None] = vmap,
        ) -> int | None:
            return _vmap[_amap[plain_off]]

        obs_regions: list[tuple[int, int]] = []
        for a in annotations:
            if a.tag == "known" or a.end <= a.start:
                continue
            start_c = _to_content(a.start)
            end_c = _to_content(a.end)
            if end_c is None:
                last = _to_content(a.end - 1)
                end_c = last + 1 if last is not None else None
            if start_c is None or end_c is None:
                _prune(f"step {mid}",
                       f"⟦{a.tag}⟧ span falls in the truncation ellipsis")
                continue
            if a.tag == "obs":
                obs_regions.append((start_c, end_c))
            elif a.tag == "claim":
                claims_by_mid.setdefault(mid, []).append(
                    (start_c, end_c, str(a.attrs.get("role", ""))),
                )
            elif a.tag == "constraint":
                constraints_new.append((mid, start_c, end_c, dict(a.attrs)))
            elif a.tag == "sym":
                syms.append((plain[a.start:a.end], dict(a.attrs)))
            else:
                _prune(f"step {mid}", f"unknown annotation tag {a.tag!r}")
        if obs_regions:
            obs_regions.sort()
            merged: list[tuple[int, int]] = []
            for span in obs_regions:
                if merged and span[0] <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))
                else:
                    merged.append(span)
            obs_by_mid[mid] = tuple(merged)

    for mid in ann_by_mid:
        _prune(f"step {mid}", "annotated message references unknown id")

    for i, msg in enumerate(messages):
        mid = str(msg.get("id", f"s{base_idx + i}"))
        role = role_map.get(str(msg.get("role", "")), StepRole.USER)
        content, tool_name = _message_step_content(msg)
        spans = obs_by_mid.get(mid, ())
        if spans and role == StepRole.TOOL_RESULT:
            spans = ()   # attested — the whole content already counts
        step = Step(
            run_id=run_id,
            step_id=mid,
            index=int(mid) if mid.isdigit() else base_idx + i,
            role=role,
            content=content,
            tool_name=tool_name,
            obs_regions=spans,
        )
        index.add_step(step)
        steps_by_id[mid] = step

    # Symbols: canonical name from the ``name`` attr when the surface is
    # not canonical; the surface itself becomes an alias. Aliases are no
    # longer hand-listed — they emerge from marked surfaces.
    for surface, attrs in syms:
        canonical = (attrs.get("name") or surface).strip()
        if not canonical:
            continue
        aliases = (
            [surface.strip()]
            if surface.strip() and normalize_name(surface) != normalize_name(canonical)
            else []
        )
        kind = attrs.get("kind", "unknown").lower()
        raw_class = attrs.get("class", "identifier")
        entity_class = (
            raw_class if raw_class in _ENTITY_CLASS_VALUES else "identifier"
        )
        sym_dict = {"name": canonical, "kind": kind, "aliases": aliases}
        ns = namespace_fn(run_id, sym_dict) if namespace_fn else ""
        index.upsert_symbol(
            name=canonical, kind=kind, aliases=aliases,
            namespace=ns, entity_class=entity_class,
        )

    # Claims: text is an exact content slice — verbatim by construction.
    for mid, spans_list in claims_by_mid.items():
        cstep = steps_by_id.get(mid)
        if cstep is None:
            continue
        for start_c, end_c, role in spans_list:
            text = cstep.content[start_c:end_c].strip()
            if not text:
                continue
            # start offset discriminates same-prefix claims within a step
            cid = stable_id("clm", run_id, mid, start_c, text[:80])
            index.claims[cid] = Claim(
                id=cid, run_id=run_id, step_id=mid, text=text, role=role,
            )

    # Constraints: task requirements, verbatim; machine-checkable
    # normalization comes from model-emitted attrs, code-validated
    # (unparseable attrs degrade to a semantic constraint, logged).
    for mid, start_c, end_c, attrs in constraints_new:
        nstep = steps_by_id.get(mid)
        if nstep is None:
            continue
        text = nstep.content[start_c:end_c].strip()
        if not text:
            continue
        normalized = _constraint_normalized(attrs)
        if attrs.get("kind") and normalized is None:
            _prune(f"step {mid}",
                   f"constraint attrs unparseable, kept as semantic: {attrs!r}")
        conid = stable_id("con", run_id, mid, start_c, text[:80])
        index.constraints[conid] = Constraint(
            id=conid,
            subject=attrs.get("subject", "answer"),
            description=text,
            normalized=normalized,
        )

    all_syms = index.registry_snapshot()
    namespaces = {str(s["name"]): namespace_fn(run_id, s) if namespace_fn else "" for s in all_syms}
    refs = _build_references(all_syms, messages)
    for ref in refs:
        rsym = index.resolve_symbol_by_name(
            ref.symbol_name,
            namespace=namespaces.get(ref.symbol_name, ""),
        )
        rstep = steps_by_id.get(ref.turn_id)
        if rsym and rstep:
            index.add_reference(
                symbol=rsym, step=rstep, text=ref.text,
                kind=ref.kind, start=ref.start,
                confidence=reference_confidence,
            )
