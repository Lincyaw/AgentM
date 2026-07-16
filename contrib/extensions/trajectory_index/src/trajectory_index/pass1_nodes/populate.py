"""Pass 1 — populate the index IR from an extraction result.

The extractor outputs a structured list of symbols and claims; this pass
lands them as first-class nodes (symbols, claims). References are then
built deterministically by grepping symbol names over the message content.

``populate_from_extraction`` takes a ``TrajectoryIndex`` and mutates it;
``index.py`` exposes it as a thin method so the public API is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..ir.models import (
    _ENTITY_CLASS_VALUES,
    Claim,
    Constraint,
    Step,
    StepRole,
    mentions_symbol,
    stable_id,
)
from ..locate import locate, strip_tags

if TYPE_CHECKING:
    from ..ir.index import TrajectoryIndex



def _message_step_content(msg: dict[str, Any]) -> tuple[str, str | None]:
    """Step content + tool name, from the single shared message walk.

    Derived from ``data.message_parts`` — the same pairs the extractor's
    view is built from — so the two representations that offset alignment
    depends on cannot drift apart.
    """
    from .serialize import message_parts

    pairs, tool_name = message_parts(msg)
    return "\n".join(c for c, _ in pairs if c), tool_name


def _locate_claim(text: str, steps: dict[str, Step], role_filter: str | None = "assistant") -> tuple[str, int, int] | None:
    """Locate a claim/constraint in steps via word-level matching."""
    for mid, step in steps.items():
        if role_filter and step.role != role_filter:
            continue
        hit = locate(text, step.content)
        if hit:
            return mid, hit[0], hit[1]
    return None


def _link_symbols(index: TrajectoryIndex, run_id: str) -> None:
    """Attach symbol_ids to claims and constraints by word-bounded mention."""
    sym_names: list[tuple[str, list[str]]] = [
        (sid, [sym.canonical_name, *sym.aliases])
        for sid, sym in index.symbols.items()
    ]
    for cid, claim in index.claims.items():
        if claim.run_id and claim.run_id != run_id:
            continue
        hits = tuple(sid for sid, names in sym_names if mentions_symbol(claim.text, names))
        if hits and hits != claim.symbol_ids:
            index.claims[cid] = Claim(
                id=claim.id, run_id=claim.run_id, step_id=claim.step_id,
                text=claim.text, role=claim.role, symbol_ids=hits,
            )
    for cid, con in index.constraints.items():
        hits = tuple(sid for sid, names in sym_names if mentions_symbol(con.description, names))
        if hits and hits != con.symbol_ids:
            index.constraints[cid] = Constraint(
                id=con.id, description=con.description,
                normalized=con.normalized, symbol_ids=hits,
            )


def populate_from_extraction(
    index: TrajectoryIndex,
    result: Any,
    messages: list[Any],
    *,
    run_id: str = "",
    namespace_fn: Any | None = None,
    message_id_start: int = 0,
    diagnostics: Any | None = None,
) -> None:
    """Populate index from an extraction result and source messages.

    ``result`` carries a structured list of symbols and claims (the new
    schema); code locates each in the original messages by name/substring
    matching. References are built deterministically by grepping symbol
    names over the message content.

    ``messages`` can be typed ``AgentMessage`` objects or pre-serialized
    dicts. ``namespace_fn(run_id, sym_dict) -> str`` optionally scopes
    symbols. ``message_id_start`` is the absolute offset of the first
    message in the full trajectory.
    """
    from loguru import logger as _clog

    from .serialize import _build_references

    if messages and not isinstance(messages[0], dict):
        from ..atom import _agentmsg_to_extraction_dict

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

    for i, msg in enumerate(messages):
        mid = str(msg.get("id", f"s{base_idx + i}"))
        role = role_map.get(str(msg.get("role", "")), StepRole.USER)
        content, tool_name = _message_step_content(msg)

        call_id: str | None = None
        blocks = msg.get("content", [])
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict) and block.get("type") in ("tool_call", "tool_use"):
                    call_id = block.get("id")
                    break

        step = Step(
            run_id=run_id,
            step_id=mid,
            index=int(mid) if mid.isdigit() else base_idx + i,
            role=role,
            content=content,
            tool_name=tool_name,
            call_id=call_id,
        )
        index.add_step(step)
        steps_by_id[mid] = step

    # Symbols: directly from result.symbols (or legacy result.annotated).
    extracted_syms = getattr(result, "symbols", None)
    for sym in (extracted_syms or []):
        canonical = strip_tags(sym.name).strip()
        if not canonical:
            continue
        kind = strip_tags(sym.kind).lower() if sym.kind else "unknown"
        raw_class = getattr(sym, "entity_class", "identifier")
        entity_class = raw_class if raw_class in _ENTITY_CLASS_VALUES else "identifier"
        aliases = [strip_tags(a) for a in sym.aliases] if sym.aliases else []
        sym_dict = {"name": canonical, "kind": kind, "aliases": aliases}
        ns = namespace_fn(run_id, sym_dict) if namespace_fn else ""
        index.upsert_symbol(
            name=canonical, kind=kind, aliases=aliases,
            namespace=ns, entity_class=entity_class,
        )

    # Claims: head/tail anchors locate assertions in assistant steps.
    extracted_claims = getattr(result, "claims", None)
    if extracted_claims is not None:
        for claim in extracted_claims:
            head = str(getattr(claim, "head", getattr(claim, "text", ""))).strip()
            if not head:
                continue
            tail = str(getattr(claim, "tail", "")).strip()
            query = f"{head}…{tail}" if tail else head
            role = getattr(claim, "role", "") or ""
            loc = _locate_claim(query, steps_by_id)
            if loc is None:
                _prune("claim", f"could not locate: {head[:40]!r}")
                continue
            mid, start_c, end_c = loc
            full_text = steps_by_id[mid].content[start_c:end_c].strip()
            cid = stable_id("clm", run_id, mid, start_c, full_text[:80])
            index.claims[cid] = Claim(
                id=cid, run_id=run_id, step_id=mid, text=full_text, role=role,
            )

    # Observations: head/tail anchors locate retrieved regions in chunk steps.
    extracted_obs = getattr(result, "observations", None)
    for obs in (extracted_obs or []):
        head = str(getattr(obs, "head", "")).strip()
        if not head:
            continue
        tail = str(getattr(obs, "tail", "")).strip()
        query = f"{head}…{tail}" if tail else head
        loc = _locate_claim(query, steps_by_id, role_filter="assistant")
        if loc is None:
            _prune("obs", f"anchor not found: {head[:40]!r}")
            continue
        obs_mid, hi, end = loc
        obs_step = steps_by_id[obs_mid]
        existing = list(obs_step.obs_regions)
        existing.append((hi, end))
        existing.sort()
        merged: list[tuple[int, int]] = []
        for span in existing:
            if merged and span[0] <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))
            else:
                merged.append(span)
        new_step = Step(
            run_id=obs_step.run_id, step_id=obs_step.step_id, index=obs_step.index,
            role=obs_step.role, content=obs_step.content, tool_name=obs_step.tool_name,
            timestamp=obs_step.timestamp, metadata=obs_step.metadata,
            obs_regions=tuple(merged),
        )
        index.steps[(obs_step.run_id, obs_step.step_id)] = new_step
        steps_by_id[obs_mid] = new_step

    # Constraints: head/tail anchors locate requirements.
    extracted_constraints = getattr(result, "constraints", None)
    for con in (extracted_constraints or []):
        head = str(getattr(con, "head", getattr(con, "text", ""))).strip()
        if not head:
            continue
        tail = str(getattr(con, "tail", "")).strip()
        query = f"{head}…{tail}" if tail else head
        loc = _locate_claim(query, steps_by_id, role_filter=None)
        if loc is None:
            _prune("constraint", f"could not locate: {head[:40]!r}")
            continue
        mid, start_c, end_c = loc
        full_text = steps_by_id[mid].content[start_c:end_c].strip()
        conid = stable_id("con", run_id, mid, start_c, full_text[:80])
        index.constraints[conid] = Constraint(
            id=conid, description=full_text,
        )

    # Values: symbol:value pairs from tool results (LLM-extracted).
    extracted_values = getattr(result, "values", None)
    for val in (extracted_values or []):
        sym_name = str(getattr(val, "sym", "")).strip()
        val_text = str(getattr(val, "value", "")).strip()
        if not sym_name or not val_text:
            continue
        sym_name = strip_tags(sym_name)
        val_text = strip_tags(val_text)
        # Locate the value text in a tool_result step (the READ side).
        loc = _locate_claim(val_text, steps_by_id, role_filter="tool_result")
        if loc is None:
            loc = _locate_claim(val_text, steps_by_id, role_filter=None)
        if loc is None:
            _prune("val", f"value not found: {sym_name}={val_text[:30]!r}")
            continue
        mid, start_c, end_c = loc
        sym = index.upsert_symbol(
            name=sym_name, kind="variable", entity_class="value",
        )
        index.add_reference(
            symbol=sym, step=steps_by_id[mid],
            text=val_text, kind="tool_output",
            start=start_c, value=val_text,
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
            )

    _link_symbols(index, run_id)
