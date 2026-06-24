"""Derived LSP-style context index for auditor prompts.

The index is a navigation surface over a trajectory prefix. It intentionally
does not decide whether a claim is correct or whether a reminder should fire.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from llmharness.schema import (
    AttentionHint,
    CandidateRef,
    CandidateState,
    ClaimRef,
    ContextIndex,
    ContractEventRef,
    ContractStatus,
    EntityRef,
    EntityType,
    IndexLink,
    LinkKind,
    ObligationRef,
    ObservationRef,
    ObservationSignal,
    TurnKind,
    TurnRef,
)

_MAX_SUMMARY_CHARS = 360
_ENTITY_RE = re.compile(
    r"""
    (?:
        \b[A-Za-z][A-Za-z0-9]*(?:[-_.:/][A-Za-z0-9]+)+\b
        |
        \b[A-Z][A-Z0-9_]{2,}\b
    )
    """,
    re.VERBOSE,
)
_VALUE_RE = re.compile(
    r"\b(?:abn|abnormal|normal|nml|ratio|count|total|errors?|latency|p\d+|avg(?:_ms)?)"
    r"[\w ./%:-]{0,24}(?:=|:)\s*-?\d+(?:\.\d+)?%?",
    re.IGNORECASE,
)
_OBLIGATION_RE = re.compile(
    r"\b(?:need to|should|must|TODO|todo|pending|will check|before finalizing|before submitting)\b",
    re.IGNORECASE,
)
_DEMOTION_RE = re.compile(
    r"\b(?:rule(?:d)? out|exclude(?:d)?|demote(?:d)?|not (?:the )?root|downstream|victim)\b",
    re.IGNORECASE,
)
_FINAL_RE = re.compile(r"\b(?:final|submit_final_report|root_cause|root cause)\b", re.IGNORECASE)
_CONTRACT_RE = re.compile(
    r"\b(?:submit_final_report|validation|schema|malformed|rejected|empty|invalid|failed|error)\b",
    re.IGNORECASE,
)
_STOP_ENTITIES = frozenset(
    {
        "abnormal_logs",
        "abnormal_metrics",
        "abnormal_metrics_histogram",
        "abnormal_metrics_sum",
        "abnormal_traces",
        "abnormal-vs-normal",
        "avg_ms",
        "call-count",
        "call-graph",
        "callee",
        "caller",
        "cnt",
        "count",
        "duration",
        "get",
        "http",
        "inter-service",
        "metric",
        "normal-only",
        "normal_logs",
        "normal_metrics",
        "normal_metrics_histogram",
        "normal_metrics_sum",
        "normal_traces",
        "null",
        "p50",
        "p95",
        "p99",
        "p99_ms",
        "parent_span_id",
        "post",
        "query_sql",
        "root-span",
        "route-path",
        "select",
        "service-level",
        "service_name",
        "span_id",
        "span_name",
        "top-level",
        "top-span",
        "trace_id",
        "value",
        "normal",
        "abnormal",
        "root-cause",
        "root_cause",
        "final-report",
        "submit-final-report",
        "source-turns",
        "source_turns",
        "attr.status_code",
        "attr.http.response.status_code",
    }
)


def build_context_index(
    *,
    trajectory: Sequence[Mapping[str, Any]] = (),
    symbols: Sequence[Mapping[str, Any]] = (),
    references: Sequence[Mapping[str, Any]] = (),
) -> ContextIndex:
    """Build a sparse navigation index from trajectory and pre-built symbol table.

    ``symbols`` and ``references`` come from the trajectory_index extension.
    Each symbol has ``{id, name, kind, aliases, summary}``, each reference has
    ``{id, symbol_id, step_id, text, kind, confidence}``.
    """

    entity_data: dict[str, dict[str, Any]] = {}
    links: list[IndexLink] = []

    def add_entity(name: str, turns: Sequence[int], preferred_type: EntityType = "unknown") -> str:
        clean = _clean_entity(name)
        key = clean.casefold()
        if not clean or key in _STOP_ENTITIES or _is_low_value_entity(clean):
            return ""
        inferred = _infer_entity_type(clean)
        ent_type: EntityType = preferred_type if preferred_type != "unknown" else inferred
        existing = entity_data.get(key)
        if existing is None:
            entity_id = f"ent:{len(entity_data) + 1}"
            entity_data[key] = {
                "id": entity_id,
                "name": clean,
                "type": ent_type,
                "turns": set(int(t) for t in turns if t >= 0),
                "aliases": set(),
            }
            return entity_id
        existing_turns = existing["turns"]
        if isinstance(existing_turns, set):
            existing_turns.update(int(t) for t in turns if t >= 0)
        if existing.get("type") == "unknown" and ent_type != "unknown":
            existing["type"] = ent_type
        return str(existing["id"])

    # --- Ingest symbols from the trajectory_index symbol table ---
    symbol_id_to_entity_id: dict[str, str] = {}
    for sym in symbols:
        sym_id = str(sym.get("id", ""))
        sym_name = str(sym.get("name", ""))
        sym_kind = str(sym.get("kind", ""))
        if not sym_name:
            continue
        preferred_type = _symbol_kind_to_entity_type(sym_kind, sym_name)
        # Symbols don't carry per-turn data; turns come from references
        entity_id = add_entity(sym_name, (), preferred_type)
        if entity_id:
            symbol_id_to_entity_id[sym_id] = entity_id
            # Register aliases
            aliases_raw = sym.get("aliases")
            if isinstance(aliases_raw, (list, tuple)):
                key = sym_name.casefold()
                existing = entity_data.get(key)
                if existing is not None:
                    alias_set = existing.get("aliases")
                    if isinstance(alias_set, set):
                        for alias in aliases_raw:
                            if isinstance(alias, str) and alias.strip():
                                alias_set.add(alias.strip())

    # --- Ingest references: associate symbols with turns ---
    # Group references by symbol for candidate-state derivation
    refs_by_symbol: dict[str, list[Mapping[str, Any]]] = {}
    for ref in references:
        sym_id = str(ref.get("symbol_id", ""))
        step_id = ref.get("step_id")
        entity_id = symbol_id_to_entity_id.get(sym_id, "")
        if entity_id and isinstance(step_id, int) and step_id >= 0:
            # Register turn association on the entity
            key_for_entity = None
            for k, v in entity_data.items():
                if v.get("id") == entity_id:
                    key_for_entity = k
                    break
            if key_for_entity is not None:
                turns_set = entity_data[key_for_entity].get("turns")
                if isinstance(turns_set, set):
                    turns_set.add(step_id)
        refs_by_symbol.setdefault(sym_id, []).append(ref)

    turn_refs = tuple(_turn_ref(i, turn) for i, turn in enumerate(trajectory))

    observations: list[ObservationRef] = []
    claims: list[ClaimRef] = []
    obligations: list[ObligationRef] = []
    contract_events: list[ContractEventRef] = []

    for turn in turn_refs:
        entity_ids = _entity_ids_from_text(turn.summary, (turn.turn_index,), add_entity)
        if turn.kind == "tool_result":
            obs = ObservationRef(
                id=f"obs:turn:{turn.turn_index}",
                turns=(turn.turn_index,),
                source="tool_result",
                summary=turn.summary,
                entities=entity_ids,
                values=_extract_values(turn.summary),
                signals=_observation_signals(turn.summary),
            )
            observations.append(obs)
            links.extend(_entity_links(obs.id, entity_ids, "mentions", "tool result mentions entity"))
        if turn.kind == "assistant" and _OBLIGATION_RE.search(turn.summary):
            obligation = ObligationRef(
                id=f"obl:turn:{turn.turn_index}",
                turns=(turn.turn_index,),
                source="agent_plan",
                text=turn.summary,
                entities=entity_ids,
            )
            obligations.append(obligation)
            links.extend(_entity_links(obligation.id, entity_ids, "mentions", "obligation mentions entity"))
        status = _contract_status(turn.summary)
        if status is not None:
            contract = ContractEventRef(
                id=f"contract:turn:{turn.turn_index}",
                turns=(turn.turn_index,),
                tool=_contract_tool(turn.summary),
                status=status,
                summary=turn.summary,
                entities=entity_ids,
            )
            contract_events.append(contract)
            links.extend(_entity_links(contract.id, entity_ids, "mentions", "contract event mentions entity"))

    # Derive candidate state from reference kinds: tool_input/tool_output → investigated
    _promote_candidates_from_references(entity_data, refs_by_symbol, symbol_id_to_entity_id)
    candidates = _candidate_refs(entity_data, observations, claims)
    entities = _entity_refs(entity_data)
    attention_hints = _attention_hints(entities, observations, candidates)

    return ContextIndex(
        turns=turn_refs,
        entities=entities,
        observations=tuple(observations),
        claims=tuple(claims),
        candidates=tuple(candidates),
        obligations=tuple(obligations),
        contract_events=tuple(contract_events),
        links=tuple(links),
        attention_hints=tuple(attention_hints),
    )


def _turn_ref(index: int, turn: Mapping[str, Any]) -> TurnRef:
    raw_index = turn.get("index")
    turn_index = raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) else index
    role = str(turn.get("role") or turn.get("type") or turn.get("kind") or "unknown")
    summary = _excerpt(_turn_summary(turn), compact=False)
    return TurnRef(
        turn_index=turn_index,
        role=role,
        kind=_infer_turn_kind(role, summary),
        summary=summary,
    )


def _turn_summary(turn: Mapping[str, Any]) -> str:
    """Return human text from a serialized trajectory turn.

    The context index relies on plain table text to detect signals such as
    ``abn_cnt=0`` versus ``nml_cnt>0``. JSON-dumping the whole turn escapes tabs
    and newlines, hiding those patterns from the lightweight regexes below.
    """
    content = turn.get("content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return _stringify(turn)
    blocks: list[str] = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        block_type = str(block.get("type", ""))
        if block_type == "tool_result":
            nested = block.get("content")
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                for item in nested:
                    if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                        blocks.append(str(item["text"]))
        elif block_type == "tool_call":
            name = str(block.get("name", "tool_call"))
            args = _stringify(block.get("arguments", {}))
            blocks.append(f"{name} {args}")
        elif isinstance(block.get("text"), str):
            blocks.append(str(block["text"]))
    return "\n".join(blocks) if blocks else _stringify(turn)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _excerpt(text: str, limit: int = _MAX_SUMMARY_CHARS, *, compact: bool = True) -> str:
    excerpt = " ".join(text.split()) if compact else text.strip()
    if len(excerpt) <= limit:
        return excerpt
    return excerpt[: limit - 3] + "..."


def _clean_entity(name: str) -> str:
    return name.strip(" \t\r\n.,;:()[]{}<>\"'`")


def _infer_turn_kind(role: str, summary: str) -> TurnKind:
    raw = f"{role} {summary}".casefold()
    if "toolresult" in raw or "tool_result" in raw or "tool result" in raw:
        return "tool_result"
    if "toolcall" in raw or "tool_call" in raw or "tool call" in raw:
        return "tool_call"
    if "system-reminder" in raw or "<system-reminder>" in raw:
        return "reminder"
    if "system" in role.casefold():
        return "system"
    if "user" in role.casefold():
        return "user"
    return "assistant"


def _infer_entity_type(name: str) -> EntityType:
    low = name.casefold()
    if low.startswith("/") or "http." in low or "http_" in low:
        return "endpoint"
    if "->" in name or "→" in name:
        return "edge"
    if any(part in low for part in ("metric", "cpu", "memory", "jvm.", "hubble", "db.", "k8s.")):
        return "metric"
    if any(part in low for part in ("error", "exception", "timeout", "refused", "reset")):
        return "log_pattern"
    if any(part in low for part in ("mutator", "networkloss", "podfailure", "fault")):
        return "fault_kind"
    if low in {"submit_final_report", "query_sql", "describe_tables", "load_skill"}:
        return "tool"
    if low in {"root_cause", "causal_graph", "nodes", "edges"}:
        return "schema_field"
    if low.startswith(("ts-", "svc:")) or "service" in low:
        return "service"
    return "unknown"


def _is_low_value_entity(name: str) -> bool:
    low = name.casefold()
    if low in _STOP_ENTITIES:
        return True
    if low.startswith("attr."):
        return True
    if re.fullmatch(r"p\d+(?:_ms)?", low):
        return True
    if re.fullmatch(r"(?:avg|sum|min|max|count|cnt)(?:_ms|_cnt|_count)?", low):
        return True
    return low.endswith(("_id", "_name")) and not low.startswith(("ts-", "/"))


def _entity_ids_from_text(
    text: str,
    turns: Sequence[int],
    add_entity: Callable[[str, Sequence[int]], str],
) -> tuple[str, ...]:
    ids: list[str] = []
    seen: set[str] = set()
    for match in _ENTITY_RE.finditer(text):
        entity_id = add_entity(match.group(0), turns)
        if entity_id and entity_id not in seen:
            ids.append(entity_id)
            seen.add(entity_id)
    return tuple(ids)


def _extract_values(text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    values: list[str] = []
    for match in _VALUE_RE.finditer(text):
        val = _excerpt(match.group(0), 120)
        if val not in seen:
            values.append(val)
            seen.add(val)
    return tuple(values[:12])


def _observation_signals(text: str) -> tuple[ObservationSignal, ...]:
    low = text.casefold()
    signals: list[ObservationSignal] = []

    def add(signal: ObservationSignal) -> None:
        if signal not in signals:
            signals.append(signal)

    if any(
        phrase in low
        for phrase in (
            "normal-only",
            "normal only",
            "no abnormal",
            "0 abnormal",
            "absent in abnormal",
            "missing child",
            "missing span",
            "vanish",
            "disappear",
            "no abnormal row",
        )
    ):
        add("missing_or_normal_only")
    if (
        re.search(r"\babn_(?:cnt|count)\b", low)
        and re.search(r"\bnml_(?:cnt|count)\b", low)
        and re.search(r"(?:^|\n)[^\n]*\t0\t[1-9]\d*(?:\t|$)", text)
    ):
        add("missing_or_normal_only")
    if any(
        phrase in low
        for phrase in (
            "dropped",
            "drop from",
            "drop ",
            "decrease",
            "reduced",
            "shrink",
            "fell",
            "fewer",
            "lower count",
        )
    ):
        add("volume_or_count_drop")
    if any(
        phrase in low
        for phrase in (
            "increased",
            "increase",
            "rose",
            "higher",
            "surge",
            "amplification",
            "more calls",
            "volume",
        )
    ):
        add("volume_or_count_increase")
    if any(
        phrase in low
        for phrase in (
            "latency",
            "duration",
            "avg_ms",
            "p95",
            "p99",
            "slow",
            "slower",
        )
    ):
        add("latency_delta")
    if any(phrase in low for phrase in ("error", "errors", "http 4", "status_code", "exception")):
        add("error_delta")
    concrete_resource = any(
        phrase in low
        for phrase in (
            "cpu",
            "memory",
            "filesystem",
            "storage",
            "disk",
            "k8s.",
            "container.",
            "hubble",
            "network.io",
            "page_fault",
        )
    )
    resource_delta_word = any(
        phrase in low
        for phrase in (
            "abnormal",
            "delta",
            "higher",
            "increase",
            "ratio",
            "usage",
            "spike",
            "surge",
            "normal",
        )
    )
    if concrete_resource or ("resource" in low and resource_delta_word):
        add("resource_delta")
    if any(
        phrase in low
        for phrase in (
            "0 errors",
            "0 technical errors",
            "no technical errors",
            "no elevated errors",
            "resources normal",
            "stable local resources",
            "not worse",
            "lower than normal",
            "normal resources",
        )
    ):
        add("weak_or_no_error")
    if _contract_status(text) is not None:
        add("schema_or_output_failure")
    return tuple(signals)


def _symbol_kind_to_entity_type(kind: str, name: str) -> EntityType:
    """Map trajectory_index symbol kind to ContextIndex EntityType."""
    mapping: dict[str, EntityType] = {
        "service": "service",
        "api": "endpoint",
        "metric": "metric",
        "error": "log_pattern",
        "tool": "tool",
        "skill": "tool",
    }
    if kind in mapping:
        return mapping[kind]
    if kind == "concept":
        low = name.casefold()
        if any(p in low for p in ("cpu_stress", "network_loss", "fault", "mutator", "podfailure")):
            return "fault_kind"
    return "unknown"


def _promote_candidates_from_references(
    entity_data: dict[str, dict[str, Any]],
    refs_by_symbol: dict[str, list[Mapping[str, Any]]],
    symbol_id_to_entity_id: dict[str, str],
) -> None:
    """Mark entities as investigated based on reference kinds from trajectory_index."""
    for sym_id, ref_list in refs_by_symbol.items():
        entity_id = symbol_id_to_entity_id.get(sym_id)
        if not entity_id:
            continue
        has_tool_ref = any(
            str(r.get("kind", "")) in ("tool_input", "tool_output")
            for r in ref_list
        )
        if has_tool_ref:
            for data in entity_data.values():
                if data.get("id") == entity_id:
                    data.setdefault("_investigated", True)
                    break


def _contract_status(text: str) -> ContractStatus | None:
    if not _CONTRACT_RE.search(text):
        return None
    low = text.casefold()
    if "repaired" in low or "accepted" in low or "success" in low:
        return "repaired"
    if "validation" in low or "schema" in low:
        return "validation_failed"
    if "malformed" in low or "invalid" in low:
        return "malformed"
    if "empty" in low:
        return "empty"
    if "rejected" in low or "failed" in low or "error" in low:
        return "rejected"
    return None


def _contract_tool(text: str) -> str:
    low = text.casefold()
    for tool in ("submit_final_report", "query_sql", "describe_tables", "load_skill"):
        if tool in low:
            return tool
    return "unknown"


def _entity_links(src: str, entity_ids: Sequence[str], kind: LinkKind, reason: str) -> list[IndexLink]:
    return [IndexLink(src=src, dst=entity_id, kind=kind, reason=reason) for entity_id in entity_ids]


_STATE_RANK: dict[CandidateState, int] = {
    "mentioned": 0,
    "investigated": 1,
    "retained": 2,
    "demoted": 3,
    "finalized": 4,
}


def _candidate_refs(
    entity_data: Mapping[str, Mapping[str, Any]],
    observations: Sequence[ObservationRef],
    claims: Sequence[ClaimRef],
) -> list[CandidateRef]:
    data: dict[str, dict[str, Any]] = {}

    def touch(
        entity_id: str,
        turns: Sequence[int],
        state: CandidateState,
        *,
        evidence_id: str | None = None,
        evidence_signals: Sequence[ObservationSignal] = (),
        reason_claim_id: str | None = None,
    ) -> None:
        if not entity_id:
            return
        state_turn = min(turns) if turns else None
        last_turn = max(turns) if turns else None
        existing = data.get(entity_id)
        if existing is None:
            data[entity_id] = {
                "entity_id": entity_id,
                "first_seen_turn": state_turn,
                "last_seen_turn": last_turn,
                "state": state,
                "state_turn": state_turn,
                "reason_claim_id": reason_claim_id,
                "evidence_ids": [evidence_id] if evidence_id else [],
                "evidence_tags": list(evidence_signals),
            }
            return
        existing_first = existing.get("first_seen_turn")
        if state_turn is not None and (existing_first is None or state_turn < existing_first):
            existing["first_seen_turn"] = state_turn
        existing_last = existing.get("last_seen_turn")
        if last_turn is not None and (existing_last is None or last_turn > existing_last):
            existing["last_seen_turn"] = last_turn
        current_state = existing.get("state")
        if isinstance(current_state, str) and _STATE_RANK[state] >= _STATE_RANK[current_state]:  # type: ignore[index]
            existing["state"] = state
            existing["state_turn"] = state_turn
            if reason_claim_id is not None:
                existing["reason_claim_id"] = reason_claim_id
        ev_ids = existing.get("evidence_ids")
        if isinstance(ev_ids, list) and evidence_id and evidence_id not in ev_ids:
            ev_ids.append(evidence_id)
        tags = existing.get("evidence_tags")
        if isinstance(tags, list):
            for signal in evidence_signals:
                if signal not in tags:
                    tags.append(signal)

    for raw in entity_data.values():
        entity_id = str(raw.get("id") or "")
        raw_turns = raw.get("turns")
        turns = tuple(sorted(raw_turns)) if isinstance(raw_turns, set) else ()
        initial_state: CandidateState = "investigated" if raw.get("_investigated") else "mentioned"
        touch(entity_id, turns, initial_state)
    for obs in observations:
        for entity_id in obs.entities:
            touch(
                entity_id,
                obs.turns,
                "investigated",
                evidence_id=obs.id,
                evidence_signals=obs.signals,
            )
    for claim in claims:
        if claim.kind == "final_answer" or claim.status == "finalized":
            state: CandidateState = "finalized"
        elif claim.kind == "demotion":
            state = "demoted"
        else:
            state = "retained"
        for entity_id in claim.entities:
            touch(entity_id, claim.turns, state, reason_claim_id=claim.id)

    return [
        CandidateRef(
            entity_id=str(item["entity_id"]),
            first_seen_turn=item.get("first_seen_turn")
            if isinstance(item.get("first_seen_turn"), int)
            else None,
            last_seen_turn=item.get("last_seen_turn")
            if isinstance(item.get("last_seen_turn"), int)
            else None,
            state=item["state"],
            state_turn=item.get("state_turn") if isinstance(item.get("state_turn"), int) else None,
            reason_claim_id=str(item["reason_claim_id"])
            if item.get("reason_claim_id") is not None
            else None,
            evidence_ids=tuple(str(e) for e in item.get("evidence_ids", []) if e),
            evidence_tags=tuple(
                tag
                for tag in item.get("evidence_tags", [])
                if tag
            ),
        )
        for item in data.values()
    ]


def _entity_refs(entity_data: Mapping[str, Mapping[str, Any]]) -> tuple[EntityRef, ...]:
    refs: list[EntityRef] = []
    for item in entity_data.values():
        raw_turns = item.get("turns")
        turns = tuple(sorted(raw_turns)) if isinstance(raw_turns, set) else ()
        raw_aliases = item.get("aliases")
        aliases = tuple(sorted(raw_aliases)) if isinstance(raw_aliases, set) else ()
        refs.append(
            EntityRef(
                id=str(item["id"]),
                name=str(item["name"]),
                type=item["type"],
                turns=turns,
                aliases=aliases,
            )
        )
    return tuple(refs)


def _attention_hints(
    entities: Sequence[EntityRef],
    observations: Sequence[ObservationRef],
    candidates: Sequence[CandidateRef],
) -> list[AttentionHint]:
    """Pre-aggregate high-value navigation cues for the auditor.

    These hints are deliberately non-verdict-bearing.  They surface patterns
    that were repeatedly useful in fork experiments while leaving the auditor to
    decide whether a reminder is justified.
    """

    entity_names = {entity.id: entity.name for entity in entities}
    entity_types = {entity.id: entity.type for entity in entities}
    hints: list[AttentionHint] = []

    def entity_priority(entity_id: str) -> tuple[int, str]:
        name = entity_names.get(entity_id, "")
        ent_type = entity_types.get(entity_id, "unknown")
        if ent_type in {"service", "endpoint", "edge"}:
            return (0, name)
        if ent_type in {"metric", "log_pattern", "fault_kind"}:
            return (1, name)
        return (2, name)

    def names_for(entity_ids: Sequence[str], *, limit: int = 10) -> tuple[str, ...]:
        names: list[str] = []
        seen: set[str] = set()
        for entity_id in sorted(entity_ids, key=entity_priority):
            name = entity_names.get(entity_id)
            if not name or name in seen:
                continue
            names.append(name)
            seen.add(name)
            if len(names) >= limit:
                break
        return tuple(names)

    def hintable_root_entity(entity_id: str) -> bool:
        return entity_types.get(entity_id) in {"service", "endpoint", "edge"}

    def evidence_excerpt(items: Sequence[ObservationRef], *, limit: int = 320) -> str:
        snippets: list[str] = []
        for item in items:
            if not (
                {"missing_or_normal_only", "volume_or_count_drop", "resource_delta"}
                & set(item.signals)
            ):
                continue
            snippets.append(_excerpt(item.summary, 120))
            if len("; ".join(snippets)) >= limit:
                break
        return _excerpt("; ".join(snippets), limit)

    def collect(obs: Sequence[ObservationRef]) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...], tuple[ObservationSignal, ...]]:
        turns: set[int] = set()
        entity_ids: list[str] = []
        obs_ids: list[str] = []
        signals: list[ObservationSignal] = []
        seen_entities: set[str] = set()
        for item in obs:
            turns.update(item.turns)
            obs_ids.append(item.id)
            for entity_id in item.entities:
                if entity_id not in seen_entities:
                    entity_ids.append(entity_id)
                    seen_entities.add(entity_id)
            for signal in item.signals:
                if signal not in signals:
                    signals.append(signal)
        return tuple(sorted(turns)), tuple(entity_ids), tuple(obs_ids), tuple(signals)

    def top_observations(items: Sequence[ObservationRef], *, limit: int) -> list[ObservationRef]:
        priority = {
            "missing_or_normal_only": 0,
            "volume_or_count_drop": 1,
            "resource_delta": 2,
            "weak_or_no_error": 3,
            "volume_or_count_increase": 4,
            "latency_delta": 5,
            "error_delta": 6,
            "schema_or_output_failure": 7,
        }

        def key(item: ObservationRef) -> tuple[int, int, int]:
            best = min((priority[s] for s in item.signals), default=99)
            return best, min(item.turns, default=999999), -len(item.entities)

        return sorted(items, key=key)[:limit]

    competing = [
        obs
        for obs in observations
        if (
            "missing_or_normal_only" in obs.signals
            or "volume_or_count_drop" in obs.signals
        )
    ]
    if competing:
        selected = top_observations(competing, limit=8)
        turns, entity_ids, obs_ids, signals = collect(selected)
        names = names_for(entity_ids, limit=12)
        hints.append(
            AttentionHint(
                id="hint:competing-observation-cluster",
                kind="competing_observation_cluster",
                turns=turns,
                summary=(
                    "Visible competing observations show missing, normal-only, "
                    f"or sharply reduced activity for: {', '.join(names) or 'named entities'}."
                ),
                entities=entity_ids,
                observation_ids=obs_ids,
                signals=signals,
            )
        )

    for candidate in candidates:
        if not hintable_root_entity(candidate.entity_id):
            continue
        tags = set(candidate.evidence_tags)
        if "volume_or_count_increase" not in tags:
            continue
        if not ({"weak_or_no_error", "latency_delta"} & tags):
            continue
        name = entity_names.get(candidate.entity_id)
        if not name:
            continue
        hints.append(
            AttentionHint(
                id=f"hint:weak-candidate:{candidate.entity_id}",
                kind="weak_candidate_signal",
                turns=tuple(
                    turn
                    for turn in (candidate.first_seen_turn, candidate.last_seen_turn)
                    if turn is not None
                ),
                summary=(
                    f"{name} has volume/count-oriented evidence plus weak or "
                    "mixed local evidence; audit whether it is a root or a "
                    "downstream workload-shift effect before finalizing."
                ),
                entities=(candidate.entity_id,),
                observation_ids=candidate.evidence_ids,
                signals=candidate.evidence_tags,
            )
        )
        if len([hint for hint in hints if hint.kind == "weak_candidate_signal"]) >= 4:
            break

    by_entity: dict[str, list[ObservationRef]] = {}
    for obs in observations:
        for entity_id in obs.entities:
            by_entity.setdefault(entity_id, []).append(obs)
    local_hints = 0
    for entity_id, items in by_entity.items():
        if not hintable_root_entity(entity_id):
            continue
        local_signal_set: set[ObservationSignal] = {
            signal for obs in items for signal in obs.signals
        }
        if not (
            ({"missing_or_normal_only", "volume_or_count_drop"} & local_signal_set)
            and "resource_delta" in local_signal_set
        ):
            continue
        name = entity_names.get(entity_id)
        if not name:
            continue
        selected = top_observations(items, limit=5)
        turns, entity_ids, obs_ids, merged_signals = collect(selected)
        evidence = evidence_excerpt(selected)
        hints.append(
            AttentionHint(
                id=f"hint:local-signal-on-disappeared:{entity_id}",
                kind="local_signal_on_disappeared_entity",
                turns=turns,
                summary=(
                    f"{name} has both disappeared/normal-only activity evidence "
                    "and local resource evidence; use it in root/effect synthesis."
                    + (f" Evidence: {evidence}" if evidence else "")
                ),
                entities=(entity_id, *tuple(eid for eid in entity_ids if eid != entity_id)),
                observation_ids=obs_ids,
                signals=merged_signals,
            )
        )
        local_hints += 1
        if local_hints >= 4:
            break

    return hints[:8]


__all__ = [
    "build_context_index",
]
