"""Extractor tools: per-firing state, witness validation, and tool surface."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from agentm.core.abi import (
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
    ToolTerminate,
)
from agentm.extensions import ExtensionManifest
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llmharness.schema import CommitmentStatus, Edge, EdgeKind, EdgeRole, Event, EventKind

from .index_store import (
    Index,
    IndexOp,
    LinkDelete,
    LinkUpsert,
    RecordDelete,
    RecordUpsert,
    fold_index,
    merge_to_phases,
    parse_op,
)

__all__: Final = [
    "Index",
    "IndexOp",
    "LinkDelete",
    "LinkUpsert",
    "RecordDelete",
    "RecordUpsert",
    "fold_index",
    "merge_to_phases",
    "parse_op",
]

# ---------------------------------------------------------------------------
# Witness validation
# ---------------------------------------------------------------------------

_WS_RUN = re.compile(r"\s+")


def normalize(s: str) -> str:
    """Lowercase + collapse runs of whitespace to a single space."""
    return _WS_RUN.sub(" ", s.lower()).strip()


def witness_data(
    cited_entities: list[str],
    src_text: str,
    dst_text: str,
) -> str | None:
    """Verify every entity in cited_entities appears in src OR dst."""
    if not cited_entities:
        return "witness/data: cited_entities must be non-empty for kind='data'"
    src_norm = normalize(src_text)
    dst_norm = normalize(dst_text)
    for entity in cited_entities:
        ent_norm = normalize(entity)
        if not ent_norm:
            return "witness/data: cited entity is empty after normalization"
        if ent_norm not in src_norm and ent_norm not in dst_norm:
            return (
                f"witness/data: cited entity {entity!r} not found in normalized "
                "src_turns OR dst_turns text"
            )
    return None


def witness_ref(
    cited_quote: str,
    src_text: str,
    dst_text: str,
) -> str | None:
    """Verify cited_quote appears verbatim (mod normalize) in src OR dst."""
    if not cited_quote:
        return "witness/ref: cited_quote must be non-empty for kind='ref'"
    quote_norm = normalize(cited_quote)
    if not quote_norm:
        return "witness/ref: cited_quote is empty after normalization"
    src_norm = normalize(src_text)
    dst_norm = normalize(dst_text)
    if quote_norm not in src_norm and quote_norm not in dst_norm:
        return (
            f"witness/ref: cited_quote {cited_quote!r} not found in normalized "
            "src_turns OR dst_turns text"
        )
    return None


# ---------------------------------------------------------------------------
# Section 6: Error formatting
# ---------------------------------------------------------------------------


def format_witness_error(
    *,
    symptom: str,
    attempt: str | None,
    state_echo: str | None,
    options: list[str],
) -> str:
    """Render a three-section actionable error message."""
    if not options:
        raise ValueError(
            "format_witness_error: 'options' must be non-empty — every "
            "tool error must name at least one concrete next action"
        )
    attempt_line = attempt if attempt else "—"
    state_line = state_echo if state_echo else "(empty)"
    labelled: list[str] = []
    for idx, opt in enumerate(options):
        labelled.append(f"    ({chr(ord('a') + idx)}) {opt}")
    return (
        f"{symptom}\n"
        "\n"
        f"  what you tried:    {attempt_line}\n"
        f"  current index:     {state_line}\n"
        "\n"
        "  next options:\n" + "\n".join(labelled)
    )


# ---------------------------------------------------------------------------
# Section 7: State echo
# ---------------------------------------------------------------------------


def _state_echo(state: ExtractionState) -> str:
    """One-line summary of the currently-folded index for this firing."""
    if not state.pending_ops:
        return ""
    records = state.pending_index.records
    links = state.pending_index.links
    if not records:
        return ""
    last_id = max(records)
    last = records[last_id]
    return (
        f"{len(records)} record(s), {len(links)} link(s); "
        f"last accepted: id={last.id} kind={last.kind.value}"
    )


# ---------------------------------------------------------------------------
# Section 8: ExtractionState
# ---------------------------------------------------------------------------

EVENT_KIND_VALUES: Final[list[str]] = [k.value for k in EventKind]
EDGE_KIND_VALUES: Final[list[str]] = [k.value for k in EdgeKind]
COMMITMENT_STATUS_VALUES: Final[list[str]] = [s.value for s in CommitmentStatus]
EDGE_ROLE_VALUES: Final[list[str]] = [r.value for r in EdgeRole]


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _validate_event_shape(idx: int, raw: dict[str, Any]) -> tuple[str | None, Event | None]:
    """Validate one record payload from upsert_record / apply_record_upsert."""
    eid_raw = raw.get("id")
    kind_raw = raw.get("kind")
    summary_raw = raw.get("summary")
    source_turns_raw = raw.get("source_turns")

    if isinstance(eid_raw, bool) or not isinstance(eid_raw, int):
        return f"upsert_record: events[{idx}].id must be an integer", None
    if eid_raw < 1:
        return f"upsert_record: events[{idx}].id must be >= 1; got {eid_raw}", None
    if not isinstance(kind_raw, str):
        return f"upsert_record: events[{idx}].kind must be a string", None
    try:
        kind = EventKind(kind_raw)
    except ValueError:
        return (
            f"upsert_record: events[{idx}].kind {kind_raw!r} not in {EVENT_KIND_VALUES}",
            None,
        )
    if not isinstance(summary_raw, str) or not summary_raw.strip():
        return f"upsert_record: events[{idx}].summary must be a non-empty string", None
    if not isinstance(source_turns_raw, list) or not source_turns_raw:
        return (
            f"upsert_record: events[{idx}].source_turns must be a non-empty array of integers",
            None,
        )
    source_turns: list[int] = []
    for t in source_turns_raw:
        if isinstance(t, bool) or not isinstance(t, int):
            return (
                f"upsert_record: events[{idx}].source_turns contains non-integer entry {t!r}",
                None,
            )
        source_turns.append(t)
    return None, Event(id=eid_raw, kind=kind, summary=summary_raw, source_turns=source_turns)


def _compute_degree_warning(
    events: list[Event],
    edges: list[Edge],
) -> str | None:
    """Flag consecutive (in=1, out=1) chain links as a soft advisory."""
    if len(events) <= 1:
        return None
    in_deg: dict[int, int] = {ev.id: 0 for ev in events}
    out_deg: dict[int, int] = {ev.id: 0 for ev in events}
    for ed in edges:
        if ed.dst in in_deg:
            in_deg[ed.dst] += 1
        if ed.src in out_deg:
            out_deg[ed.src] += 1
    chain_links: list[Event] = [ev for ev in events if in_deg[ev.id] == 1 and out_deg[ev.id] == 1]
    if not chain_links:
        return None
    lines = [
        f"  event[{ev.id}] kind={ev.kind.value} '{ev.summary[:70]}': in=1, out=1"
        for ev in chain_links
    ]
    return (
        f"Soft warning: {len(chain_links)} chain-link event(s) "
        "(in-degree=1 AND out-degree=1) detected. If two adjacent "
        "``act`` records have nothing branching between them, consider "
        "merging them into one coalesced ``act`` (record every probe "
        "and result in time order in the summary). If a real "
        "``hyp`` / ``dec`` reasoning move was made between them but "
        "you didn't emit a record for it, add one in a follow-up "
        "firing. Aim for compact indexes but do NOT fabricate refs "
        "just to satisfy this heuristic.\n"
        "Chain-link events:\n" + "\n".join(lines)
    )


@dataclass
class ExtractionState:
    """Per-firing in-memory state for the extractor."""

    # turn_index -> raw turn text for witness substring checks.
    turn_texts: dict[int, str] = field(default_factory=dict)

    # Recent index slice presented to the extractor this firing.
    recent_records: tuple[Event, ...] = ()
    recent_links: tuple[Edge, ...] = ()

    # Next available global event id.
    next_event_id: int = 1

    # Frozen results populated by finalize.
    events: tuple[Event, ...] = ()
    edges: tuple[Edge, ...] = ()
    dropped_edges: tuple[dict[str, Any], ...] = ()
    committed: bool = False

    # Index edit log.
    recent_record_dict: dict[int, Event] = field(default_factory=dict)
    recent_link_dict: dict[tuple[int, int, str], Edge] = field(default_factory=dict)
    pending_ops: list[IndexOp] = field(default_factory=list)
    pending_index: Index = field(default_factory=Index)

    ops_file: str | None = None

    def __post_init__(self) -> None:
        if self.recent_records and not self.recent_record_dict:
            self.recent_record_dict = {ev.id: ev for ev in self.recent_records}
        if self.recent_links and not self.recent_link_dict:
            self.recent_link_dict = {
                (ed.src, ed.dst, ed.kind.value): ed for ed in self.recent_links
            }
        if self.recent_record_dict or self.recent_link_dict or self.pending_ops:
            self._refold()

    @property
    def recent_graph(self) -> tuple[Event, ...]:
        """Compatibility alias for old replay code."""
        return self.recent_records

    @recent_graph.setter
    def recent_graph(self, value: tuple[Event, ...]) -> None:
        self.recent_records = value

    @property
    def recent_graph_dict(self) -> dict[int, Event]:
        """Compatibility alias for old replay code."""
        return self.recent_record_dict

    @recent_graph_dict.setter
    def recent_graph_dict(self, value: dict[int, Event]) -> None:
        self.recent_record_dict = value

    @property
    def recent_edges_dict(self) -> dict[tuple[int, int, str], Edge]:
        """Compatibility alias for old replay code."""
        return self.recent_link_dict

    @recent_edges_dict.setter
    def recent_edges_dict(self, value: dict[tuple[int, int, str], Edge]) -> None:
        self.recent_link_dict = value

    @property
    def pending_graph(self) -> Index:
        """Compatibility alias for old replay code."""
        return self.pending_index

    @pending_graph.setter
    def pending_graph(self, value: Index) -> None:
        self.pending_index = value

    def _persist_op(self, op: IndexOp) -> None:
        if self.ops_file is None:
            return
        try:
            p = Path(self.ops_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(op.to_dict(), ensure_ascii=False) + "\n")
        except OSError:
            logger.opt(exception=True).warning(f"ops file write failed: {self.ops_file}")

    # -- Public mutators ----------------------------------------------------

    def finalize(self) -> str | None:
        """Freeze pending state and commit."""
        if self.committed:
            return "finalize: firing already finalized"

        firing_record_ids = {op.id for op in self.pending_ops if isinstance(op, RecordUpsert)}
        firing_record_ids -= {op.id for op in self.pending_ops if isinstance(op, RecordDelete)}
        records, links_view = self._folded_view()
        firing_events = [records[rid] for rid in sorted(firing_record_ids) if rid in records]
        firing_edges = [
            ed
            for (src, dst, _kind), ed in links_view.items()
            if src in firing_record_ids or dst in firing_record_ids
        ]
        self.events = tuple(firing_events)
        self.edges = tuple(firing_edges)
        self.dropped_edges = ()
        self.committed = True
        return None

    def compute_degree_warning(self) -> str | None:
        """Return chain-link warning for the committed index, or None."""
        return _compute_degree_warning(list(self.events), list(self.edges))

    def reset_pending(self) -> None:
        """Drop this firing's ops so the LLM can re-submit from scratch."""
        self.pending_ops = []
        self._refold()

    def salvage(self) -> None:
        """Commit-on-stop: finalize if not already committed."""
        if not self.committed:
            self.finalize()

    # -- Event-sourcing apply_* surface -------------------------------------

    def _refold(self) -> None:
        """Recompute pending_index from recent + pending_ops."""
        prefix: list[IndexOp] = []
        for rid, ev in self.recent_record_dict.items():
            prefix.append(
                RecordUpsert(
                    id=rid,
                    kind=ev.kind.value,
                    summary=ev.summary,
                    source_turns=tuple(ev.source_turns),
                    external_refs=ev.external_refs,
                    status=ev.status.value if ev.status else None,
                )
            )
        for (src, dst, kind), ed in self.recent_link_dict.items():
            prefix.append(
                LinkUpsert(
                    src=src,
                    dst=dst,
                    kind=kind,
                    reason=ed.reason,
                    cited_entities=ed.cited_entities,
                    cited_quote=ed.cited_quote,
                    src_turns=ed.src_turns,
                    dst_turns=ed.dst_turns,
                    role=ed.role.value if ed.role else None,
                )
            )
        self.pending_index = fold_index(prefix + self.pending_ops)

    def _ops_digest(self, op: str) -> dict[str, Any]:
        return {
            "ok": True,
            "op": op,
            "pending_ops": len(self.pending_ops),
            "index_records": len(self.pending_index.records),
            "index_links": len(self.pending_index.links),
        }

    def _folded_view(self) -> tuple[dict[int, Event], dict[tuple[int, int, str], Edge]]:
        """Return (records, links) of the current folded index."""
        return self.pending_index.records, self.pending_index.links

    def apply_record_upsert(self, raw: dict[str, Any]) -> dict[str, Any] | str:
        """Validate + apply one RecordUpsert against the folded view."""
        if self.committed:
            return "apply_record_upsert: firing already finalized"
        err, ev = _validate_event_shape(0, raw)
        if err is not None:
            return err
        assert ev is not None

        records, _links = self._folded_view()
        deleted_in_firing = {op.id for op in self.pending_ops if isinstance(op, RecordDelete)}
        max_seen = max(
            [0]
            + list(records.keys())
            + [op.id for op in self.pending_ops if isinstance(op, RecordUpsert)]
            + list(deleted_in_firing)
            + [self.next_event_id - 1]
        )
        allowed_ids = set(records.keys()) | deleted_in_firing | {max_seen + 1}
        if ev.id not in allowed_ids:
            return (
                f"apply_record_upsert: id={ev.id} is neither an existing record "
                f"in the folded index (recent + pending), a record deleted in "
                f"this firing, nor the next available id ({max_seen + 1}). "
                "Pick one: edit an existing id, re-use a just-deleted id, or "
                "append at the next free slot."
            )

        status_raw = raw.get("status")
        if status_raw is not None and status_raw not in COMMITMENT_STATUS_VALUES:
            return (
                f"apply_record_upsert: status {status_raw!r} not in "
                f"{COMMITMENT_STATUS_VALUES}"
            )

        op = RecordUpsert(
            id=ev.id,
            kind=ev.kind.value,
            summary=ev.summary,
            source_turns=tuple(ev.source_turns),
            status=status_raw,
        )
        self.pending_ops.append(op)
        self._persist_op(op)
        self._refold()
        return self._ops_digest("record_upsert")

    def apply_record_delete(self, record_id: int) -> dict[str, Any] | str:
        """Validate + apply one RecordDelete against the folded view."""
        if self.committed:
            return "apply_record_delete: firing already finalized"
        records, _links = self._folded_view()
        if record_id not in records:
            return (
                f"apply_record_delete: unknown record id {record_id}. "
                f"Existing ids in the folded index: {sorted(records.keys())}"
            )
        op = RecordDelete(id=record_id)
        self.pending_ops.append(op)
        self._persist_op(op)
        self._refold()
        return self._ops_digest("record_delete")

    def apply_link_upsert(self, raw: dict[str, Any]) -> dict[str, Any] | str:
        """Validate + apply one LinkUpsert against the folded view."""
        if self.committed:
            return "apply_link_upsert: firing already finalized"

        src = _coerce_int(raw.get("src"))
        dst = _coerce_int(raw.get("dst"))
        kind_raw = raw.get("kind")
        if src is None or dst is None:
            return "apply_link_upsert: 'src' and 'dst' must be integers"
        try:
            kind = EdgeKind(kind_raw)
        except ValueError:
            return f"apply_link_upsert: kind {kind_raw!r} not in {EDGE_KIND_VALUES}"

        records, _links = self._folded_view()
        if src not in records or dst not in records:
            return (
                f"apply_link_upsert: src={src} and dst={dst} must both exist "
                f"in the folded index. Existing ids: {sorted(records.keys())}"
            )
        src_event = records[src]
        dst_event = records[dst]

        cited_entities_raw = raw.get("cited_entities", [])
        cited_quote = str(raw.get("cited_quote", "") or "")
        if kind is EdgeKind.DATA:
            if not isinstance(cited_entities_raw, list) or not cited_entities_raw:
                return "apply_link_upsert: kind='data' requires non-empty cited_entities"
            if any(not isinstance(e, str) or not e for e in cited_entities_raw):
                return "apply_link_upsert: cited_entities must be non-empty strings"
            cited_entities = tuple(str(e) for e in cited_entities_raw)
        else:
            if not cited_quote:
                return "apply_link_upsert: kind='ref' requires non-empty cited_quote"
            src_text = self._concat_turn_texts(src_event.source_turns)
            dst_text = self._concat_turn_texts(dst_event.source_turns)
            werr = witness_ref(cited_quote, src_text, dst_text)
            if werr is not None:
                return f"apply_link_upsert: {werr}"
            cited_entities = tuple(str(e) for e in (cited_entities_raw or []))

        reason = raw.get("reason", "")
        if not isinstance(reason, str):
            return "apply_link_upsert: reason must be a string"

        role_raw = raw.get("role")
        if role_raw is not None and role_raw not in EDGE_ROLE_VALUES:
            return (
                f"apply_link_upsert: role {role_raw!r} not in "
                f"{EDGE_ROLE_VALUES}"
            )

        op = LinkUpsert(
            src=src,
            dst=dst,
            kind=kind.value,
            reason=reason,
            cited_entities=cited_entities,
            cited_quote=cited_quote,
            src_turns=tuple(src_event.source_turns),
            dst_turns=tuple(dst_event.source_turns),
            role=role_raw,
        )
        self.pending_ops.append(op)
        self._persist_op(op)
        self._refold()
        return self._ops_digest("link_upsert")

    def apply_link_delete(self, selector: dict[str, Any]) -> dict[str, Any] | str:
        """Validate + apply one LinkDelete against the folded view."""
        if self.committed:
            return "apply_link_delete: firing already finalized"

        src = _coerce_int(selector.get("src"))
        dst = _coerce_int(selector.get("dst"))
        kind_raw = selector.get("kind")
        if src is None or dst is None:
            return "apply_link_delete: 'src' and 'dst' must be integers"
        if not isinstance(kind_raw, str) or not kind_raw:
            return "apply_link_delete: 'kind' is required and must be a string"
        try:
            EdgeKind(kind_raw)
        except ValueError:
            return f"apply_link_delete: kind {kind_raw!r} not in {EDGE_KIND_VALUES}"

        _records, links = self._folded_view()
        if (src, dst, kind_raw) not in links:
            return (
                f"apply_link_delete: link ({src}, {dst}, {kind_raw}) not found in the folded index"
            )
        op = LinkDelete(src=src, dst=dst, kind=kind_raw)
        self.pending_ops.append(op)
        self._persist_op(op)
        self._refold()
        return self._ops_digest("link_delete")

    def _concat_turn_texts(self, turn_indices: list[int] | tuple[int, ...]) -> str:
        return " ".join(self.turn_texts.get(idx, "") for idx in turn_indices)


# ---------------------------------------------------------------------------
# Section 9: Tool Pydantic models + builders
# ---------------------------------------------------------------------------

UPSERT_RECORD_TOOL_NAME = "upsert_record"
UPSERT_LINK_TOOL_NAME = "upsert_link"
DELETE_RECORD_TOOL_NAME = "delete_record"
DELETE_LINK_TOOL_NAME = "delete_link"
RESET_EXTRACTION_TOOL_NAME = "reset_extraction"
FINALIZE_EXTRACTION_TOOL_NAME = "finalize_extraction"
FINALIZE_EXTRACTION_REASON = "llmharness:finalize_extraction"

_EventKindLiteral = Literal["task", "hyp", "act", "dec", "concl"]
_EdgeKindLiteral = Literal["data", "ref"]

# -- upsert_record ------------------------------------------------------------


class UpsertRecordArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(
        description=(
            "Record id. If this id already exists in the current index "
            "(this firing or any prior firing), the record is updated "
            "in place. Otherwise the id must equal the next available "
            "id (max existing id + 1) or a record id you deleted earlier "
            "in this firing (re-use after delete is the merge-duplicate "
            "path). Gaps over still-live ids are rejected."
        ),
    )
    kind: _EventKindLiteral = Field(
        description=(
            "Closed-set event kind classified by ACTION SIGNATURE, "
            "not by what the agent says it is doing."
        ),
    )
    summary: str = Field(
        description=(
            "Natural-language paragraph describing this event, with "
            "LENGTH PROPORTIONAL TO source_turns COUNT. A "
            "single-turn branch event (task / hyp / dec / concl) is "
            "one focused sentence with the concrete claim. A linear "
            "act that COALESCES N consecutive turns must be a "
            "paragraph that walks through what happened across those "
            "N turns: roughly one short sentence per covered turn. "
            "Name every distinct tool_call's concrete parameters "
            "verbatim (services, time windows, query filters, file "
            "paths, error codes, span/log/metric names) AND quote the "
            "key numbers each result returned."
        ),
    )
    source_turns: list[int] = Field(
        description=(
            "Trajectory indices this event was extracted from. "
            "Non-empty and contiguous ([first, first+1, ..., last] "
            "with no gaps)."
        ),
    )
    status: Literal["exploratory", "tentative", "committed", "finalized"] | None = Field(
        default=None,
        description=(
            "Commitment status — REQUIRED for hyp/dec/concl records. "
            "exploratory=mentioned but not pursued, "
            "tentative=under active investigation, "
            "committed=later reasoning depends on this, "
            "finalized=part of the final answer. "
            "Must be set for hyp/dec/concl; omit only for task/act."
        ),
    )


def _upsert_record_attempt_echo(args: UpsertRecordArgs) -> str:
    summary_preview = args.summary[:40] + ("..." if len(args.summary) > 40 else "")
    return (
        f"upsert_record(id={args.id}, kind={args.kind}, "
        f"source_turns={args.source_turns}, summary={summary_preview!r})"
    )


def build_upsert_record_tool(state: ExtractionState) -> FunctionTool:
    """Build a FunctionTool for upsert_record closing over state."""

    async def _handler(args: dict[str, Any]) -> ToolResult:
        try:
            parsed = UpsertRecordArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"upsert_record rejected: {exc}")],
                is_error=True,
            )
        raw = parsed.model_dump()
        result = state.apply_record_upsert(raw)
        if isinstance(result, str):
            options = [
                "re-call upsert_record with id = max(existing ids) + 1 to append a fresh record",
                "re-call upsert_record with an id already present in the folded index to edit in place",
                "delete_record(<id>) first if you intended to replace and re-use that id",
            ]
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=format_witness_error(
                            symptom=result,
                            attempt=_upsert_record_attempt_echo(parsed),
                            state_echo=_state_echo(state),
                            options=options,
                        ),
                    )
                ],
                is_error=True,
            )
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        )

    return FunctionTool(
        name=UPSERT_RECORD_TOOL_NAME,
        description=(
            "Insert or replace one event record in the current index. "
            "Editing a prior-firing record is supported — the id resolves "
            "against the folded view (this firing + every prior firing). "
            "Recorded as a RecordUpsert op."
        ),
        parameters=UpsertRecordArgs,
        fn=_handler,
    )


# -- upsert_link ------------------------------------------------------------


class UpsertLinkArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src: int = Field(
        description="Source record id — must exist in the current index (this firing or any prior firing).",
    )
    dst: int = Field(
        description=(
            "Destination record id — must exist in the current index "
            "(this firing or any prior firing)."
        ),
    )
    kind: _EdgeKindLiteral = Field(
        description=(
            "Link kind. 'ref' for textual-witness links (cited_quote); "
            "'data' for entity-witness links (cited_entities)."
        ),
    )
    reason: str = Field(
        description="One short sentence explaining the causal connection.",
    )
    cited_entities: list[str] = Field(
        description=(
            "Concrete identifiers (service names, span ids, file "
            "paths, error codes, etc.) shared by src and dst. "
            "Required non-empty when kind='data'; each entity must "
            "appear (case+whitespace normalized substring) in at "
            "least one of the src or dst record's source_turns text. "
            "Pass [] when kind='ref'."
        ),
    )
    cited_quote: str = Field(
        description=(
            "Verbatim substring of at least one endpoint record's "
            "source_turns text (case+whitespace normalized). "
            "Required when kind='ref'. "
            "Paraphrasing or reformatting is rejected at op-build "
            "time. Pass \"\" when kind='data'."
        ),
    )
    role: Literal["supports", "weakens", "depends", "narrows"] = Field(
        description=(
            "Context-link role — REQUIRED. "
            "supports=evidence positively confirms the claim, "
            "weakens=evidence partially contradicts, "
            "depends=logical dependency, "
            "narrows=eliminates alternatives."
        ),
    )


def _upsert_link_attempt_echo(args: UpsertLinkArgs) -> str:
    if args.kind == "data":
        return (
            f"upsert_link(src={args.src}, dst={args.dst}, kind=data, "
            f"cited_entities={args.cited_entities!r})"
        )
    quote_preview = args.cited_quote[:60] + ("..." if len(args.cited_quote) > 60 else "")
    return f"upsert_link(src={args.src}, dst={args.dst}, kind=ref, cited_quote={quote_preview!r})"


def build_upsert_link_tool(state: ExtractionState) -> FunctionTool:
    """Build a FunctionTool for upsert_link closing over state."""

    async def _handler(args: dict[str, Any]) -> ToolResult:
        try:
            parsed = UpsertLinkArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"upsert_link rejected: {exc}")],
                is_error=True,
            )
        raw = parsed.model_dump()
        result = state.apply_link_upsert(raw)
        if isinstance(result, str):
            records, _links = state._folded_view()
            existing = sorted(records.keys())[:8]
            if parsed.kind == "ref":
                options = [
                    "re-call upsert_link with a cited_quote that is a verbatim "
                    "substring of at least one endpoint's source_turns text (after "
                    "case+whitespace normalisation)",
                    "switch to kind='data' if no shared literal quote exists, "
                    "and pass cited_entities=[<shared identifier>, ...]",
                    f"verify src/dst are existing record ids — folded view contains {existing}",
                ]
            else:
                options = [
                    "re-call upsert_link with non-empty cited_entities — each "
                    "entry must be a concrete identifier present in at least one "
                    "endpoint's source_turns text",
                    "switch to kind='ref' with cited_quote if a verbatim shared "
                    "substring exists in at least one endpoint",
                    f"verify src/dst are existing record ids — folded view contains {existing}",
                ]
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=format_witness_error(
                            symptom=result,
                            attempt=_upsert_link_attempt_echo(parsed),
                            state_echo=_state_echo(state),
                            options=options,
                        ),
                    )
                ],
                is_error=True,
            )
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        )

    return FunctionTool(
        name=UPSERT_LINK_TOOL_NAME,
        description=(
            "Insert or replace one witness-bearing link keyed by (src, dst, kind). "
            "Both endpoint records must already exist in the current index (this firing "
            "or any prior firing). kind='data' requires non-empty cited_entities; "
            "kind='ref' requires cited_quote to appear in at least one endpoint "
            "record's source_turns text."
        ),
        parameters=UpsertLinkArgs,
        fn=_handler,
    )


# -- delete_record -------------------------------------------------------------


class DeleteRecordArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(
        description=(
            "Event record id to delete. May reference a record from this "
            "firing or from any prior firing — the resulting "
            "RecordDelete cascades to all incident links at fold time."
        ),
    )


def build_delete_record_tool(state: ExtractionState) -> FunctionTool:
    """Build a FunctionTool for delete_record closing over state."""

    async def _handler(args: dict[str, Any]) -> ToolResult:
        try:
            parsed = DeleteRecordArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"delete_record rejected: {exc}")],
                is_error=True,
            )
        result = state.apply_record_delete(parsed.id)
        if isinstance(result, str):
            records, _links = state._folded_view()
            sample_ids = sorted(records.keys())[:5]
            options = [
                f"re-call delete_record with one of the existing ids: {sample_ids}",
                "if you meant to skip this delete, just proceed with upsert_record / upsert_link",
            ]
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=format_witness_error(
                            symptom=result,
                            attempt=f"delete_record(id={parsed.id})",
                            state_echo=_state_echo(state),
                            options=options,
                        ),
                    )
                ],
                is_error=True,
            )
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        )

    return FunctionTool(
        name=DELETE_RECORD_TOOL_NAME,
        description=(
            "Delete one event record from the current index (this firing or any "
            "prior firing). Every link incident to that record is removed "
            "automatically at fold time."
        ),
        parameters=DeleteRecordArgs,
        fn=_handler,
    )


# -- delete_link -------------------------------------------------------------


class DeleteLinkArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src: int = Field(description="Source record id of the link to delete.")
    dst: int = Field(description="Destination record id of the link to delete.")
    kind: _EdgeKindLiteral = Field(
        description=(
            "Required: the link kind to delete. Without this the "
            "selector is ambiguous — the same (src, dst) pair may "
            "carry both a 'data' and a 'ref' link in the persistent "
            "op log."
        ),
    )


def build_delete_link_tool(state: ExtractionState) -> FunctionTool:
    """Build a FunctionTool for delete_link closing over state."""

    async def _handler(args: dict[str, Any]) -> ToolResult:
        try:
            parsed = DeleteLinkArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"delete_link rejected: {exc}")],
                is_error=True,
            )
        raw = parsed.model_dump()
        result = state.apply_link_delete(raw)
        if isinstance(result, str):
            _records, links = state._folded_view()
            present = sorted(links.keys())[:8]
            options = [
                f"re-call delete_link with an existing (src, dst, kind) "
                f"triple — folded view contains {present}",
                "if you wanted to remove a record entirely, call delete_record(<id>) "
                "instead and let the link cascade off automatically",
            ]
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=format_witness_error(
                            symptom=result,
                            attempt=f"delete_link(src={parsed.src}, dst={parsed.dst}, kind={parsed.kind})",
                            state_echo=_state_echo(state),
                            options=options,
                        ),
                    )
                ],
                is_error=True,
            )
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        )

    return FunctionTool(
        name=DELETE_LINK_TOOL_NAME,
        description=(
            "Delete one link identified by (src, dst, kind). 'kind' is mandatory "
            "because the same (src, dst) pair may carry both a 'data' and a 'ref' link."
        ),
        parameters=DeleteLinkArgs,
        fn=_handler,
    )


# -- reset_extraction --------------------------------------------------------


class ResetExtractionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def build_reset_extraction_tool(state: ExtractionState) -> FunctionTool:
    """Build a FunctionTool for reset_extraction closing over state."""

    async def _handler(args: dict[str, Any]) -> ToolResult:
        try:
            ResetExtractionArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"reset_extraction rejected: {exc}")],
                is_error=True,
            )
        if state.committed:
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=format_witness_error(
                            symptom=(
                                "reset_extraction: firing already finalized; "
                                "reset is not possible after finalize_extraction succeeded"
                            ),
                            attempt="reset_extraction()",
                            state_echo=_state_echo(state),
                            options=[
                                "stop calling tools — the firing has already terminated",
                            ],
                        ),
                    )
                ],
                is_error=True,
            )
        state.reset_pending()
        return ToolResult(content=[TextContent(type="text", text='{"ok": true, "reset": true}')])

    return FunctionTool(
        name=RESET_EXTRACTION_TOOL_NAME,
        description=(
            "Rarely needed; use delete_record + upsert_record to repair instead. "
            "Drops all pending records / links so you can re-emit the firing's "
            "index from scratch. Reserved as a last-resort escape hatch when "
            "the accumulated draft is fundamentally unrecoverable."
        ),
        parameters=ResetExtractionArgs,
        fn=_handler,
    )


# -- finalize_extraction -----------------------------------------------------


class FinalizeExtractionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def build_finalize_extraction_tool(state: ExtractionState) -> FunctionTool:
    """Build a FunctionTool for finalize_extraction closing over state."""

    async def _handler(args: dict[str, Any]) -> ToolTerminate | ToolResult:
        try:
            FinalizeExtractionArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"finalize_extraction rejected: {exc}")],
                is_error=True,
            )
        err = state.finalize()
        if err is not None:
            return ToolResult(
                content=[TextContent(type="text", text=err)],
                is_error=True,
            )
        digest = (
            f'{{"ok": true, "records": {len(state.events)}, '
            f'"links": {len(state.edges)}, '
            f'"dropped": {len(state.dropped_edges)}}}'
        )
        warning = state.compute_degree_warning()
        text = "Index committed. Note: " + warning + "\n\n" + digest if warning else digest
        return ToolTerminate(
            result=ToolResult(content=[TextContent(type="text", text=text)]),
            reason=FINALIZE_EXTRACTION_REASON,
        )

    return FunctionTool(
        name=FINALIZE_EXTRACTION_TOOL_NAME,
        description=(
            "Terminate the extractor firing. Call this AFTER emitting every "
            "record/link with upsert_record / upsert_link (and any merges via "
            "delete_record / delete_link). The handler commits the witness-valid "
            "index and ends the firing."
        ),
        parameters=FinalizeExtractionArgs,
        fn=_handler,
        metadata={"terminates": True},
    )


# ---------------------------------------------------------------------------
# Section 10: Atom (MANIFEST + install)
# ---------------------------------------------------------------------------

STATE_SERVICE_KEY = "llmharness.extractor_state"


class ExtractorToolsConfig(BaseModel):
    model_config = {"extra": "allow"}


MANIFEST = ExtensionManifest(
    name="extractor_tools",
    description="Register the extractor context-index editing tools.",
    registers=(
        "tool:upsert_record",
        "tool:upsert_link",
        "tool:delete_record",
        "tool:delete_link",
        "tool:reset_extraction",
        "tool:finalize_extraction",
    ),
    requires=("extractor_context",),
    config_schema=ExtractorToolsConfig,
)

_BUILDERS: Final = [
    build_upsert_record_tool,
    build_upsert_link_tool,
    build_delete_record_tool,
    build_delete_link_tool,
    build_reset_extraction_tool,
    build_finalize_extraction_tool,
]


def install(api: ExtensionAPI, config: ExtractorToolsConfig) -> None:
    from .context import STATE_SERVICE_KEY as _CTX_KEY

    state = api.get_service(_CTX_KEY)
    if not isinstance(state, ExtractionState):
        raise RuntimeError("extractor_tools: no ExtractionState in service registry")
    for build in _BUILDERS:
        api.register_tool(build(state))
