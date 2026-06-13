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

from .graph import (
    EdgeDelete,
    EdgeUpsert,
    Graph,
    GraphOp,
    NodeDelete,
    NodeUpsert,
    fold_graph,
    merge_to_phases,
    parse_op,
)

# Re-export for backward compatibility
__all__: Final = [
    "EdgeDelete",
    "EdgeUpsert",
    "Graph",
    "GraphOp",
    "NodeDelete",
    "NodeUpsert",
    "fold_graph",
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
        f"  current graph:     {state_line}\n"
        "\n"
        "  next options:\n" + "\n".join(labelled)
    )


# ---------------------------------------------------------------------------
# Section 7: State echo
# ---------------------------------------------------------------------------


def _state_echo(state: ExtractionState) -> str:
    """One-line summary of the currently-folded graph for this firing."""
    if not state.pending_ops:
        return ""
    nodes = state.pending_graph.nodes
    edges = state.pending_graph.edges
    if not nodes:
        return ""
    last_id = max(nodes)
    last = nodes[last_id]
    return (
        f"{len(nodes)} node(s), {len(edges)} edge(s); "
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
    """Validate one node payload from upsert_node / apply_node_upsert."""
    eid_raw = raw.get("id")
    kind_raw = raw.get("kind")
    summary_raw = raw.get("summary")
    source_turns_raw = raw.get("source_turns")

    if isinstance(eid_raw, bool) or not isinstance(eid_raw, int):
        return f"upsert_node: events[{idx}].id must be an integer", None
    if eid_raw < 1:
        return f"upsert_node: events[{idx}].id must be >= 1; got {eid_raw}", None
    if not isinstance(kind_raw, str):
        return f"upsert_node: events[{idx}].kind must be a string", None
    try:
        kind = EventKind(kind_raw)
    except ValueError:
        return (
            f"upsert_node: events[{idx}].kind {kind_raw!r} not in {EVENT_KIND_VALUES}",
            None,
        )
    if not isinstance(summary_raw, str) or not summary_raw.strip():
        return f"upsert_node: events[{idx}].summary must be a non-empty string", None
    if not isinstance(source_turns_raw, list) or not source_turns_raw:
        return (
            f"upsert_node: events[{idx}].source_turns must be a non-empty array of integers",
            None,
        )
    source_turns: list[int] = []
    for t in source_turns_raw:
        if isinstance(t, bool) or not isinstance(t, int):
            return (
                f"upsert_node: events[{idx}].source_turns contains non-integer entry {t!r}",
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
        "``act`` nodes have nothing branching between them, consider "
        "merging them into one coalesced ``act`` (record every probe "
        "and result in time order in the summary). If a real "
        "``hyp`` / ``dec`` reasoning move was made between them but "
        "you didn't emit a node for it, add one in a follow-up "
        "firing. Aim for compact graphs but do NOT fabricate refs "
        "just to satisfy this heuristic.\n"
        "Chain-link events:\n" + "\n".join(lines)
    )


@dataclass
class ExtractionState:
    """Per-firing in-memory state for the extractor."""

    # turn_index -> raw turn text for witness substring checks.
    turn_texts: dict[int, str] = field(default_factory=dict)

    # Recent-graph slice presented to the extractor this firing.
    recent_graph: tuple[Event, ...] = ()

    # Next available global event id.
    next_event_id: int = 1

    # Frozen results populated by finalize.
    events: tuple[Event, ...] = ()
    edges: tuple[Edge, ...] = ()
    dropped_edges: tuple[dict[str, Any], ...] = ()
    committed: bool = False

    # Event-sourcing op log.
    recent_graph_dict: dict[int, Event] = field(default_factory=dict)
    recent_edges_dict: dict[tuple[int, int, str], Edge] = field(default_factory=dict)
    pending_ops: list[GraphOp] = field(default_factory=list)
    pending_graph: Graph = field(default_factory=Graph)

    ops_file: str | None = None

    def __post_init__(self) -> None:
        if self.recent_graph and not self.recent_graph_dict:
            self.recent_graph_dict = {ev.id: ev for ev in self.recent_graph}
        if self.recent_graph_dict or self.recent_edges_dict or self.pending_ops:
            self._refold()

    def _persist_op(self, op: GraphOp) -> None:
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

        firing_node_ids = {op.id for op in self.pending_ops if isinstance(op, NodeUpsert)}
        firing_node_ids -= {op.id for op in self.pending_ops if isinstance(op, NodeDelete)}
        nodes, edges_view = self._folded_view()
        firing_events = [nodes[nid] for nid in sorted(firing_node_ids) if nid in nodes]
        firing_edges = [
            ed
            for (src, dst, _kind), ed in edges_view.items()
            if src in firing_node_ids or dst in firing_node_ids
        ]
        self.events = tuple(firing_events)
        self.edges = tuple(firing_edges)
        self.dropped_edges = ()
        self.committed = True
        return None

    def compute_degree_warning(self) -> str | None:
        """Return chain-link warning for the committed graph, or None."""
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
        """Recompute pending_graph from recent + pending_ops."""
        prefix: list[GraphOp] = []
        for nid, ev in self.recent_graph_dict.items():
            prefix.append(
                NodeUpsert(
                    id=nid,
                    kind=ev.kind.value,
                    summary=ev.summary,
                    source_turns=tuple(ev.source_turns),
                    external_refs=ev.external_refs,
                    status=ev.status.value if ev.status else None,
                )
            )
        for (src, dst, kind), ed in self.recent_edges_dict.items():
            prefix.append(
                EdgeUpsert(
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
        self.pending_graph = fold_graph(prefix + self.pending_ops)

    def _ops_digest(self, op: str) -> dict[str, Any]:
        return {
            "ok": True,
            "op": op,
            "pending_ops": len(self.pending_ops),
            "graph_nodes": len(self.pending_graph.nodes),
            "graph_edges": len(self.pending_graph.edges),
        }

    def _folded_view(self) -> tuple[dict[int, Event], dict[tuple[int, int, str], Edge]]:
        """Return (nodes, edges) of the current folded graph."""
        return self.pending_graph.nodes, self.pending_graph.edges

    def apply_node_upsert(self, raw: dict[str, Any]) -> dict[str, Any] | str:
        """Validate + apply one NodeUpsert against the folded view."""
        if self.committed:
            return "apply_node_upsert: firing already finalized"
        err, ev = _validate_event_shape(0, raw)
        if err is not None:
            return err
        assert ev is not None

        nodes, _edges = self._folded_view()
        deleted_in_firing = {op.id for op in self.pending_ops if isinstance(op, NodeDelete)}
        max_seen = max(
            [0]
            + list(nodes.keys())
            + [op.id for op in self.pending_ops if isinstance(op, NodeUpsert)]
            + list(deleted_in_firing)
            + [self.next_event_id - 1]
        )
        allowed_ids = set(nodes.keys()) | deleted_in_firing | {max_seen + 1}
        if ev.id not in allowed_ids:
            return (
                f"apply_node_upsert: id={ev.id} is neither an existing node "
                f"in the folded graph (recent + pending), a node deleted in "
                f"this firing, nor the next available id ({max_seen + 1}). "
                "Pick one: edit an existing id, re-use a just-deleted id, or "
                "append at the next free slot."
            )

        status_raw = raw.get("status")
        if status_raw is not None and status_raw not in COMMITMENT_STATUS_VALUES:
            return (
                f"apply_node_upsert: status {status_raw!r} not in "
                f"{COMMITMENT_STATUS_VALUES}"
            )

        op = NodeUpsert(
            id=ev.id,
            kind=ev.kind.value,
            summary=ev.summary,
            source_turns=tuple(ev.source_turns),
            status=status_raw,
        )
        self.pending_ops.append(op)
        self._persist_op(op)
        self._refold()
        return self._ops_digest("node_upsert")

    def apply_node_delete(self, node_id: int) -> dict[str, Any] | str:
        """Validate + apply one NodeDelete against the folded view."""
        if self.committed:
            return "apply_node_delete: firing already finalized"
        nodes, _edges = self._folded_view()
        if node_id not in nodes:
            return (
                f"apply_node_delete: unknown node_id {node_id}. "
                f"Existing ids in the folded graph: {sorted(nodes.keys())}"
            )
        op = NodeDelete(id=node_id)
        self.pending_ops.append(op)
        self._persist_op(op)
        self._refold()
        return self._ops_digest("node_delete")

    def apply_edge_upsert(self, raw: dict[str, Any]) -> dict[str, Any] | str:
        """Validate + apply one EdgeUpsert against the folded view."""
        if self.committed:
            return "apply_edge_upsert: firing already finalized"

        src = _coerce_int(raw.get("src"))
        dst = _coerce_int(raw.get("dst"))
        kind_raw = raw.get("kind")
        if src is None or dst is None:
            return "apply_edge_upsert: 'src' and 'dst' must be integers"
        try:
            kind = EdgeKind(kind_raw)
        except ValueError:
            return f"apply_edge_upsert: kind {kind_raw!r} not in {EDGE_KIND_VALUES}"

        nodes, _edges = self._folded_view()
        if src not in nodes or dst not in nodes:
            return (
                f"apply_edge_upsert: src={src} and dst={dst} must both exist "
                f"in the folded graph. Existing ids: {sorted(nodes.keys())}"
            )
        src_event = nodes[src]
        dst_event = nodes[dst]

        cited_entities_raw = raw.get("cited_entities", [])
        cited_quote = str(raw.get("cited_quote", "") or "")
        if kind is EdgeKind.DATA:
            if not isinstance(cited_entities_raw, list) or not cited_entities_raw:
                return "apply_edge_upsert: kind='data' requires non-empty cited_entities"
            if any(not isinstance(e, str) or not e for e in cited_entities_raw):
                return "apply_edge_upsert: cited_entities must be non-empty strings"
            cited_entities = tuple(str(e) for e in cited_entities_raw)
        else:
            if not cited_quote:
                return "apply_edge_upsert: kind='ref' requires non-empty cited_quote"
            src_text = self._concat_turn_texts(src_event.source_turns)
            dst_text = self._concat_turn_texts(dst_event.source_turns)
            werr = witness_ref(cited_quote, src_text, dst_text)
            if werr is not None:
                return f"apply_edge_upsert: {werr}"
            cited_entities = tuple(str(e) for e in (cited_entities_raw or []))

        reason = raw.get("reason", "")
        if not isinstance(reason, str):
            return "apply_edge_upsert: reason must be a string"

        role_raw = raw.get("role")
        if role_raw is not None and role_raw not in EDGE_ROLE_VALUES:
            return (
                f"apply_edge_upsert: role {role_raw!r} not in "
                f"{EDGE_ROLE_VALUES}"
            )

        op = EdgeUpsert(
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
        return self._ops_digest("edge_upsert")

    def apply_edge_delete(self, selector: dict[str, Any]) -> dict[str, Any] | str:
        """Validate + apply one EdgeDelete against the folded view."""
        if self.committed:
            return "apply_edge_delete: firing already finalized"

        src = _coerce_int(selector.get("src"))
        dst = _coerce_int(selector.get("dst"))
        kind_raw = selector.get("kind")
        if src is None or dst is None:
            return "apply_edge_delete: 'src' and 'dst' must be integers"
        if not isinstance(kind_raw, str) or not kind_raw:
            return "apply_edge_delete: 'kind' is required and must be a string"
        try:
            EdgeKind(kind_raw)
        except ValueError:
            return f"apply_edge_delete: kind {kind_raw!r} not in {EDGE_KIND_VALUES}"

        _nodes, edges = self._folded_view()
        if (src, dst, kind_raw) not in edges:
            return (
                f"apply_edge_delete: edge ({src}, {dst}, {kind_raw}) not found in the folded graph"
            )
        op = EdgeDelete(src=src, dst=dst, kind=kind_raw)
        self.pending_ops.append(op)
        self._persist_op(op)
        self._refold()
        return self._ops_digest("edge_delete")

    def _concat_turn_texts(self, turn_indices: list[int] | tuple[int, ...]) -> str:
        return " ".join(self.turn_texts.get(idx, "") for idx in turn_indices)


# ---------------------------------------------------------------------------
# Section 9: Tool Pydantic models + builders
# ---------------------------------------------------------------------------

UPSERT_NODE_TOOL_NAME = "upsert_node"
UPSERT_EDGE_TOOL_NAME = "upsert_edge"
DELETE_NODE_TOOL_NAME = "delete_node"
DELETE_EDGE_TOOL_NAME = "delete_edge"
RESET_EXTRACTION_TOOL_NAME = "reset_extraction"
FINALIZE_EXTRACTION_TOOL_NAME = "finalize_extraction"
FINALIZE_EXTRACTION_REASON = "llmharness:finalize_extraction"

_EventKindLiteral = Literal["task", "hyp", "act", "dec", "concl"]
_EdgeKindLiteral = Literal["data", "ref"]

# -- upsert_node ------------------------------------------------------------


class UpsertNodeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(
        description=(
            "Node id. If this id already exists in the current graph "
            "(this firing or any prior firing), the node is updated "
            "in place. Otherwise the id must equal the next available "
            "id (max existing id + 1) or a node id you deleted earlier "
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
            "Commitment status for hyp/dec/concl nodes. "
            "exploratory=mentioned but not pursued, "
            "tentative=under active investigation, "
            "committed=later reasoning depends on this, "
            "finalized=part of the final answer. "
            "Ignored for task/act nodes. "
            "Omit to leave unchanged on existing nodes."
        ),
    )


def _upsert_node_attempt_echo(args: UpsertNodeArgs) -> str:
    summary_preview = args.summary[:40] + ("..." if len(args.summary) > 40 else "")
    return (
        f"upsert_node(id={args.id}, kind={args.kind}, "
        f"source_turns={args.source_turns}, summary={summary_preview!r})"
    )


def build_upsert_node_tool(state: ExtractionState) -> FunctionTool:
    """Build a FunctionTool for upsert_node closing over state."""

    async def _handler(args: dict[str, Any]) -> ToolResult:
        try:
            parsed = UpsertNodeArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"upsert_node rejected: {exc}")],
                is_error=True,
            )
        raw = parsed.model_dump()
        result = state.apply_node_upsert(raw)
        if isinstance(result, str):
            options = [
                "re-call upsert_node with id = max(existing ids) + 1 to append a fresh node",
                "re-call upsert_node with an id already present in the folded graph to edit in place",
                "delete_node(<id>) first if you intended to replace and re-use that id",
            ]
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=format_witness_error(
                            symptom=result,
                            attempt=_upsert_node_attempt_echo(parsed),
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
        name=UPSERT_NODE_TOOL_NAME,
        description=(
            "Insert or replace one event node in the current graph. "
            "Editing a prior-firing node is supported — the id resolves "
            "against the folded view (this firing + every prior firing). "
            "Recorded as a NodeUpsert op."
        ),
        parameters=UpsertNodeArgs,
        fn=_handler,
    )


# -- upsert_edge ------------------------------------------------------------


class UpsertEdgeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src: int = Field(
        description="Source node id — must exist in the current graph (this firing or any prior firing).",
    )
    dst: int = Field(
        description=(
            "Destination node id — must exist in the current graph "
            "(this firing or any prior firing)."
        ),
    )
    kind: _EdgeKindLiteral = Field(
        description=(
            "Edge kind. 'ref' for textual-witness edges (cited_quote); "
            "'data' for entity-witness edges (cited_entities)."
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
            "least one of the src or dst node's source_turns text. "
            "Pass [] when kind='ref'."
        ),
    )
    cited_quote: str = Field(
        description=(
            "Verbatim substring of at least one endpoint node's "
            "source_turns text (case+whitespace normalized). "
            "Required when kind='ref'. "
            "Paraphrasing or reformatting is rejected at op-build "
            "time. Pass \"\" when kind='data'."
        ),
    )
    role: Literal["supports", "weakens", "depends", "narrows"] | None = Field(
        default=None,
        description=(
            "Causal role of this edge. "
            "supports=evidence positively confirms the claim, "
            "weakens=evidence partially contradicts, "
            "depends=logical dependency (default), "
            "narrows=eliminates alternatives. "
            "Omit to default to depends."
        ),
    )


def _upsert_edge_attempt_echo(args: UpsertEdgeArgs) -> str:
    if args.kind == "data":
        return (
            f"upsert_edge(src={args.src}, dst={args.dst}, kind=data, "
            f"cited_entities={args.cited_entities!r})"
        )
    quote_preview = args.cited_quote[:60] + ("..." if len(args.cited_quote) > 60 else "")
    return f"upsert_edge(src={args.src}, dst={args.dst}, kind=ref, cited_quote={quote_preview!r})"


def build_upsert_edge_tool(state: ExtractionState) -> FunctionTool:
    """Build a FunctionTool for upsert_edge closing over state."""

    async def _handler(args: dict[str, Any]) -> ToolResult:
        try:
            parsed = UpsertEdgeArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"upsert_edge rejected: {exc}")],
                is_error=True,
            )
        raw = parsed.model_dump()
        result = state.apply_edge_upsert(raw)
        if isinstance(result, str):
            nodes, _edges = state._folded_view()
            existing = sorted(nodes.keys())[:8]
            if parsed.kind == "ref":
                options = [
                    "re-call upsert_edge with a cited_quote that is a verbatim "
                    "substring of at least one endpoint's source_turns text (after "
                    "case+whitespace normalisation)",
                    "switch to kind='data' if no shared literal quote exists, "
                    "and pass cited_entities=[<shared identifier>, ...]",
                    f"verify src/dst are existing node ids — folded view contains {existing}",
                ]
            else:
                options = [
                    "re-call upsert_edge with non-empty cited_entities — each "
                    "entry must be a concrete identifier present in at least one "
                    "endpoint's source_turns text",
                    "switch to kind='ref' with cited_quote if a verbatim shared "
                    "substring exists in at least one endpoint",
                    f"verify src/dst are existing node ids — folded view contains {existing}",
                ]
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=format_witness_error(
                            symptom=result,
                            attempt=_upsert_edge_attempt_echo(parsed),
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
        name=UPSERT_EDGE_TOOL_NAME,
        description=(
            "Insert or replace one witness-bearing edge keyed by (src, dst, kind). "
            "Both endpoint nodes must already exist in the current graph (this firing "
            "or any prior firing). kind='data' requires non-empty cited_entities; "
            "kind='ref' requires cited_quote to appear in at least one endpoint "
            "node's source_turns text."
        ),
        parameters=UpsertEdgeArgs,
        fn=_handler,
    )


# -- delete_node -------------------------------------------------------------


class DeleteNodeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(
        description=(
            "Event node id to delete. May reference a node from this "
            "firing or from any prior firing — the resulting "
            "NodeDelete cascades to all incident edges at fold time."
        ),
    )


def build_delete_node_tool(state: ExtractionState) -> FunctionTool:
    """Build a FunctionTool for delete_node closing over state."""

    async def _handler(args: dict[str, Any]) -> ToolResult:
        try:
            parsed = DeleteNodeArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"delete_node rejected: {exc}")],
                is_error=True,
            )
        result = state.apply_node_delete(parsed.id)
        if isinstance(result, str):
            nodes, _edges = state._folded_view()
            sample_ids = sorted(nodes.keys())[:5]
            options = [
                f"re-call delete_node with one of the existing ids: {sample_ids}",
                "if you meant to skip this delete, just proceed with upsert_node / upsert_edge",
            ]
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=format_witness_error(
                            symptom=result,
                            attempt=f"delete_node(id={parsed.id})",
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
        name=DELETE_NODE_TOOL_NAME,
        description=(
            "Delete one event node from the current graph (this firing or any "
            "prior firing). Every edge incident to that node is removed "
            "automatically at fold time."
        ),
        parameters=DeleteNodeArgs,
        fn=_handler,
    )


# -- delete_edge -------------------------------------------------------------


class DeleteEdgeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src: int = Field(description="Source node id of the edge to delete.")
    dst: int = Field(description="Destination node id of the edge to delete.")
    kind: _EdgeKindLiteral = Field(
        description=(
            "Required: the edge kind to delete. Without this the "
            "selector is ambiguous — the same (src, dst) pair may "
            "carry both a 'data' and a 'ref' edge in the persistent "
            "op log."
        ),
    )


def build_delete_edge_tool(state: ExtractionState) -> FunctionTool:
    """Build a FunctionTool for delete_edge closing over state."""

    async def _handler(args: dict[str, Any]) -> ToolResult:
        try:
            parsed = DeleteEdgeArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                content=[TextContent(type="text", text=f"delete_edge rejected: {exc}")],
                is_error=True,
            )
        raw = parsed.model_dump()
        result = state.apply_edge_delete(raw)
        if isinstance(result, str):
            _nodes, edges = state._folded_view()
            present = sorted(edges.keys())[:8]
            options = [
                f"re-call delete_edge with an existing (src, dst, kind) "
                f"triple — folded view contains {present}",
                "if you wanted to remove a node entirely, call delete_node(<id>) "
                "instead and let the edge cascade off automatically",
            ]
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=format_witness_error(
                            symptom=result,
                            attempt=f"delete_edge(src={parsed.src}, dst={parsed.dst}, kind={parsed.kind})",
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
        name=DELETE_EDGE_TOOL_NAME,
        description=(
            "Delete one edge identified by (src, dst, kind). 'kind' is mandatory "
            "because the same (src, dst) pair may carry both a 'data' and a 'ref' edge."
        ),
        parameters=DeleteEdgeArgs,
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
            "Rarely needed; use delete_node + upsert_node to repair instead. "
            "Drops all pending events / edges so you can re-emit the firing's "
            "graph from scratch. Reserved as a last-resort escape hatch when "
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
            f'{{"ok": true, "events": {len(state.events)}, '
            f'"edges": {len(state.edges)}, '
            f'"dropped": {len(state.dropped_edges)}}}'
        )
        warning = state.compute_degree_warning()
        text = "Graph committed. Note: " + warning + "\n\n" + digest if warning else digest
        return ToolTerminate(
            result=ToolResult(content=[TextContent(type="text", text=text)]),
            reason=FINALIZE_EXTRACTION_REASON,
        )

    return FunctionTool(
        name=FINALIZE_EXTRACTION_TOOL_NAME,
        description=(
            "Terminate the extractor firing. Call this AFTER emitting every "
            "node/edge with upsert_node / upsert_edge (and any merges via "
            "delete_node / delete_edge). The handler commits the witness-valid "
            "graph and ends the firing."
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
    description="Register the extractor graph-editing tools.",
    registers=(
        "tool:upsert_node",
        "tool:upsert_edge",
        "tool:delete_node",
        "tool:delete_edge",
        "tool:reset_extraction",
        "tool:finalize_extraction",
    ),
    requires=("extractor_context",),
    config_schema=ExtractorToolsConfig,
)

_BUILDERS: Final = [
    build_upsert_node_tool,
    build_upsert_edge_tool,
    build_delete_node_tool,
    build_delete_edge_tool,
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
