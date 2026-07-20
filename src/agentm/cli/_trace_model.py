# code-health: ignore-file[AM025] -- CLI renders typed-union trace records from query/store boundaries
"""Read-side view models for trajectory-oriented trace UIs.

This module is intentionally independent from Textual. It turns durable
trajectory records into flat rows that can be rendered by a TUI, printed by a
CLI command, or reused by future extension-specific views.
"""

from __future__ import annotations

import json
import re
import shlex
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from agentm.core.abi.messages import (
    ImageContent,
    OpaqueThinkingBlock,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
)
from agentm.core.abi.termination import ProviderRequestFailed
from agentm.core.abi.trajectory import Turn, TurnCheckpoint
from agentm.core.abi.trigger import UserInput

TraceRecordStatus = Literal["committed", "incomplete"]
TraceRowKind = Literal[
    "system",
    "trigger",
    "user",
    "assistant",
    "thinking",
    "tool_call",
    "tool_result",
    "error",
    "control",
    "metric",
    "policy",
]


@dataclass(frozen=True, slots=True)
class TraceMetrics:
    committed_turns: int = 0
    incomplete_turns: int = 0
    rows: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    by_tool: Mapping[str, int] = field(default_factory=dict)
    by_kind: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TraceTurnSummary:
    key: str
    turn_index: int
    turn_id: str
    status: TraceRecordStatus
    rounds: int
    tool_calls: int
    tool_errors: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    model: str | None = None
    cause: str | None = None
    trigger: str = ""

    @property
    def state_label(self) -> str:
        return self.cause or self.status


@dataclass(frozen=True, slots=True)
class TraceRow:
    key: str
    kind: TraceRowKind
    title: str
    preview: str
    content: str
    turn_index: int | None = None
    round_index: int | None = None
    message_index: int | None = None
    status: TraceRecordStatus | None = None
    role: str | None = None
    tool_name: str | None = None
    display_name: str | None = None
    is_error: bool = False
    cause: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def location(self) -> str:
        if self.turn_index is None:
            return "-"
        if self.round_index is None:
            return f"T{self.turn_index}"
        return f"T{self.turn_index} R{self.round_index}"

    @property
    def searchable_text(self) -> str:
        parts = [
            self.key,
            self.kind,
            self.title,
            self.preview,
            self.content,
            self.role or "",
            self.tool_name or "",
            self.display_name or "",
            self.status or "",
            self.cause or "",
            self.location,
        ]
        for key, value in self.metadata.items():
            parts.append(str(key))
            parts.append(str(value))
        return "\n".join(parts).lower()


@dataclass(frozen=True, slots=True)
class TraceSnapshot:
    session_id: str
    turns: tuple[TraceTurnSummary, ...]
    rows: tuple[TraceRow, ...]
    metrics: TraceMetrics

    @property
    def status_label(self) -> str:
        if self.metrics.incomplete_turns:
            return "running"
        return "complete" if self.metrics.committed_turns else "empty"


@dataclass(frozen=True, slots=True)
class TokenPredicate:
    field: Literal["tokens", "in", "out", "cache"]
    op: Literal[">", ">=", "<", "<=", "="]
    value: int


@dataclass(frozen=True, slots=True)
class TraceQuery:
    raw: str = ""
    terms: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    causes: tuple[str, ...] = ()
    turn_indices: tuple[int, ...] = ()
    round_indices: tuple[int, ...] = ()
    id_fragments: tuple[str, ...] = ()
    errors_only: bool = False
    token_predicates: tuple[TokenPredicate, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.terms,
                self.roles,
                self.kinds,
                self.tools,
                self.statuses,
                self.causes,
                self.turn_indices,
                self.round_indices,
                self.id_fragments,
                self.errors_only,
                self.token_predicates,
            )
        )


@dataclass(frozen=True, slots=True)
class TraceView:
    id: str
    title: str
    rows: tuple[TraceRow, ...]
    summary: str = ""
    empty_text: str = "No rows match the current query."


TraceViewBuilder = Callable[[TraceSnapshot, TraceQuery], TraceView]


@dataclass(frozen=True, slots=True)
class TraceViewSpec:
    """A pluggable read-only view over a trajectory snapshot."""

    id: str
    title: str
    description: str
    shortcut: str
    build: TraceViewBuilder


class TraceViewProvider(Protocol):
    """Extension hook for future trace views, for example policy metrics."""

    def trace_view_specs(self) -> Sequence[TraceViewSpec]: ...


class TraceViewRegistry:
    def __init__(self, specs: Iterable[TraceViewSpec] = ()) -> None:
        self._specs: dict[str, TraceViewSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: TraceViewSpec) -> None:
        if not spec.id:
            raise ValueError("trace view id must be non-empty")
        if spec.id in self._specs:
            raise ValueError(f"duplicate trace view id: {spec.id}")
        self._specs[spec.id] = spec

    def extend(self, provider: TraceViewProvider) -> None:
        for spec in provider.trace_view_specs():
            self.register(spec)

    def specs(self) -> tuple[TraceViewSpec, ...]:
        return tuple(self._specs.values())

    def get(self, view_id: str) -> TraceViewSpec:
        return self._specs[view_id]


_TOKEN_PATTERN = re.compile(r"^(tokens|in|out|cache)(>=|<=|>|<|=)(\d+)$")


def parse_trace_query(raw: str) -> TraceQuery:
    """Parse a compact query string.

    Supported filters:
      role:<name>, kind:<name>, tool:<name>, status:<committed|incomplete>,
      cause:<name>, turn:<n>, round:<n>, id:<fragment>, error/is:error, and
      tokens/in/out/cache comparisons such as tokens>100000.
    """

    if not raw.strip():
        return TraceQuery(raw=raw)
    try:
        parts = shlex.split(raw)
    except ValueError:
        parts = raw.split()

    terms: list[str] = []
    roles: list[str] = []
    kinds: list[str] = []
    tools: list[str] = []
    statuses: list[str] = []
    causes: list[str] = []
    turn_indices: list[int] = []
    round_indices: list[int] = []
    id_fragments: list[str] = []
    token_predicates: list[TokenPredicate] = []
    errors_only = False

    for part in parts:
        lowered = part.lower()
        if lowered in {"error", "errors", "is:error", "is:err"}:
            errors_only = True
            continue
        token_match = _TOKEN_PATTERN.match(lowered)
        if token_match:
            field, op, value = token_match.groups()
            token_predicates.append(
                TokenPredicate(
                    field=field,  # type: ignore[arg-type]
                    op=op,  # type: ignore[arg-type]
                    value=int(value),
                )
            )
            continue
        if ":" not in part:
            terms.append(lowered)
            continue
        field_name, value = part.split(":", 1)
        field_name = field_name.lower()
        value_lower = value.lower()
        if field_name == "role":
            roles.append(value_lower)
        elif field_name == "kind":
            kinds.append(value_lower)
        elif field_name == "tool":
            tools.append(value_lower)
        elif field_name == "status":
            statuses.append(value_lower)
        elif field_name == "cause":
            causes.append(value_lower)
        elif field_name == "turn":
            _append_int(turn_indices, value)
        elif field_name == "round":
            _append_int(round_indices, value)
        elif field_name == "id":
            id_fragments.append(value_lower)
        elif field_name in {"text", "q", "path"}:
            terms.append(value_lower)
        else:
            terms.append(lowered)

    return TraceQuery(
        raw=raw,
        terms=tuple(terms),
        roles=tuple(roles),
        kinds=tuple(kinds),
        tools=tuple(tools),
        statuses=tuple(statuses),
        causes=tuple(causes),
        turn_indices=tuple(turn_indices),
        round_indices=tuple(round_indices),
        id_fragments=tuple(id_fragments),
        errors_only=errors_only,
        token_predicates=tuple(token_predicates),
    )


def filter_trace_rows(
    rows: Iterable[TraceRow],
    query: TraceQuery,
) -> tuple[TraceRow, ...]:
    if query.is_empty:
        return tuple(rows)
    return tuple(row for row in rows if trace_row_matches(row, query))


def trace_row_matches(row: TraceRow, query: TraceQuery) -> bool:
    if query.roles and (row.role or "").lower() not in query.roles:
        return False
    if query.kinds and row.kind.lower() not in query.kinds:
        return False
    if query.tools and (row.tool_name or "").lower() not in query.tools:
        return False
    if query.statuses and (row.status or "").lower() not in query.statuses:
        return False
    if query.causes and (row.cause or "").lower() not in query.causes:
        return False
    if query.turn_indices and row.turn_index not in query.turn_indices:
        return False
    if query.round_indices and row.round_index not in query.round_indices:
        return False
    if query.errors_only and not row.is_error:
        return False
    haystack = row.searchable_text
    if query.id_fragments and not all(item in haystack for item in query.id_fragments):
        return False
    if query.terms and not all(term in haystack for term in query.terms):
        return False
    return all(
        _token_predicate_matches(row, predicate) for predicate in query.token_predicates
    )


def build_trace_snapshot(
    session_id: str,
    turns: Sequence[Turn],
    checkpoints: Sequence[TurnCheckpoint] = (),
) -> TraceSnapshot:
    records: list[Turn | TurnCheckpoint] = [*turns, *checkpoints]
    records.sort(key=lambda record: record.index)

    rows: list[TraceRow] = []
    summaries: list[TraceTurnSummary] = []
    shown_system_hashes: set[int] = set()

    for record in records:
        status: TraceRecordStatus = (
            "committed" if isinstance(record, Turn) else "incomplete"
        )
        cause = (
            type(record.outcome.cause).__name__ if isinstance(record, Turn) else None
        )
        tool_calls = sum(len(round_.tool_results) for round_ in record.rounds)
        tool_errors = sum(
            1
            for round_ in record.rounds
            for tool_record in round_.tool_results
            if tool_record.result.is_error
        )
        trigger_text = _trigger_text(record)
        summary = TraceTurnSummary(
            key=f"turn:{record.index}:{record.id}",
            turn_index=record.index,
            turn_id=record.id,
            status=status,
            rounds=len(record.rounds),
            tool_calls=tool_calls,
            tool_errors=tool_errors,
            input_tokens=record.meta.total_input_tokens,
            output_tokens=record.meta.total_output_tokens,
            cache_read_tokens=record.meta.cache_read_tokens,
            cache_write_tokens=record.meta.cache_write_tokens,
            model=record.meta.model_id,
            cause=cause,
            trigger=trigger_text,
        )
        summaries.append(summary)

        system_prompt = record.meta.system_prompt
        if system_prompt is not None:
            system_hash = hash(system_prompt)
            if system_hash not in shown_system_hashes:
                shown_system_hashes.add(system_hash)
                rows.append(
                    _row(
                        record,
                        status,
                        "system",
                        title="SYSTEM",
                        content=system_prompt,
                        cause=cause,
                    )
                )

        if trigger_text:
            rows.append(
                _row(
                    record,
                    status,
                    "user" if isinstance(record.trigger, UserInput) else "trigger",
                    title=record.trigger.source.upper(),
                    content=trigger_text,
                    role="user" if isinstance(record.trigger, UserInput) else "control",
                    cause=cause,
                )
            )

        for round_index, round_ in enumerate(record.rounds):
            assistant_text: list[str] = []
            thinking_text: list[str] = []
            for block in round_.response.content:
                if isinstance(block, TextContent):
                    assistant_text.append(block.text)
                elif isinstance(block, ThinkingBlock):
                    thinking_text.append(block.text)
                elif isinstance(block, OpaqueThinkingBlock):
                    thinking_text.append(f"[opaque reasoning: {block.provider}]")
                elif isinstance(block, ToolCallBlock):
                    if assistant_text:
                        rows.append(
                            _row(
                                record,
                                status,
                                "assistant",
                                title="ASSISTANT",
                                content="\n".join(assistant_text),
                                round_index=round_index,
                                role="assistant",
                                cause=cause,
                            )
                        )
                        assistant_text = []
                    if thinking_text:
                        rows.append(
                            _row(
                                record,
                                status,
                                "thinking",
                                title="THINKING",
                                content="\n".join(thinking_text),
                                round_index=round_index,
                                role="assistant",
                                cause=cause,
                            )
                        )
                        thinking_text = []
                    rows.append(
                        _row(
                            record,
                            status,
                            "tool_call",
                            title=f"CALL {block.name}",
                            content=json.dumps(
                                dict(block.arguments),
                                ensure_ascii=False,
                                indent=2,
                            ),
                            round_index=round_index,
                            role="assistant",
                            tool_name=block.name,
                            cause=cause,
                            metadata={"tool_call_id": block.id},
                        )
                    )

            if thinking_text:
                rows.append(
                    _row(
                        record,
                        status,
                        "thinking",
                        title="THINKING",
                        content="\n".join(thinking_text),
                        round_index=round_index,
                        role="assistant",
                        cause=cause,
                    )
                )
            if assistant_text:
                rows.append(
                    _row(
                        record,
                        status,
                        "assistant",
                        title="ASSISTANT",
                        content="\n".join(assistant_text),
                        round_index=round_index,
                        role="assistant",
                        cause=cause,
                    )
                )

            for tool_record in round_.tool_results:
                rows.append(
                    _row(
                        record,
                        status,
                        "tool_result",
                        title=f"RESULT {tool_record.call.name}",
                        content=_tool_result_text(tool_record.result.content),
                        round_index=round_index,
                        role="tool_result",
                        tool_name=tool_record.call.name,
                        is_error=tool_record.result.is_error,
                        cause=cause,
                        metadata={
                            "tool_call_id": tool_record.call.id,
                            "backgrounded": tool_record.backgrounded,
                        },
                    )
                )

        if isinstance(record, Turn):
            content = f"committed: {cause}"
            if isinstance(record.outcome.cause, ProviderRequestFailed):
                content += (
                    f"\n{record.outcome.cause.error_type}: "
                    f"{record.outcome.cause.detail}"
                )
            rows.append(
                _row(
                    record,
                    status,
                    "control",
                    title="TURN COMMITTED",
                    content=content,
                    role="control",
                    cause=cause,
                    is_error=isinstance(record.outcome.cause, ProviderRequestFailed),
                )
            )
        else:
            rows.append(
                _row(
                    record,
                    status,
                    "control",
                    title="TURN CHECKPOINT",
                    content="incomplete checkpoint",
                    role="control",
                    cause=None,
                )
            )

    return TraceSnapshot(
        session_id=session_id,
        turns=tuple(summaries),
        rows=tuple(_with_message_indices(rows)),
        metrics=_compute_metrics(rows, summaries),
    )


def default_trace_view_registry() -> TraceViewRegistry:
    return TraceViewRegistry(default_trace_view_specs())


def default_trace_view_specs() -> tuple[TraceViewSpec, ...]:
    return (
        TraceViewSpec(
            id="trajectory",
            title="Trajectory",
            description="All user, assistant, tool, and control rows.",
            shortcut="1",
            build=_build_trajectory_view,
        ),
        TraceViewSpec(
            id="tools",
            title="Tools",
            description="Tool calls and tool results.",
            shortcut="2",
            build=_build_tools_view,
        ),
        TraceViewSpec(
            id="errors",
            title="Errors",
            description="Tool errors and terminal failure rows.",
            shortcut="3",
            build=_build_errors_view,
        ),
        TraceViewSpec(
            id="metrics",
            title="Metrics",
            description="Token, tool, and status summary rows.",
            shortcut="4",
            build=_build_metrics_view,
        ),
        TraceViewSpec(
            id="policy",
            title="Policy",
            description="Extension-owned policy metrics and findings.",
            shortcut="5",
            build=_build_policy_placeholder_view,
        ),
    )


def _build_trajectory_view(snapshot: TraceSnapshot, query: TraceQuery) -> TraceView:
    rows = filter_trace_rows(snapshot.rows, query)
    return TraceView(
        id="trajectory",
        title="Trajectory",
        rows=rows,
        summary=_summary_text(snapshot, rows),
    )


def _build_tools_view(snapshot: TraceSnapshot, query: TraceQuery) -> TraceView:
    rows = filter_trace_rows(
        (row for row in snapshot.rows if row.kind in {"tool_call", "tool_result"}),
        query,
    )
    return TraceView(
        id="tools", title="Tools", rows=rows, summary=_summary_text(snapshot, rows)
    )


def _build_errors_view(snapshot: TraceSnapshot, query: TraceQuery) -> TraceView:
    rows = filter_trace_rows(
        (row for row in snapshot.rows if row.is_error or row.kind == "error"),
        query,
    )
    return TraceView(
        id="errors", title="Errors", rows=rows, summary=_summary_text(snapshot, rows)
    )


def _build_metrics_view(snapshot: TraceSnapshot, query: TraceQuery) -> TraceView:
    metrics = snapshot.metrics
    rows = [
        TraceRow(
            key="metric:turns",
            kind="metric",
            title="Turns",
            preview=(
                f"{metrics.committed_turns} committed, "
                f"{metrics.incomplete_turns} incomplete"
            ),
            content=(
                f"committed_turns: {metrics.committed_turns}\n"
                f"incomplete_turns: {metrics.incomplete_turns}\n"
                f"status: {snapshot.status_label}"
            ),
        ),
        TraceRow(
            key="metric:tokens",
            kind="metric",
            title="Tokens",
            preview=(
                f"in {metrics.input_tokens:,} / out {metrics.output_tokens:,} / "
                f"cache {metrics.cache_read_tokens:,}"
            ),
            content=(
                f"input_tokens: {metrics.input_tokens:,}\n"
                f"output_tokens: {metrics.output_tokens:,}\n"
                f"cache_read_tokens: {metrics.cache_read_tokens:,}\n"
                f"cache_write_tokens: {metrics.cache_write_tokens:,}"
            ),
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            cache_read_tokens=metrics.cache_read_tokens,
            cache_write_tokens=metrics.cache_write_tokens,
        ),
        TraceRow(
            key="metric:tools",
            kind="metric",
            title="Tools",
            preview=f"{metrics.tool_calls} calls, {metrics.tool_errors} errors",
            content=_counter_lines("tool", metrics.by_tool)
            or "No tool calls recorded.",
            is_error=metrics.tool_errors > 0,
        ),
        TraceRow(
            key="metric:kinds",
            kind="metric",
            title="Rows",
            preview=f"{metrics.rows} rows",
            content=_counter_lines("kind", metrics.by_kind),
        ),
    ]
    filtered = filter_trace_rows(rows, query)
    return TraceView(
        id="metrics",
        title="Metrics",
        rows=filtered,
        summary=_summary_text(snapshot, filtered),
    )


def _build_policy_placeholder_view(
    snapshot: TraceSnapshot,
    query: TraceQuery,
) -> TraceView:
    rows = filter_trace_rows(
        [
            TraceRow(
                key="policy:placeholder",
                kind="policy",
                title="Policy Metrics",
                preview="No policy trace view provider is registered yet.",
                content=(
                    "Policy metrics can plug in by registering TraceViewSpec "
                    "objects against the trace view registry."
                ),
            )
        ],
        query,
    )
    return TraceView(
        id="policy",
        title="Policy",
        rows=rows,
        summary=_summary_text(snapshot, rows),
        empty_text="No policy metrics match the current query.",
    )


def _row(
    record: Turn | TurnCheckpoint,
    status: TraceRecordStatus,
    kind: TraceRowKind,
    *,
    title: str,
    content: str,
    round_index: int | None = None,
    role: str | None = None,
    tool_name: str | None = None,
    is_error: bool = False,
    cause: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> TraceRow:
    preview = _preview(content)
    return TraceRow(
        key=f"{record.index}:{round_index if round_index is not None else '-'}:{kind}:{len(content)}:{title}",
        kind=kind,
        title=title,
        preview=preview,
        content=content,
        turn_index=record.index,
        round_index=round_index,
        status=status,
        role=role,
        tool_name=tool_name,
        is_error=is_error,
        cause=cause,
        input_tokens=record.meta.total_input_tokens,
        output_tokens=record.meta.total_output_tokens,
        cache_read_tokens=record.meta.cache_read_tokens,
        cache_write_tokens=record.meta.cache_write_tokens,
        metadata=metadata or {},
    )


def _with_message_indices(rows: Sequence[TraceRow]) -> tuple[TraceRow, ...]:
    result: list[TraceRow] = []
    for index, row in enumerate(rows):
        result.append(
            TraceRow(
                key=f"{index}:{row.key}",
                kind=row.kind,
                title=row.title,
                preview=row.preview,
                content=row.content,
                turn_index=row.turn_index,
                round_index=row.round_index,
                message_index=index,
                status=row.status,
                role=row.role,
                tool_name=row.tool_name,
                display_name=row.display_name,
                is_error=row.is_error,
                cause=row.cause,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                cache_read_tokens=row.cache_read_tokens,
                cache_write_tokens=row.cache_write_tokens,
                metadata=row.metadata,
            )
        )
    return tuple(result)


def _compute_metrics(
    rows: Sequence[TraceRow],
    summaries: Sequence[TraceTurnSummary],
) -> TraceMetrics:
    by_tool: Counter[str] = Counter()
    by_kind: Counter[str] = Counter(row.kind for row in rows)
    for row in rows:
        if row.kind == "tool_result" and row.tool_name:
            by_tool[row.tool_name] += 1
    return TraceMetrics(
        committed_turns=sum(
            1 for summary in summaries if summary.status == "committed"
        ),
        incomplete_turns=sum(
            1 for summary in summaries if summary.status == "incomplete"
        ),
        rows=len(rows),
        tool_calls=sum(summary.tool_calls for summary in summaries),
        tool_errors=sum(summary.tool_errors for summary in summaries),
        input_tokens=sum(summary.input_tokens for summary in summaries),
        output_tokens=sum(summary.output_tokens for summary in summaries),
        cache_read_tokens=sum(summary.cache_read_tokens for summary in summaries),
        cache_write_tokens=sum(summary.cache_write_tokens for summary in summaries),
        by_tool=dict(by_tool),
        by_kind=dict(by_kind),
    )


def _trigger_text(record: Turn | TurnCheckpoint) -> str:
    if isinstance(record.trigger, UserInput):
        parts: list[str] = []
        for block in record.trigger.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
            elif isinstance(block, ImageContent):
                parts.append(f"[image {block.mime_type}, {len(block.data)} bytes]")
            else:
                parts.append(f"[{type(block).__name__}]")
        return "\n".join(parts)
    return record.trigger.source


def _tool_result_text(content: Sequence[TextContent | ImageContent]) -> str:
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ImageContent):
            parts.append(f"[image {block.mime_type}, {len(block.data)} bytes]")
        else:
            parts.append(f"[{type(block).__name__}]")
    return "\n".join(parts)


def _preview(content: str, *, limit: int = 160) -> str:
    text = " ".join(line.strip() for line in content.splitlines() if line.strip())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _append_int(values: list[int], raw: str) -> None:
    try:
        values.append(int(raw))
    except ValueError:
        return


def _token_predicate_matches(row: TraceRow, predicate: TokenPredicate) -> bool:
    value = {
        "tokens": row.input_tokens + row.output_tokens,
        "in": row.input_tokens,
        "out": row.output_tokens,
        "cache": row.cache_read_tokens,
    }[predicate.field]
    if predicate.op == ">":
        return value > predicate.value
    if predicate.op == ">=":
        return value >= predicate.value
    if predicate.op == "<":
        return value < predicate.value
    if predicate.op == "<=":
        return value <= predicate.value
    return value == predicate.value


def _counter_lines(label: str, values: Mapping[str, int]) -> str:
    return "\n".join(f"{label}:{key} {value}" for key, value in sorted(values.items()))


def _summary_text(snapshot: TraceSnapshot, rows: Sequence[TraceRow]) -> str:
    return (
        f"{len(rows)} row(s) | {snapshot.metrics.committed_turns} committed | "
        f"{snapshot.metrics.incomplete_turns} incomplete | "
        f"{snapshot.metrics.tool_errors} tool error(s)"
    )


__all__ = [
    "TraceMetrics",
    "TraceQuery",
    "TraceRecordStatus",
    "TraceRow",
    "TraceRowKind",
    "TraceSnapshot",
    "TraceTurnSummary",
    "TraceView",
    "TraceViewProvider",
    "TraceViewRegistry",
    "TraceViewSpec",
    "build_trace_snapshot",
    "default_trace_view_registry",
    "default_trace_view_specs",
    "filter_trace_rows",
    "parse_trace_query",
    "trace_row_matches",
]
