# Single Event Log

## Status

Landed (2026-05-24). Phased across four commits on the OTLP/JSON cutover
branch:

- **PR-A** (`be78860`) — file exporters in `core.runtime.otel_export` +
  `setup_session_telemetry`. Isolated; not wired into the runtime yet.
- **Commit 1** (`9bae7dc`) — bus-owned `dispatch_id` on `EventBus` with
  `current_dispatch_id()` accessor. Superseded by PR-E (`531378b`):
  the id moved onto the `Event` ABI as a real field, the bus stash and
  accessor were removed.
- **Commit 2a** (`ad6c5cb`) — `ExtensionAPI.get_session_telemetry()` service
  facade so atoms can reach the per-session OTel `Tracer` + `Logger`
  without violating §11. Lazy construction, substrate-owned teardown.
- **Commit 2b** (`1f9caa3`) — atomic OTLP/JSON cutover: rewritten
  `observability.py` writer, every reader (session_manager, indexer,
  tool_query_traces, tool_eval_run, tool_guard_watch, llmharness CLIs, rca
  graders) cut over in lock-step, all test fixtures rewritten. New
  `tests/unit/extensions/test_observability_semconv.py` locks down the
  on-disk semconv contract.
- **Commit 3** (this commit) — design doc + index.yaml sync.
- **PR-H** (2026-05-24) — process-level `TracerProvider` / `LoggerProvider`;
  `agentm.session.id` moves from OTel resource attribute to per-span /
  per-log attribute. See "Process-level providers" below.
- **PR-F** (2026-05-24) — declarative `Event.to_otel` mapping; the
  observability atom collapses to a thin dispatcher. See "Followups
  identified during landing" → PR-F for the full record, and the
  "Event → OTel mapping" section for the new pattern.

## Motivation

Before the merge, a single session produced **two parallel JSONL streams**:

| Path | Content | Write path |
|---|---|---|
| `~/.agentm/sessions/--<cwd>--/<ts>_<id>.jsonl` | trajectory — one entry per `AgentMessage` | `SessionManager._append_record`: synchronous `open("a") + write + close` per message |
| `<cwd>/.agentm/observability/<trace>.jsonl` | OTel-shaped events — spans, dispatches, handler invocations, llm.request.start/end | observability builtin: bounded async queue, drops on full, `flush()` per record |

The two streams were largely **overlapping**: every trajectory entry was
mirrored into observability via `emit:tool_call`, `emit:tool_result`,
`emit:before_send_to_llm` etc. The split was denormalization, not logical
separation. It carried four costs:

1. **Redundant message bytes** — `before_send_to_llm` snapshotted the whole
   `messages` list every turn. N-turn session = O(N²) message bytes. The
   shim hid this by replacing message bodies with `chars + sha256_prefix` in
   the observability event, but that broke replay-from-observability.
2. **Two durability policies** — trajectory writes were synchronous-on-call
   (slow but lossless); observability writes were async-bounded (fast but
   drops under backpressure). Two write paths to reason about.
3. **Two retention policies** — ops glob had to know both directories.
4. **Cross-correlation cost** — debugging required aligning two files by
   timestamp.

## Decision

Single per-session event log, OTLP/JSON ndjson on disk.

- **One file per session**, at `<cwd>/.agentm/observability/<session_id>.jsonl`.
- **`SessionManager._append_record` no longer writes its own file.** Instead
  it dispatches a `MessageAppendedEvent` through the EventBus; the
  observability atom subscribes and writes an `agentm.message.appended` log
  record via the per-session OTel `Logger`.
- **`SessionManager._load` reads the same single file**, filters
  `scopeLogs[*].logRecords[*]` for `eventName == "agentm.session.header"`
  and `"agentm.message.appended"`, reconstructs the in-memory trajectory.
- **All event writes go through the SDK `BatchProcessor`** — bounded async
  queue with batched flush handled by the standard `opentelemetry-sdk`
  primitives (`BatchSpanProcessor`, `BatchLogRecordProcessor`), wrapped at
  the substrate boundary by `BlockingBatch*` subclasses from PR-A.
- **Backpressure = block, not drop.** The SDK's stock processors drop on
  queue overflow; PR-A's `BlockingBatch*` subclasses spin-wait on queue
  length so producers block when the queue is full. Default queue size is
  100k; overflow at this size is a real problem to surface.
- **Drain on shutdown.** The substrate's `_SessionTelemetryHolder`
  installs a `SessionShutdownEvent` handler at `BusPriority.POST` that
  calls `SessionTelemetry.shutdown()` — drains both batch processors
  before closing the exporters. Observability handlers at `NORMAL`
  priority get to emit `agentm.session.end` before the drain runs.
- **Drop the redaction layer for `before_send_to_llm`.** That channel no
  longer carries a `messages` payload at all on the dispatch record (the
  field is stripped); consumers reconstruct from prior
  `agentm.message.appended` records. The `redact_messages` helper still
  serves `ApiSendUserMessage` and `LlmRequestStart/End` payloads when
  `redact_prompts=True`.

## Trade-offs considered

- **Sync vs async writes for message events**: rejected sync — agent loop
  blocked on disk on every message. Async with the SDK BatchProcessor +
  blocking backpressure is the right tier for both event classes.
- **Two queues with one file (separate lanes for message vs other events)**:
  rejected as unnecessary complexity. One BatchProcessor serves both
  spans and logs fine if sized for peak.
- **Keep two files but unify queue**: rejected — the goal is one file so
  consumers stop having to join.
- **Body as gen_ai parts schema vs SessionEntry dict verbatim**: chose
  SessionEntry dict verbatim (Q4 during PR-B planning). The gen_ai parts
  schema loses information vs. our richer block types (`ThinkingBlock`,
  `ImageContent` with mime types, etc.); future gen_ai-shaped consumers
  can read a separate span attribute on `chat` if we ever need one.

## Wire format

The on-disk event log is **OTLP/JSON ndjson**: every line is a self-contained
JSON object that would be a valid element inside an
`ExportTraceServiceRequest.resource_spans[]` or
`ExportLogsServiceRequest.resource_logs[]` array. Span lines and log lines
share one file and interleave in arrival order; the OTLP top-level shape
(`scopeSpans` vs `scopeLogs`) disambiguates them on read. Collector pipelines
(e.g. `filelog` receiver + `otlpjson` decoder) consume the file directly.

Each line carries a `resource` block duplicating the **process-scoped**
attributes:

| Attribute | Source |
|---|---|
| `service.name` | constant `"agentm"` |
| `service.version` | `importlib.metadata.version("agentm")` |

The resource carries only what is genuinely process-global: the OTel
ecosystem assumes one `service.name` per process, and AgentM honours that
(PR-H). Session-scoped metadata lives on **per-span / per-log attributes**,
not on the resource:

| Attribute | Lives on | Stamped by |
|---|---|---|
| `agentm.session.id` | every span and every log record | observability atom (and `SessionManager` for the direct-write path) |
| `agentm.session.scenario` | spans/records where it carries information | observability atom |
| `agentm.session.purpose`, `agentm.session.root_id`, ... | spans/records where it carries information | observability atom |

Lines do **not** include the outer `{"resourceSpans": [...]}` /
`{"resourceLogs": [...]}` envelope — that wrap is the exporter request, not
the on-disk shape.

### Process-level providers

A single Python process can host many AgentSessions (gateway daemon, RCA
runner, eval driver). They all share **one** `TracerProvider` and one
`LoggerProvider`. Each session attaches its own
`BlockingBatchSpanProcessor` + `BlockingBatchLogRecordProcessor` to those
shared providers; each processor forwards exclusively to its own per-session
file exporter and **filters by `agentm.session.id`** (records lacking the
matching attribute are dropped at the processor before they enter the batch
queue). This keeps three things consistent simultaneously:

- OTel ecosystem compatibility — one `service.name` per process is what
  collectors, samplers, and the GenAI semconv assume.
- File partitioning — a session's file holds only that session's records,
  so consumers walking one file never see cross-session noise.
- Lazy startup — `setup_process_telemetry()` is invoked exactly once per
  process (on the first `setup_session_telemetry` call); subsequent
  sessions are cheap (one processor pair + one file handle each).

Session shutdown is a **partial** teardown: the per-session processors are
flushed and removed from the global providers (rebuilding the multi-processor
tuple under its internal lock), the file handles close, but the providers
themselves stay alive for the next session. Process shutdown drains the
providers once via an `atexit` hook registered inside
`setup_process_telemetry`.

**Shape source.** Canonical OTLP/JSON is produced by feeding `ReadableSpan` /
`ReadableLogRecord` through the standard `opentelemetry-exporter-otlp`
proto encoders (`encode_spans`, `encode_logs`) and converting the resulting
protobuf message via `google.protobuf.json_format.MessageToDict`. The
exporter then splits `resource_spans[]` / `resource_logs[]` into one line
each. Field names match the OTLP/JSON spec exactly (`traceId`, `spanId`,
`startTimeUnixNano`, ...); `traceId` and `spanId` are base64-encoded per the
proto-JSON default.

Sample (one span line, one log line):

```ndjson
{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"agentm"}},{"key":"service.version","value":{"stringValue":"0.1.0"}},{"key":"agentm.session.id","value":{"stringValue":"sess-abc"}},{"key":"agentm.scenario.name","value":{"stringValue":"local"}}]},"scopeSpans":[{"scope":{"name":"agentm","version":"0.1.0"},"spans":[{"traceId":"jPTe/iWSTaCL7oH6aPJ76g==","spanId":"jtiuCvsDU1M=","name":"chat m-stub","kind":"SPAN_KIND_CLIENT","startTimeUnixNano":"1779589308307455359","endTimeUnixNano":"1779589308307470397","attributes":[{"key":"gen_ai.operation.name","value":{"stringValue":"chat"}},{"key":"gen_ai.request.model","value":{"stringValue":"m-stub"}}],"status":{},"flags":256}]}]}
{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"agentm"}},{"key":"service.version","value":{"stringValue":"0.1.0"}},{"key":"agentm.session.id","value":{"stringValue":"sess-abc"}}]},"scopeLogs":[{"scope":{"name":"agentm","version":"0.1.0"},"logRecords":[{"timeUnixNano":"1779589316951916032","severityNumber":"SEVERITY_NUMBER_INFO","severityText":"INFO","eventName":"agentm.message.appended","body":{"kvlistValue":{"values":[{"key":"type","value":{"stringValue":"message"}},{"key":"id","value":{"stringValue":"abc"}}]}},"attributes":[{"key":"agentm.session.id","value":{"stringValue":"sess-abc"}},{"key":"agentm.message.id","value":{"stringValue":"abc"}}],"observedTimeUnixNano":"1779589316951963264"}]}]}
```

## Event → OTel mapping

Each concrete `Event` subclass declares its own translation via a
`to_otel(telemetry: SessionTelemetry) -> None` method (PR-F). The
observability atom is a thin dispatcher: it subscribes one uniform
handler per typed channel that calls `event.to_otel(telemetry)`, and
new `Event` classes reach the wire by adding their channel to
`_TO_OTEL_CHANNELS` — no central switch to edit. The base method on
`Event` is a documented no-op for events that have no OTel meaning
(kernel-internal lifecycle, mutation channels covered by the bus
observer, `stream_delta` excluded as bloat).

**Lifecycle pairing.** Start/End event pairs (`BeforeAgentStart` /
`AgentEnd`, `LlmRequestStart` / `LlmRequestEnd`, `ToolCall` /
`ToolResult`) share a single OTel span through the
`SessionTelemetry.span_tracker` dict, keyed by `(span_kind, correlator)`:
the Start event's `to_otel` calls `tracer.start_span(...)` and stashes
the span via `telemetry.open_span(kind, key, span)`; the End event
recovers it with `telemetry.pop_span(kind, key)`, stamps terminal
attributes, and `.end()` s it. `SessionShutdownEvent.to_otel` calls
`telemetry.close_open_spans(...)` so abnormal exits (crash, signal)
never leak open spans.

The wire-format table below is reference for consumers; the source of
truth is the per-class `to_otel` body. The mapping is intentionally
sparse — only the events whose record is structurally meaningful land
on disk; high-frequency channels like `stream_delta` are excluded from
`event.dispatch` recording.

**Spans** (paired start/end, written through
`api.get_session_telemetry().tracer`):

| Span name | Kind | Paired events | Key attributes |
|---|---|---|---|
| `invoke_agent <scenario>` | INTERNAL | `BeforeAgentStartEvent` → `AgentEndEvent` | `gen_ai.operation.name=invoke_agent`, `gen_ai.agent.name`, `gen_ai.conversation.id`, cause kind on end |
| `chat <model>` | CLIENT | `LlmRequestStartEvent` → `LlmRequestEndEvent` | `gen_ai.operation.name=chat`, `gen_ai.request.model`, `gen_ai.provider.name`, `agentm.turn.index`, `agentm.turn.id`, chunk/duration on end |
| `execute_tool <tool>` | INTERNAL | `ToolCallEvent` → `ToolResultEvent` (by `tool_call_id`) | `gen_ai.operation.name=execute_tool`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.call.arguments` (JSON-encoded), `gen_ai.tool.call.result` on close |

**Log records** (severity INFO unless noted; written through
`api.get_session_telemetry().logger`):

| Event name | Source | Body | Key attributes |
|---|---|---|---|
| `agentm.session.header` | `SessionHeaderEmittedEvent` | SessionHeader dict verbatim | `agentm.session.id`, `agentm.session.header.id` |
| `agentm.message.appended` | `MessageAppendedEvent` | SessionEntry dict verbatim (`{type, id, parent_id, timestamp, payload}`) | `agentm.session.id`, `agentm.message.id`, `agentm.message.parent_id`, `agentm.message.type` |
| `agentm.session.start` | install-time | session identity dict | `agentm.session.id`, `agentm.session.root_id`, `agentm.session.purpose`, `agentm.session.scenario` |
| `agentm.session.ready` | `SessionReadyEvent` | tool / command / extension lists, model | `agentm.session.id`, `agentm.session.tool_count`, `agentm.session.extension_count` |
| `agentm.session.fingerprint` | `SessionReadyEvent` (after compute) | atom hash map + task_meta | `agentm.session.id`, `agentm.task.class`, `agentm.task.eval_run_id`, `agentm.task.eval_task_id` |
| `agentm.session.end` | `SessionShutdownEvent` | session identity + duration | `agentm.session.id`, `agentm.session.duration_ns` |
| `agentm.turn.summary` | `TurnEndEvent` | tool_calls, tool_call_count, tool_error_count, stop_reason, usage (input/output tokens) | `agentm.turn.*`, `gen_ai.usage.*` |
| `agentm.agent.end` | `AgentEndEvent` | cause dict (via `to_jsonable`) | `agentm.agent.cause_kind`, `agentm.agent.cause_final`, `agentm.agent.message_count` |
| `agentm.extension.install` | `ExtensionInstallEvent` | config | `agentm.extension.module_path`, `agentm.extension.phase`, `agentm.extension.trigger`, `agentm.extension.duration_ns` |
| `agentm.extension.reload` (atom) | `ExtensionReloadEvent` | name + hashes + fingerprint_after | `agentm.atom.name`, `agentm.atom.new_hash`, `agentm.atom.old_hash`, `agentm.atom.tier` |
| `agentm.extension.unload` | `ExtensionUnloadEvent` | trigger + tier | `agentm.extension.name`, `agentm.extension.module_path`, `agentm.extension.trigger` |
| `agentm.atom.reload` | `ExtensionReloadEvent` (paired with reload above) | fingerprint_after | `agentm.atom.name`, `agentm.atom.new_hash`, ... |
| `agentm.api.register` | `ApiRegisterEvent` | payload | `agentm.api.kind`, `agentm.api.name`, `agentm.api.extension` |
| `agentm.api.send_user_message` | `ApiSendUserMessageEvent` | content (redacted when `redact_prompts=True`) | `agentm.api.extension`, `agentm.api.content_chars` |
| `agentm.diagnostic` | `DiagnosticEvent` | message + level | `agentm.diagnostic.level`, `agentm.diagnostic.source` |
| `agentm.event.dispatch` | bus observer `on_emit_end` | — | `agentm.event.channel`, `agentm.event.dispatch_id`, `agentm.handler.count`, `agentm.event.payload` (JSON-encoded) |
| `agentm.handler.invoke` | bus observer `on_handler_done` | — | `agentm.event.channel`, `agentm.event.dispatch_id`, `agentm.handler.name`, `agentm.handler.duration_ns`, `agentm.handler.raised`, optional `agentm.handler.error.*` |
| `agentm.handler.mutated` | bus observer (mutable channels only) | — | `agentm.event.channel`, `agentm.event.dispatch_id`, `agentm.handler.name`, `agentm.handler.mutations` (JSON list) |

The `gen_ai.*` namespace tracks the OpenTelemetry GenAI semconv where
mapping is unambiguous (operation name, request model, tool name, usage
tokens). Anything AgentM-specific lives under `agentm.*`.

## Bus correlation

The `dispatch_id` lives on the `Event` ABI as a real field; the
`EventBus` reassigns it on every emit. This closes the bus-stash workaround
that the original landing carried (PR-E, see "Followups identified during
landing" below).

**Mechanism.** `Event` is `@dataclass(slots=True)` with a single field:

```python
dispatch_id: str = field(
    default_factory=lambda: uuid.uuid4().hex, kw_only=True
)
```

`kw_only=True` keeps the field out of subclass positional argument order
so the ~30 concrete `Event` subclasses inherit cleanly without
re-ordering. The `default_factory` exists so events that are constructed
but never emitted — test fixtures, standalone payloads — still carry a
sane id. On every `emit` / `emit_sync` entry, the bus reassigns
`event.dispatch_id = uuid.uuid4().hex` when the payload is an `Event`
instance, so re-emitting the same instance produces a fresh id (the
field default is overwritten). Non-`Event` payloads (raw dicts on
extension-invented channels) flow through untouched; observability
reads via `getattr(event, "dispatch_id", "")` and stamps an empty
string in that case.

**Why not a bus-owned stack.** The previous design carried
`_dispatch_stack: list[str]` on `EventBus` plus a `current_dispatch_id()`
accessor so observers could read the in-flight id. Two reasons that
mechanism was a workaround, not the destination:

1. The id is conceptually per-emission — it travels with the event
   through every handler. Stashing it on the bus forced observers to
   look it up out-of-band; the event was already in scope.
2. Re-entrant emits needed a stack to distinguish nested ids from outer
   ids. With the id on the event, each event carries its own; nested
   emits of *different* events are naturally isolated, and the bus has
   no per-dispatch state to leak across emits.

**Why `Event` is no longer `frozen`.** Python's `dataclass` forbids
mixing frozen and non-frozen across an inheritance chain. The bus needs
to overwrite `dispatch_id` on every emit, so the base must be
non-frozen; every concrete `Event` subclass therefore drops
`frozen=True` as well. Nothing in the codebase relied on Event
instances being hashable or immutable — `TerminationCause` and
`LoopAction` payloads (which are still frozen) are passed *inside*
events, not as events themselves.

**Observer hook signature.** `EventBusObserver.on_handler_done` now
takes `event` as a parameter so observers can read `event.dispatch_id`
without an out-of-band accessor. `on_emit_end` already received `event`.

**How consumers join.** The observability atom reads
`event.dispatch_id` from inside its `EventBusObserver` hooks
(`on_emit_end` for the `agentm.event.dispatch` record,
`on_handler_done` for each `agentm.handler.invoke`,
`_record_mutation` for `agentm.handler.mutated`). All three records
stamp the same value into their `agentm.event.dispatch_id` attribute,
so a downstream query

```
WHERE eventName = "agentm.handler.invoke"
  AND attributes["agentm.event.dispatch_id"] = $id
```

recovers every handler that fanned out from one dispatch, and the matching
`agentm.event.dispatch` record names which channel + how many handlers.

**Tests.** `tests/unit/core/test_event_dispatch_id.py` locks down the
properties consumers depend on: stable within one dispatch, fresh
across dispatches (including same-instance re-emit), construction
default for never-emitted events, nested-emit isolation, exception
safety, and non-Event payload pass-through. The observer-side join is
verified end-to-end by
`test_observability_semconv.py::test_dispatch_id_links_dispatch_and_handler_records`.

## What this is NOT

- Not a schema change to the underlying `Event` dataclasses — kernel
  events stay as they are. `MessageAppendedEvent` was already in place
  before this work; PR-B/C just wired it through OTel.
- Not a change to retention or where files live in the filesystem
  hierarchy. The path is still `<cwd>/.agentm/observability/<session_id>.jsonl`.
- Not a stable public schema for third parties. The on-disk shape **is**
  OTLP/JSON ndjson, which is a stable open format — but the
  `agentm.*` event names and attribute namespace are project-internal
  and may evolve. Downstream consumers should pin AgentM versions or
  read through the (forthcoming) TraceReader API; see followups.

## Validation

Locked down by the test suite:

- `tests/unit/core/test_otel_export.py` — PR-A wire format + blocking
  backpressure + idempotent shutdown.
- `tests/unit/core/test_event_dispatch_id.py` — Commit 1 dispatch_id
  properties.
- `tests/unit/extensions/test_observability_semconv.py` — gen_ai semconv
  attribute presence on `chat` / `execute_tool` / `invoke_agent` spans,
  log record shape for header / message.appended / turn.summary /
  fingerprint, dispatch_id join.
- `tests/unit/core/test_event_to_otel.py` — PR-F declarative-mapping
  contract: every concrete `Event` subclass either overrides
  `to_otel` or is in the intentional no-op set; lifecycle pairing via
  the `span_tracker` round-trips cleanly for the three real span
  families.
- `tests/unit/core/test_single_event_log.py` — `SessionManager._load`
  walks interleaved OTLP/JSON lines and rebuilds the trajectory
  identically to a clean in-memory build.
- `tests/unit/core/catalog/test_indexer.py` — indexer idempotence still
  holds after the cutover; rebuild from raw OTLP traces reproduces the
  same metrics.jsonl rows.
- `tests/integration/test_observability_cli.py` /
  `test_cli_slash_commands.py` / `test_cc_extension_cli.py` /
  `test_guard_watch.py` — end-to-end via stub providers; assert on the
  new event-name + span-name vocabulary.

Stub-provider smoke (per spec): the integration tests above spin up an
HTTP stub provider in-process, drive `agentm` via subprocess, and inspect
the resulting jsonl. Every run produces at least one `chat <model>` span
with `gen_ai.operation.name=chat`, one `agentm.session.ready` log record,
one `agentm.turn.summary` per turn, and one `agentm.session.end` at
shutdown — all with the expected gen_ai semconv attributes.

## TraceReader API

**Status:** Landed (PR-G, see "Followups identified during landing" below).
Previously every reader (`tool_query_traces`, `tool_eval_run`,
`tool_guard_watch`, the llmharness CLIs, both rca graders, plus the core
substrate's `SessionManager` and the catalog indexer) carried its own
copy of the proto-JSON tagged-union unwrapper and its own `scopeSpans` /
`scopeLogs` walk. PR-G extracted that into
`src/agentm/core/runtime/trace_reader.py` and re-exported the read-only
surface (`TraceReader`, `Span`, `LogRecord`, `attr`) through
`agentm.core.abi`, so atoms stay §11-clean while sharing the parser.

The surface is intentionally small and lazy — every iterator is a
generator over the file, so callers never materialise a whole trace into
memory unless they explicitly ask via a `load_*` accessor.

Public surface:

```python
from agentm.core.abi import TraceReader, Span, LogRecord, attr

reader = TraceReader(path)

# Generic walks (filter by name + attribute equality, lazy):
reader.iter_spans(name=None, attribute_filters=None)
reader.iter_log_records(name=None, attribute_filters=None, parent_span_id=None)
reader.iter_all()  # interleaves spans + logs in file order

# Convenience accessors (eager, list-returning where order matters):
reader.load_session_header()       # -> dict | None
reader.load_session_fingerprint()  # -> dict | None
reader.load_messages()             # -> list[dict]
reader.load_turn_summaries()       # -> list[dict]
reader.chat_calls()                # -> Iterator[Span]
reader.tool_calls()                # -> Iterator[(Span, LogRecord?, LogRecord?)]

attr(obj, key, default=None)       # terse attribute lookup
```

`Span` and `LogRecord` are frozen dataclasses with attributes already
unwrapped from OTLP tagged-union form: a `kvlistValue` becomes a `dict`,
an `intValue` becomes an `int`, etc. The raw element is preserved in
`.raw` for callers that need to peek at fields the dataclass does not
surface.

The reader skips malformed lines silently — the writer is append-only,
so a partial last line during a crash should not crash readers.

## Followups identified during landing

Recorded here so they don't get lost; PR briefs to follow when dispatched.

- **PR-E** — landed `531378b`. `Event` is now `@dataclass(slots=True)`
  with a `dispatch_id` field defaulted via `field(default_factory=...,
  kw_only=True)`; the `EventBus` reassigns the field on every emit. The
  `_dispatch_stack` + `current_dispatch_id()` workaround is gone. See
  "Bus correlation" above for the new mechanism.
- **PR-F** — Landed (2026-05-24). Declarative `Event → OTel` mapping
  via a per-class `to_otel(telemetry)` method on every concrete `Event`
  subclass. The `observability.py` atom is reduced to a thin dispatcher:
  it stamps session-scoped metadata onto :class:`SessionTelemetry`,
  subscribes one uniform handler per typed channel that delegates to
  `event.to_otel(telemetry)`, and still owns the bus-observer side
  (`agentm.event.dispatch` / `agentm.handler.invoke` /
  `agentm.handler.mutated`) plus the atom-hash bookkeeping needed for
  `agentm.session.fingerprint` and `agentm.atom.reload`. Lifecycle
  pairing for Start/End span families uses a new
  :meth:`SessionTelemetry.open_span` / :meth:`pop_span` /
  :meth:`close_open_spans` helper keyed by `(span_kind, correlator)`;
  the `_Observer` class no longer reaches for the chat/tool/invoke
  span dicts that the old translator held in install-scope closures.
  Coverage is enforced by
  `tests/unit/core/test_event_to_otel.py::test_every_event_class_has_to_otel`:
  every concrete `Event` subclass either overrides `to_otel` or appears
  in the explicit intentional no-op set (kernel-internal lifecycle
  events whose semantics are covered by the bus-observer records).
  Adding a new `Event` without considering wire format trips the test.
  On-disk wire format is unchanged — semconv tests in
  `tests/unit/extensions/test_observability_semconv.py` continue to
  lock down the byte-shape contract.
- **PR-G** — Landed. `TraceReader` API extracted into
  `core.runtime.trace_reader` and re-exported via `agentm.core.abi`;
  every reader (`SessionManager`, the catalog indexer, the three
  `tool_*` atoms, the llmharness CLIs, both rca graders, the rca_hfsm
  Phase 2 runner) now depends on it. See the "TraceReader API" section
  above for the surface; `tests/unit/core/test_trace_reader.py` locks
  it down.
- **PR-H** — Landed. One process-level `TracerProvider` +
  `LoggerProvider`; resource carries only `service.name=agentm` +
  `service.version`. Per-session isolation via per-session
  `BlockingBatchSpanProcessor` / `BlockingBatchLogRecordProcessor`
  attached to the shared providers, each filtering on
  `agentm.session.id` so cross-session traffic does not bleed into the
  wrong file. `agentm.session.id` is stamped as a span/log attribute by
  the observability atom and the `SessionManager` direct-write path.
  Session shutdown drains and removes the per-session processors; the
  providers themselves are torn down once at process exit via an
  `atexit` hook. See the "Process-level providers" subsection above
  for the multi-session model; locked down by
  `tests/unit/core/test_otel_export.py::test_two_sessions_share_one_tracer_provider`
  and `test_session_shutdown_does_not_affect_other_session`.

## Related concepts

- `observability.md` — the consumer side; this design changes its write
  path and on-disk shape, not the underlying event semantics.
- `pluggable_architecture.md` — `SessionManager` lives in `core.runtime/`;
  this change touches that substrate.
- `agent_loop.md` — `message.appended` event is emitted from the kernel's
  trajectory-append path; the writer is now an OTel `Logger` invocation
  rather than a hand-rolled JSONL write.
