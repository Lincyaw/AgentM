# Single Event Log

## Status

Landed (2026-05-24). Phased across four commits on the OTLP/JSON cutover
branch:

- **PR-A** (`be78860`) — file exporters in `core.runtime.otel_export` +
  `setup_session_telemetry`. Isolated; not wired into the runtime yet.
- **Commit 1** (`9bae7dc`) — bus-owned `dispatch_id` on `EventBus` with
  `current_dispatch_id()` accessor. No consumer reads it yet.
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

Each line carries a `resource` block duplicating the session-scoped
attributes:

| Attribute | Source |
|---|---|
| `service.name` | constant `"agentm"` |
| `service.version` | `importlib.metadata.version("agentm")` |
| `agentm.session.id` | the session id |
| `agentm.scenario.name` | scenario name, when known |

The ~150 bytes/line of resource duplication is accepted; sessions are a few
MB. Lines do **not** include the outer `{"resourceSpans": [...]}` /
`{"resourceLogs": [...]}` envelope — that wrap is the exporter request, not
the on-disk shape.

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
{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"agentm"}},{"key":"service.version","value":{"stringValue":"0.1.0"}},{"key":"agentm.session.id","value":{"stringValue":"sess-abc"}},{"key":"agentm.scenario.name","value":{"stringValue":"general_purpose"}}]},"scopeSpans":[{"scope":{"name":"agentm","version":"0.1.0"},"spans":[{"traceId":"jPTe/iWSTaCL7oH6aPJ76g==","spanId":"jtiuCvsDU1M=","name":"chat m-stub","kind":"SPAN_KIND_CLIENT","startTimeUnixNano":"1779589308307455359","endTimeUnixNano":"1779589308307470397","attributes":[{"key":"gen_ai.operation.name","value":{"stringValue":"chat"}},{"key":"gen_ai.request.model","value":{"stringValue":"m-stub"}}],"status":{},"flags":256}]}]}
{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"agentm"}},{"key":"service.version","value":{"stringValue":"0.1.0"}},{"key":"agentm.session.id","value":{"stringValue":"sess-abc"}}]},"scopeLogs":[{"scope":{"name":"agentm","version":"0.1.0"},"logRecords":[{"timeUnixNano":"1779589316951916032","severityNumber":"SEVERITY_NUMBER_INFO","severityText":"INFO","eventName":"agentm.message.appended","body":{"kvlistValue":{"values":[{"key":"type","value":{"stringValue":"message"}},{"key":"id","value":{"stringValue":"abc"}}]}},"attributes":[{"key":"agentm.session.id","value":{"stringValue":"sess-abc"}},{"key":"agentm.message.id","value":{"stringValue":"abc"}}],"observedTimeUnixNano":"1779589316951963264"}]}]}
```

## Event → OTel mapping

The observability atom translates AgentM bus events into OTel spans (timed
work) and log records (events with bodies). The mapping is intentionally
sparse — only the events whose record is structurally meaningful land on
disk; high-frequency channels like `stream_delta` are excluded from
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

The bus owns the `dispatch_id`, not the `Event` ABI. This is the Commit 1
decision recorded here for posterity.

**Mechanism.** `EventBus` carries a `_dispatch_stack: list[str]` — a stack
of in-flight dispatch ids. `emit` and `emit_sync` push a fresh
`uuid.uuid4().hex` on entry and pop on exit (in a `finally` clause, so a
buggy handler can't leak the id). A public accessor
`bus.current_dispatch_id() -> str | None` returns the top of the stack, or
`None` when no emit is on the call stack.

**Why a stack, not a single field.** A handler that re-emits on the same
bus needs its own nested id; once the inner emit returns, the outer id
must resume. Single-field designs lose the outer id during the nested
call and miscorrelate the surrounding `agentm.handler.invoke` records.

**Why not on the `Event` ABI.** During PR-B planning we surveyed adding
`dispatch_id: str = field(default_factory=...)` to the `Event` base. Two
problems:

1. `Event` is a bare marker class; the ~50 subclasses are
   `@dataclass(slots=True[, frozen=True])` with their own required fields.
   Adding a default field to a non-dataclass base while keeping non-default
   subclass fields would force converting `Event` to
   `@dataclass(kw_only=True, slots=True)` — a sweeping change with ripple
   through every event construction site.
2. "Generated when the event is constructed" and "EventBus passes the
   same id to all handlers" are different mechanisms. The first means
   each `MyEvent(...)` call gets its own id (so re-emitting the same
   instance produces stale ids); the second means one id per emit. The
   bus owns the emit loop, so the bus is the natural owner of the id.

**How consumers join.** The observability atom reads
`api.events.current_dispatch_id()` from inside its `EventBusObserver`
hooks (`on_emit_end` for the `agentm.event.dispatch` record, and
`on_handler_done` for each `agentm.handler.invoke`). Both records stamp
the same value into their `agentm.event.dispatch_id` attribute, so a
downstream query

```
WHERE eventName = "agentm.handler.invoke"
  AND attributes["agentm.event.dispatch_id"] = $id
```

recovers every handler that fanned out from one dispatch, and the matching
`agentm.event.dispatch` record names which channel + how many handlers.

**Tests.** `tests/unit/core/test_event_dispatch_id.py` locks down the three
properties consumers depend on: stable within one dispatch, fresh across
dispatches, nested-dispatch isolation, exception-safe pop, and the
observer-side join is verified end-to-end by
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

## Known follow-up: extract a TraceReader API

The OTLP-unwrap helper (a small recursive function that walks the proto-JSON
tagged-union shape into plain Python) is currently duplicated across
`tool_query_traces`, `tool_eval_run`, `tool_guard_watch`, the llmharness
CLIs, and both rca graders. The duplication is intentional under §11 (atoms
can't import from `core.runtime.*`), but it's a maintenance hazard: a bug
in one copy stays hidden until that specific reader hits the affected
shape, and any future shape extension (e.g. encoding bytes attributes)
needs N edits.

**PR-G in the followup queue** will lift this into a published `TraceReader`
API — likely a small read-only surface in a new published package or under
`core.lib` — so atoms can depend on the unwrap layer the same way they
already depend on `agentm.core.lib.to_jsonable`. The §11 boundary stays
satisfied because the new surface is pure-function read-only over OTLP/JSON,
not a window into runtime state.

Tracking note: when PR-G lands, every `_unwrap_otlp` copy in this
repository should be deleted in the same commit. Grep for `_unwrap_otlp`
to find them.

## Followups identified during landing

Recorded here so they don't get lost; PR briefs to follow when dispatched.

- **PR-E** — `Event` ABI refactor: convert base to `@dataclass(kw_only=True,
  slots=True)` and add a real `dispatch_id` field, closing the bus-stash
  workaround documented in "Bus correlation" above.
- **PR-F** — Declarative `Event → OTel` mapping via per-class `to_otel()`
  method (or registered translator), eliminating the giant translator
  switch in `observability.install`.
- **PR-G** — `TraceReader` API in core, closing the OTLP-unwrap duplication
  flagged above.
- **PR-H** — Process-level `TracerProvider` with `session_id` moved from
  resource attribute to span attribute, aligning with OTel ecosystem
  assumptions about one provider per process.

## Related concepts

- `observability.md` — the consumer side; this design changes its write
  path and on-disk shape, not the underlying event semantics.
- `pluggable_architecture.md` — `SessionManager` lives in `core.runtime/`;
  this change touches that substrate.
- `agent_loop.md` — `message.appended` event is emitted from the kernel's
  trajectory-append path; the writer is now an OTel `Logger` invocation
  rather than a hand-rolled JSONL write.
