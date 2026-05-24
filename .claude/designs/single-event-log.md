# Single Event Log

## Status

Proposed. Implementation pending — see plan `2026-05-24-single-event-log.md`.

## Motivation

Today a single session produces **two parallel JSONL streams**:

| Path | Content | Write path |
|---|---|---|
| `~/.agentm/sessions/--<cwd>--/<ts>_<id>.jsonl` | trajectory — one entry per `AgentMessage` | `SessionManager._append_record`: synchronous `open("a") + write + close` per message |
| `<cwd>/.agentm/observability/<trace>.jsonl` | OTel-shaped events — spans, dispatches, handler invocations, llm.request.start/end | observability builtin: bounded async queue, drops on full, `flush()` per record |

The two streams are largely **overlapping**: every trajectory entry is mirrored
into observability via `emit:tool_call`, `emit:tool_result`,
`emit:before_send_to_llm` etc. The split is denormalization, not logical
separation. It carries several costs:

1. **Redundant message bytes** — `before_send_to_llm` snapshots the whole
   `messages` list every turn. N-turn session = O(N²) message bytes. Today
   we hide this by replacing message bodies with `chars + sha256_prefix` in
   the observability event, but that breaks replay-from-observability.
2. **Two durability policies** — trajectory writes are synchronous-on-call
   (slow but lossless); observability writes are async-bounded (fast but
   drops under backpressure). Two write paths to reason about.
3. **Two retention policies** — ops glob has to know both directories.
4. **Cross-correlation cost** — debugging requires aligning two files by
   timestamp.

## Decision

Replace both with a single per-session event log.

- **One JSONL file per session**, under
  `<cwd>/.agentm/observability/<trace>.jsonl` (existing path is fine).
- **`SessionManager._append_record` no longer writes its own file.** Instead
  it dispatches a `MessageAppendedEvent` through the EventBus; observability
  writer treats it like any other event.
- **`SessionManager.continue_recent` reads the same single file**, filters
  entries where `kind == "message.appended"`, reconstructs the in-memory
  trajectory.
- **All event writes go through one bounded async queue** with **batched
  flush** (write up to N entries or wait T ms before fsync).
- **Backpressure = block, not drop.** Today observability drops events when
  the queue is full; this is acceptable only because trajectory is the
  authoritative copy. Once merged, drops would lose conversation state, so
  the writer must block the producer instead. Queue size set generously
  (~100k entries) so this is rare; if it fires, it is a real problem to
  surface.
- **Drain on shutdown.** `atexit` + signal hook empty the queue before
  process exit.
- **Drop the redaction layer.** `before_send_to_llm` no longer carries a
  `messages` payload at all; consumers reconstruct from prior
  `message.appended` events. (Observers that need the current trajectory
  state get it from the in-memory `SessionManager`, not from the event
  payload.)

## Trade-offs considered

- **Sync vs async writes for message events**: rejected sync — agent loop
  blocked on disk on every message. Async with generous queue + blocking
  backpressure is the right tier for both event classes.
- **Two queues with one file (separate lanes for message vs other events)**:
  rejected as unnecessary complexity. One queue serves both fine if sized
  for peak.
- **Keep two files but unify queue**: rejected — the goal is one file so
  consumers stop having to join.

## What this is NOT

- Not a schema change to existing event records (other than adding the new
  `message.appended` kind and removing the redundant `messages` field from
  `before_send_to_llm`).
- Not a change to the OTel-shaped `schema: "otel/span/v0"` envelope.
- Not a change to retention or where files live in the filesystem hierarchy.

## Validation

Before landing:

1. **Throughput bench** — verifier scenario, ~30-turn session, measure
   wall-clock difference vs. baseline. Acceptance: ≤5% regression.
2. **Queue water-mark under load** — synthetic burst of 10k events;
   confirm queue never blocks under realistic peak.
3. **Crash safety** — SIGTERM during a burst; replay confirms last-batch
   loss is bounded by `T ms` flush interval.

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
{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"agentm"}},{"key":"service.version","value":{"stringValue":"0.1.0"}},{"key":"agentm.session.id","value":{"stringValue":"sess-abc"}},{"key":"agentm.scenario.name","value":{"stringValue":"general_purpose"}}]},"scopeSpans":[{"scope":{"name":"agentm","version":"0.1.0"},"spans":[{"traceId":"jPTe/iWSTaCL7oH6aPJ76g==","spanId":"jtiuCvsDU1M=","name":"turn","kind":"SPAN_KIND_INTERNAL","startTimeUnixNano":"1779589308307455359","endTimeUnixNano":"1779589308307470397","attributes":[{"key":"turn.index","value":{"intValue":"3"}}],"status":{},"flags":256}]}]}
{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"agentm"}},{"key":"service.version","value":{"stringValue":"0.1.0"}},{"key":"agentm.session.id","value":{"stringValue":"sess-abc"}}]},"scopeLogs":[{"scope":{"name":"agentm","version":"0.1.0"},"logRecords":[{"timeUnixNano":"1779589316951916032","severityNumber":"SEVERITY_NUMBER_INFO","severityText":"INFO","body":{"stringValue":"tool_call read"},"attributes":[{"key":"tool.name","value":{"stringValue":"read"}}],"observedTimeUnixNano":"1779589316951963264"}]}]}
```

## Implementation phases

- **PR-A — file exporters (this PR).** Add `core.runtime.otel_export` with
  `FileSpanExporter`, `FileLogExporter`, blocking batch processors, and
  `setup_session_telemetry`. No production wiring; module sits unused but
  fully covered by fail-stop tests at
  `tests/unit/core/test_otel_export.py`.
- **PR-B — observability rewrite.** Cut over the observability builtin and
  `SessionManager._append_record` to publish via the OTel
  Tracer/Logger built by `setup_session_telemetry`. Map each bus event to
  spans or logs per the table in `observability.md`. Old custom JSONL
  writer deleted; the in-memory trajectory rebuild path moves to reading
  log records with `agentm.event="message.appended"`.
- **PR-C — reader cutover.** `SessionManager.continue_recent` and any
  downstream consumers (tuner, evolution indexer) read the new format
  directly. Legacy `~/.agentm/sessions/` JSONL retired.

## Open questions for PR-B

The kernel event surface (`src/agentm/core/abi/events.py`) carries enough
identity for span/log correlation in most cases, but a few gaps need a
PR-B-time decision:

- **Turn correlation.** `LlmRequestStart/EndEvent`, `TurnStart/EndEvent`,
  `StreamDeltaEvent`, and `BeforeSendToLlmEvent` all carry `turn_index` and
  `turn_id`. PR-B should use OTel's native parent span context — a span
  per turn, with LLM-request and tool-call spans as children — rather than
  duplicating `turn_id` into log attributes. The `turn_id` field already
  exists; no event surgery needed.
- **Tool correlation.** `ToolCallEvent` / `ToolResultEvent` /
  `ToolErrorEvent` already share `tool_call_id`. PR-B opens a span keyed
  off this id at `ToolCallEvent` time and closes it at the matching
  `ToolResultEvent`. The provider's id is the trace correlation key,
  no new field needed.
- **Session correlation.** `agentm.session.id` is on the resource block,
  so every span and log line already names its owning session. No
  per-event field required.
- **Missing: bus-dispatch correlation.** Today's observability sink writes
  `event.dispatch` / `handler.invoke` pairs and joins them by an
  ephemeral dispatch id assigned at the bus boundary. The bus does **not**
  expose that id on the event surface — PR-B either adds it to the
  `Event` base class or accepts that bus-internal joins are reconstructed
  from temporal adjacency at read time. Decision deferred to PR-B.
- **Missing: provider request id.** `LlmRequestStartEvent` does not
  carry a provider-side request id (Anthropic / OpenAI request_id
  headers). If we want to cross-reference provider audit logs we need
  to thread that id through `StreamFn` and into `LlmRequestEndEvent`.
  Not blocking on PR-B; flag for the cost-attribution work.

## Related concepts

- `observability.md` — the consumer side; this design only changes its
  write path, not its event semantics.
- `pluggable_architecture.md` — `SessionManager` lives in `core.runtime/`;
  this change touches that substrate.
- `agent_loop.md` — `message.appended` event is emitted from the kernel's
  trajectory-append path.
