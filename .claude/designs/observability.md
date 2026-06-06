# Observability

## Goal

Capture **everything** that happens during one AgentM session — extension
installs, every event-bus dispatch, every handler invocation (with
attribution), errors swallowed by the bus — into a single JSONL file per
session, flushed per record so a crash never loses data. The schema is
OpenTelemetry-flavored (span-shaped) so the file can be ported to OTLP
later with no field renaming.

Observability is **complementary** to `trajectory`, not a replacement:

| Aspect | `trajectory` | `observability` |
| --- | --- | --- |
| Subscribers | Curated channel list | Every channel, automatically |
| Scope | Event payloads | Events + per-handler I/O + ext lifecycle |
| Persistence | Single `agent_end` flush | Per-record streaming flush |
| Audience | Replay / debugging trajectories | Diagnose harness/extension design |
| Schema | Custom flat `{timestamp, channel, event}` | OTel span-shaped |

## Single principle: everything goes through the bus

The atom is a **pure subscriber + EventBusObserver** registered through
`ExtensionAPI.add_observer`. Every diagnostic signal (extension installs,
ExtensionAPI registrations, send_user_message, LLM request start/end, turn
lifecycle) is published as a normal bus event by the harness/kernel. Any
extension can subscribe to any of them — the observability atom is not
privileged and does not monkey-patch the bus or `ExtensionAPI.on`.

## Architecture

### EventBus instrumentation hook

`core/abi/events.py` adds:

- An `EventBusObserver` Protocol with three hooks (`on_emit_start`,
  `on_handler_done`, `on_emit_end`). Observers are installed through
  `ExtensionAPI.add_observer`; hooks fire inside `emit`/`emit_sync` and
  observer crashes are swallowed.
- An `emit_sync(channel, event)` method that runs only sync handlers
  (coroutine returns are skipped with a warning, then closed). Lets sync
  code paths (`ExtensionAPI.register_*`, `send_user_message`) publish
  events without forcing the API surface async.

### New bus events

| Event | Layer | Emitted by | Channel |
| --- | --- | --- | --- |
| `ExtensionInstallEvent` | harness | `session.py` around each `load_extension` (start/end/error) | `extension_install` |
| `ApiRegisterEvent` | harness | `_ExtensionAPIImpl.register_tool/command/provider/message_renderer` via `emit_sync` | `api_register` |
| `ApiSendUserMessageEvent` | harness | `_ExtensionAPIImpl.send_user_message` via `emit_sync` | `api_send_user_message` |
| `LlmRequestStartEvent` | kernel | `AgentLoop` right before draining `stream_fn` | `llm_request_start` |
| `LlmRequestEndEvent` | kernel | `AgentLoop` after `stream_fn` finishes (success or error in try/except) | `llm_request_end` |

### Handler attribution

Per-handler spans come from the observer hook that sees the actual handler
object after dispatch. Handler-to-extension attribution is best-effort and
uses metadata already present on handlers; the atom no longer rewrites
`api.on` registrations.

### The atom

`extensions/builtin/observability.py`:

- Opens `<cwd>/.agentm/observability/<trace_id>.jsonl`.
- Attaches an `_Observer` via `api.add_observer` → `event.dispatch` +
  `handler.invoke` for every channel automatically.
- Subscribes to all the harness/kernel signal events listed above and
  emits a friendly OTel record per kind.
- Subscribes to `turn_start`/`tool_call`/`tool_result`/`turn_end` to
  produce a `turn.summary` synthetic span (tool counts, stop_reason,
  content block types, duration).
- Writes one JSON line per event, flushed immediately. `session.start`
  is the only direct sink write — no other extension is loaded yet to
  hear it.

### Record kinds

All records share the OTel field set: `schema, kind, trace_id, span_id,
parent_span_id, name, start_time_unix_nano, end_time_unix_nano,
attributes, status`.

| `kind` | Source |
| --- | --- |
| `session.start` / `session.ready` / `session.end` | install + session_ready + session_shutdown |
| `extension.install` | `extension_install` (start/end/error) |
| `event.dispatch` | EventBus.emit/emit_sync (one per emit, parents handler.invoke) |
| `handler.invoke` | EventBus per-handler invocation (toggle via `include_handler_records`) |
| `api.register` | `api_register` event |
| `api.send_user_message` | `api_send_user_message` event |
| `llm.request.start` / `llm.request.end` | `llm_request_start` / `llm_request_end` events |
| `turn.summary` | aggregated from turn_start / tool_call / tool_result / turn_end |

## Configuration

```yaml
- module: agentm.extensions.builtin.observability
  config:
    path: ".agentm/observability/{session_id}.jsonl"  # default
    include_handler_records: true                       # default
```

Loaded **first** in `local/manifest.yaml` so it observes every subsequent
extension's install. Observability cannot observe its own install start
(the observer is attached *during* its install) — minor and acceptable.

## Backpressure & blocking

The hot path (`bus.emit` → observer → sink) cannot afford synchronous
file I/O on the asyncio event-loop thread. Three guards:

1. **Async sink**: `_Sink` enqueues records to a bounded `queue.Queue`
   (default `max_queue=10_000`); a daemon thread does the `json.dumps` +
   `write` + `flush`. Hot-path cost is `put_nowait` (O(1)). Slow disks
   no longer block the agent loop.
2. **Drop on full**: when the queue is full (writer falling behind), records
   are dropped and a final `sink.drop_summary` record is appended on close
   so the count is recoverable. Better than OOM.
3. **`atexit.register(close)`**: process exit flushes the tail; we don't
   silently lose the last batch.

`emit_sync` adds a `strict_sync` mode (off by default; enabled via
`observability.strict_sync_handlers: true`). Off → async handlers logged-
and-skipped (cheap, lossy). On → bus raises `RuntimeError` after the
emit, surfacing the bug at dev time without leaving partial dispatches.

Mutation diffing is restricted to `_MUTABLE_CHANNELS` — the documented
mutable events (`before_agent_start`, `context`, `tool_call`, `tool_result`,
`before_compact`). Immutable channels skip both `_serialize` snapshots so
the per-handler cost stays close to the no-diff baseline. Diffs run in
both the success path and the exception path (in `try/finally`), so a
handler that raises after partially mutating the event is still observable.

## Future port to OpenTelemetry

The on-disk schema is OTel span shape verbatim. To port:

1. Replace the `_Sink` with an OTel SDK span exporter.
2. Map `kind` to `SpanKind.INTERNAL` (events) or `CLIENT` (LLM calls,
   when added later).
3. Promote `trace_id`/`span_id` from random hex to OTel ID generators.

No record-shape migration is needed.

## Non-goals

- Sampling / rate limiting (tests disable handler records via config when
  the volume isn't desired; no smarter knob today).
- Bus observer composition (one observer at a time; `set_observer` replaces
  rather than chains). Future: a fan-out observer atom.
- Capturing LLM raw request/response bodies. The kernel already emits
  `before_send_to_llm` / `turn_end` events with the relevant payloads;
  bodies can be added later by widening those events.
