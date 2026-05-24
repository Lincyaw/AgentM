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

## Related concepts

- `observability.md` — the consumer side; this design only changes its
  write path, not its event semantics.
- `pluggable_architecture.md` — `SessionManager` lives in `core.runtime/`;
  this change touches that substrate.
- `agent_loop.md` — `message.appended` event is emitted from the kernel's
  trajectory-append path.
