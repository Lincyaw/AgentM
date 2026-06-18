# Gateway Agent Proxy Architecture Review

**Status:** PROPOSED
**Created:** 2026-06-18
**Builds on:** [pluggable-architecture.md](pluggable-architecture.md),
[session-inbox.md](session-inbox.md),
[single-process-gateway.md](single-process-gateway.md),
[interactive-subagent.md](interactive-subagent.md),
[go-terminal-tui.md](go-terminal-tui.md)

## 0. Decision

The gateway should expose a stable **Agent Proxy** semantic layer over AgentM
Core. It should not become a second agent runtime, and it should not push
client-specific operations into Core.

The current Core surface is already close to the right level of composition:

- `AgentSession.prompt(...)`: push one user inbox item and await completion.
- `AgentSession.tick(...)`: resume without a new prompt.
- `AgentSession.interrupt()`: preempt an in-flight driver round.
- `AgentSession.idle(...)`: await parked + inbox empty + no tracked background
  work.
- `AgentSession.inbox.push(InboxItem(source="user", ...))`: generic input
  delivery.
- `AgentSession.bus`: event subscription surface.
- `ExtensionAPI.post_inbox(...)`, `send_user_message(...)`,
  `spawn_child_session(...)`: atom-facing composition points.

So the priority is not "add a Core API for every UX feature." The priority is
to define the gateway-side proxy semantics that compose these primitives into a
stable contract for every client.

## 1. Problem Statement

The Go terminal TUI currently exposes controls that suggest Claude Code-like
live intervention, but the underlying gateway semantics are not complete enough
to make those controls reliable.

The visible symptom is: when the model is already working, sending another
message queues locally in the TUI; the user cannot clearly choose "let this run
then apply my next instruction" versus "stop now and apply this new instruction."

That is not only a terminal bug. It exposes a boundary problem:

- Core has the mechanisms: inbox, persistent driver, interrupt signal, events.
- Gateway currently exposes only a narrow inbound shape:
  `content`, `button_value`, `control`.
- Terminal-go fills the missing policy locally with a client queue.
- Root sessions and child sessions are not routed through the same gateway path:
  root input uses `sess.prompt(...)`, while child input uses `child.inbox.push(...)`.
- Several UI affordances are only stubs or shallow mappings because the gateway
  does not yet publish a complete capability/read model.

The fix should start at the gateway semantics. Client UX should be an adapter
over those semantics.

## 2. Existing Surface Review

| Layer | Existing capability | Current concern |
|---|---|---|
| Core `AgentSession` | Persistent driver blocks on `SessionInbox.wait_nonempty`; `prompt()` pushes inbox and waits; `interrupt()` sets the driver signal; `idle()` gives a real quiescence boundary. | Good primitive set. The missing piece is mostly a host-facing semantic facade, not more one-off Core methods. |
| Core `SessionInbox` | Single entry point for user/background/monitor/subagent messages; FIFO drain at turn boundary; `kick()` wakeup; background work accounting. | Gateway should use this uniformly for root and child sessions. |
| Core `ExtensionAPI` | `post_inbox`, `send_user_message`, `spawn_child_session`, service registry, event subscription. | Atom-facing API is already composable. Avoid expanding it for client UX. |
| Gateway wire inbound | `content`, `button_value`, `control="interrupt"`. | Too action-poor. It cannot express submit policy, request correlation, target state, or generic interaction responses. |
| Gateway runtime | `handle_inbound` dispatches interrupt inline; root prompts create `_prompt_session` tasks that await `sess.prompt`; child input pushes directly into child inbox. | Mixed root/child semantics and prompt-await coupling leak into client behavior. |
| Gateway outbound | Event-derived `metadata.kind` values with durable vs ephemeral routing. | Clients need a durable or replayable state snapshot; `session_ready` and `agent_end` are currently ephemeral even though they affect recoverable UI state. |
| Terminal-go | Bubble Tea TUI supports queue, interrupt key, child tabs, approvals, commands, model picker, tool views. | Some controls are real, some are local approximations, and some depend on partial capability data. |

## 3. Key Findings

### F1. Core API expansion is not the first move

Core already has the building blocks required for live intervention:

1. queue/pending input: `SessionInbox.push`
2. cooperative next-turn delivery: default inbox drain
3. hard preemption: `AgentSession.interrupt`
4. resume processing: persistent driver wakes on inbox content
5. completion observation: event bus + `agent_end`
6. quiescence: `AgentSession.idle`

Adding Core APIs like `interrupt_then_submit`, `enqueue_message`, or
`client_queue_policy` would encode gateway/client policy into Core. Those should
be composed in the gateway proxy layer.

### F2. Gateway root input should stop depending on `prompt()` as the host model

`prompt()` is correct SDK sugar for a caller that wants to submit one user turn
and await completion. A long-lived gateway is different: it should submit input,
acknowledge acceptance, observe events, and maintain a read model.

Using `await sess.prompt(...)` in gateway root handling creates two problems:

- the gateway task lifetime is tied to the turn lifetime;
- the natural operation becomes "submit and wait" rather than "submit an intent
  to an addressable session."

For a proxy, the primary operation should be `inbox.push(...)` plus event/read
model observation. `prompt()` can remain available for CLI/SDK use.

### F3. Root and child sessions need one target model

The interactive-subagent design already states the right model:

> Every session, main or child, is addressable; input is delivered to that
> session's inbox; whoever owns the driver drains it.

Gateway should promote that from a child-specific mechanism to the general
session model:

- A target is a session, not a "root chat" versus "child tab" special case.
- A gateway `session_key` maps to a target session id.
- Child `session_id` can also be used directly as a target.
- Input delivery is always inbox-based.
- The driver owner is independent from the input sender.

### F4. The wire protocol needs client intent, not client behavior

Today an inbound is classified by field presence:

- `control == "interrupt"` -> interrupt
- `button_value` -> approval response
- leading slash in `content` -> command
- otherwise -> prompt

That is too implicit for advanced clients. The gateway needs explicit intent:

- `submit` a user message
- `interrupt` a running turn
- `submit` with a policy such as cooperative or interrupt-first
- `resolve_interaction`
- `run_command`
- `snapshot` or `query_state` if the client needs active state refresh

This can be added compatibly: keep the existing fields as v2 shorthands, add an
explicit `action`/`request_id`/`policy` shape, and have the router normalize both
old and new forms into one internal command.

### F5. Clients need a read model, not just event deltas

Terminal-go should not reconstruct everything from best-effort stream frames.
At minimum it needs a session snapshot that answers:

- what session am I attached to?
- is there an active turn?
- is it idle, running, waiting for approval, interrupted, or errored?
- what is the active turn id?
- what tools/commands/model/capabilities are available?
- are there pending interactions?
- what child sessions exist, and what are their states?

This read model can be gateway-owned and event-projected. It does not need to be
a broad Core API. The gateway is already the process that observes all session
events and routes them to clients.

### F6. Human interactions should be generalized

Approvals currently use `button_value`; elicitation in terminal-go falls back to
text. That will not scale.

The gateway should expose one interaction model:

- `interaction_request`: durable outbound, with `interaction_id`, type, title,
  payload schema, actions, requester identity, target session/turn.
- `interaction_response`: inbound action with `interaction_id`, chosen action,
  payload, sender identity.

Approval is one interaction type. Elicitation, OAuth, form input, plan approval,
and future review gates can reuse the same mechanism.

### F7. Capability surfaces should be honest

Terminal-go currently inherits a cagent-shaped app surface. Some methods are
real gateway calls; others are local stubs or name-only projections. This makes
the UI appear more capable than the gateway contract actually is.

Gateway should publish capabilities explicitly:

- static capabilities in `welcome`;
- session capabilities in `session_snapshot` or `session_ready`;
- interaction capabilities per request;
- unsupported operations as explicit capability absence, not local no-ops.

Only after this is stable should clients expose polished controls.

### F8. "Single-process gateway" should mean deployment, not ownership

`single-process-gateway.md` correctly removed the old worker split. But the
phrase "the gateway IS the SDK process" should not be read as "the gateway owns
agent policy." The intended boundary is:

- Core owns mechanism.
- Extensions/scenarios own agent policy.
- Gateway owns proxying, routing, delivery, state projection, and human
  interaction transport.
- Clients own presentation and local ergonomics.

## 4. Proposed Gateway Agent Proxy Semantics

The gateway should wrap each live `AgentSession` in an internal proxy facade.
This is not a new Core interface at first; it is a gateway-side boundary that
can later be promoted only if a second host proves the need.

### 4.1 Internal Target

```python
@dataclass(frozen=True)
class AgentTarget:
    session_id: str
    session_key: str | None
    parent_session_id: str | None
    kind: Literal["root", "child"]
```

The gateway resolves every inbound to an `AgentTarget` before deciding what to
do. This is **internal gateway state**, not a new wire routing field. The v2
envelope keeps routing on `Envelope.session_key`; root chat keys and child ids
become two addressing modes for the same internal target concept.

### 4.2 Submit

```python
@dataclass(frozen=True)
class SubmitIntent:
    request_id: str
    content: str
    source: str = "user"
    policy: Literal["cooperative", "interrupt_first"] = "cooperative"
```

Semantics:

- `cooperative`: append a user inbox item. The current turn continues; the new
  message is drained at a turn boundary.
- `interrupt_first`: call `interrupt()` if a turn is active, then append the user
  inbox item. The new message is the user's explicit correction after stopping
  the current trajectory.

The gateway returns an acceptance outcome immediately:

```python
@dataclass(frozen=True)
class SubmitAccepted:
    request_id: str
    target: AgentTarget
    policy: str
    accepted_at: float
```

The final model response still arrives through outbound events.

For explicit client requests, `request_id` is an idempotency key scoped by
`peer_id + session_key + action`. Replaying the same semantic request must return
the same acknowledgement and must not append a second inbox item or resolve an
interaction twice. Legacy inbounds without `request_id` keep today's at-most-once
envelope semantics; the gateway may synthesize a correlation id for logging, but
that synthesized id is not a retry contract.

### 4.3 Interrupt

```python
@dataclass(frozen=True)
class InterruptIntent:
    request_id: str
    reason: str | None = None
```

Semantics:

- If a round is active, set the session signal.
- If idle, do not fail the client; return `noop_idle`.
- Emit a state update so every connected client sees the interruption.

Core may eventually return a structured interrupt result, but the gateway can
start by deriving the result from its own read model.

### 4.4 Command

Slash commands remain gateway-routed first, then session-routed if known by the
session. Expanded prompt commands re-enter the same `SubmitIntent` path instead
of calling `prompt()` directly.

### 4.5 Interaction

Approvals, elicitation, OAuth, and review gates normalize to:

```python
@dataclass(frozen=True)
class InteractionRequest:
    interaction_id: str
    target: AgentTarget
    turn_id: str | None
    kind: str
    payload: dict[str, Any]
    actions: list[InteractionAction]
```

```python
@dataclass(frozen=True)
class InteractionResponse:
    request_id: str
    interaction_id: str
    action: str
    payload: dict[str, Any]
```

Existing approval cards can be adapted onto this without removing
`button_value` immediately.

### 4.6 Read Model

Gateway maintains:

```python
@dataclass
class SessionProxyState:
    target: AgentTarget
    phase: Literal["idle", "running", "waiting_interaction", "interrupting", "errored"]
    active_turn_id: str | None
    active_tools: list[dict[str, Any]]
    pending_interactions: list[str]
    capabilities: dict[str, Any]
    children: list[AgentTarget]
    last_error: str | None
```

This state is projected from events plus gateway-local actions. It can be sent:

- on `welcome` reconnect as a snapshot list;
- after session creation;
- after turn start/end;
- after interrupt request/result;
- after interaction request/response;
- after child start/end.

This avoids making best-effort stream frames the only source of truth for UI
state.

## 5. Wire Shape

Keep the existing envelope. Add explicit body fields compatibly.

Routing stays on the envelope:

- root chat: `Envelope.session_key` is the chat client's opaque conversation key;
- child session: `Envelope.session_key` is the child `session_id`, as already
  used by interactive subagents.

No `target` object is added to the body. The gateway derives its internal
`AgentTarget` from the envelope and its session/child registries.

### 5.1 Inbound

Legacy remains valid:

```json
{"content": "hello"}
{"control": "interrupt"}
{"button_value": "approval:approve"}
```

New explicit form:

```json
{
  "action": "submit",
  "request_id": "cli-123",
  "content": "stop doing X; do Y instead",
  "policy": "interrupt_first"
}
```

```json
{
  "action": "interaction_response",
  "request_id": "cli-124",
  "interaction_id": "approval-abc",
  "response": {"action": "approve", "payload": {}}
}
```

### 5.2 Outbound

Add three stateful outbound kinds:

- `request_ack`: durable or at least replayable until the client has seen it.
- `session_snapshot`: durable/current-state frame for reconnect and view repair.
- `interaction_request` / `interaction_resolved`: durable human gates.

Keep high-volume stream/tool deltas ephemeral. Reconsider `agent_end` and
`session_ready`: either make them durable or make `session_snapshot` carry all
state they mutate so missed ephemeral frames are harmless.

## 6. Roadmap

### Phase 0 - Inventory and semantic tests

Goal: freeze what Core already provides and identify actual gaps before adding
API.

Tasks:

1. Write a small API inventory for `AgentSession`, `SessionInbox`, `ExtensionAPI`,
   gateway inbound/outbound, and terminal-go controller assumptions.
2. Add fail-stop tests around today's behavior:
   - root prompt while running
   - child inbox input while running
   - interrupt while running
   - interrupt while idle
   - reconnect after `session_ready` was missed
3. Decide whether any state cannot be derived outside Core.

Exit criteria:

- list of Core additions is empty or justified one by one;
- current root/child semantic mismatch is captured in tests.

### Phase 1 - Gateway internal AgentProxy facade

Goal: create one gateway boundary over root and child sessions.

Tasks:

1. Add an internal `AgentProxy`/`SessionHandle` abstraction in `src/agentm/gateway/`.
2. Resolve every inbound to an `AgentTarget`.
3. Route root input through `session.inbox.push(InboxItem(source="user", ...))`
   instead of `await session.prompt(...)`.
4. Keep `prompt()` for SDK/CLI use; do not remove it.
5. Re-enter expanded commands through the same submit path.
6. Keep external wire/client behavior unchanged in this phase. Terminal-go may
   still queue locally until Phase 2/5 expose explicit submit policies.

Exit criteria:

- root and child input share one gateway delivery mechanism;
- gateway no longer needs `await session.prompt(...)` as the root-host model;
- no terminal-go queue behavior is removed before policy intent exists;
- no new Core API has been added.

### Phase 2 - Structured intent wire

Goal: make client intent explicit while preserving old peers.

Tasks:

1. Extend `InboundBody` with `action`, `request_id`, `policy`,
   and `interaction_id`.
2. Make `Router.dispatch` normalize legacy and explicit forms into the same
   internal gateway command.
3. Define `request_id` idempotency semantics for explicit mutating actions
   (`submit`, `interrupt`, `interaction_response`) in the same change that adds
   `request_ack`.
4. Add `request_ack` outbound.
5. Add gateway-side tests for all legacy and explicit combinations, including
   duplicate explicit requests with the same `request_id`.

Exit criteria:

- terminal/Feishu can stay on legacy fields during migration;
- terminal-go can opt into `policy="interrupt_first"` when the user chooses to
  stop and redirect.
- retrying the same explicit request cannot double-submit or double-resolve.

### Phase 3 - Gateway read model and snapshots

Goal: clients can render state from a stable snapshot plus live deltas.

Tasks:

1. Build a per-session read model projected from events and proxy actions.
2. Send `session_snapshot` on connect/reconnect and after major state changes.
3. Include capabilities, phase, active turn, pending interactions, and children.
4. Update durability classification so replay/reconnect cannot lose critical UI
   state.

Exit criteria:

- a terminal reconnect can show whether the agent is running or waiting;
- missing ephemeral stream frames does not corrupt controls;
- capability-driven UI can hide unsupported actions.

### Phase 4 - Unified human interaction

Goal: approvals and elicitation share one semantic mechanism.

Tasks:

1. Introduce `interaction_request` and `interaction_response`.
2. Adapt existing approval cards onto the generic interaction model.
3. Replace terminal-go elicitation text fallback with structured responses.
4. Keep `button_value` compatibility until all peers migrate.

Exit criteria:

- every human gate has an id, type, actions, and payload;
- clients do not need special protocol branches for approvals versus forms.

### Phase 5 - Terminal-go UX over the new contract

Goal: improve terminal experience after gateway semantics are stable.

Tasks:

1. Replace the implicit local queue with explicit user choices:
   - send after current turn
   - interrupt and send
   - cancel current turn
2. Render `request_ack` and `session_snapshot` state.
3. Make controls capability-driven.
4. Align child and root session controls.
5. Remove or hide cagent-surface stubs that the gateway cannot support.

Exit criteria:

- pressing Enter during a run has deterministic semantics;
- Esc has visible acknowledgement and state transition;
- users can intervene in root and child sessions consistently.

### Phase 6 - Cross-client hardening

Goal: keep Gateway semantics client-neutral.

Tasks:

1. Verify Feishu peer behavior on legacy fields.
2. Add a protocol compatibility matrix.
3. Document how new clients implement submit/interrupt/snapshot/interaction.
4. Add reconnect stress tests across legacy and explicit peers.

Exit criteria:

- no terminal-specific semantics in Core;
- no Feishu-specific semantics in Core;
- gateway remains the only proxy/scheduling/distribution boundary.

## 7. Core API Change Policy

Only add Core API if the gateway cannot safely compose existing primitives.

Candidate additions, in priority order:

1. **Structured interrupt result**:
   `AgentSession.interrupt() -> InterruptResult`, where result is one of
   `accepted`, `noop_idle`, `closed`. This improves acknowledgements but is not
   required for the first proxy pass.
2. **Non-blocking driver state read**:
   a tiny read-only property for `running/parked/closed`, if event projection is
   insufficient for race-safe child kick decisions.
3. **Inbox observation helpers**:
   only if gateway status genuinely needs queue depth or pending-user counts.

Rejected as Core APIs:

- `interrupt_then_submit(...)`
- `enqueue_user_message(...)`
- `client_queue_policy(...)`
- terminal/Feishu-specific capability methods
- interaction UI concepts

These belong in gateway or clients.

## 8. Acceptance Scenarios

1. User sends message A; while A is running, sends message B with cooperative
   policy. A completes or reaches a turn boundary, then B is processed. Client
   received request ack for B immediately.
2. User sends message A; while A is running, sends message B with
   interrupt-first policy. Gateway interrupts A, appends B, emits state update,
   and the next model turn sees B.
3. User presses Esc while idle. Gateway returns a no-op interrupt ack; UI does
   not pretend a turn was cancelled.
4. User interacts with a live child session. The same submit/interrupt semantics
   apply as root sessions.
5. User reconnects after missing `session_ready`. Gateway sends a snapshot with
   commands/tools/model/session state, so the UI is correct.
6. Approval and elicitation both arrive as interaction requests; the client
   responds through the same inbound action.
7. Feishu legacy `content` and `button_value` inbounds still work.

## 9. Documentation Follow-up

If this proposal is accepted, update:

- [single-process-gateway.md](single-process-gateway.md): clarify that
  "single process" is a deployment/implementation choice, while the gateway's
  semantic role is proxy/router/state projector.
- [interactive-subagent.md](interactive-subagent.md): generalize the child
  session target model into the root session path.
- [go-terminal-tui.md](go-terminal-tui.md): replace local queue-first behavior
  with capability-driven submit policies.
- [textual-tui.md](textual-tui.md): mark the old interrupt affordance as a
  proxy semantic, not only a control marker.

## 10. Implementation Evidence, 2026-06-18

Phase 2/3 now have both unit coverage and live-wire evidence against a real
gateway process at `unix:///tmp/agentm-debug.sock` using the `litellm-dsv4flash`
profile.

- Real provider path: a manually framed prompt to DeepSeek-V4-flash emitted
  `stream_text` and `assistant_text`; sending one prompt, immediately
  interrupting, then sending a new instruction did not wedge behind the old run.
- Duplicate request id: two explicit `action="run_command"` inbounds carrying
  the same `request_id="dup-proof-1"` and `/gateway_debug` content produced
  exactly two `request_ack` frames (`accepted`, then `duplicate`) and exactly
  one `command_result`. The duplicate replay did not execute the command a
  second time.
- Gateway debug: explicit `action="run_command"` with `/gateway_debug all`
  returned `request_ack` plus a `command_result` containing `**sessions**` and
  `**globals**`; the live payload included session projections, phases,
  `inflight_tasks`, `outbox_ready`, `total_pending_approvals`, and
  `tracked_sessions`.

Verification commands:

```bash
uv run pytest tests/unit/gateway --tb=short
uv run ruff check src/agentm/gateway tests/unit/gateway
cd contrib/gateway-peers/terminal-go && go test ./internal/adapter
```
