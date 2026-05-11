# Design: Gateway & Channels (Feishu / Lark and beyond) — channels v0

**Status**: SHIPPED (PR #137)
**Created**: 2026-05-11
**Superseded by**: [`client-server-architecture.md`](client-server-architecture.md)
for v1 (process-level split). This doc continues to describe the
**v0 in-process design** that ships today; it stays load-bearing
during the migration window because v0 channels remain supported until
every shipping platform has a v1 client. Once that migration
completes, this doc moves to `designs/historical/`.
**Reference codebase**: [`HKUDS/nanobot`](https://github.com/HKUDS/nanobot) — multi-channel chat agent. Local clone read at `/tmp/refs/nanobot` on 2026-05-11.

---

## 1. Purpose

`contrib/channels/` is a long-running daemon that mediates between a chat
platform (Feishu today; Slack / Telegram / WeChat tomorrow) and one
`AgentSession` per `(channel, chat_id [, thread_id])`. The agent has no
notion of "the chat"; channels have no notion of "the agent". Both ends
speak only `InboundMessage` / `OutboundMessage` over a two-queue
`MessageBus`.

This document is the **boundary contract** for that subsystem. It does
not specify any individual channel beyond illustrative examples — those
live in their own files under `contrib/channels/src/agentm_channels/channels/`.

---

## 2. First Principles

1. **Channel-agnostic**: every concept above the `BaseChannel` line —
   `MessageBus`, `Gateway`, `ApprovalBridge`, command routing —
   knows nothing about Feishu / Slack / Telegram. Adding a channel is
   one Python file plus an entry in YAML config.
2. **Single-file channel contract**: each channel lives in
   `channels/<name>.py`, no peer imports, no reach into `gateway.py` or
   `approval.py`. Mirrors AgentM's §11 atom contract.
3. **Typed wire format**: no magic-string dispatch on `metadata` dict
   keys. Control vs content distinction is `OutboundKind` enum;
   approval buttons are typed `Button` dataclasses; sticky control
   state is on the route, not the message metadata. Compare with
   nanobot's `metadata["_progress"] / ["_stream_delta"] / ["_streamed"]`
   forest.
4. **Cross-loop safety**: `lark_oapi` dispatches inbound on a
   background-thread loop; `MessageBus` detects foreign callers and
   bridges via `call_soon_threadsafe` onto the home loop. Channels
   never need to know this.

---

## 3. Module Layout

```
contrib/channels/src/agentm_channels/
├── bus.py                # InboundMessage, OutboundMessage, Button, MessageBus
├── base.py               # BaseChannel ABC (start / stop / send / send_delta)
├── registry.py           # pkgutil + entry_points discovery (built-in beats plugin on collision)
├── manager.py            # ChannelManager: instantiate enabled channels, fan out outbound
├── gateway.py            # MessageBus ↔ AgentSession bridge; route table
├── approval.py           # ApprovalBridge: typed Button round-trip, identity check
├── chat_session_map.py   # (channel, chat_id) → AgentSession session_id, JSON-persistent
├── commands/             # Slash command layer (see command-routing.md)
├── cli.py                # `agentm-gateway` console entry
└── channels/
    ├── feishu.py         # lark_oapi adapter (only channel implemented today)
    └── stub.py           # in-memory channel for tests
```

The four "kernel-of-the-gateway" files (`bus`, `base`, `gateway`,
`approval`) are the boundary contract. Everything else is replaceable
without forking.

---

## 4. Wire Types (`bus.py`)

### 4.1 InboundMessage

```python
@dataclass(slots=True)
class InboundMessage:
    channel: str               # "feishu" — routes the reply back
    sender_id: str             # stable user id within channel
    chat_id: str               # DM / group / channel id
    content: str               # flattened plaintext (channel responsible for rendering richtext → text)
    timestamp: datetime
    media: list[str]
    metadata: dict[str, Any]   # channel-private extras (feishu_message_id, …)
    button_value: str | None   # set when this inbound is a button click round-trip
    session_key_override: str | None   # defaults to "{channel}:{chat_id}"

    @property
    def session_key(self) -> str: ...
```

`session_key` is what the gateway uses to look up the AgentSession.
Channels that scope tighter than `chat_id` (e.g. one session per
Feishu thread) set `session_key_override` like `"feishu:c123::thread456"`.

### 4.2 OutboundMessage + OutboundKind

```python
class OutboundKind(str, Enum):
    MESSAGE = "message"
    TURN_COMPLETE = "turn_complete"   # control signal — tear down ACK affordances

@dataclass(slots=True)
class OutboundMessage:
    channel: str
    chat_id: str
    content: str = ""
    reply_to: str | None = None
    media: list[str]
    metadata: dict[str, Any]
    buttons: list[Button]
    kind: OutboundKind = OutboundKind.MESSAGE
```

`kind` is the dispatch axis. Anything that would have been a magic
metadata key (progress / stream delta / streamed / tool hint) becomes a
new `OutboundKind` variant when needed. This keeps the channel switch
exhaustive and grep-friendly.

### 4.3 Button (typed approval primitive)

```python
@dataclass(frozen=True, slots=True)
class Button:
    label: str
    value: str                  # round-trips as InboundMessage.button_value
    style: Literal["primary", "danger", "default"] = "default"
```

Channels translate `Button` into native UI (Feishu interactive card,
Slack action block, Telegram inline keyboard, plaintext numbered list
as last resort). The approval bridge owns the encoding of `value`;
gateways must not parse `button_value` themselves.

### 4.4 MessageBus (cross-loop safe)

Two `asyncio.Queue`s (inbound, outbound), owned by the gateway. The
critical subtlety: `lark_oapi.ws.client` dispatches WS events on a
background-thread asyncio loop. `await queue.put()` from a foreign
loop *appears* to succeed, but the home-loop consumer's waiter never
wakes — silent hang. `MessageBus._publish` detects a foreign caller
and bridges via `loop.call_soon_threadsafe(queue.put_nowait, msg)` on
the home loop. This is in the bus, not in each channel, so it is
correct by construction.

Both queues are **unbounded today**. Future work (P1) is to add a
configurable cap with a drop-oldest policy for outbound — see plan
batch 3.

---

## 5. BaseChannel Contract

```python
class BaseChannel(ABC):
    name: str = "base"        # must match the module filename
    display_name: str = "Base"

    # required
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def send(self, msg: OutboundMessage) -> None: ...

    # optional, opt-in
    async def send_delta(self, chat_id: str, delta: str, metadata: dict | None = None) -> None: ...
    @property
    def supports_streaming(self) -> bool: ...

    # helper provided by base, called by concrete impls on inbound
    async def _handle_message(self, *, sender_id, chat_id, content,
                              media=None, metadata=None,
                              session_key=None, button_value=None) -> None: ...

    def is_allowed(self, sender_id: str) -> bool: ...

    @classmethod
    def default_config(cls) -> dict[str, Any]: ...
```

`_handle_message` does the `allow_from` check, then publishes one
`InboundMessage`. Subclasses must call it from their inbound callback;
they must not publish to the bus directly.

Streaming (`send_delta`) is opt-in: the base default is a no-op, and
`supports_streaming` returns True only when the subclass overrides
`send_delta` *and* config has `streaming: true`. The
`send_delta` protocol (deltas, end signals, throttling) is spelled
out in §8 as future work.

---

## 6. Gateway (`gateway.py`)

The gateway is the agent-side half of the bus. Its job per inbound
message:

1. **Decode** — if `button_value` is set, route to `ApprovalBridge`
   and short-circuit.
2. **Filter** — if `content` starts with `/`, route to `CommandRouter`
   (see `command-routing.md`); control commands return without
   touching the session, prompt commands rewrite `content` and fall
   through.
3. **Route lookup** — `(_routes: dict[session_key, _Route])`. Create
   on first contact: build an `EventBus`, call
   `session_factory(cwd, bus, resume_id)`, wire `TurnEndEvent` /
   `DiagnosticEvent` / `ToolCallEvent` handlers, persist
   `session_key → session_id` to `ChatSessionMap`.
4. **Dispatch** — set per-turn `ApprovalContext` on the route, await
   `route.session.prompt(content)`. The route lock serializes turns
   within one chat; **inbound dispatch itself is `asyncio.create_task`'d**
   so a route that is awaiting an approval future does not block
   button clicks routed to other (or even the same) route.
5. **Reply** — `TurnEndEvent` → `OutboundKind.MESSAGE` with
   assistant text; `DiagnosticEvent(level∈{warning,error})` →
   `MESSAGE` prefixed with `⚠`. On turn end, emit one
   `OutboundKind.TURN_COMPLETE` so channels can tear down "thinking"
   affordances (Feishu ACK emoji).

Tool calls and tool results are **not** echoed back to the chat by
design. Operators reading the trajectory JSONL see everything; users
see what the assistant's next turn summarises. Compare with nanobot's
tool-hint cards — we deliberately opt out.

---

## 7. ApprovalBridge (`approval.py`)

Channel-agnostic human-in-the-loop gate for tool calls.

- Hooks `ToolCallEvent`. For each call, consults `ApprovalPolicy` —
  `always_allow` / `always_block` / `require_approval` (or `"*"`).
- For `ask` decisions: posts one `OutboundMessage` with two `Button`s
  (`Approve` / `Deny`) and awaits an `asyncio.Future` keyed by
  `approval_id` (`f"approval-{id(future):x}"`). Encoding of the
  button `value` (`"<approval_id>:<decision>"`) is **private** to
  `approval.py`.
- On click: `try_resolve_inbound(msg)` matches `button_value`,
  identity-checks `sender_id == ctx.sender_id` (so a bystander cannot
  approve someone else's tool call), resolves the future.
- On timeout: future is cancelled, the call is denied.

### 7.1 O(1) routing (PR #1 addition)

Today every inbound with `button_value` walks the full route table.
The bridge will publish each pending `approval_id` to a global
`Gateway._approval_index: dict[approval_id, ApprovalBridge]` on
creation and pop on resolution; `_dispatch` queries the index first
and only falls back to broadcast on miss (preserving current
correctness if a bug skips an unregister).

### 7.2 Text fallback (future)

Channels without native button UI (email, raw IRC, basic Telegram) need
`/approve <id>` / `/deny <id>` text commands. `ApprovalBridge` will
expose `try_resolve_text(approval_id, decision, sender_id)` so the
command layer can wire builtin commands without coupling to the
private encoding. Listed in command-routing.md, not implemented in PR #1.

---

## 8. What we deliberately do not have yet

| Feature | nanobot does it | We will when |
|---|---|---|
| Streaming text (`send_delta`) | yes, CardKit + delta-coalesce | AgentM `EventBus` exposes token-delta events |
| Persistent message history per chat (separate from AgentM session) | yes, JSONL per chat | Verified that AgentM's `SessionManager` resume does not round-trip our chat history. Until then, ChatSessionMap (session_key → session_id) is enough. |
| Group `respond_when: mention` | yes, `_is_bot_mentioned` | Feishu group support is a real use case (in plan batch 3) |
| Bounded queues + drop policy | no (also unbounded) | Outbound backpressure becomes observable |
| ASR / audio transcription | yes | Never — not in AgentM's scope |
| `sendProgress` / tool-hint cards | yes | Never — operator reads JSONL, user reads summaries |
| Multi-replica / leader election | no | Production HA pressure shows up |

---

## 9. Differences from nanobot worth recording

We started from nanobot's `bus + channels + manager` shape and kept
the discovery/registry/manager pattern verbatim. Differences:

1. **Typed `OutboundKind`** replaces magic `metadata["_progress"]` /
   `["_stream_delta"]` / `["_streamed"]` dispatch. nanobot's
   `_dispatch_outbound` is a long `if msg.metadata.get(...)` chain;
   ours is a `match msg.kind`.
2. **Typed `Button`** replaces `buttons: list[list[str]]` and label
   string-matching for callback resolution. nanobot has no formal
   tool-call approval workflow (its `ask_user` is an agent-loop stop
   reason).
3. **`ApprovalBridge`** is a separate channel-agnostic module with
   per-call identity check, timeout, and a private button-value
   encoding. Built around `ToolCallEvent` so any tool can be gated.
4. **Cross-loop bus** centralizes the `lark_oapi` foreign-loop fix;
   nanobot inlines `run_coroutine_threadsafe` at each callback.
5. **No tool I/O echo to chat**. Deliberate product choice; nanobot
   sends progress cards by default.
6. **Smaller surface**: 1.5k lines vs nanobot's 14.6k in channels/.
   Feature parity is **not** a goal; correctness + replaceability is.

---

## 10. Open questions

1. When AgentM eventually emits assistant-token deltas (today only
   `TurnEndEvent` carries final assistant text), do we expose a new
   `OutboundKind.STREAM_DELTA` + `STREAM_END`, or absorb deltas
   inside `MESSAGE` with a state field? Lean toward separate kinds —
   matches the rest of the dispatch design.
2. Outbound bounded queue drop policy: drop-oldest (less stale
   content) vs drop-newest (preserve "first response"). Lean
   drop-oldest, validate against real Feishu rate-limit traces.
3. Is `ChatSessionMap` enough for resume? Needs verification that
   AgentM `SessionManager.resume(session_id)` actually round-trips
   message history. If not, we either land an SDK-level fix or
   keep our own per-chat history. Owner: this design doc author,
   next pass.

---

## 11. Cross-references

- `command-routing.md` — slash command layer riding on top of the gateway.
- `pluggable-architecture.md` — the SDK boundary the channels sit above.
- `self-modifiable-architecture.md` — relevant when atom-as-command
  lands (atoms mounted via `/atom:install` mutate the catalog).
