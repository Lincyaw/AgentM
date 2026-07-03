# Design: Single-Process Gateway (channels v2)

**Status**: IMPLEMENTED (channels v2, merged 2026-05-28)
**Created**: 2026-05-28
**Supersedes**: channels v1 (peer-mesh daemon/worker split) and channels v0 (in-process channels) — both fully retired by this rewrite; see git history for the prior designs.

## 0. Mandate

The user directed: cleanest, simplest, most decoupled future shape. **No historical decisions preserved.** Compat shims, deprecation warnings, dual-stack code paths — all forbidden. Single-jump rewrite.

The architectural realisation this doc captures: **the daemon is the gateway and the gateway is one process**. No separate "worker" process. Earlier drafts of this design carried v1's daemon/worker process split as an unexamined assumption — it was bought process-level isolation, distributability, and zero-SDK-coupling at the cost of a wire-glued inner-bus and a SessionDriver/spawn lifecycle. For this codebase's actual use case (single user / small team / Feishu chat surface) none of those v1 properties earn their rent. They are deleted.

What survives, on first principles:

| Concept | Survives because |
|---|---|
| Wire protocol (length-prefixed JSON envelopes) | Chat-client vendor SDKs (lark_oapi, Textual) must stay out of the SDK process; the wire is the cleanest isolation. |
| Per-peer durable outbox | Chat clients disconnect / crash / reconnect; the daemon must not drop outbound mid-flight. |
| Per-peer at-most-once inbox ledger | Same reason, opposite direction. |
| Pluggable `OutboxStore` / `InboxLog` / `Authenticator` Protocols | Real extension axes; documented surfaces. |
| `chat_client` peer kind | Three (feishu, terminal, future slack/discord) ship; vendor SDK isolation. |
| Per-session AgentSession in memory | The session IS its in-memory state; persistence is for crash recovery. |
| `ChatSessionMap` (session_key → session_id) | Long-lived; survives daemon restart so resume-from-history works. |
| Gateway schedule store (`schedules.json`) | Durable host-level wakeups belong to the long-lived gateway, not to per-session monitor atoms. |

What is deleted, with prejudice:

| Concept | Deleted because |
|---|---|
| `agent_worker` peer kind | No separate worker process exists in this design. |
| `agentm-worker` package | Collapses into the SDK as `agentm gateway` mode + `wire_driver` atom. |
| `Gateway` class (the in-process session host) | The gateway IS the SDK process; the concept doesn't need its own class. |
| `MessageBus` (internal queue between channels and gateway) | Internal bus mirroring external wire was the v0 abstraction; gone with v0. |
| `BaseChannel` / `ChannelManager` / `_WireChannel` / `channels/` subpackage | "Channel as in-process plugin" is a dead concept. |
| `agentm_channels.channels` entry-point group | Same. |
| `--inproc-worker` flag | No worker concept means no in-process vs out-of-process mode. |
| `--resume` as a normal-path flag | Resume only happens on crash recovery; normal sessions live in daemon memory across turns. |
| `_session_id_hint` / `body["channel"]=="_a2a"` magic strings | Replaced by typed envelope fields and same-process session lookup. |
| Worker pool / spawn-on-demand / inactivity timeout machinery | No worker = no spawn = no pool = no timer. |
| Cross-process peer_send | Same-process dict lookup; cross-process is future work, not v2. |

---

## 1. Process topology

```
                  ┌──────────────────────────────────────────────────┐
                  │            agentm gateway (one process)          │
                  │                                                  │
                  │   WireServer                                     │
                  │   Router + SessionManager                        │
                  │   ApprovalManager + CommandRouter                │
                  │   Outbox + Inbox + ChatSessionMap + Scheduler    │
                  │   PeerRegistry + Auth                            │
                  │                                                  │
                  │   sessions: dict[session_key, AgentSession]      │
                  │     each AgentSession has wire_driver atom       │
                  │     installed so its events emit envelopes       │
                  │                                                  │
                  │   imports agentm SDK (this IS the SDK process)   │
                  └──────────────────────────────────────────────────┘
                                       ▲
                ━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━
                │     chat_client peers (one process per platform)
                │
   ┌────────────┴──────┐  ┌─────────────────┐  ┌──────────────────┐
   │  agentm-feishu    │  │ agentm-terminal │  │  (future Slack)  │
   │   lark_oapi       │  │  textual / rich │  │                  │
   └───────────────────┘  └─────────────────┘  └──────────────────┘
```

**Two process kinds, no more.** A gateway process (one) plus one chat client process per platform you want to expose. The gateway holds every session in memory; chat clients are dumb adapters.

Process-launch contract:

| Process | Binary | Purpose |
|---|---|---|
| Gateway | `agentm gateway --bind <url>` | Long-lived. Holds all sessions. |
| Chat client | `agentm-feishu --connect <same-url>` / `agentm-terminal --connect <same-url>` | One per platform. |

Gateway persistent state (`wire-outbox.sqlite`, `wire-inbox.sqlite`,
`session_map.json`, and `schedules.json`) defaults to `$AGENTM_HOME/gateway`
for both the direct `agentm gateway` process and the local daemon/supervisor
path. `--state-dir` is the explicit override when an operator wants an
alternate location.

Deployment contract: do not hand-write service files. Use
`agentm gateway --cwd <workspace> --install-systemd`, which writes managed
user units for the gateway and Feishu peer, pins both to
`unix://%t/agentm/gw.sock`, and loads `<workspace>/.env`. Literal `/tmp/*.sock`
paths are fine for ad-hoc local development but are not the deployment path.

The standalone-prompt mode (`agentm prompt "..."`) still exists for non-chat usage — that's the existing CLI, untouched by this doc.

---

## 2. Wire envelope v2

Single-jump version bump from v1 to v2. v1 envelopes are **rejected** at hello with `error{code="unsupported_wire_version"}`. No translation layer.

### 2.1 Framing

Unchanged from v1: 4-byte big-endian length + N bytes UTF-8 JSON body. `wire/framing.py` carries over verbatim.

### 2.2 Envelope schema

```json
{
  "v": 2,
  "id": "<uuid hex, 12 chars>",
  "kind": "<one of §2.3>",
  "ts": 1748400000.42,
  "session_key": "<opaque string, chat-client computed>",
  "scenario": "<scenario name, optional, only on inbound>",
  "body": { ... }
}
```

**Mandatory on every envelope**: `v`, `id`, `kind`, `ts`.

**Conditional**:

| Field | Required when | Set by | Purpose |
|---|---|---|---|
| `session_key` | `kind=inbound` or `kind=outbound` | chat client computes (§3.4); daemon echoes back on outbound | Stable conversation identity |
| `scenario` | `kind=inbound` for a chat the gateway has not seen before | chat client (from its config) | Tells gateway which scenario to construct the new session in |

**`body`** is **kind-specific** and carries **no routing fields**. Anything routing-relevant lives on the envelope.

Compared to v1, the envelope is intentionally minimal: gone are `to`, `correlation_id`, `hops`, `root_session_key`, `session_id`. Their use cases:

* `to` was for cross-peer routing — only `chat_client` peers exist, addressing is by session_key.
* `correlation_id` / `root_session_key` / `hops` were for cross-process peer_send and a2a — same-process dict lookup needs none of them.
* `session_id` was a worker resume hint — sessions live in daemon memory; resume only happens on crash recovery, handled by ChatSessionMap internally.

Re-add any of these only when an actual feature needs them. v2 does not.

### 2.3 Kinds

Seven kinds. Half of v1's set.

**Peer → Gateway**

| kind | body shape | purpose |
|---|---|---|
| `hello` | `{peer_name, peer_version, capabilities, auth?}` | First message after connect. No `peer_kind` field — only `chat_client` peers exist. |
| `inbound` | §2.4 | A message the user sent on the chat platform. |
| `ack` | `{envelope_id}` | Confirms one server-originated envelope id. |
| `pong` | `{}` | Reply to `ping`. |

**Gateway → Peer**

| kind | body shape | purpose |
|---|---|---|
| `welcome` | `{server_version, wire_version, peer_id, session_resume[], capabilities?}` | Reply to `hello`. `capabilities` (optional) carries the gateway's static, session-independent view — `{models[], model, scenario, scenarios:[{name,source,manifest_path,description}], commands:[{name,kind,summary}], skills:[{name,summary}]}` — so a chat client can populate its model picker, scenario picker, command palette, and skill list *before* its first message creates a session. `model` and `scenario` here are gateway defaults; `/model` and `/scenario` maintain per-`session_key` overrides. Session-specific capabilities (the scenario's tools, in-session commands) still arrive later on the `session_ready` outbound. Sourced from `GatewayRuntime.describe_capabilities` via the `WireServer(capabilities_provider=…)` hook. |
| `outbound` | §2.5 | Render-and-send-this in the chat platform. |
| `error` | `{code, message, fatal}` | Protocol-level error. |
| `ping` | `{}` | Liveness probe. |

Gone from v1: `bye` (close-on-socket is fine), `delivery_batch` / `ack_batch` (pipelined push is enough at this scale), `control` (no worker to control), `subscribe` / `event` (no admin plane in v2).

### 2.4 `inbound` body shape

```json
{
  "channel": "feishu",
  "chat_id": "oc_xxx",
  "thread_id": "th_yyy",
  "sender_id": "ou_zzz",
  "sender_name": "Alice",
  "content": "<user text>",
  "button_value": "<approval-id>:approve",
  "raw": { ... }
}
```

* `channel` / `chat_id` / `thread_id` / `sender_id` / `sender_name` are the platform identifiers the chat client extracts from its SDK callbacks.
* `content` is the user's text (markdown allowed).
* `button_value` is set **only** on button-click inbounds; absent otherwise.
* `raw` is optional, platform-specific debug echo. Gateway ignores.

**`session_key`** is at envelope level (not in body) because it's daemon-side routing, not user-visible content.

### 2.5 `outbound` body shape

```json
{
  "channel": "feishu",
  "chat_id": "oc_xxx",
  "thread_id": "th_yyy",
  "content": "<assistant text or markdown>",
  "buttons": [
    {"label": "Approve", "value": "appr-deadbeef:approve", "style": "primary"},
    {"label": "Deny",    "value": "appr-deadbeef:deny",    "style": "danger"}
  ],
  "metadata": {
    "kind": "assistant_text" | "command_result" | "approval_request" | "approval_resolved" | "diagnostic_warning" | "diagnostic_error"
  }
}
```

`metadata.kind` is a typed discriminator: chat clients use it to pick a rendering style (plain text vs alert card vs interactive approval). `metadata` does not carry routing fields.

`command_result` is the output of a **control command** (`/status`, `/help`, `/context`, `/model`, `/new`, `/resume`, …) — distinct from `assistant_text` (the LLM speaking). A chat client renders it as a *system notice*: no agent author, no working spinner, and it is not part of the conversation transcript fed back to the model. It is durable (a control reply is a reliability-floor response, like `assistant_text`). The gateway emits it directly from the command router (`CommandContext.notice`), not through a session's `wire_driver`. Control-command *errors* still use `diagnostic_error`.

### 2.6 Delivery semantics

Outbound to a chat client flows through **one ordered channel per peer**: a single in-memory FIFO send queue (`send_queue.SendQueue`) drained by one sender task that is the peer's only socket writer. Enqueue order == event order == delivery order — so no frame overtakes another, and the receiver never has to reorder (no wire sequence number is needed in a single-process gateway).

**Ordering and reliability are orthogonal**, carried on the *same* queue:

- A **durable** frame (`assistant_text` / `approval_*` / `diagnostic_*`, see `wire/types.py:DURABLE_OUTBOUND_KINDS`) is first written to the per-peer SQLite outbox (the replay floor) and enqueued carrying its row id; the sender acks the row only after a successful write. On socket failure the row is left in the outbox (nacked for immediate re-lease) and replayed — in FIFO order, ahead of new live frames — when the peer reconnects (`WireServer._prefill_from_outbox`). This preserves **at-least-once**, idempotent on envelope `id`. A row's `attempts` advances once per reconnect-lease; a row that exceeds the attempt cap is dead-lettered at prefill rather than replayed forever. (No live retry loop exists — the sender exits on failure and waits for reconnect — so there is no backoff schedule to tune.)
- An **ephemeral** frame (`stream_text` / `tool_call` / `agent_end` …) carries no row id, is never persisted, and is **best-effort**: under backpressure the queue sheds its *oldest ephemeral* item (durable items are never dropped — they are already on disk), bounding memory without ever losing a durable record or reordering survivors.

> **Why one channel.** v1/earlier-v2 split delivery into two paths — durable via the outbox/async worker, ephemeral written straight to the socket. They reordered relative to each other (the async-outbox round-trip lags the direct write), so a later-produced `agent_end` reliably overtook an earlier durable `assistant_text`. Unifying onto one FIFO fixes ordering at the source; the outbox is demoted to a persistence side-channel for reconnect replay, not a delivery path.

Inbound from chat client to gateway is **at-most-once via ack-on-process** — gateway writes to inbox ledger, processes, then acks. Unchanged from v1.

---

## 3. Gateway components

Package `agentm`, all daemon machinery under `src/agentm/gateway/`. Imports `agentm.core` freely — this **is** the SDK process.

### 3.1 Component map

| Class | File | Job |
|---|---|---|
| `WireServer` | `gateway/server.py` | Accept connections, per-peer read/write loops, framing, hello/welcome handshake, auth, ping/pong, outbox delivery worker. |
| `Router` | `gateway/router.py` | Pure function: given an envelope + state (PeerRegistry, ChatSessionMap), decide what to do. Three cases (§3.2). |
| `SessionManager` | `gateway/session_manager.py` | Holds `sessions: dict[session_key, AgentSession]`. `get_or_create(session_key, scenario, inbound)` is the only public method that matters. |
| `ApprovalManager` | `gateway/approval.py` | Per-tool-call Future map; renders approval cards to the chat; resolves on button click. No cross-process plumbing — all same-process. |
| `CommandRouter` | `gateway/commands/router.py` | `/`-prefixed inbounds intercepted before they reach the session (§3.5). |
| `Outbox` (impl `SqliteOutbox`) | `gateway/outbox/sqlite.py` | Implements `OutboxStore` Protocol. |
| `Inbox` (impl `SqliteInbox`) | `gateway/outbox/sqlite.py` | Implements `InboxLog` Protocol. |
| `ChatSessionMap` | `gateway/chat_session_map.py` | SQLite-backed `session_key → session_id`. Long-lived, survives daemon restart. |
| `PeerRegistry` | `gateway/peer.py` | All connected chat clients + their hello capabilities. |
| `Authenticator` (Protocol) | `gateway/auth/__init__.py` | peercred / token / allow-all. |

**Banned**: nothing. The gateway IS the SDK process; the constraint that v1 had ("daemon doesn't know what an AgentSession is") is intentionally dropped.

### 3.2 Router

Three cases, exhaustive:

```python
def dispatch(env: Envelope, sender: Peer) -> RouterAction:
    if env.kind == "inbound":
        if env.body.get("button_value"):
            return RouterAction.RESOLVE_APPROVAL(env)
        if is_slash_command(env.body["content"]):
            return RouterAction.RUN_COMMAND(env)
        return RouterAction.PROMPT_SESSION(env)
    raise ProtocolError(f"unexpected sender kind for {env.kind}")
```

Single function, < 30 LoC, exhaustively testable.

### 3.3 SessionManager

```python
class SessionManager:
    def __init__(self, *, chat_map: ChatSessionMap, scenarios: ScenarioRegistry,
                 outbound_sink: Callable[[Envelope], Awaitable[None]]):
        self._sessions: dict[str, AgentSession] = {}
        self._chat_map = chat_map
        self._scenarios = scenarios
        self._outbound_sink = outbound_sink

    async def get_or_create(self, session_key: str, scenario: str,
                            inbound: InboundEnvelope) -> AgentSession:
        sess = self._sessions.get(session_key)
        if sess is not None:
            return sess

        # First inbound for this session_key in this process lifetime.
        # Check whether it's actually new vs a daemon-restart resume.
        prior_session_id = self._chat_map.get(session_key)
        if prior_session_id is not None:
            sess = await self._resume(prior_session_id, scenario)
        else:
            sess = await self._create_new(session_key, scenario)
            self._chat_map.set(session_key, sess.session_id)

        # Stamp the wire_driver atom so this session's events fan out
        # via outbound_sink, scoped to this session_key.
        sess.set_service("wire_outbound", self._outbound_sink)
        sess.set_service("session_key", session_key)
        sess.install_atom("wire_driver")

        self._sessions[session_key] = sess
        return sess

    async def shutdown_session(self, session_key: str) -> None:
        sess = self._sessions.pop(session_key, None)
        if sess is not None:
            await sess.shutdown()
        # Note: ChatSessionMap entry STAYS so /new-then-message resumes
        # transcript; only an explicit /end (CommandRouter §3.5) clears it.
```

`_create_new` and `_resume` differ only in whether they pass `session_id=...` to the scenario's session factory. Otherwise identical.

### 3.4 session_key — the mapping the user asked about

The fundamental mapping that distinguishes one chat conversation from another is `session_key → AgentSession`. The semantics:

* **`session_key` is computed by the chat client** based on its platform's notion of "conversation identity". The gateway treats it as opaque.
* **Default composition rule** (each chat client implements):
  | Surface | Composition |
  |---|---|
  | 1-on-1 DM | `<channel>:<chat_id>` (e.g. `feishu:p2p_oc_xxx`) |
  | Group chat, shared session | `<channel>:<chat_id>` |
  | Group chat, per-user sessions | `<channel>:<chat_id>:<sender_id>` |
  | Threaded conversation | `<channel>:<chat_id>:<thread_id>` |
* **Composition rule is chat-client config**, not envelope-level: each client has a `session_scope: chat | thread | user` setting that controls how it composes the key from its platform IDs.
* **The gateway does not parse** session_key — it's just a dict key and a SQLite row key.

This decouples "what counts as one conversation" (a chat-platform concern) from the gateway (which only cares about identity).

### 3.5 CommandRouter

Single concern: intercept `inbound` whose `body.content` starts with `/`.

| Command | Handling |
|---|---|
| `/help`, `/status` | Compose an outbound locally; never reach a session. |
| `/new` | `SessionManager.shutdown_session(session_key)`; leave ChatSessionMap intact; reply confirmation. Next message re-resumes from transcript. |
| `/end` | `SessionManager.shutdown_session(session_key)`; **also** clear `ChatSessionMap[session_key]`; reply confirmation. Next message starts a fresh session. |
| `/model [name]` | With no args, list configured model profiles and mark the current `session_key`'s active profile. With a name, validate the profile, store a per-`session_key` factory override, shut down the current session, clear the chat mapping, and let the next user message create a fresh session under the selected model. |
| `/scenario [name]` | With no args, list discoverable scenarios and mark the current `session_key`'s active scenario. With a name, validate via `validate_scenario(name)`, store a per-`session_key` scenario override, shut down the current session, clear the chat mapping, and let the next user message create a fresh session under the selected scenario. |
| `/schedule ...` | Manage durable gateway-owned scheduled prompts for the current `session_key`. Jobs are persisted in `schedules.json`, carry the route metadata needed to reconstruct an inbound, and fire by enqueueing a synthetic prompt through the same SessionManager path as external messages. This is distinct from the `monitor` atom's in-memory per-session wakeups. |
| `/skill:X`, `/<markdown_cmd>` | Look up handler, expand body, replace `env.body.content` with expanded text, mark `metadata.expanded_from = "/skill:X"`, fall through to `PROMPT_SESSION` as a normal inbound. |
| `/atom:install X`, `/atom:uninstall X` | `sess.install_atom(X)` / `sess.uninstall_atom(X)` directly. Reply confirmation. |
| Unknown `/foo` | Reply `diagnostic_error{"Unknown command: /foo"}`. Never propagate to LLM. |

Discovery (markdown / skill / atom command lists) lives in `gateway/commands/registry.py`. Handlers move from `contrib/channels/src/agentm_channels/commands/` to `src/agentm/gateway/commands/` in the rewrite.

### 3.6 ApprovalManager

```python
class ApprovalManager:
    def __init__(self, outbound_sink): ...
    
    async def request(self, *, session_key: str, sender_id: str,
                      tool_call: ToolCallEvent) -> bool:
        approval_id = f"appr-{uuid4().hex[:12]}"
        future: asyncio.Future[bool] = asyncio.Future()
        self._pending[approval_id] = (future, sender_id)
        await self._outbound_sink(self._render_card(approval_id, session_key, tool_call))
        return await asyncio.wait_for(future, timeout=APPROVAL_TIMEOUT_S)

    def resolve(self, button_value: str, clicker_sender_id: str) -> None:
        approval_id, decision = button_value.split(":")
        entry = self._pending.pop(approval_id, None)
        if entry is None:
            return  # stale, already resolved or timed out
        future, requester_sender_id = entry
        if clicker_sender_id != requester_sender_id:
            return  # identity mismatch, silently drop
        future.set_result(decision == "approve")
```

Same-process Future. No cross-peer rewriting (no worker), no `root_session_key` (no a2a). The wire_driver atom (§5) calls `request(...)` when it sees a `ToolCallEvent` that needs approval.

### 3.7 What the gateway does not do

* Spawn worker processes. There are none.
* Maintain a worker pool. Same.
* Hop counting. No a2a hops in v2.
* Cross-peer approval rewriting. Same-process now.

---

## 4. `wire_driver` atom

The new code. **One §11 atom**, ~150 LoC, replaces what would have been the entire `agentm-worker` package.

`src/agentm/extensions/builtin/wire_driver.py`:

```python
MANIFEST = ExtensionManifest(
    name="wire_driver",
    description="Translate AgentSession events into wire envelopes.",
)

def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    outbound_sink: OutboundSink = api.get_service("wire_outbound")
    session_key: str           = api.get_service("session_key")
    approval_mgr: ApprovalManager | None = api.get_service("approval_manager")

    async def emit(body_kind: str, content: str, **extra: Any) -> None:
        await outbound_sink(Envelope(
            v=2, id=new_id(), kind="outbound", ts=time.time(),
            session_key=session_key,
            body={"content": content, "metadata": {"kind": body_kind}, **extra},
        ))

    @api.on(AssistantMessageEvent)
    async def on_assistant(ev: AssistantMessageEvent) -> None:
        await emit("assistant_text", ev.message.text_content())

    @api.on(ToolCallEvent)
    async def on_tool_call(ev: ToolCallEvent) -> None:
        if approval_mgr is not None and approval_mgr.requires(ev.tool_name):
            ok = await approval_mgr.request(
                session_key=session_key,
                sender_id=api.get_service("current_sender_id"),
                tool_call=ev,
            )
            if not ok:
                ev.deny("user denied")

    @api.on(LogWarningEvent)
    async def on_warn(ev: LogWarningEvent) -> None:
        await emit("diagnostic_warning", ev.message)

    @api.on(LogErrorEvent)
    async def on_err(ev: LogErrorEvent) -> None:
        await emit("diagnostic_error", ev.message)
```

That's it. Translation glue, fully expressible inside the §11 atom contract. No new ExtensionAPI surface — just `set_service` / `get_service` / `@on`, all of which already exist.

**`peer_send` is removed, not ported** (was planned to move from `contrib/channels-clients/worker/src/agentm_worker/peer_send_atom.py` into builtin). It has no users — no scenario mounts it, no test covers it — and same-process delegation with a `wait_for_reply` future is semantically a sibling of the existing `sub_agent` atom (spawn a child unit, await its result), so a same-process port would have been a redundant second way to do what `sub_agent` already does. The old `agentm_worker/peer_send_atom.py` is deleted with the rest of the worker package. Cross-peer / agent-to-agent messaging is **out of scope** until a concrete need appears; if one does, resolve the `sub_agent` overlap first (one tool, not two) before reintroducing anything.

---

## 5. Chat client peers

Packages: `agentm-feishu`, `agentm-terminal` (and any future platform) at `contrib/gateway-peers/<name>/`. Same shape as today; updated for v2 envelope.

### 5.1 Components per chat client

| Class | Job |
|---|---|
| `WireClient` | (shared lib in `agentm`) Connect, framing, retry, ack pump. |
| `PlatformAdapter` | Receive platform events (lark_oapi message callback / stdin readline / Textual input event), wrap as `inbound` envelopes (computing `session_key` per §3.4), send via WireClient. |
| Outbound handler | Receive `outbound` envelopes from WireClient, render to platform UI (lark card / Rich panel / stdout JSON). |

A chat client implementer's checklist:

1. Provide platform credentials (`lark_app_id`, `slack_bot_token`, …) via config.
2. Map platform message events to `inbound` envelope body fields per §2.4.
3. Compute `session_key` from platform identifiers per §3.4 default rule (or override).
4. Render `outbound` content + buttons per platform UI conventions; surface `metadata.kind` distinction.
5. Round-trip button clicks as `inbound` envelopes carrying `body.button_value`.

The chat client knows nothing about: sessions, scenarios, approvals, commands, atoms. It is dumb.

---

## 6. Pluggability axes

Three Protocols, documented and complete. No accidental extension surfaces.

| Protocol | File | Default impl | Purpose |
|---|---|---|---|
| `OutboxStore` | `gateway/outbox/protocol.py` | `SqliteOutbox` | Per-peer at-least-once durable delivery (replay-on-reconnect side-channel). Minimal surface — `enqueue` (returns row id) / `lease` / `ack` / `nack` / `dead_letter` / `pending_count` / `close`. The v1 delivery-worker hint methods (`set_notifier` / `backoff_delay` / `next_retry_at_min`) were removed when the lease-poll worker was replaced by the unified send queue — a backend implements only what the sender actually calls. |
| `InboxLog` | `gateway/outbox/protocol.py` | `SqliteInbox` | Per-peer at-most-once ledger. |
| `Authenticator` | `gateway/auth/__init__.py` | `AllowAllAuthenticator`, `UnixPeerCredAuthenticator`, `TokenAuthenticator` | Hello-time identity check. |

`ServerTransport` / `ClientTransport` are also Protocols but **internal** to the wire layer (`gateway/transport/base.py`); operators don't swap them, new transports ship as subpackages.

Single entry-point group: `agentm.gateway.commands` (markdown/skill/atom command handlers).

---

## 7. What gets deleted

Hard list. The Phase-1 worker must delete these files outright.

### 7.1 Files

| Path (current) | Reason |
|---|---|
| `contrib/channels/src/agentm_channels/base.py` | `BaseChannel` — retired concept |
| `contrib/channels/src/agentm_channels/manager.py` | `ChannelManager` — retired |
| `contrib/channels/src/agentm_channels/registry.py` | Discovery for retired concept |
| `contrib/channels/src/agentm_channels/channels/` (subpackage) | `StubChannel` moves to test fixtures |
| `contrib/channels/src/agentm_channels/bus.py` | `MessageBus` — internal queue retired |
| `contrib/channels/src/agentm_channels/gateway.py` | `Gateway` class — concept retired |
| `contrib/channels/src/agentm_channels/wire_bridge.py` | Bridge between v0 manager and wire — both ends deleted |
| `contrib/channels/src/agentm_channels/session_bindings.py` | session→peer binding for worker peers; no worker peers exist |
| `contrib/channels/src/agentm_channels/worker_registry.py` | No worker peers |
| `contrib/channels-clients/worker/` (whole package) | Deleted entirely. `peer_send_atom` is **not** ported (§4 — no users, redundant with `sub_agent`). |

### 7.2 Code constructs

* `agentm_channels.channels` entry-point group declaration in `pyproject.toml`.
* `_WireChannel` synthetic class.
* `--inproc-worker` flag and its plumbing.
* `peer_kind` field on hello envelopes (only one kind exists).
* `--resume` as a CLI flag on any binary (resume is an internal-to-gateway mechanism, not user-facing).
* All `# type: ignore[attr-defined]` reaches into `_chat_map`, `_pending`, `_extension_api` — replaced by public accessors on the new classes.
* `body["_session_id_hint"]`, `body["channel"] == "_a2a"`, `body["correlation_id"]`, `body["root_session_key"]`, `body["hops"]` magic strings.

### 7.3 Tests

Per [[feedback_aggressive_test_pruning]]: target reduction from ~200+ to ~40 fail-stop tests.

Delete: every test for `BaseChannel`, `ChannelManager`, `_WireChannel`, `MessageBus`, `Gateway`, `WorkerRegistry`, `session_bindings`, agent-worker peer dispatch, cross-process peer_send, a2a hop limit, root_session_key forwarding.

Keep / write (target invariants):

| Invariant | Test |
|---|---|
| Wire envelope round-trip | encode→decode→encode produces identical bytes; v2-only |
| Outbox at-least-once + dedup | crash mid-delivery → reconnect re-delivers; dup envelope id is no-op |
| Inbox ack-on-process | crash mid-dispatch → restart re-processes unacked inbound |
| Router three-case dispatch | inbound-with-content / inbound-with-button_value / unknown → correct RouterAction |
| SessionManager get_or_create | first call creates + maps in ChatSessionMap; second call returns same instance; after restart with same session_key resumes by session_id |
| CommandRouter handler classes | /help local, /skill expansion + fall-through, /new shutdown without map clear, /end shutdown with map clear, unknown /foo rejected |
| ApprovalManager | request resolves on matching click; identity mismatch silently dropped; timeout returns False |
| wire_driver atom | AssistantMessageEvent → outbound envelope with correct shape; ToolCallEvent → approval request when policy says so |
| WireServer auth | unix peercred match/mismatch; token match/mismatch |
| Wire version negotiation | v1 hello → reject; v2 hello → welcome |

---

## 8. What gets renamed

### 8.1 Packages

| Current | New |
|---|---|
| `agentm-channels` (package `agentm_channels`) | **deleted**; contents move into `agentm` SDK under `src/agentm/gateway/` |
| `agentm-worker` (package `agentm_worker`) | **deleted**; `peer_send_atom` not ported (§4) |
| Entry-point group `agentm_channels.commands` | `agentm.gateway.commands` |

### 8.2 Directories

| Current | New |
|---|---|
| `contrib/channels/` | **deleted** (contents move into SDK) |
| `contrib/channels-clients/terminal/` | `contrib/gateway-peers/terminal/` |
| `contrib/channels-clients/feishu/` | `contrib/gateway-peers/feishu/` |
| `contrib/channels-clients/worker/` | **deleted** |
| `contrib/skills/feishu-cli/` (orphan) | `contrib/gateway-peers/feishu/src/agentm_feishu/skills/feishu-cli/` (feishu peer hands a skill_path to the gateway in its hello so CommandRouter discovers it) |

### 8.3 CLI surface

| Current | New |
|---|---|
| `agentm-gateway --bind ...` (separate binary) | `agentm gateway --bind ...` (subcommand of `agentm`) |
| local daemon management | `agentm daemon start/status/stop/restart/socket` |
| `agentm-worker --connect ...` | **deleted** |
| `agentm-feishu --connect ...` | unchanged; can use `agentm daemon socket` / `AGENTM_SOCKET` |
| `agentm-terminal --connect ...` | unchanged; defaults to the local daemon socket when `--connect` is omitted |

The `agentm` console script gains `gateway` and `daemon` subcommands alongside
the existing prompt and trace surfaces. `agentm gateway` is the foreground
server. `agentm daemon` manages the local reloadable supervisor used by
single-host clients. `agentm terminal` is a convenience wrapper around
`agentm daemon start` plus `agentm-terminal`.

### 8.4 CI / config touch points

* `pyproject.toml`: workspace members shrink (channels + worker packages removed); `tool.pytest.ini_options` ignores updated.
* `CLAUDE.md`: CI lint scope drops `contrib/channels/src/` and `contrib/channels-clients/worker/src/`.
* `.claude/index.yaml`: §11 below.
* All `from agentm_channels.X import Y` → `from agentm.gateway.X import Y` (then verify import works).
* `.claude/designs/runtime-context-atom.md`, `command-routing.md`, any other doc referencing the old paths.

---

## 9. What gets added

* `src/agentm/gateway/` — new subpackage holding everything from §3.
* `src/agentm/gateway/wire/types.py` — typed `InboundBody`, `OutboundBody` dataclasses (from the deleted `bus.py`).
* `src/agentm/extensions/builtin/wire_driver.py` — §4. (`peer_send` removed, not ported — see §4.)
* `src/agentm/cli/gateway.py` — `agentm gateway` subcommand glue.
* `src/agentm/gateway_daemon.py` / `src/agentm/cli_daemon.py` — shared local
  daemon paths, status, start/stop/restart/socket CLI.
* `src/agentm/gateway_supervisor.py` — local-development process supervisor
  used by `agentm daemon`: keeps a stable unix socket, starts the ordinary
  `agentm gateway` worker, and restarts that worker when watched source/config
  files change. This is not a distributed worker pool and does not execute
  sessions outside the single gateway process; it exists so code changes apply
  without manually restarting the TUI client.

No new third-party deps.

---

## 10. Migration

There isn't one. v1 and v0 are deleted in the same PR that ships v2. The repo at the end of the Phase-1 PR has **no v1 wire code, no v0 channel code, no worker package, no compat layer**.

This is acceptable because AgentM is pre-1.0, internal-only (single user / small team), and a compat layer would carry the very abstractions being deleted.

---

## 11. Concept-graph updates (`.claude/index.yaml`)

The live graph carries one entry for this design, `single_process_gateway`
(related to `pluggable_architecture`, `command_routing`,
`single_file_extension_contract`, `sub_agent_lifecycle`, `session_inbox`).
The superseded `gateway_channels` (v0) and `client_server_architecture` (v1)
concepts were removed when their designs were retired. `.claude/index.yaml`
is the source of truth for the current description — this doc does not
duplicate it.
