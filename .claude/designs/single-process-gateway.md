# Design: Single-Process Gateway (channels v2)

**Status**: ACCEPTED
**Created**: 2026-05-28
**Supersedes**: [`historical/client-server-architecture.md`](historical/client-server-architecture.md) (channels v1) and [`historical/gateway-channels.md`](historical/gateway-channels.md) (channels v0). Both now live in `designs/historical/`.

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
                  │   Outbox + Inbox + ChatSessionMap                │
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
| Gateway | `agentm gateway --bind unix:///tmp/gw.sock` | Long-lived. Holds all sessions. |
| Chat client | `agentm-feishu --connect …` / `agentm-terminal --connect …` | One per platform. |

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
| `welcome` | `{server_version, wire_version, peer_id, session_resume[]}` | Reply to `hello`. |
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
    "kind": "assistant_text" | "approval_request" | "approval_resolved" | "diagnostic_warning" | "diagnostic_error"
  }
}
```

`metadata.kind` is a typed discriminator: chat clients use it to pick a rendering style (plain text vs alert card vs interactive approval). `metadata` does not carry routing fields.

### 2.6 Delivery semantics

Outbound from gateway to chat client is **at-least-once via durable outbox** — per-peer SQLite-backed queue, deliver-and-ack, retry with exponential backoff, idempotent on envelope `id`. Unchanged from v1's outbox layer (§4.5 of `client-server-architecture.md`), which works fine and is one of the surviving good parts.

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

The new code. **One §11 atom**, ~80 LoC, replaces what would have been the entire `agentm-worker` package.

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
| `OutboxStore` | `gateway/outbox/protocol.py` | `SqliteOutbox` | Per-peer at-least-once durable delivery. All performance hint methods (`set_notifier`, `backoff_delay`, `next_retry_at_min`) declared on the Protocol with default implementations — no `getattr` probing. |
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
| `agentm-worker --connect ...` | **deleted** |
| `agentm-feishu --connect ...` | unchanged |
| `agentm-terminal --connect ...` | unchanged |

The `agentm` console script gains `gateway` as a subcommand alongside the existing `prompt` and `trace`.

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
* `scripts/agentm-all-in-one` (optional) — convenience shell wrapper that `popen`s gateway + chosen chat client in one command, for single-user installs. Not a daemon mode — just a process supervisor.

No new third-party deps.

---

## 10. Migration

There isn't one. v1 and v0 are deleted in the same PR that ships v2. The repo at the end of the Phase-1 PR has **no v1 wire code, no v0 channel code, no worker package, no compat layer**.

This is acceptable because AgentM is pre-1.0, internal-only (single user / small team), and a compat layer would carry the very abstractions being deleted.

---

## 11. Concept-graph updates (`.claude/index.yaml`)

Two existing entries flipped:

* `gateway_channels` — already `historical`; no change needed, but add `superseded_by: single_process_gateway`.
* `client_server_architecture` → `status: historical`, `superseded_by: single_process_gateway`.

One new entry:

```yaml
  single_process_gateway:
    description: "channels v2. Single-process gateway: one agentm gateway daemon holds all sessions in memory and serves all chat-client peers. Wire protocol exists only for chat-client vendor SDK isolation (lark_oapi, Textual). No separate worker process — earlier daemon/worker split carried v1 abstractions whose isolation/distribution properties were not actually needed at this codebase's scale. Daemon imports agentm.core directly. Sessions live as dict[session_key, AgentSession] in daemon memory; persistence via ChatSessionMap (session_key → session_id) for crash recovery only. session_key is chat-client-computed (channel:chat_id default, optionally thread_id/sender_id), opaque to gateway. wire_driver atom (~80 LoC, §11 contract) is the only new code: translates session events to outbound envelopes via api.set_service / api.get_service / @api.on. peer_send atom moves from contrib/.../worker/ to builtin, rewritten for same-process dict lookup. sub_agent works unchanged (child sessions in same dict). Three pluggability Protocols: OutboxStore, InboxLog, Authenticator — all with default impls, no hidden optional-method surfaces. Seven wire kinds (down from 11): hello/welcome, inbound/outbound, ack, ping/pong, error. Envelope is minimal: v, id, kind, ts, session_key, scenario, body; routing primitives (to/correlation_id/hops/root_session_key/session_id) all deleted because there's no cross-process routing. No worker pool, no spawn-on-demand, no inactivity timeout, no --resume normal path, no --inproc-worker mode. Approval: same-process Future map in ApprovalManager; identity check against original requester's sender_id; chat-client renders card. Commands: /help, /status synthesised locally; /new shuts down session keeping ChatSessionMap (next message resumes from transcript); /end shuts down + clears map (fresh session); /skill:X and /<markdown> expanded server-side and fall through to session; /atom:install/uninstall mutates session atom registry directly. CLI: agentm gateway subcommand alongside existing agentm prompt and agentm trace; agentm-feishu and agentm-terminal stay as separate binaries (vendor SDK isolation only); agentm-worker deleted. Single-jump rewrite — no compat with v0 or v1; v1 hello → error{unsupported_wire_version}. Fail-stop test set is ~40 tests covering wire round-trip, outbox/inbox semantics, Router three-case dispatch, SessionManager get_or_create + resume, CommandRouter handler classes, ApprovalManager identity check + timeout, wire_driver atom translation, auth, wire version negotiation."
    design: "designs/single-process-gateway.md"
    replaces: [client_server_architecture, gateway_channels]
    related_concepts: [pluggable_architecture, command_routing, single_file_extension_contract, sub_agent_lifecycle, session_inbox]
    plans: ["plans/2026-05-28-single-process-gateway.md"]
    tasks: []
```

(Note: the design file path in the index entry above is `single-process-gateway.md`. This doc was initially named `daemon-as-router.md` while the design was being negotiated; the Phase-1 worker may either keep that filename or rename to `single-process-gateway.md` and update the index reference. The filename is a cosmetic decision the worker can make.)

---

## 12. Implementation contract for the Phase-1 worker

The Phase-1 dev-worker is dispatched against this design as its complete specification. The worker:

1. **Reads this doc** as the authoritative contract.
2. **Operates in a worktree** branched from latest `main`.
3. **Lands one PR** containing the entire rewrite (no incremental landing).
4. **Commits incrementally inside the worktree** for review readability. Suggested commit sequence:
   1. delete v0 + v1 dead code (`base.py`, `manager.py`, `_WireChannel`, etc.); tests fail loudly
   2. move surviving channel infrastructure (wire/, transport/, auth/, outbox/, commands/) into `src/agentm/gateway/`
   3. delete `agentm-channels` and `agentm-worker` packages (pyproject + workspace + CLI entry points); add `agentm gateway` subcommand
   4. write `Router` + `SessionManager` + `ApprovalManager` (CommandRouter + Outbox carry over with minor rename)
   5. write `wire_driver` atom (`peer_send_atom` is dropped, not ported — §4)
   6. update chat clients (terminal, feishu) for v2 envelope and new package paths
   7. prune tests + add fail-stop tests per §7.3
   8. update design docs (this one moves to PROPOSED→ACCEPTED, gateway-channels + client-server-architecture move to historical/, index.yaml updated)
5. **Gates landing on**:
   * `uv run ruff check` clean across the whole repo
   * `uv run mypy` clean across the whole repo (per existing per-workspace configs)
   * `uv run pytest --tb=short` passes
   * E2E smoke the worker self-invents: two-process boot (`agentm gateway` + a stub chat client driver), one inbound → one outbound round-trip, one approval round-trip, one `/help` command round-trip. Worker shows the trace.
6. **Does not** self-review; does not merge; returns the branch + commit SHAs + test summary for human review.

A Phase-2 worker, if needed, handles polish items (`chat_id_prefix` → `channel_name` rename in feishu config, `ws_patch.py` upstream issue, socket buffer constant naming, envelope id `uuid4`, `_install_lark_log_filters` Handler fix, `_assert_handler` test extraction). Most of these are subsumed by Phase 1's rewrite; only platform-specific polish remains.

---

## 13. Open questions

None. All forks resolved.
