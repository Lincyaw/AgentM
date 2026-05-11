# Design: Gateway / Client Process Split (channels v1)

**Status**: PROPOSED
**Created**: 2026-05-11
**Supersedes**: `gateway-channels.md` (which now describes v0 — the
in-process design shipped in PR #137).

This is a **design-only** document. Implementation lands in subsequent
PRs guided by `plans/2026-05-11-gateway-client-server.md`. Open
questions that need user sign-off are tagged `[OPEN]`.

---

## 1. Why split

The v0 design ships gateway and every channel (Feishu, terminal, stub)
as one Python process. That has worked, but it has structural
problems we've now hit or are about to hit:

1. **Crash isolation.** A bug in `lark_oapi` WS handling — connection
   reset, threadpool exhaustion, unhandled exception during a card
   render — currently takes down `Gateway` + every active session.
2. **Hot restart.** Restarting "the Feishu adapter" today means
   restarting the entire daemon. Every chat session is reset, every
   pending approval future is dropped.
3. **Auth surface.** The gateway daemon needs Lark `app_secret` only
   because Feishu lives in the same process. A pure operator who
   never wants to expose secrets to the gateway has no way to slot in
   a Feishu adapter she trusts.
4. **Testability.** Channel implementations and the gateway can be
   unit-tested separately, but **integration** testing today requires
   spinning up the whole stack. With a process boundary, an
   integration test client is one socket and a JSON dialect away.
5. **Multi-language clients.** Today every channel is Python because
   it has to be importable into the gateway process. Once the wire is
   a serialized protocol over IPC, the next Slack adapter can be a
   Go binary, the next WeChat adapter a Node script. The gateway
   doesn't care.

The pivot: **the gateway becomes a server**, channels become
**clients**, and the existing in-process `Bus` / `BaseChannel`
abstraction lives on as the *server-side* internal handler that
demultiplexes connected clients.

---

## 2. First principles

1. **Wire protocol is the contract.** Same status as the §11 atom
   contract or the `ExtensionAPI` Protocol. Versioned. Schema-locked.
   Breaking changes are semver-major.
2. **One transport family, multiple bind targets.** JSON over a
   length-framed stream socket. Unix socket by default for local
   deployments, TCP optional behind a feature flag for remote
   deployments. No HTTP, no gRPC in v1 — both pull in framework
   dependencies that the kernel does not need to take on.
3. **The gateway is a routing bus, not a session host.** Every
   process connecting to the gateway is a *peer*. Peers come in
   kinds — `chat_client` (Feishu, terminal, …), `agent_worker` (a
   process running the AgentM loop with real LLM calls),
   `control` (admin / observability). The wire is identical for all
   kinds; what differs is the `peer_kind` declared at hello and how
   the gateway routes messages addressed to or from that kind.
   Generalised from the "chat client" framing in earlier drafts
   because agent runtimes need to be distributable on the same
   wire (§7.5, §8 phase 5).
4. **Clients are dumb adapters.** A chat-side client speaks one chat
   platform ("speak lark_oapi to/from Feishu") and the gateway
   protocol. It owns no session state, no approval state, no command
   routing. If it crashes mid-conversation, the gateway carries on;
   the client reconnects and resumes. Agent-worker peers are the
   same shape — they connect, declare what scenarios/models they can
   handle, and the gateway dispatches sessions to them.
5. **No silent reach-arounds.** The gateway never opens a network
   connection to a chat platform. Every byte from the world reaches
   the gateway through a peer process speaking the wire. This is
   the same boundary as §11 — kernel never touches scenario code
   directly.
6. **Backwards compatibility is finite.** v0 in-process channels stay
   functional during the migration window but are tagged deprecated
   and removed after every shipping channel has a v1 client. No
   permanent dual-stack.

---

## 3. Topology

```
                                       ┌──────────────────────────────────────┐
                                       │  agentm-gateway (daemon, one host)   │
                                       │                                      │
                                       │   PeerRegistry  (every kind)         │
                                       │   ChatSessionMap                     │
                                       │   ApprovalBridge                     │
                                       │   CommandRouter                      │
                                       │   Router (by peer_id / capability)   │
                                       │   ┌─────────────┐                    │
                                       │   │ SocketServer│ ◄── unix or tcp
                                       │   └─────────────┘                    │
                                       └──────────────────────────────────────┘
                                                ▲                    ▲
                            ━━━━━━━━━━━━━━━━━━━━┛                    ┗━━━━━━━━━━━━━━━━━━━━━━━━━
                            │   chat side (peer_kind=chat_client)    │   agent side (peer_kind=agent_worker)
                            │                                        │
                ┌───────────┴─────────┐  ┌─────────────┐    ┌────────┴─────────┐  ┌─────────────────┐
                │ agentm-feishu       │  │ agentm-http │    │ agent-worker-A    │  │ agent-worker-B  │
                │  lark_oapi WS       │  │  webhooks   │    │  scenarios=[…]    │  │  scenarios=[…]  │
                │                     │  │             │    │  model=opus       │  │  model=sonnet   │
                └─────────────────────┘  └─────────────┘    │  host=gpu-pod-1   │  │  host=laptop    │
                            │                                └──────┬───────────┘  └──────┬──────────┘
                  Feishu open platform                              │ AgentSession × N      │ AgentSession × M
                                                                     │ (live LLM loop)       │
                                                                     │
                                                                     └─── agent_worker_C ⇄ agent_worker_D
                                                                          (a2a tool calls — §7.6)
```

The gateway runs on one host as a single process. **Every other
process — chat adapters and agent runtimes alike — is a peer.** They
connect from anywhere (same host, sibling pod, laptop on the WAN over
TLS-fronted TCP) and speak the same wire. The gateway routes between
them; it owns no LLM and no chat secret directly.

Three peer kinds today:

* `chat_client` — speaks a chat platform (Feishu, terminal, HTTP
  webhook). Maps user activity to/from `inbound`/`outbound`.
* `agent_worker` — runs the AgentM loop. Imports `agentm.harness`,
  holds `AgentSession`s, calls real LLM providers. Declares
  capabilities at hello: which scenarios it can serve, which models
  it has credentials for, max concurrent sessions.
* `control` — admin/observability. Subscribes to gateway events,
  introspects routing state, can force-shutdown peers. Not on the
  v1 critical path; reserved in the protocol.

The wire stays the same envelope for all three. The only kind-aware
logic in the server is **routing policy**:

* `chat_client` inbound → dispatched to an `agent_worker` peer that
  matches the route's `(scenario, model)` constraint.
* `agent_worker` outbound → routed to the originating `chat_client`
  (via the session_key map).
* Either kind can send to a `peer://<id>` address — that's the a2a
  primitive (§7.6).

---

## 4. Wire protocol v0

### 4.1 Framing

Length-prefixed JSON. Each message:

```
4-byte big-endian length     N
N bytes UTF-8 JSON body
```

Why not line-delimited (`\n`-terminated) JSON like the in-process
JSON terminal mode? Two reasons:

* Embedded newlines in message content (markdown, code blocks) would
  need escaping; explicit length avoids the question entirely.
* Length-prefixed framing trivially supports binary payloads when v2
  needs them (media uploads, audio); line-delimited would mean
  reframing later.

### 4.2 Message envelope

Every JSON body wraps:

```json
{
  "v": "v0",
  "id": "client-msg-12",
  "kind": "<kind>",
  "ts": "2026-05-11T14:32:18Z",
  "body": { ... }
}
```

* `v` — protocol version. Servers reject unknown majors.
* `id` — sender-assigned, opaque. Used to correlate replies/acks.
* `kind` — discriminator (one of the kinds in §4.3).
* `ts` — ISO 8601 UTC, advisory; servers use their own clock for
  ordering when authoritative.
* `body` — kind-specific payload.

### 4.3 Kinds

The kind set is **peer-symmetric** — both chat clients and agent
workers send `inbound` and receive `outbound`. The difference is
semantic (chat clients translate to/from a chat platform; agent
workers translate to/from an `AgentSession`), not protocol-shape.

**Peer → Server**

| kind | Purpose |
|---|---|
| `hello` | First message. Carries `peer_kind` (`chat_client` / `agent_worker` / `control`), `peer_name`, version, capabilities, optional auth. Server replies `welcome` or `error`. |
| `inbound` | A message the peer wants the gateway to route. Body shape unchanged from v0 `InboundMessage` (sender_id, chat_id, content, media, metadata, session_key_override, button_value). New optional `to` field on the envelope (§4.3.1) lets a peer specify a destination peer for a2a; absence means "router policy decides". |
| `presence` | Advisory: "user is typing", "agent is thinking", "worker drained". Server may relay to interested subscribers. |
| `ack` | Confirms receipt of a server-originated message id. |
| `bye` | Graceful shutdown. Server flushes pending outbound for this peer, then closes. |
| `subscribe` | (control / agent-worker) Subscribe to a routing event stream — peer connect/disconnect, session migrations, approval gates. |

**Server → Peer**

| kind | Purpose |
|---|---|
| `welcome` | Reply to `hello`. Carries server version, negotiated wire version, the peer's assigned `peer_id`, optional `session_resume` list (§4.4). |
| `outbound` | Render-and-send-this. For `chat_client` peers, body is `OutboundMessage` to render in the chat. For `agent_worker` peers, body is the same shape — what the worker's local code paths would have received as inbound for the session. The wire kind is `outbound` either way because "what the gateway is sending to the peer" is the semantic. |
| `error` | Protocol-level error (bad hello, auth failure, malformed payload, unknown routing target). Code + message + whether the connection survives. |
| `ping` | Liveness check. Peer replies `pong`. |
| `event` | Streaming routing event for `subscribe`d peers — peer-up / peer-down / session-migrated. |

`pong` is the symmetric reply to `ping`.

### 4.3.1 Addressing

Every `inbound`/`outbound` envelope may carry an optional `to`
address:

* `chat://<channel_name>/<chat_id>` — deliver to the chat client
  servicing that chat (e.g. `chat://feishu/oc_xxx`).
* `session://<session_id>` — deliver to whichever peer currently owns
  this session. Used by the gateway internally; peers don't usually
  set this.
* `peer://<peer_id>` — deliver directly to a named peer. The receiver
  sees an `inbound` whose `metadata.from_peer = <sender_peer_id>`.
  This is the a2a primitive.
* `kind://<peer_kind>` — broadcast to every peer of that kind. Used
  by control plane operations ("drain all agent_workers"). Gated
  behind `control` permission.

If `to` is absent, the gateway falls back to the v0 routing policy:
chat inbound → owning agent worker; agent outbound → originating
chat client. This means everything in PR #137 keeps working without
the new fields touched.

### 4.4 hello / welcome handshake

**Chat client hello:**

```json
{
  "v":"v0", "id":"h1", "kind":"hello", "ts":"...",
  "body":{
    "peer_kind":"chat_client",
    "peer_name":"feishu",
    "peer_version":"0.1.0",
    "wire_versions":["v0"],
    "auth": {"method":"token","token":"..."} ,
    "capabilities":["streaming","buttons","markdown"]
  }
}
```

**Agent worker hello:**

```json
{
  "v":"v0", "id":"h2", "kind":"hello", "ts":"...",
  "body":{
    "peer_kind":"agent_worker",
    "peer_name":"worker-gpu-1",
    "peer_version":"0.2.0",
    "wire_versions":["v0"],
    "auth": {"method":"token","token":"..."} ,
    "capabilities":{
      "scenarios":["general_purpose","rca","feishu_chat"],
      "models":["claude-opus-4-7","claude-sonnet-4-6"],
      "max_concurrent_sessions": 8,
      "supports_a2a": true
    }
  }
}
```

**Welcome:**

```json
{
  "v":"v0", "id":"w1", "kind":"welcome", "ts":"...", "in_reply_to":"h1",
  "body":{
    "server_version":"0.2.0",
    "wire_version":"v0",
    "peer_id":"feishu-a8f3e2",
    "session_resume": ["feishu:c123","feishu:c456"]
  }
}
```

* `peer_id` is gateway-assigned, stable for the connection lifetime
  (and across reconnects when the peer auth matches a previously
  seen peer). It's the routing address other peers use to reach this
  one via `peer://<peer_id>`.
* `session_resume` lets a reconnecting peer know which routes the
  server still believes belong to it — important for crash recovery
  (§7).
* Capability shape is open-ended per `peer_kind`. The server stores
  them verbatim and matches on them during routing.

### 4.5 Delivery semantics

* Server → Client `outbound` is **at-least-once**. The server retains
  unacked outbound for a small bounded window (default 100 messages
  per client) and replays them after reconnect.
* Client → Server `inbound` is **at-most-once** within a session — a
  duplicate `inbound.id` from the same client is dropped after
  logging. Clients responsible for not re-sending after a confirmed
  ack.
* Out-of-order tolerated: the server reorders nothing; each `outbound`
  carries enough metadata (chat_id, turn marker) for the client to
  render in receipt order.

### 4.6 Why JSON, not protobuf / msgpack / cap'n proto

* JSON is debuggable with `tcpdump | jq`. For a chat substrate this
  matters more than throughput.
* No code generation; clients in any language can hand-write a
  client without a toolchain dependency.
* The current `OutboundMessage`/`Button`/etc dataclasses already
  serialize naturally to JSON. We're not inventing schemas.

Trade-off: ~5–10× the bandwidth of msgpack. For chat-rate traffic
(messages per second, not thousand-per-second) this is irrelevant.

---

## 5. Transport

### 5.1 Default: Unix socket

* Path: `/var/run/agentm/gateway.sock` (configurable).
* Permission: 0660 owned by `agentm:agentm` (or whatever group the
  operator chooses).
* Trust: peer-credential auth on Linux/macOS — the kernel tells the
  server the connecting client's uid/gid. If it matches the
  configured `peer_uid`/`peer_gid`, no token needed.

### 5.2 Optional: TCP

* Behind `--bind tcp://host:port`. Off by default.
* Token auth mandatory in TCP mode. Plain `Bearer` in the `hello`
  body; the user supplies the token via env var to both gateway
  (allow-list) and client (presented).
* No TLS termination in the gateway — operator is expected to put
  this behind nginx / stunnel / WireGuard. This is the standard
  Unix-tool decomposition; SMTP/Postfix is the reference model. If
  TLS-in-gateway becomes a real requirement we'll add it explicitly.

### 5.3 Why not HTTP/WebSocket?

* Adds a framework dependency the kernel does not currently take.
* Browser-driven clients are not on the v1 roadmap (a v2 web client
  would need WebSocket; we can add a WebSocket-on-the-same-socket
  upgrade in v2 without breaking v1).
* Length-prefixed JSON is one screen of code per language.

### 5.4 Why not gRPC

* Pulls in protobuf and a runtime per language.
* The whole point of the v1 wire is human-debuggable. gRPC works
  against that.
* If someone wants to write a gRPC adapter on top of our v0 wire,
  they can — but the gateway speaks JSON.

---

## 6. Auth and identity

### 6.1 Client identity (who is connecting)

Three options, all supported:

1. **Unix peer-cred** (default for `unix://`). Kernel provides the
   client's uid; matches against `peer_uid` allow-list.
2. **Token**. Bearer in `hello.auth.token`. Required for TCP.
   Optional for Unix when peer-cred is in use (defense in depth).
3. **mTLS via stunnel**. Operator concern, gateway doesn't see it.

### 6.2 Channel identity (what platform are we)

`hello.body.client_name` is the channel name (`feishu`, `slack`,
`terminal`, …). Combined with the gateway's `clients.<name>` config:

```yaml
clients:
  feishu:
    auth: { method: peer_uid, allow: [1000] }
    can_send_as: ["feishu"]      # channel namespaces this client may
                                  # set on InboundMessage.channel
  terminal:
    auth: { method: token, allow: ["${TERM_TOKEN}"] }
    can_send_as: ["terminal"]
```

A client connecting and trying to send `inbound.channel = "slack"`
when its config only allows `feishu` is rejected with `error`
(`code = forbidden_channel`).

### 6.3 User identity (who sent the message)

Unchanged from v0: `inbound.sender_id` is the platform user id. The
client is responsible for canonicalising. The gateway never
authenticates the *user*, only the *client*.

`[OPEN]` Do we want per-user ACL at the gateway? Today `allow_from`
is per-channel. v1 keeps that — but in the future we may want
"these sender_ids may use atom commands" etc. Defer to a follow-up.

---

## 7. Crash, reconnect, multi-client semantics

### 7.1 Gateway crash

Sessions are persisted by `ChatSessionMap` (already exists). On
restart, the gateway has no live clients; any client connecting after
gets a `welcome.session_resume` listing the routes the gateway still
remembers. The client decides whether to "claim" them by resending
pending outbound state (typing indicators etc.).

### 7.2 Client crash

The gateway notices the socket close. Per-route state stays alive
(session, pending approval futures), but **outbound destined for a
disconnected client is queued up to a bounded ring** (`outbound_buffer
= 100` per route). When the client reconnects with the same
`client_name` + identity, the gateway replays the queue.

If two clients try to register the same `client_name` simultaneously,
the second is rejected (`error code = duplicate_client`). This is
deliberate: chat platforms have one canonical adapter at a time. If
the operator wants HA, they front the gateway with a leader-elect
sidecar.

### 7.3 Multi-client to the same chat

`[OPEN]` This is the hardest semantic question. Two reasonable
behaviours:

**(A) Single-writer per chat (recommended)**: at any moment, exactly
one client process is the adapter for a given `(channel, chat_id)`.
Trivial routing: outbound goes to whichever client connected with
`client_name == route.channel`. This is what we get for free with
"one client per channel name".

**(B) Pool of adapters**: multiple clients may register the same
`client_name` and the gateway round-robins or sticky-routes among
them. Enables horizontal scaling but explodes the consistency story.

Recommendation: **ship (A)**. If/when (B) becomes a real requirement,
we add a sticky-route field to `hello` ("I am `feishu` adapter
instance `aZ`, route shards 0–127 to me"). Not v1.

### 7.4 Approval round-trip across the wire

A `ToolCallEvent` causes the gateway's `ApprovalBridge` to publish
one `outbound` to the chat's client with `kind = approval_request`
and two typed `Button`s. The button-value encoding is unchanged from
v0 (`<approval_id>:<approve|deny>`).

The user clicks. The client (Feishu adapter) receives the card
callback, sends an `inbound` with `button_value = "<...>:approve"`.
The gateway's existing `Gateway._dispatch` button-value branch picks
it up and resolves the approval future.

Identity check still applies: the bridge stores `sender_id` of the
requester at request time; a click from a different `sender_id` is
ignored.

**Network-partition concern.** If the client connection drops between
sending the approval card and receiving the click, the click is
delayed until reconnect+replay. Pending approval future times out as
usual; user sees "approval timed out" the same as if they sat on
their hands. The gateway never sees the click → it's a deny. This is
acceptable failure semantics.

### 7.5 Distributed agent workers

An `agent_worker` peer is a process running the AgentM loop. It
holds `AgentSession` instances locally and calls real LLM
providers. It connects to the gateway *out* — typically over Unix
socket for same-host, TCP-via-stunnel for off-host.

**Session ownership.** Sessions live in the worker, not the
gateway. The gateway records `session_key → owning peer_id` in an
extended `ChatSessionMap` and acts as a router; the worker holds
the conversation history, runs `prompt()`, and emits assistant
text. Worker death takes its sessions with it (unless persistence
is configured at the worker level — orthogonal to this design).

**Dispatch.** When a `chat_client` inbound arrives, the gateway:

1. Looks up the session_key in the map. If a worker owns it and
   the worker is connected → forward the `inbound` to that worker.
2. If no worker owns it (new session) → pick a worker whose
   capabilities match the session's `(scenario, model)` constraint.
   Selection policy is configurable; default is `least_loaded`
   among matching workers.
3. If no eligible worker is connected → buffer in the gateway's
   pending queue for up to `pending_inbound_ttl_seconds` (default
   30), reply with an `error` of code `no_capable_worker` to the
   chat client after the timeout.
4. If a worker accepts the dispatch, the gateway records ownership
   and forwards subsequent inbound for that session_key to the
   same worker (sticky session).

**Worker death / migration.** Two policies:

* **(A) Strict** (default): if the owning worker disconnects, the
  session is marked `lost` in the map. New inbound on that
  session_key returns `error: session_lost`. The chat client is
  expected to start fresh (or the user types `/new`). This matches
  the user's mental model of "the bot crashed, start over."
* **(B) Re-dispatch** (opt-in): when the worker disconnects and the
  session has a serializable transcript (worker-side persistence
  config), the gateway re-dispatches the session to another
  eligible worker with the transcript as resume context. Requires
  agreement on the resume protocol; deferred from v1 minimum.

**Capability matching.** Workers declare scenarios + models. The
gateway's routing config can additionally constrain — e.g. "only
worker peers from this allow-list serve `feishu_chat` sessions" —
to prevent a malicious worker from grabbing chat traffic it should
not see. Standard `allow_from`-style policy, extended to workers.

**Load balancing.** Default policy is `least_loaded` measured by
`active_session_count` reported via `presence`. Operators wanting
deterministic routing (one worker per chat, useful for debugging)
flip to `sticky_first_worker`. More-sophisticated policies (queue
depth, model cost, region affinity) are config-pluggable but not
shipped in v1.

### 7.6 Agent-to-agent messaging (a2a)

An agent_worker can send a message addressed to another peer (an
agent_worker, or even a chat client) by emitting an `outbound` with
`to: peer://<peer_id>` or `to: chat://<channel>/<chat_id>`. The
gateway routes it to the destination peer; the destination receives
it as an `inbound` whose `metadata.from_peer = <sender_peer_id>`.

**The tool form.** The atom `tool_peer_send` (ships in a later
phase, **not** v1 minimum) exposes this to the LLM as a tool:

```
tool name: peer_send
args:
  to_peer_id: "<peer-id>"            # or to_chat: "<channel>/<chat>"
  content: "<message>"
  wait_for_reply: true               # block until destination replies, or fire-and-forget
  timeout_seconds: 60
returns: { reply_content?, reply_peer_id, status }
```

When `wait_for_reply: true`, the calling worker pauses the calling
session until the destination peer emits an outbound back to the
caller. The gateway correlates by a `correlation_id` on the
envelope (added when the tool fires).

**Why route through the gateway, not direct peer-to-peer?**

* Centralised auth — the gateway already knows which peers may
  talk to which. Direct peer dialing means another auth surface.
* Centralised observability — every message between agents lands
  in the same observability stream the gateway already produces.
* Centralised approval — if agent A's `peer_send` should require
  human approval (because it's about to spend money on agent B's
  expensive model), `ApprovalBridge` is the existing gate. Direct
  peer-to-peer skips it.
* Routing remains a property of the cluster, not of individual
  peer identities — workers can come and go without other workers
  needing to know.

**A2A failure semantics.**

* Destination peer not connected: gateway returns `error:
  unknown_peer` to the sender. No queueing.
* Destination connected but rejects (`error: forbidden_target`):
  gateway returns the rejection to the sender; the sender's tool
  call gets a structured error.
* Destination drops mid-exchange (`wait_for_reply`): pending tool
  times out per `timeout_seconds`. Standard tool-timeout path.

**Loop prevention.** The gateway sets a `hops` counter in the
envelope, increments on each a2a forward, and refuses to forward
beyond `max_a2a_hops` (default 5). Stops infinite agent ping-pong
in one place.

`[OPEN]` Should the *sender* peer's identity (`from_peer`) be
forwarded verbatim, or rewritten by the gateway to hide identities
across security boundaries? Recommend verbatim within an
operator-managed cluster; rewrite-only behind a future "untrusted
worker" boundary.

### 7.7 Cross-peer approval

When an agent_worker A's `peer_send` to agent_worker B causes B's
tool call that needs human approval, who approves? The chat user
(the *root* of the dispatch chain) is the only available human.
Approval card must travel:

* `agent_worker B` → emits `ToolCallEvent` to gateway
* gateway sees the request came from a session whose root is
  `chat://feishu/oc_xxx`, fans approval card to that chat
* chat user clicks Approve
* gateway resolves B's approval future; B's tool proceeds

This requires the gateway to track the **root chat** of every
session, including a2a-spawned ones. Achieved by propagating a
`root_session_key` through the envelope when a session originates
from another session via a2a. One field on the inbound envelope,
filled by the gateway when the a2a tool fires.

Recommendation: hold approvals only at the root chat. An a2a child
session does not get its own approval surface; if there is no root
chat (worker-initiated session with no human in the loop), the
approval policy must be `always_allow` or `always_block` —
`require_approval` becomes an error at session creation.

---

## 8. Migration plan (channels v0 → v1 → peer mesh)

Six phases, each shippable independently. The first four are the
"v0 → v1 channels" path locked in earlier drafts; phases 5–6 are
the peer-mesh extension. Each phase is meant to be one PR.

### Phase 1 — Protocol + server, in-tree client lib

* Implement `agentm_channels.wire` (framing + envelope + kinds,
  *including* `peer_kind`, the optional `to` address, and the
  `correlation_id`+`hops`+`root_session_key` fields from §7.6/7.7
  — adding them in the wire from day one is much cheaper than
  retrofitting later).
* Implement `SocketServer` inside the existing gateway daemon.
* Implement `agentm_channels.client` Python lib (used by tests and
  by Phase 2/3/5 clients).
* `BaseChannel` stays. The gateway can run *either* in-process
  channels (v0) *or* a `SocketServer` (v1) selected by config.
* Wire test: in-tree client lib drives the running gateway through a
  Unix socket; existing 61 tests pass + new wire tests.

### Phase 2 — Terminal extracted to `agentm-terminal`

* New process: `agentm-terminal`. Wraps the wire client lib + the
  stdin/stdout JSON dialect already in PR #137. Same UX (`--format
  json|text`).
* Gateway no longer ships a `TerminalChannel` in-process — `--terminal`
  flag becomes "start `agentm-terminal` as a subprocess and connect
  to the gateway". Operators who used the old `--terminal` get a
  deprecation warning and a translation note.
* `[OPEN]` Should `--terminal` keep working as a single-process
  convenience (spawn both as a subprocess pair from one command)?
  Recommend yes; it's the smoke-test path.

### Phase 3 — Feishu extracted to `agentm-feishu`

* New process: `agentm-feishu`. Holds the `lark_oapi` dep. Reads
  `app_id`/`app_secret` from its own config. Connects to the
  gateway over Unix socket. The existing FeishuChannel code moves
  here almost verbatim; the only behavioural change is that
  outbound now arrives as `outbound` over a socket instead of as
  `OutboundMessage` from a Queue.
* Gateway pyproject no longer requires `lark_oapi` (it moves to
  `agentm-feishu`'s pyproject). Smaller deps for everyone not
  running Feishu.

### Phase 4 — Deprecate v0 in-process channels

* `BaseChannel` and the `channels/` directory get a "Use clients
  instead" deprecation note.
* StubChannel stays for now — it's the test fixture, in-process by
  necessity.
* `gateway.yaml` `channels:` key emits a deprecation warning on
  startup; supported until the next minor.

### Phase 5 — Agent workers as a peer kind

* The gateway today owns `AgentSession`. After this phase, the
  gateway only **routes** to a peer that holds the session.
* New process: `agentm-worker`. Wraps the wire client lib;
  `peer_kind = agent_worker`. On accepted dispatch (§7.5) it
  constructs an `AgentSession` locally, runs `prompt()`, fans
  `outbound` over the wire.
* Gateway gains: capability matching, sticky session routing,
  worker pool config, `least_loaded` selection, `presence`-driven
  load tracking.
* Default migration: `agentm-gateway --inproc-worker` keeps the
  in-process worker path so single-host deployments don't pay the
  process cost. Off-host deployments turn it off and connect one
  or more external `agentm-worker` processes.
* This is the phase where the gateway pyproject can drop the LLM
  provider deps as well — only `agentm-worker` needs anthropic /
  openai SDKs.

### Phase 6 — A2A tool calls

* Ships `tool_peer_send` atom (`MANIFEST.mountable_via_command=False`
  by default; operators opt in per-deployment for safety).
* Server enforces the loop guard (`max_a2a_hops`, default 5) and
  cross-peer approval forwarding (§7.7) — both happen at the
  envelope level, so no changes to existing approval-bridge code.
* Documents the "agent calls another agent through the bus" pattern
  as a first-class scenario; ships an example multi-agent scenario
  YAML.
* `[OPEN]` Should `from_peer` identity be exposed verbatim or
  rewritten across security boundaries? See §7.6.

---

## 9. What stays out of v1

* **Web client.** A WebSocket upgrade on the same socket would be
  natural in v2.
* **Per-user ACL beyond `allow_from`.** Mentioned in §6.3.
* **HA / multi-replica gateway.** §7.3 (B).
* **TLS in the gateway.** §5.2.
* **Streaming token deltas.** Already deferred at v0; no new
  obstacle here, but no implementation either.
* **Audio / image inline payloads.** Wire framing supports binary in
  v2 (length-prefixed makes it cheap); kinds + schema deferred.

---

## 10. Open questions for review

Each marked `[OPEN]` above. Summarised here for the design PR
checkout:

1. **Multi-client to same chat (§7.3).** Recommend (A) single-writer
   per channel name. Confirm or argue for (B).
2. **`--terminal` convenience wrapper (Phase 2).** Recommend keeping
   it as a "spawn pair" convenience. Confirm or remove.
3. **Per-user ACL (§6.3).** Defer to a follow-up — confirm OK.
4. **Token rotation / cred refresh** for TCP mode. Today `hello`
   carries the static token. Live rotation needs a kind like
   `reauth`. Defer to v2 unless explicitly needed.
5. **Worker death policy (§7.5).** Default (A) Strict — session is
   lost when its worker dies. (B) Re-dispatch via serializable
   transcript is opt-in and deferred from v1 minimum. Confirm
   Strict as default.
6. **A2A `from_peer` identity (§7.6).** Recommend verbatim within
   one operator-managed cluster; "rewrite across boundaries" hook
   reserved for future. Confirm verbatim default.
7. **Approval forwarding to root chat (§7.7).** Recommend approvals
   only at the root chat; if no root chat exists,
   `require_approval` is an error at session creation. Confirm.
8. **Wire fields added Day 1.** Phase 1 should land
   `peer_kind`/`to`/`correlation_id`/`hops`/`root_session_key` even
   though Phases 5–6 are when they get *used* — retrofitting the
   envelope later is cheap, but rolling out a v0→v0.1 mid-cluster
   is not. Confirm the "all envelope fields shipped in Phase 1"
   approach.

---

## 11. Cross-references

* `gateway-channels.md` — v0 design, in-process. Stays as-is during
  the migration window with a "superseded by" pointer at the top.
* `command-routing.md` — slash command layer. Lives **server-side**,
  unchanged. Clients never see commands; they just forward
  `inbound.content`. Server intercepts before LLM dispatch.
* `pluggable-architecture.md` — the IPC wire is at the same boundary
  layer as `BaseChannel` was. It does not replace any internal port
  in the SDK.
