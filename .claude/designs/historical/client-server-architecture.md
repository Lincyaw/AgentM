# Design: Gateway / Client Process Split (channels v1)

**Status**: HISTORICAL (superseded by channels v2)
**Created**: 2026-05-11
**Supersedes**: [`gateway-channels.md`](gateway-channels.md) (channels v0).
**Superseded by**: [`../single-process-gateway.md`](../single-process-gateway.md)
(channels v2 — single-process gateway, no separate worker process). The
v1 daemon/worker split documented here is retained for historical
reference; the abstractions it introduced (`agent_worker` peer kind,
SessionDriver/multiplexer worker, cross-process peer_send,
`root_session_key` approval rewriting) are deleted from the codebase in
the v2 rewrite.

This is a **design-only** document. Original implementation guidance
referenced `plans/2026-05-11-gateway-client-server.md` (which still
exists as historical record). Decisions recorded in §10 are now also
historical.

## 0. Design discipline

Standing constraint (applies to everything below): **smallest correct
surface + few well-defined extension points**. Easier to add a kind
than to remove one; easier to keep a default opinionated than to
delete a config knob that an operator has come to depend on. Concrete
applications:

* **Wire kinds**: only those v1 explicitly uses are in the protocol.
  Plausible-future kinds (broadcast addressing, control-plane event
  streams) are listed under §9 as v2 candidates.
* **Configuration**: opinionated defaults over knobs. Each knob in
  this doc names the concrete operator scenario that justifies it;
  knobs without one are deleted before Phase 1.
* **Extension points**: two Protocols (`OutboxStore`, `InboxLog`),
  one routing-policy function, one auth method enum. That's it.
  Not "everything is a plugin".

When future review wants to add a feature: first ask whether the
existing wire already supports the use case as a degenerate combination
of current kinds + fields. If yes, document the recipe; do not add
a kind.

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
* `agent_worker` — runs the AgentM loop. Imports `agentm.core`,
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
  "v": 1,
  "id": "client-msg-12",
  "kind": "<kind>",
  "ts": 1746974538.42,
  "body": { ... }
}
```

* `v` — integer wire-protocol version. Current: `1`. Servers reject
  unknown versions immediately. Integer (not `"v0"` string) so
  validation is a single equality check.
* `id` — sender-assigned, opaque. Used to correlate replies/acks.
* `kind` — discriminator (one of the kinds in §4.3).
* `ts` — float, epoch seconds. Advisory; servers use their own clock
  for ordering when authoritative. (Float over ISO string for the
  same reason as `v`: cheap validation, one canonical representation.)
* `body` — kind-specific payload.

### 4.3 Kinds

The kind set is **peer-symmetric** — both chat clients and agent
workers send `inbound` and receive `outbound`. The difference is
semantic (chat clients translate to/from a chat platform; agent
workers translate to/from an `AgentSession`), not protocol-shape.

**Peer → Server**

| kind | Purpose |
|---|---|
| `hello` | First message. Carries `peer_kind` (`chat_client` / `agent_worker`), `peer_name`, version, capabilities, optional auth. Server replies `welcome` or `error`. |
| `inbound` | A message the peer wants the gateway to route. Body shape unchanged from v0 `InboundMessage`. Optional `to` field on the envelope (§4.3.1) specifies a destination peer for a2a; absence means "router policy decides". |
| `ack` | Confirms receipt of one server-originated message id. |
| `ack_batch` | Confirms receipt of a `delivery_batch` (§4.5.3). |
| `bye` | Graceful shutdown. Server flushes pending outbound for this peer, then closes. |

**Server → Peer**

| kind | Purpose |
|---|---|
| `welcome` | Reply to `hello`. Carries server version, negotiated wire version, peer's assigned `peer_id`, optional `session_resume` list. |
| `outbound` | Render-and-send-this. For `chat_client` peers, body is `OutboundMessage`. For `agent_worker` peers, body is the same shape — what the worker's local code paths would have received as inbound. |
| `delivery_batch` | Catch-up envelope: multiple `outbound` items grouped for one round-trip (§4.5.3). |
| `error` | Protocol-level error. Code + message + whether the connection survives. |
| `ping` | Liveness check. Peer replies `pong`. |

`pong` is the symmetric reply to `ping`.

**What's deliberately not here**:

* No `subscribe` / `event` kinds — operators tail gateway logs for
  peer-up/down/dead-letter notifications in v1. If a real control
  plane consumer appears (live dashboard, autoscaler), we add the
  pair then.
* No `presence` advisory channel — "typing indicators" / "worker
  drained" are optimisations chat platforms or operators rarely
  block on; defer until needed.
* `control` peer kind reserved in the field but not implemented in
  v1. Admin actions go through the operator-facing CLI hitting the
  gateway's local API surface (the same one `--check` uses); they
  don't need a wire surface yet.

### 4.3.1 Addressing

Every `inbound`/`outbound` envelope may carry an optional `to`
address. Three schemes in v1, all point-to-point:

* `chat://<channel_name>/<chat_id>` — deliver to the chat client
  servicing that chat (e.g. `chat://feishu/oc_xxx`).
* `session://<session_id>` — deliver to whichever peer currently owns
  this session. Used by the gateway internally; peers rarely set this.
* `peer://<peer_id>` — deliver directly to a named peer. The receiver
  sees an `inbound` whose `metadata.from_peer = <sender_peer_id>`.
  This is the a2a primitive.

If `to` is absent, the gateway falls back to the v0 routing policy:
chat inbound → owning agent worker; agent outbound → originating
chat client. PR #137 behaviour is preserved.

Broadcast addressing (`kind://`, `*`) deliberately omitted from v1 —
no v1 feature uses it. If a future "drain all agent workers" admin
op appears, we add it then.

### 4.4 hello / welcome handshake

**Chat client hello:**

```json
{
  "v":1, "id":"h1", "kind":"hello", "ts":1746974538.42,
  "body":{
    "peer_kind":"chat_client",
    "peer_name":"feishu",
    "peer_version":"0.1.0",
    "wire_versions":[1],
    "auth": {"method":"token","token":"..."} ,
    "capabilities":["streaming","buttons","markdown"]
  }
}
```

**Agent worker hello:**

```json
{
  "v":1, "id":"h2", "kind":"hello", "ts":1746974538.42,
  "body":{
    "peer_kind":"agent_worker",
    "peer_name":"worker-gpu-1",
    "peer_version":"0.2.0",
    "wire_versions":[1],
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
  "v":1, "id":"w1", "kind":"welcome", "ts":1746974538.42, "in_reply_to":"h1",
  "body":{
    "server_version":"0.2.0",
    "wire_version":1,
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

### 4.5 Delivery semantics — durable outbox

The naive "in-memory bounded ring per peer" approach earlier drafts
described is too fragile. Real failure modes the gateway needs to
survive without losing messages:

* Gateway restart with peers connected and undelivered traffic in
  flight.
* Long peer outage (a Feishu adapter down for an hour while the
  agent finishes a long-running task and posts results).
* Network flap that drops the socket mid-write — the kernel buffer
  flushed and we don't know how much the peer received.
* Burst of outbound from a fast agent overwhelming a slow chat
  client and back-pressuring everything else.

The gateway therefore maintains a **durable outbox per peer**, not
an in-memory ring:

* **Storage.** SQLite at `<state_dir>/outbox.sqlite` by default.
  One row per pending message: `(peer_id, msg_id, kind, body,
  enqueued_at, last_attempt_at, attempts, status)`. WAL mode for
  concurrent reader (delivery worker) + writer (gateway).
* **Insert path.** Every outgoing envelope addressed to a peer is
  written to outbox first, then attempted over the socket. The
  insert is the source of truth; the socket write is a fast-path
  hint.
* **Delivery worker.** Per peer, a loop that pulls
  `status=pending OR status=retry` messages in `(enqueued_at,
  msg_id)` order, writes them on the socket, marks
  `status=in_flight`. On `ack` from the peer, mark `status=acked`
  and prune (configurable retention window so traces / replays can
  read recent ones).
* **Retry policy.** On write failure or missed ack within
  `ack_timeout` (default 30s): exponential backoff starting at 1s,
  capped at 60s, with full jitter. `attempts` increments. After
  `max_attempts` (default 12, ~12 minutes of retries), the message
  is moved to a **dead-letter** table and an `event` is emitted on
  the `subscribe` stream so operators see it.
* **At-least-once with idempotency.** A peer that receives a
  duplicate `msg_id` (it had already acked but the ack got lost
  before we recorded it) treats it as a no-op and re-acks.
  Idempotency-key field on `outbound` is the envelope `id`; peers
  dedup on `(server_peer_id, id)`.
* **Bounded write-burst observation.** When the outbox depth for a
  peer exceeds `peer_outbox_high_water` (default 1000), the delivery
  worker emits a `slow_consumer` log line and sets an observational
  `backpressure` flag on the peer session. v1 does **not** stop
  pulling from the outbox under back-pressure (doing so would
  deadlock the queue — the only way to drain it is to keep pulling).
  Real back-pressure (gating enqueue at the producer) is a v2
  concern, gated on whether a producer ever genuinely overruns
  consumers in practice.
* **Delivery-loop disconnect.** A single write failure
  (BrokenPipeError / ConnectionResetError) terminates the peer's
  delivery loop and tears down the peer session. The outbox row is
  nack'd with backoff so it redelivers on reconnect — retrying the
  same write against a dead writer would exhaust max_attempts in
  milliseconds. (Implementation: `_ConnectionLost` sentinel in
  `WireServer._delivery_loop`.)
* **Crash recovery.** Gateway restart re-opens SQLite, finds
  `status=in_flight OR pending`, resets them to `pending`, the
  delivery worker resumes from there. No message is lost; some may
  be re-delivered (handled by idempotency).

### 4.5.1 Inbound durability

Peer → gateway `inbound` is, by contrast, **at-most-once with
ack-on-process**:

* When a peer sends `inbound`, the gateway writes it to an inbox
  log (same SQLite) before processing.
* After successful processing (dispatched to a session or rejected
  with a structured error), the gateway emits `ack` with the
  envelope's `id`.
* A peer that does not see the ack should resend with the same `id`.
  The gateway dedupes on `(peer_id, id)` — duplicates are no-op
  acked.
* If processing fails (crash mid-dispatch), the inbox entry stays
  unacked. On restart the gateway replays from the inbox before
  accepting new traffic.

This is a small ledger, not a queue — the gateway processes inbound
synchronously most of the time. The inbox exists so that
*ack-on-process* is honest: we don't ack until we've actually done
the work, and on crash we redo unacked work.

### 4.5.2 Ordering

* Per `(peer_id, session_key)`: ordered. The delivery worker
  serializes by session.
* Across sessions on the same peer: best-effort FIFO from the
  outbox table; may be reordered after retry.
* Across peers: no global ordering claim.

### 4.5.3 Batch delivery (catch-up)

When a peer reconnects after an outage with many queued messages,
the gateway should not pay one round-trip per message — that turns
a 30-second outage into a slow drain.

**Two mechanisms compose:**

1. **Pipelined push.** The default. The delivery worker writes the
   next outbound to the socket without waiting for the previous
   ack to land. Bounded by `peer_in_flight_max` (default 64) — the
   peer may have up to N unacked messages at once. Acks arrive
   asynchronously and slide the window. This is the same pattern
   as HTTP/2 or AMQP prefetch; no new wire kind required, just
   permission to not block on ack.

2. **Explicit batch envelope.** New optional kind `delivery_batch`
   (server → peer) and `ack_batch` (peer → server) for catch-up
   scenarios where the gateway *wants* the peer to consider the
   batch as a unit:

   ```json
   {
     "v":1, "id":"b1", "kind":"delivery_batch", "ts":1746974538.42,
     "body":{
       "reason":"reconnect_catchup",
       "session_key":"feishu:c123",
       "items":[
         {"id":"m1","kind":"outbound","body":{...}},
         {"id":"m2","kind":"outbound","body":{...}},
         {"id":"m3","kind":"outbound","body":{...}}
       ]
     }
   }
   ```

   Peer responds with a single `ack_batch` (acking the batch_id +
   list of item ids it accepted) or, if it cannot accept all
   atomically, with individual `ack`s for the ones it processed
   and the server keeps the rest pending.

**When the server uses a batch vs pipelined push:**

| Condition | Mechanism |
|---|---|
| Normal steady-state (outbox depth < 16) | Pipelined push, msg-by-msg |
| Reconnect with > 1 message pending | One `delivery_batch` per session_key, up to `batch_max_items` (default 64) per batch |
| Outbox depth > `peer_outbox_high_water` (slow consumer) | Multiple batches in flight, throttled per `peer_in_flight_max` |
| Cross-session catch-up | One batch per `(peer_id, session_key)` — preserves the per-session ordering claim while allowing many sessions to drain in parallel |

**Why this matters for the user's scenario.** Agent worker
crashes; user sends 5 chat messages; agent worker reconnects. The
gateway has all 5 queued. The agent worker receives them as one
`delivery_batch` (because reason=reconnect_catchup, session_key
matches). The agent has the option to:

* Process each message as a separate turn (the dumb default — fine
  if "5 messages" semantically means "5 conversations").
* **Consolidate**: notice the batch shape, fold the 5 inbound into
  one user turn ("the user sent these 5 things while I was down,
  here's a combined prompt"). This is a *scenario-level* decision,
  exposed to atoms via an event (`InboundBatchEvent` with the
  whole list) before per-message dispatch. Atom default: pass
  through one-by-one. Worker scenarios that want to consolidate
  set a handler.

**Idempotency under batching.** Per-item `id` is still the dedup
key. A batch that gets re-delivered (no `ack_batch` seen) is
de-duped item by item at the peer; items already processed are
no-op'd.

**Why not just rely on TCP buffering?**

* The peer's local processing rate, not the socket buffer, is the
  bottleneck. Pipelined push hides the latency *but* still requires
  the peer to process N times. Batch envelope lets the peer take N
  messages and decide to do *less* work than N processings.
* `reason` field on `delivery_batch` lets the receiving scenario
  distinguish `reconnect_catchup` (consolidate) from
  `slow_consumer_drain` (process individually but in bulk).
* Operational: one `ack_batch` replacing N `ack`s reduces protocol
  chatter for what is otherwise an O(N) wire pattern.

### 4.5.3 Why not an external broker (Redis / NATS / RabbitMQ / Kafka)

Considered and rejected for v1, kept on the table for v2 plug-in:

* **Operational cost.** A second daemon to deploy / monitor / fail
  over, for a workload that fits in a single SQLite file for years
  of production traffic.
* **Latency.** Direct-socket dispatch is microseconds in the
  happy path; broker-mediated is milliseconds, and the happy path
  is what users feel.
* **Dependency surface.** Pulling in `redis-py` / `nats-py` /
  `aiokafka` per language for the future polyglot clients defeats
  the "any language, hand-rolled adapter" goal.
* **Scale.** The gateway is one process per operator deployment; a
  single SQLite WAL handles tens of thousands of messages/second
  comfortably. The number of chat messages an organisation
  generates is small relative to that.

**Decided** (§10): Should we ship a **pluggable outbox backend** interface in
Phase 1 (so an operator can swap SQLite for Redis Streams / NATS
JetStream later without modifying gateway core)? See §10
— it's two extra interfaces (`OutboxStore` + `InboxLog`) and a
factory; the default implementation is SQLite, alternatives are
contrib. Keeps the door open without committing to a broker dep.

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

**Decided** (§10): Do we want per-user ACL at the gateway? Today `allow_from`
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

**Decided** (§10): This is the hardest semantic question. Two reasonable
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

**A2A failure semantics — two delivery shapes.**

Fire-and-forget (`wait_for_reply: false`) goes through the durable
outbox (§4.5):

* Destination offline at send time: message persists in outbox,
  delivers on reconnect.
* Destination permanently gone (`max_attempts` exhausted, default
  ~12 min of retries): message lands in dead-letter, an `event` is
  emitted, the originating session sees a delayed
  `peer_send_failed` notification (the tool call has already
  returned `status=queued`).
* Idempotency: the envelope's `id` doubles as the dedup key —
  re-sending the same `id` is a no-op at the destination.

RPC-style (`wait_for_reply: true`) is **not** durably queued
because the calling session is blocked on the reply:

* Destination not connected at send time: tool returns
  `error: unknown_peer` immediately. No outbox insert.
* Destination connected but unresponsive: tool times out per
  `timeout_seconds` (default 60). The tool call gets a structured
  timeout error; the destination's pending work continues but its
  reply, if it ever arrives, is dropped (correlation_id no longer
  has a waiter).
* Destination crash mid-call: same as timeout — sender's tool sees
  timeout, no reply lands.

The distinction matches the user's mental model: "fire a message"
should survive a partition; "call another agent and wait" is RPC
with the usual RPC failure semantics.

**Loop prevention.** The gateway sets a `hops` counter in the
envelope, increments on each a2a forward, and refuses to forward
beyond `max_a2a_hops` (default 5). Stops infinite agent ping-pong
in one place.

**Decided** (§10): Should the *sender* peer's identity (`from_peer`) be
forwarded verbatim, or rewritten by the gateway to hide identities
across security boundaries? Resolved §10 #6 — verbatim within an
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
* **Decided** (§10): Should `--terminal` keep working as a single-process
  convenience (spawn both as a subprocess pair from one command)?
  Resolved §10 #2 — kept.

#### Phase 2 — landed (2026-05-11)

The two-process invocation pattern is the real shape after PR #142.
The gateway and each chat client run as separate processes communicating
over a Unix-domain socket using the v1 wire protocol (§4):

    # Terminal 1 — long-running daemon
    agentm-gateway --bind unix:///tmp/agentm/gw.sock

    # Terminal 2 — user-facing CLI
    agentm-terminal --connect unix:///tmp/agentm/gw.sock

The gateway no longer hosts in-process channels in this mode; each
chat platform is a peer process speaking the wire (§4.4 handshake).

The in-process `channels:` config block in `gateway.yaml` is
deprecated. Each platform's in-process channel emits a
`DeprecationWarning` on start (StubChannel exempted — it remains the
test fixture). Phase 4 removes the in-process path entirely.

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

#### Phase 4 — landed (2026-05-11)

* **Wire-bridge feature completeness.** `_WireChannel.send` now
  serializes `OutboundMessage.buttons` (typed shape — `label`/`value`/
  `style`) and passes `OutboundMessage.metadata` through verbatim, so
  the approval round-trip works end-to-end across the wire — no client
  needed a code change because both `agentm-terminal` and
  `agentm-feishu` already consumed `body["buttons"]`/`body["metadata"]`
  from Phase 2/3. `WireBridge.handle_inbound` reads
  `body["button_value"]` and forwards it as
  `InboundMessage.button_value`, closing the click→approval-future
  loop over the wire. Fail-stop test:
  `contrib/channels/tests/test_wire_bridge_buttons.py`.
* **`BaseChannel` / `ChannelManager` deprecation.**
  `BaseChannel.__init_subclass__` raises `DeprecationWarning` for any
  non-stub subclass, pointing operators to
  `agentm-gateway --bind unix:///path` + a platform client.
  `ChannelManager.__init__` raises a separate `DeprecationWarning`
  when any non-stub channel is constructed from the yaml-driven path
  (`inject_channel` is exempt — that's the wire bridge's seam). The
  CLI additionally prints `agentm-gateway: warn: in-process channels
  are deprecated …` to stderr at startup whenever
  `gateway.yaml.channels:` is non-empty, so the migration is visible
  to humans tailing logs.
* **Historical archive.** `gateway-channels.md` moves to
  `designs/historical/gateway-channels.md`; `index.yaml` flips
  `gateway_channels.status` to `historical` and adds
  `client_server_architecture.replaces: [gateway_channels]`.
* **Out of scope.** `BaseChannel` / `ChannelManager` are NOT removed;
  that's a future minor. `StubChannel` (and the synthetic
  `_WireChannel` from the bridge) opt out of the warning via
  `_is_stub_fixture = True`.

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

* Ships `tool_peer_send` atom (`MANIFEST.mountable_via_command=True`;
  operators opt in by listing the atom on the gateway's
  `commands.atoms.allow` so `/atom:install tool_peer_send` succeeds).
* Server enforces the loop guard (`max_a2a_hops`, default 10) and
  cross-peer approval forwarding (§7.7) — both happen at the
  envelope level, so no changes to existing approval-bridge code.
* Documents the "agent calls another agent through the bus" pattern
  as a first-class scenario; the worker README §A2A shows a
  researcher → coder → reviewer chain.
* **Decided** (§10): Should `from_peer` identity be exposed verbatim or
  rewritten across security boundaries? See §7.6.

#### Phase 6 — landed (2026-05-11)

Implementation summary; brief reference of what changed in code so a
reviewer can audit without re-reading every file. Detailed design
lives in §7.6 (correlation_id / hops) and §7.7 (cross-peer approval).

* **Gateway (`agentm_channels.wire_bridge.WireBridge`).** Splits
  inbound handling into a chat-originated path (Phase 5a, with the
  Phase 6 `root_session_key` tagging and hop bump added) and a
  worker-originated path that routes by `env.to`. The worker path
  rejects inbounds missing `root_session_key` with a
  `KIND_ERROR{reason="missing_root_session_key"}`. Both paths
  increment `hops` on the forwarded envelope and drop with
  `KIND_ERROR{reason="hop_limit_exceeded"}` once `hops > max_a2a_hops`
  (configurable via the new `--max-a2a-hops` CLI flag; default 10).
* **Approval override (§7.7).** Worker outbounds whose body carries
  `metadata.kind == "approval_request"` are rewritten to the chat
  channel/chat_id derived from `root_session_key` and enqueued to
  the chat client's peer outbox. The original
  `channel`/`chat_id`/`worker_peer_id` are preserved in
  `metadata.origin` for auditability. The user's button click flows
  back to the emitting worker via the existing synthetic-channel
  send path — no new wire kinds, no new envelope fields.
* **A2A reply routing.** Worker outbounds with both
  `correlation_id` and a worker peer_id in `to` are forwarded to
  that worker so its pending `peer_send` future resolves. Default
  (Phase 5a) chat-by-channel-name routing remains the fallback.
* **`tool_peer_send` atom.** Single-file §11 atom at
  `contrib/channels-clients/worker/src/agentm_worker/peer_send_atom.py`.
  Looks up the `peer_messaging` service from the session registry at
  install time; refuses to install when absent so misconfiguration
  surfaces immediately. The protocol exposes three methods:
  `new_correlation_id() -> str`, `send_peer(*, to, content,
  correlation_id) -> None`, `await_peer_reply(correlation_id,
  timeout_seconds) -> dict`. Mountable via `/atom:install` (gated by
  the gateway's existing `commands.atoms.allow` list).
* **Worker runner (`agentm_worker.runner.WorkerRunner`).**
  Implements `PeerMessaging` directly. Wraps the supplied
  `session_factory` so every freshly-created `AgentSession` gets the
  runner stamped in as its `peer_messaging` service via the host-side
  `AgentSession.set_service` accessor (added in this phase — see
  ExtensionAPI minimality note). Holds `correlation_id → Future`
  pending-reply map; `KIND_OUTBOUND` envelopes with a matching id
  resolve the future instead of feeding into a local session. Late
  replies are logged and dropped.
* **ExtensionAPI surface.** Single host-side accessor:
  `AgentSession.set_service(name, obj)` mirrors the atom-side
  `ExtensionAPI.set_service` (refuses to clobber). No new Protocols
  in core; the `peer_messaging` Protocol lives in the atom file and
  the runner duck-types it. This keeps the SDK surface unchanged
  except for one already-symmetric method.

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

## 10. Decisions recorded (2026-05-11 design review)

All ten open questions resolved toward the simplest viable answer
consistent with §0 design discipline. Phase 1 implementation works
from this list as the contract.

| # | Decision | Rationale |
|---|---|---|
| 1 | **Single-writer per channel name** (§7.3 option A) | Chat platforms are one-bot-per-credential; pool semantics fight the platform itself. HA = leader-elect two gateways, not pool one channel. |
| 2 | **Keep `--terminal` convenience** in Phase 2 | One-line smoke test path stays. ~50 LoC of subprocess management — cheap. |
| 3 | **Per-user ACL deferred** | `allow_from` per channel + approval-bridge identity check cover real threats today. Add when a concrete need appears, not before. |
| 4 | **TCP token rotation deferred** | Disconnect-reconnect with new token works. `reauth` kind only when ops actually demands it. |
| 5 | **Worker death = Strict** (§7.5 option A) | "Bot crashed, start over" matches user mental model. `redispatch` requires a transcript-resume sub-protocol; out of scope for v1. |
| 6 | **A2A `from_peer` verbatim** (§7.6) | Cluster-internal transparency is correct default. `rewrite_across_boundary` config knob reserved as future hook, not implemented. |
| 7 | **Approvals only at root chat** (§7.7) | Multiple approval surfaces = user confusion. No root chat + `require_approval` = config error at session start. Forces explicit operator choice. |
| 8 | **All envelope fields ship Day 1** | Mid-cluster wire bumps are expensive; ~100 LoC of upfront parsing is cheap. Phase 1 lands `peer_kind`/`to`/`correlation_id`/`hops`/`root_session_key` even where unused. |
| 9 | **Pluggable outbox: `OutboxStore` + `InboxLog` Protocols, SQLite default** | ~150 LoC interface, zero runtime overhead. Operators with existing JetStream / Redis can swap later. In-memory mock for tests. |
| 10 | **Outbox defaults** | `ack_timeout=30s`, `max_attempts=12` (~12 min total), `peer_outbox_high_water=1000`, `peer_in_flight_max=64`, `batch_max_items=64`, `acked_retention=7d`, `dead_letter_retention=30d`. Reviewed after 1-2 weeks of production data. |

Decisions worth flagging for **future review** (not blockers for
Phase 1):

* Decision 6 leaves a hook for `rewrite_across_boundary`; the wire
  field exists, the rewrite policy doesn't. When/if a hosted
  multi-tenant scenario appears, design the rewriter.
* Decision 1's "single writer per channel name" assumes a chat
  platform has one bot identity. If a future platform requires
  multiple shards per bot, revisit.
* Decision 5 (Strict worker death) becomes painful for long-running
  agent tasks. When that pressure appears, Phase 5+1 ships
  serializable-transcript opt-in.

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
