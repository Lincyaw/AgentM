# Plan: Single-Process Gateway (channels v2)

**Date**: 2026-05-28
**Design**: [`designs/single-process-gateway.md`](../designs/single-process-gateway.md)
**Concept**: `single_process_gateway` (replaces `client_server_architecture`, `gateway_channels`)

## What landed

Single-PR rewrite of the channel/gateway layer. The daemon is the
gateway and the gateway is one process — no separate worker process.

### Deleted
- v0/v1 dead code: `bus.py` (MessageBus), `base.py` (BaseChannel),
  `manager.py` (ChannelManager), `registry.py`, `channels/` (StubChannel),
  `gateway.py` (Gateway class), `wire_bridge.py`, `session_bindings.py`,
  `worker_registry.py`, and their tests (a2a routing, gateway e2e,
  manager allow-from, wire-bridge security, catchup batching).
- Packages: `agentm-channels`, `agentm-worker`,
  `contrib/channels-clients/worker/`.

### Moved / collapsed into the SDK
- `agentm-channels` wire/transport/auth/outbox/commands +
  server/client/peer/chat_session_map -> `src/agentm/gateway/`.
- `peer_send_atom` -> `src/agentm/extensions/builtin/peer_send.py`
  (rewritten for same-process dict lookup).
- `contrib/channels-clients/{terminal,feishu}` ->
  `contrib/gateway-peers/{terminal,feishu}` (peers depend on `agentm`).
- `contrib/skills/feishu-cli/SKILL.md` -> inside the feishu peer.

### Added
- `src/agentm/gateway/`: `router.py`, `session_manager.py`,
  `approval.py` (ApprovalManager), `cli.py` (`agentm gateway` subcommand),
  `wire/types.py` (typed InboundBody/OutboundBody/Button).
- `src/agentm/extensions/builtin/wire_driver.py` (§11-clean §4 atom).
- `tests/unit/gateway/` — 43 fail-stop tests (§7.3 keep-list).

### Wire v2
- envelope: `v`, `id`, `kind`, `ts`, `session_key?`, `scenario?`, `body`.
  Routing primitives (`to`/`correlation_id`/`hops`/`root_session_key`/
  `session_id`/`peer_kind`) deleted.
- seven kinds: hello/welcome, inbound/outbound, ack, ping/pong, error.
- v1 hello -> `error{unsupported_wire_version}`.
- OutboxStore `set_notifier`/`backoff_delay`/`next_retry_at_min` promoted
  to first-class Protocol methods with defaults (no getattr probing).

## Out of scope (Phase 2, per design §12)
Feishu in-place card streaming (one card per turn for now), the various
platform-specific polish items.
