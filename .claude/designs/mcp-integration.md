# Design: MCP Integration (Draft)

**Status**: DRAFT
**Created**: 2026-05-12
**Builds on**: [pluggable-architecture.md](pluggable-architecture.md), [extension-as-scenario.md](extension-as-scenario.md)
**Lives at**: `contrib/extensions/mcp_bridge/` (nested package, opt-in via `--extension`)

---

## 1. What MCP is, in AgentM's vocabulary

The Model Context Protocol exposes a remote process as a **bundle of tools** (plus prompts, resources, sampling — out of scope for v0). Each tool has a name, a JSON Schema, and a `tools/call` RPC.

In AgentM terms: MCP belongs on the **Tool Environment** axis (§3.2 of the pluggability contract). It is *not* a new pluggability axis. A single bridge atom takes responsibility for "talk MCP, surface results as `Tool` instances".

The bridge takes an **install-time snapshot** of each server's tool set and freezes it into the catalog like any other atom. MCP's runtime mutability (`notifications/tools/list_changed`) is intentionally ignored — see §3.

## 2. Non-goals

- **No new ABI.** The `Tool` protocol (`core/abi/tool.py`) already matches MCP's call shape: `name / description / parameters(JSON Schema) / async execute(args)`. Nothing in `core/` changes.
- **No MCP server side.** AgentM is the *client*. Exposing AgentM's own tools as an MCP server is a separate, later design.
- **No prompts / resources / sampling in v0.** Just tools. The other MCP primitives can be added as additional atoms (`mcp_prompts`, `mcp_resources`) without re-touching the bridge.
- **No `general_purpose` enrollment.** The default scenario stays MCP-free. Users opt in.
- **No runtime tool-set tracking.** Servers that push `tools/list_changed` mid-session are not honored; the user must restart the session to pick up the new set.

## 3. Pluggability decision: install-time snapshot

Two shapes were considered:

| Shape | Pros | Cons |
|---|---|---|
| **Install-time snapshot** *(chosen)* | Deterministic atom hash; clean evidence attribution; aligned with AgentM's freeze model; no post-freeze ABI gymnastics; simpler permission story. | Server-side tool-set changes require session restart. |
| Runtime discovery | Tracks `list_changed` live. | Catalog freeze can't fingerprint live tool set; needs separate schema-digest evidence dimension; opens unregister-ABI rabbit hole; complicates permission default. |

Snapshot is consistent with how every other atom in AgentM behaves: the atom's hash pins what it registered. Picking up new MCP tools requires the same restart cost as picking up a new builtin atom — a familiar, acceptable cost.

## 4. Atom shape

```
contrib/extensions/mcp_bridge/
├── __init__.py          # exports MANIFEST + install(api, config)
├── manifest.py          # ExtensionManifest definition
├── bridge.py            # install() body, lifecycle, tool registration
├── client.py            # MCP client (stdio/http/sse transports)
└── tool.py              # MCPTool(Tool) — adapts tools/call to ToolResult
```

Multi-file is permissible because this lives under `contrib/extensions/<name>/` as a **package-form atom** (mounted via `--extension contrib.extensions.mcp_bridge`), not a `builtin/` single-file atom. The §11 single-file contract applies to `extensions/builtin/`; package-form contrib atoms have their own boundary (no atom-to-atom imports, no `core.runtime.*`, no `core._internal`).

### 4.1 Config schema

```yaml
- name: contrib.extensions.mcp_bridge
  config:
    servers:
      - name: filesystem
        transport: stdio
        command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        env: {}
      - name: github
        transport: http
        url: "https://mcp.example.com/github"
        headers:
          Authorization: "${GITHUB_MCP_TOKEN}"
    naming: "mcp__{server}__{tool}"   # default; rendered into Tool.name
    timeout_seconds: 30
```

Per-server credentials read from env vars via `${...}` interpolation — secrets never sit in YAML.

### 4.2 Lifecycle

`async def install(api, config)`:

1. Construct an `MCPSessionManager(servers=...)`. For each server, open the transport, send `initialize`, store the live `ClientSession`. **If any server fails to connect, install fails loudly** — snapshot semantics demand a fully-known tool set before freeze.
2. For each server, call `tools/list` *once*. For each tool, build an `MCPTool` and call `api.register_tool(tool)`. The tool name is `mcp__<server>__<tool>` (collision rule: prefix disambiguates across servers; intra-server name collisions are impossible by spec).
3. Publish the manager under `api.set_service("mcp", manager)` so other atoms (later: `mcp_prompts`) can reuse the live clients.
4. Register a session-close hook (`api.on("session.closed", ...)` or equivalent) to gracefully shut down each transport.

Notably absent: no background task watches for `tools/list_changed`. If the bridge receives that notification it logs an `mcp.tools.list_changed_ignored` event and does nothing else.

### 4.3 `MCPTool.execute`

```python
async def execute(self, args, *, signal=None) -> ToolResult:
    try:
        result = await self._client.call_tool(self._remote_name, args, timeout=self._timeout)
    except MCPProtocolError as exc:
        return ToolResult(content=[TextContent(text=str(exc))], is_error=True)
    return ToolResult(
        content=_translate_content_blocks(result.content),
        is_error=result.is_error,
        extras={"mcp": {"server": self._server}},
    )
```

`signal` is honored by racing the RPC against `signal.wait()`. `extras["mcp"]` is intentionally minimal — atom hash already pins the schema version of every tool in the snapshot, so per-call schema digests are not needed.

## 5. Substrate touch points

Zero. The freeze model already does what we need:

- `core/abi/tool.py` — unchanged.
- `core/runtime/extension.py` — unchanged. `register_tool` is called only at install time, just like every other tool atom.
- `core/runtime/catalog/` — unchanged. The snapshot's tool list lands in the catalog by the existing mechanism.
- `core/runtime/atom_reloader.py` — unchanged. Reloading the bridge atom re-runs install, which re-snapshots; this is correct behavior.

`install()` being optionally async is already supported (`extension.py:633`). The only new dependency is the MCP client lib in `contrib/extensions/mcp_bridge/pyproject.toml` — it does not leak into the SDK.

## 6. Observability

Lifecycle events on the bus:

- `mcp.server.connected` / `mcp.server.disconnected`
- `mcp.tools.snapshotted` (per server, with tool count and names)
- `mcp.call.error` (transport-level, distinct from `is_error=True` in `ToolResult`)
- `mcp.tools.list_changed_ignored` (informational; fires if a server pushes the notification)

Per-call observability flows through the existing `tool_call` / `tool_result` events; the bridge does not add a parallel channel.

## 7. Permission integration

`mcp__*` tools are **deny-by-default**. A scenario that loads the bridge must also configure the `permission` atom to allow specific MCP tools or prefixes, e.g.:

```yaml
- name: permission
  config:
    allow:
      - "mcp__filesystem__*"
      - "mcp__github__create_issue"
```

Rationale: MCP servers are external processes whose behavior AgentM cannot audit. The bridge intentionally does not auto-allow its own registrations.

## 8. Failure modes

| Failure | Behavior |
|---|---|
| Any server unreachable at `install()` | `install` raises; session fails to start. This is the correct snapshot semantics — partial tool sets are worse than no session. |
| Server crashes mid-session | The corresponding tools start returning `is_error=True` with a transport-error payload. The bridge attempts reconnect with backoff; tools remain registered (their schema doesn't change). |
| Tool JSON Schema invalid at snapshot time | `install` raises with the offending server/tool. Forces the user to fix config rather than silently degrade. |
| Session cancellation (`signal` fires) | Tool implementations race `call_tool` against `signal.wait()`; on cancel, send `notifications/cancelled` upstream and return an error `ToolResult`. |
| Server pushes `tools/list_changed` | Logged and ignored. User must restart the session to pick up changes. |

## 9. Follow-ups

- Defer: `mcp_prompts` / `mcp_resources` atoms for the other MCP primitives.
- Defer: AgentM-as-MCP-server inverse direction.
- Test scope: one stub-MCP-server E2E that asserts the snapshot → catalog → tool call path. No per-transport unit tests.

## 10. Rollout

1. Land `contrib/extensions/mcp_bridge/` as draft; `--extension contrib.extensions.mcp_bridge` mounts it on top of any scenario.
2. Add `contrib/scenarios/mcp_demo/manifest.yaml` composing `general_purpose` + the bridge with a stdio filesystem server for smoke-testing.
3. After v0 stabilises, revisit prompts/resources atoms and the inverse direction.
