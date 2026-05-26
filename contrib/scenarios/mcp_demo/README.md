# `mcp_demo` scenario

A runnable demo that wires `contrib.extensions.mcp_bridge` to a real MCP
server (`mcp-server-fetch`, Python, via `uvx`) and exposes its single tool
to the agent as `mcp__fetch__fetch`.

See `.claude/designs/mcp-integration.md` for the design.

## Prerequisites

- `uvx` on PATH (ships with `uv`).
- An LLM provider configured through `~/.agentm/config.toml` (preferred for
  long-lived profiles) or env vars such as `AGENTM_PROVIDER` plus the
  matching API key.
- First run downloads `mcp-server-fetch` (~5s); subsequent runs are warm.

## Run the agent

```bash
uv run agentm --scenario mcp_demo "Fetch https://example.com and tell me what's on the page."
```

The agent will pick `mcp__fetch__fetch` for the URL retrieval and use its
native `read` / `bash` tools for anything else.

## Smoke test (no API key needed)

Verifies the bridge → server → tool-call path without an LLM:

```bash
uv run --package agentm-mcp-bridge python contrib/scenarios/mcp_demo/smoke.py
```

Expected output (the npm-banner pydantic warnings during cold-cache `uvx`
runs are harmless — the bridge ignores stdout junk before JSON-RPC begins):

```
>>> install(mcp_bridge) against uvx mcp-server-fetch...
>>> discovered 1 tool(s):
    - mcp__fetch__fetch: Fetches a URL from the internet ...
>>> invoking mcp__fetch__fetch on https://example.com ...
>>> is_error=False, content blocks=1
    first block (truncated 300): 'Contents of https://example.com/: ...'
>>> closed cleanly.
```

## Swapping servers

Edit `manifest.yaml` under `extensions[*].config.servers`. Two ready-to-go
alternatives are commented in the manifest header:

- `@modelcontextprotocol/server-everything` — official Anthropic test
  server exercising every MCP primitive (Node, via `npx`).
- `@modelcontextprotocol/server-filesystem` — sandboxed read/write
  inside a target directory (Node, via `npx`).

If you swap, also revise the URL or tool-name examples in the prompt you
send the agent; the bridge's `mcp__<server>__<tool>` namespace will
update automatically based on what the new server advertises.

## What this demo deliberately does *not* do

- **No permission gating.** The `permission` atom currently uses
  exact-name matching only, so an `mcp__*` allowlist would also have to
  enumerate every native AgentM tool to avoid breaking the session. The
  design's deny-by-default rule (§7) is something production scenarios
  will need to honor once the permission atom gains glob support.
- **No prompts / resources / sampling.** The bridge only surfaces MCP
  *tools*. Other primitives are deferred to later atoms
  (`mcp_prompts`, `mcp_resources`).
- **No runtime tool-set tracking.** `notifications/tools/list_changed`
  is logged and ignored. Restart the session to pick up server-side
  changes.
