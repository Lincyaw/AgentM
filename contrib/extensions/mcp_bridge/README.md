# mcp_bridge

MCP (Model Context Protocol) client bridge for AgentM. Mount the atom and the
remote server's tools land in the AgentM catalog as
`mcp__<server>__<tool>` names.

```
agentm --extension contrib.extensions.mcp_bridge ...
```

See [`.claude/designs/mcp-integration.md`](../../../.claude/designs/mcp-integration.md)
for the full design (install-time snapshot semantics, the deny-by-default
permission story, what's intentionally out of scope for v0).
