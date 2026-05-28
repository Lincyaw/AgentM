# agentm-terminal

Terminal (stdin/stdout) client process for the AgentM channels gateway.

Connects to an `agentm gateway --bind unix://…` over the v2 wire
protocol and renders outbound messages on stdout. Replaces the legacy
in-process `TerminalChannel` driver.

```sh
agentm gateway --bind unix:///tmp/gw.sock &
agentm-terminal --connect unix:///tmp/gw.sock
```

Output formats:

* `--format text` (default if stdout is a TTY) — human-friendly, ANSI
  colors, `agent ▸` prefix, numbered button rendering.
* `--format json` (default when piped) — one JSON object per line on
  stdout, suitable for scripts and other agents.

Exit codes (per `cli-design` rule group 3):

| Code | Meaning |
|------|---------|
| 0 | Clean shutdown (stdin EOF or `--no-input`) |
| 1 | Generic unexpected error |
| 2 | Argument / syntax error |
| 4 | Auth / handshake rejected by gateway |
| 6 | User interrupt (SIGINT) |
| 7 | Cannot connect (socket missing / refused) |
