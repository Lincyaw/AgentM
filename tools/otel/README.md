# Local OTel collector for AgentM

`agentm.extensions.builtin.otel_tracing` ships real OTLP spans for every
session, turn, LLM request, tool call, handler invocation, and bus dispatch.
This directory gives you a one-command collector + Jaeger UI to receive them.

## Start

```bash
docker compose -f tools/otel/docker-compose.yaml up -d
uv sync --extra otel
uv run agentm "hello"             # default scenario mounts otel_tracing
open http://localhost:16686       # Jaeger UI, service = "agentm"
```

Stop with `docker compose -f tools/otel/docker-compose.yaml down`.

## What you get

Span tree per session (see `src/agentm/extensions/builtin/otel_tracing.py`):

```
agentm.session
├── agentm.extension.install:<module>
├── agentm.atom.reload:<name>
├── agentm.api.register:*
├── agentm.event:<channel>
│   └── agentm.handler:<channel>
└── agentm.turn
    ├── agentm.llm.request          (GenAI semconv attributes)
    └── agentm.tool.execute         (args + result preview, bounded)
```

`stream_delta` channels are excluded by default (one span per LLM token
would flood any collector); re-enable via the atom's `exclude_channels`
config.

## Cross-session linkage

Sub-agent + cognitive-audit child sessions reuse the same `trace_id`
as the parent. The child's session-root span carries
`agentm.parent_session_id`; the OTel parent context is wired so Jaeger
nests them under the parent's session span automatically.

## Config knobs (all optional)

| Env var | Atom config | Default |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `endpoint` | `http://localhost:4317` (gRPC) |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `protocol` | `grpc` (or `http/protobuf`) |
| `OTEL_EXPORTER_OTLP_HEADERS` | `headers` | none |
| — | `service_name` | `agentm` |
| — | `insecure` | `true` |
| — | `include_event_spans` | `true` |
| — | `exclude_channels` | `{"stream_delta"}` |

Send to a remote collector (Grafana Cloud / Honeycomb / etc.) by setting
`OTEL_EXPORTER_OTLP_ENDPOINT` + `OTEL_EXPORTER_OTLP_HEADERS` — the
docker-compose here is for local dev only.

## Useful Jaeger queries

* `gen_ai.request.model = "<your-model>"` — every LLM call.
* `agentm.tool.name = "submit_verdict"` — only cognitive-audit verdicts.
* tag `agentm.root_session_id = <hex>` — every span in one agent tree
  (parent + sub-agents + audit children).
* span name `agentm.handler:diagnostic` — every `_record_failure` emission
  from llmharness (silent-fail no longer silent).
