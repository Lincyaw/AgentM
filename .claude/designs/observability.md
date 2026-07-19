# Observability

## Contract

| Concern | Decision |
| --- | --- |
| Source of truth | Exactly one selected `TrajectoryStore` owns complete resumable session and turn state. |
| Diagnostic plane | `SessionTelemetry` emits logs and spans independently of trajectory persistence. |
| Recovery | Resume, fork, cache, and compaction never read OTLP or ClickHouse observability tables. |
| Correlation | Both planes carry stable session, root, parent, and turn ids where applicable. |
| Privacy | Prompt-bearing diagnostic payloads are redacted by default; observability is not a hidden trajectory replica. |
| Failure | A configured trajectory-store failure is fatal to the turn commit. Telemetry delivery failure cannot be treated as a successful trajectory commit. |

## Ports

| Port | Responsibility | Implementations |
| --- | --- | --- |
| `TrajectoryStore` | Atomic authoritative session/turn persistence | Memory, JSONL, PostgreSQL |
| `TrajectoryQueryStore` | Read complete sessions and turns from the selected trajectory backend | `TrajectoryStoreQueryAdapter` |
| `SessionTelemetry` | Atom-facing diagnostic emission | OTel SDK implementation installed by the observability atom |
| `ObservabilityQueryStore` | Read diagnostic events and spans | Local OTLP JSONL, collector-managed ClickHouse |
| `TraceQueryStore` | Present both read planes without merging their storage | `CompositeTraceQueryStore` delegates to one trajectory query source and one observability query source |

## Storage

| Backend | Stored data | Consistency role |
| --- | --- | --- |
| JSONL or PostgreSQL | Full `SessionMeta`, `Turn`, and configured node/head state | Authoritative, selected once per session |
| Local OTLP JSONL | Session-scoped diagnostic logs and spans | Optional diagnostics |
| ClickHouse `otel_logs` / `otel_traces` | Collector-managed diagnostic logs and spans | Optional, eventually visible query backend |

No AgentM `sessions` or `turns` tables are created in ClickHouse. A deployment
that independently builds an analytics warehouse from PostgreSQL is an
external data product, not an SDK fallback or a second persistence contract.

## Export Modes

| Mode | Behavior |
| --- | --- |
| `local_file` | Write the local OTLP JSONL diagnostic file. |
| `otlp` | Export diagnostics only to the configured collector. |
| `auto` | Use OTLP when the collector is reachable; otherwise use local diagnostic JSONL. |

The `auto` decision changes only the diagnostic sink. It never changes the
selected trajectory backend.
