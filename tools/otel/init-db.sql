-- Materialized columns + indexes for fast session-level queries.
-- Run once after the OTEL collector creates the default tables.
-- Idempotent — safe to re-run.

-- otel_logs: session_id extracted from LogAttributes map
ALTER TABLE otel.otel_logs
  ADD COLUMN IF NOT EXISTS _session_id String
  MATERIALIZED LogAttributes['agentm.session.id'];

ALTER TABLE otel.otel_logs
  ADD INDEX IF NOT EXISTS idx_session_id _session_id
  TYPE bloom_filter GRANULARITY 4;

-- otel_traces: session_id extracted from SpanAttributes map
ALTER TABLE otel.otel_traces
  ADD COLUMN IF NOT EXISTS _session_id String
  MATERIALIZED SpanAttributes['agentm.session.id'];

ALTER TABLE otel.otel_traces
  ADD INDEX IF NOT EXISTS idx_session_id _session_id
  TYPE bloom_filter GRANULARITY 4;
