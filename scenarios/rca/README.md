# agentm-rca

`agentm-rca` is a consumer of the AgentM SDK, not part of the SDK itself.

This package owns RCA-specific code and dependencies, including the DuckDB-backed
observability tools migrated from the earlier monolithic implementation.
The root `agentm` package stays free of RCA, observability, and DuckDB concerns.

## What lives here

- RCA-only tool implementations under `src/agentm_rca/`
- RCA-only dependencies such as `duckdb`, `pyarrow`, and `tiktoken`
- Scenario-local tests and fixtures under `tests/`

## What does not live here

- Changes to `src/agentm/` for RCA-specific behavior
- DuckDB or observability dependencies added to the SDK
- Tight coupling from the SDK back into RCA
