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

## Eval-config environment variables

The YAML configs under `contrib/scenarios/rca/eval/` and
`contrib/scenarios/rca/eval/baseline/` use shell-style ``${VAR:-default}``
placeholders for paths and model identifiers so the same config is portable
across machines. The external `rca llm-eval` runner does **not** expand them
itself — pre-process the file with `envsubst` (or `gettext-base`) before
piping it in:

```bash
RCA_DATASET_ROOT=/abs/path/to/rca \
MODEL_NAME=Doubao-Seed-2.0-pro \
JUDGE_MODEL=Doubao-Seed-2.0-pro \
envsubst < contrib/scenarios/rca/eval/config.yaml \
  | uv run rca llm-eval run - -a agentm -l 1
```

Recognized variables:

| Variable            | Used in                          | Default                |
|---------------------|----------------------------------|------------------------|
| `RCA_DATASET_ROOT`  | `source_path`                    | (required, no default) |
| `MODEL_NAME`        | top-level `model_name`           | `Doubao-Seed-2.0-pro`  |
| `JUDGE_MODEL`       | `judge_model.model_provider.model` | `Doubao-Seed-2.0-pro` |

`RCA_DATASET_ROOT` has no default — set it explicitly or the resulting YAML
will contain an empty string and the runner will reject the config.
