# Issue #87 — Builtin Atoms Consume ExtensionAPI C-1 Capabilities

## Scope

Rework five builtin atom files around the C-1 surfaces:

- `observability`: use `api.add_observer`, `ResourceWriter.current_version_for_path`, and `core.lib._to_jsonable`; document the `path` config placeholder.
- `sub_agent`: use `api.spawn_child_session(**kwargs)`, resolve `system_prompt` via `discover_builtin`, move default inherited extensions into scenario config, and use `core.lib._to_jsonable` for tool payloads.
- `artifact_store`: register the live store through `api.set_service("artifact_store", store)` and have tools resolve it through `api.get_service`.
- `micro_compact` / `llm_compaction`: rely on the shared JSON serializer instead of local ladders where applicable.

## C18 Decision

Keep `src/agentm/extensions/builtin/sub_agent.py` as a single atom file in this round.
The split threshold is **at least 1500 LOC**; issue #87 does not split the atom even though it is over 1000 LOC, preserving the §11 single-file builtin contract and avoiding an unrequested refactor.

## Verification Plan

- Grep gates from the issue acceptance criteria.
- `validate_atom_file` for `observability`, `sub_agent`, `artifact_store`, `micro_compact`, and `llm_compaction`.
- Regression tests for observability trace shape, sub-agent spawn kwargs/config behavior, and artifact-store per-session service isolation.
- `uv run ruff check src/`, `uv run mypy src/`, and `uv run pytest --tb=short`.
