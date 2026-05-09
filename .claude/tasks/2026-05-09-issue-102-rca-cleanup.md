# Issue 102 — RCA scenario cleanup

## Plan note

Vendor contract policy choice: `contrib.extensions.rcabench_contract` keeps RCA
scenario startup resilient, but missing `rcabench_platform.v3.sdk.evaluation.v2`
no longer fails silently. On import failure it emits a `DiagnosticEvent` with
`level="warning"` and injects an explicit `<contract status="unavailable" ... />`
placeholder into the system prompt so trajectory/observability output records
why the official contract is absent.

## Design changes

- Shared the rcabench contract prompt through one contrib atom used by both RCA
  scenario manifests.
- Added typed `ResolveSubagentEvent` and moved RCA persona resolution and prompt
  hooks to `*.CHANNEL` constants.
- Let `sub_agent.available_inherited_extensions.<name>` omit config; the child
  inherits the already loaded parent atom config by manifest name.
- Removed scenario-local lazy wrappers and mounted `agentm_rca.tools.*` directly.
- Moved RCA SQL row/token limits to package defaults with rationale comments;
  scenario manifests only override `exclude`.

## Verification

- `uv run ruff check <changed-files>`
- `uv run mypy <changed-files>`
- `uv run pytest --tb=short tests/unit/extensions/test_issue_102_rca_cleanup.py contrib/scenarios/rca/tests/test_rca_orchestrator_setup.py tests/integration/test_sub_agent_lifecycle.py tests/integration/test_sub_agent_budgets.py`
