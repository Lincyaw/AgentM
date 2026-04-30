# Task: Phase 2.0c — Single-file extension contract scaffolding

**Plan**: [2026-04-30-phase2-parallel-extensions.md](../plans/2026-04-30-phase2-parallel-extensions.md)
**Design**: [extension-as-scenario.md](../designs/extension-as-scenario.md) §11
**Agent**: implementer (sonnet) → reviewer (opus)
**Status**: COMPLETED 2026-04-30

## Why

Future agents will edit AgentM extensions on their own. The fewer files an
extension touches, and the more its contract is mechanically verifiable,
the smaller the agent's exploration space — and the smaller the chance a
self-edit breaks the harness.

This task pre-builds the §11 single-file contract so every Phase 2 atom
that lands afterwards is auto-discovered, manifest-declared, and lint-gated
from day one.

## Deliverables

### 0c.1 — `agentm.extensions` public surface

`src/agentm/extensions/__init__.py` exports:

- `ExtensionManifest` (frozen dataclass: `name`, `description`, `registers`,
  `config_schema`, `requires`, `conflicts`)
- `parse_register_tag(tag) -> (kind, id)` — single source of truth for the
  `<kind>:<id>` grammar
- `VALID_REGISTER_KINDS` — `{tool, event, command, provider, renderer}`

Atom authors only ever import from this module. No private helpers leak.

### 0c.2 — Auto-discovery (`extensions.discover`)

`discover_builtin() -> dict[str, BuiltinEntry]` walks `extensions/builtin/`,
imports every module that does not start with `_` and is not a subpackage,
and returns a name → entry map. Memoized at process scope; tests call
`reset_cache()` between mutations.

Subpackages and missing manifests are **errors**, not silent skips —
discovery is part of the contract.

### 0c.3 — Validator (`extensions.validate`)

`validate_builtin() -> list[ValidationIssue]` runs all eight §11.4 checks:

1. No subpackages under `builtin/`.
2. `install(api, config)` exists and is a 2-arg callable.
3. `MANIFEST` exists and is an `ExtensionManifest` (enforced by discover).
4. `MANIFEST.name` matches module stem (enforced by discover).
5. Imports are within the §11.1 allow-list (AST scan, not runtime).
6. Every `MANIFEST.registers` tag parses as `<kind>:<id>`.
7. `requires` / `conflicts` reference known atom names; no self-conflict.
8. `config_schema` is `None` or a dict with top-level `type`/`properties`.

Each issue carries the rule number (e.g. `"11.4.5-import"`) so an agent
reading the failure can route the fix to the right line of the design doc.

### 0c.4 — Test gate

`tests/unit/extensions/test_extension_contract.py::test_builtin_catalog_passes_section_11_contract`
asserts `validate_builtin() == []`. **Every Phase 2 PR runs this** —
adding an atom that violates §11 fails CI / pytest with the specific rule
violated, before any human review.

`tests/unit/extensions/test_manifest_helpers.py` covers the public surface
(tag parser, manifest defaults).

## Verification

```bash
uv run ruff check src/agentm/extensions/ tests/unit/extensions/
uv run mypy src/agentm/extensions/
uv run pytest tests/unit/extensions/ -q
```

All clean. Catalog is currently empty (no atoms landed yet), so the
contract test passes vacuously — that is intentional. As atoms land in
Groups A / B / C / D1, this test becomes the gate.

## HARD constraints

- **No atom-to-atom imports**: an extension MUST NOT import from another
  `agentm.extensions.builtin.*` module. Cross-extension dependencies go
  through events on the bus (`api.events`) or capabilities (`api.session.*`).
- **No `agentm.harness.session` import** from atoms: extensions never
  reach inside the orchestrator. They consume `ExtensionAPI` only.
- **Manifest is the contract, not the docstring**: agents reading or
  generating an atom rely on `MANIFEST` to know what it is. Keep manifest
  fields accurate or the validator will catch the drift.

## Group-task template addendum (locks for A/B/C/D1)

Every group's task file MUST require:
- one new file at `src/agentm/extensions/builtin/<name>.py`
- a module-level `MANIFEST: ExtensionManifest`
- imports restricted to the §11.1 allow-list
- one new test file at `tests/unit/extensions/builtin/<name>/test_*.py`
- the §11 contract test still green after the atom lands

If a group's atoms collectively require a shared helper, the helper goes
into `agentm.harness.events` (if it is event-shaped) or stays inlined per
file. **Do not introduce a `_shared.py` under `builtin/`.**
