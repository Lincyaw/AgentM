# Task: tool-catalog-atom — Read-only catalog browser for the agent

**Date**: 2026-05-01
**Status**: PENDING
**Plan**: [self-mod-mvp](../plans/2026-05-01-self-mod-mvp.md)
**Design**: [evolution-substrate](../designs/evolution-substrate.md) §6.1 (subset)
**Assignee**: implementer
**Wave**: 3 (parallel)
**Size**: S
**Depends on**: [catalog-storage](2026-05-01-catalog-storage.md), [manifest-schema](2026-05-01-manifest-schema.md)

## Objective

Land a tier-1 single-file atom that exposes only the read API of the catalog (`list_versions`, `get_manifest`, `runs_for`) to the agent. **No write path** in MVP — `propose_change`, `compare`, `find_best`, `decisions_for` are deferred.

This atom is the proof that the autonomy layer can consume `agentm.core.catalog` through a narrow, stable surface without violating §11 layer purity.

## Inputs to read

- `evolution-substrate.md` §6.1 (full API; pick only the three MVP methods)
- `src/agentm/core/catalog/__init__.py` (the surface landed in `catalog-storage`)
- `src/agentm/extensions/builtin/tool_read.py` (an exemplar single-file atom — same shape we follow)
- `src/agentm/extensions/__init__.py` (post-`manifest-schema` `ExtensionManifest` with `tier`)

## Outputs

### New files

| Path | Purpose |
|---|---|
| `src/agentm/extensions/builtin/tool_catalog.py` | The atom |
| `tests/unit/extensions/builtin/tool_catalog/__init__.py` | Package marker |
| `tests/unit/extensions/builtin/tool_catalog/test_tool_catalog.py` | Unit tests |

### Modified files

| Path | Change |
|---|---|
| `src/agentm/extensions/scenarios/general_purpose.yaml` | Append one line: `- module: agentm.extensions.builtin.tool_catalog` (no config) |

## Concrete shape — the atom

```python
"""Read-only browser of the .agentm/catalog/ for the agent.

Tier 1, MVP scope. Phase 2 adds compare() / find_best() / propose_change()
as additional registered tools — keeping this atom in compliance with
§11's "single responsibility" rule means we keep the manifest tight here
(only browse-style tools).
"""

MANIFEST = ExtensionManifest(
    name="tool_catalog",
    description=(
        "Browse the catalog of atom versions and the traces they ran "
        "under. Read-only in MVP; compare()/propose_change() arrive in "
        "Phase 2."
    ),
    registers=(
        "tool:catalog_list_versions",
        "tool:catalog_get_manifest",
        "tool:catalog_runs_for",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "root": {"type": "string"},
        },
        "additionalProperties": False,
    },
    api_version=1,
    affects=(),
    tier=1,
)

def install(api, config):
    root = Path(config.get("root", ".agentm/catalog"))
    if not root.is_absolute():
        root = Path(api.cwd) / root

    api.register_tool(FunctionTool(
        name="catalog_list_versions",
        description="List all known versions of an atom in the catalog.",
        parameters={...},
        execute=lambda args: _list_versions(root, args["atom"])
    ))
    api.register_tool(FunctionTool(
        name="catalog_get_manifest",
        description="Get the full manifest for one (atom, version).",
        parameters={...},
        execute=lambda args: _get_manifest(root, args["atom"], args["version"])
    ))
    api.register_tool(FunctionTool(
        name="catalog_runs_for",
        description="List trace ids that ran with an exact atom-set fingerprint.",
        parameters={...},
        execute=lambda args: _runs_for(root, args["fingerprint"])
    ))
```

The three `_<name>` helpers are module-private functions delegating to `agentm.core.catalog.list_atoms` and direct path reads (using `_layout` constants — but `_layout` is private to `core.catalog`, so the atom uses only the public re-exports).

If `agentm.core.catalog.list_atoms` does not yet take the args we need, this task adds them (in `catalog-storage`'s `__init__.py` — extending the API). Specifically:

```python
# extension to agentm.core.catalog public surface, lands in catalog-storage if not already there
def list_atoms(root: Path | None = None) -> list[CatalogAtom]: ...
def list_versions(name: str, root: Path | None = None) -> list[CatalogVersion]: ...
def get_manifest_at(name: str, version: str, root: Path | None = None) -> dict: ...
def runs_for(fingerprint: dict | str, root: Path | None = None) -> list[str]: ...
```

If these are missing from `catalog-storage`'s landed surface, this task includes a small follow-on patch to `core/catalog/__init__.py` to expose them. **The atom file itself only ever imports from `agentm.core.catalog` and `agentm.core.kernel` — no `_layout` access.**

## Concrete shape — JSON-Schema for parameters

```python
_LIST_VERSIONS_PARAMS = {
    "type": "object",
    "properties": {
        "atom": {"type": "string", "description": "Atom name (e.g. 'tool_read')"},
    },
    "required": ["atom"],
    "additionalProperties": False,
}

_GET_MANIFEST_PARAMS = {
    "type": "object",
    "properties": {
        "atom": {"type": "string"},
        "version": {"type": "string", "description": "Content hash"},
    },
    "required": ["atom", "version"],
    "additionalProperties": False,
}

_RUNS_FOR_PARAMS = {
    "type": "object",
    "properties": {
        "fingerprint": {
            "type": "object",
            "description": "Active-set fingerprint dict OR atom@hash string",
        },
    },
    "required": ["fingerprint"],
    "additionalProperties": False,
}
```

## Test cases

In `tests/unit/extensions/builtin/tool_catalog/test_tool_catalog.py`:

| Test | Asserts | Scenario |
|---|---|---|
| `test_M3_list_versions_includes_current_after_seed` | After seeding `.agentm/catalog/atoms/tool_read/abc123def456/`, `catalog_list_versions(atom="tool_read")` returns a list containing `abc123def456` | M3 |
| `test_list_versions_unknown_atom_returns_empty_list` | Missing atom → `[]`, not error | (resilience) |
| `test_get_manifest_returns_yaml_content` | After seeding `manifest.yaml`, `catalog_get_manifest` returns the dict | (mechanism) |
| `test_get_manifest_unknown_version_returns_error_result` | Missing version → `ToolResult(is_error=True, ...)` | (resilience) |
| `test_runs_for_returns_trace_ids` | After symlinking 2 traces under `<atom>/<hash>/runs/`, `catalog_runs_for({"<name>": "<name>@<hash>"})` returns 2 ids | (mechanism) |
| `test_atom_passes_section_11_validator` | Vacuous — runs the existing validator gate which auto-includes the new file | M5 |
| `test_atom_imports_only_core_catalog_public_surface` | AST grep of the file: every `agentm.core.catalog.*` import must be from the top-level `agentm.core.catalog` module (not `_layout`, not `freeze`, not `hashing` directly). | (layer purity) |

## Acceptance Conditions

- [ ] `uv run pytest tests/unit/extensions/builtin/tool_catalog/ -v` all green
- [ ] `uv run pytest tests/unit/extensions/test_extension_contract.py -q` green (validator gate including the new atom)
- [ ] `uv run ruff check src/agentm/extensions/builtin/tool_catalog.py` clean
- [ ] `uv run mypy src/agentm/extensions/builtin/tool_catalog.py` clean
- [ ] LOC budget: file is ≤ 300 LoC (per §11 size hint)
- [ ] No imports from `agentm.harness.*` beyond `agentm.harness.extension`
- [ ] No imports from `agentm.extensions.builtin.*` (atom-to-atom forbidden)
- [ ] No imports from `agentm.core.catalog._layout` or `agentm.core.catalog.freeze` or `agentm.core.catalog.indexer` (layer purity — only the public surface)
- [ ] `general_purpose.yaml` loads cleanly via `load_scenario` and the new atom is among the loaded modules

## Acceptance scenarios covered

- **M3** — `test_M3_list_versions_includes_current_after_seed`
- **M5** — vacuous in `test_extension_contract`

## Notes

- **Three tools, NOT one** — pi-mono's pattern is one tool per discoverable action. Bundling `list_versions`/`get_manifest`/`runs_for` into one mega-tool with a discriminator field would make the LLM's tool selection harder. Three small tools with explicit parameters is correct.
- **No `propose_change`**: this is the ONLY mediated write path in the design (evolution-substrate §6.2). Adding it requires the `decisions.jsonl` writer + tier-2 enforcement + the constitution's lock on catalog paths — all Phase 2. Document its absence prominently in the manifest description and the file's docstring.
- **No state**: the atom is stateless. Each tool call reads the catalog from disk. Performance is fine — `.agentm/catalog/atoms/` directory listings are cheap.
- **Future expansion path** (out of scope): when Phase 2 adds `compare`/`find_best`/`propose_change`, this atom file gains those tools. If the file approaches 300 LoC, split into `tool_catalog_browse.py` + `tool_catalog_decide.py`. Document this as a Phase 2 risk, not a current concern.
```

============ END FILE ============