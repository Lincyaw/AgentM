# Task: manifest-schema — ExtensionManifest fields + validator + tier-2 declarations

**Date**: 2026-05-01
**Status**: PENDING
**Plan**: [self-mod-mvp](../plans/2026-05-01-self-mod-mvp.md)
**Designs**:
- [self-modifiable-architecture](../designs/self-modifiable-architecture.md) §4 (versioned API), §7.3 (tier declaration)
- [evolution-substrate](../designs/evolution-substrate.md) §3.1 (manifest fields)
**Assignee**: implementer
**Wave**: 1 (sequential, after `core-manifest`)
**Size**: M
**Depends on**: [core-manifest](2026-05-01-core-manifest.md)

## Objective

Grow `ExtensionManifest` with three fields (`api_version`, `affects`, `tier`), all with backward-compatible defaults so the existing ~25 atoms continue to load with no source change. Add a validator check that rejects atoms with `api_version > current` or `api_version < current - grace`. Add explicit `tier=2` declarations to the five named atoms (`permission`, `cost_budget`, `tool_filter`, `llm_compaction`, `claude_agents`).

This task is intentionally mechanical: a one-PR change touching the manifest dataclass, the validator, the contract test, and exactly five atom files (the tier-2 ones).

## Inputs to read

- This plan's R4 (defaults strategy)
- `src/agentm/extensions/__init__.py` — current `ExtensionManifest`
- `src/agentm/extensions/validate.py` — current validation flow
- `src/agentm/core/catalog/manifest.py` — `extension_api_current`, `extension_api_grace` (landed in `core-manifest`)
- The five atoms named in `core-manifest.yaml::reload.tier_2_atoms`:
  - `src/agentm/extensions/builtin/permission.py`
  - `src/agentm/extensions/builtin/cost_budget.py`
  - `src/agentm/extensions/builtin/tool_filter.py`
  - `src/agentm/extensions/builtin/llm_compaction.py`
  - `src/agentm/extensions/builtin/claude_agents.py`

## Outputs

### Modified files

| Path | Change |
|---|---|
| `src/agentm/extensions/__init__.py` | Add to `ExtensionManifest`: `api_version: int = 1`, `affects: tuple[str, ...] = ()`, `tier: int = 1`. Keep `frozen=True, slots=True`. Order: place new fields **after** existing optional fields (`config_schema`, `requires`, `conflicts`) to keep positional construction stable for atoms that use keyword args. |
| `src/agentm/extensions/validate.py` | Add §11.4.9 check: load `core_manifest.extension_api_current` + `extension_api_grace`; for each `BuiltinEntry`, if `manifest.api_version > current` → `11.4.9-api-version-too-new`; if `manifest.api_version < current - grace` → `11.4.9-api-version-too-old`. Add §11.4.10 check: warn (issue with rule `11.4.10-tier-list-mismatch`) if a `tier=2` atom is **not** listed in `core_manifest.tier_2_atoms` or vice versa. The mismatch is a warning so it does not block CI on a transient diff during this PR's own rollout. |
| `src/agentm/extensions/builtin/permission.py` | Append `tier=2` to `MANIFEST = ExtensionManifest(...)`. |
| `src/agentm/extensions/builtin/cost_budget.py` | Append `tier=2`. |
| `src/agentm/extensions/builtin/tool_filter.py` | Append `tier=2`. |
| `src/agentm/extensions/builtin/llm_compaction.py` | Append `tier=2`. |
| `src/agentm/extensions/builtin/claude_agents.py` | Append `tier=2`. |
| `tests/unit/extensions/test_manifest_helpers.py` (existing) | Add tests for the three new fields' defaults. |
| `tests/unit/extensions/test_extension_contract.py` (existing) | Add `test_S7_api_version_too_new_rejected` constructing a synthetic atom fixture under `tests/unit/extensions/_fixtures/` with `api_version=99` and asserting the validator emits `11.4.9-api-version-too-new`. Add the symmetric "too-old" fixture if grace=1 leaves a non-empty rejected window. |

## Concrete dataclass change

```python
@dataclass(frozen=True, slots=True)
class ExtensionManifest:
    name: str
    description: str
    registers: tuple[str, ...]
    config_schema: dict[str, Any] | None = None
    requires: tuple[str, ...] = field(default_factory=tuple)
    conflicts: tuple[str, ...] = field(default_factory=tuple)
    # New fields — all defaulted, all backward compatible.
    api_version: int = 1
    affects: tuple[str, ...] = field(default_factory=tuple)
    tier: int = 1
```

Rationale for tuple instead of list/dict for `affects`: matches the pattern used by `registers` and `requires`; frozen-and-hashable; keeps the manifest a value type. The richer `{primary: [...], secondary: [...]}` shape from evolution-substrate §3.1 is **deferred to Phase 2** — when `compare()` lands and actually consumes the structure. MVP indexer attributes universal counters and does not read `affects`.

## Concrete validator addition (§11.4.9)

```python
# pseudocode in plan; do not implement here
from agentm.core.catalog.manifest import load_core_manifest

manifest_constants = load_core_manifest()
current = manifest_constants.extension_api_current
grace = manifest_constants.extension_api_grace

if entry.manifest.api_version > current:
    issues.append(ValidationIssue(... rule="11.4.9-api-version-too-new",
        message=f"atom requires api_version {entry.manifest.api_version}, "
                f"current is {current}"))
if entry.manifest.api_version < current - grace:
    issues.append(ValidationIssue(... rule="11.4.9-api-version-too-old",
        message=f"atom api_version {entry.manifest.api_version} is older than "
                f"the grace window (current={current}, grace={grace})"))
```

## Acceptance Conditions

- [ ] `uv run pytest tests/unit/extensions/ -v` passes including the new contract assertions
- [ ] `uv run pytest tests/unit/core/catalog/ -v` still passes
- [ ] `uv run ruff check src/agentm/extensions/ tests/unit/extensions/` clean
- [ ] `uv run mypy src/agentm/extensions/` clean
- [ ] All ~25 existing atoms load via `discover_builtin()` without errors (their `MANIFEST` constructors take defaults)
- [ ] The five tier-2 atoms have explicit `tier=2`; the contract gate's tier-list-mismatch check is silent (lists agree with `core-manifest.yaml`)
- [ ] No new third-party dependency

## Acceptance scenarios covered

- **S7** — synthetic atom with `api_version=99` rejected by validator (`test_S7_api_version_too_new_rejected`)

## Notes

- **No reload code yet** — that lands in `transactional-reload`.
- **`affects` does not need declarations on existing atoms** — the indexer in MVP attributes universal counters (n_runs, completion). Phase 2 will require atoms to declare what they affect when `compare()` lands.
- **`tier=1` defaults for everything not on the tier-2 list** — explicit. The validator's `11.4.10-tier-list-mismatch` warns on disagreement; an agent-driven tier downgrade (`tier=2 → tier=1`) on a tier-2 atom would surface here in addition to the `propose_change`-level rejection (Phase 2). Document this two-layer defense in the validator's docstring.
- **No atom file's `install()` body changes** — only `MANIFEST = ExtensionManifest(...)` constructors.
```

============ END FILE ============