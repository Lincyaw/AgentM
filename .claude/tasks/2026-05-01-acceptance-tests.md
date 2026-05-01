# Task: acceptance-tests — End-to-end coverage of S1-S10 + E1-E10

**Date**: 2026-05-01
**Status**: PENDING
**Plan**: [self-mod-mvp](../plans/2026-05-01-self-mod-mvp.md)
**Designs**:
- [self-modifiable-architecture](../designs/self-modifiable-architecture.md) §9
- [evolution-substrate](../designs/evolution-substrate.md) §9
**Assignee**: tdd → reviewer
**Wave**: 3 (parallel — gates MVP done)
**Size**: M
**Depends on**: all earlier tasks (the test file imports the contracts they land)

## Objective

Land one integration-test file `tests/integration/test_self_mod_mvp.py` covering every S1-S10 and E1-E10 scenario reachable in the MVP, plus `pytest.skip(...)` placeholders with explicit reasons for the Phase 2 / Phase 3 ones. The skip count is the tracked regression marker for entering Phase 2.

This task does **NOT** add new product code. It only verifies the integrated behavior of everything that landed in waves 1–2 plus the indexer + tool_catalog atom from wave 3. The unit tests in earlier tasks cover individual behaviors; this file verifies the **composition**.

## Inputs to read

- The acceptance map in plan §6
- All earlier task files (so each test can target a real symbol)
- `tests/integration/` (the existing integration directory if any — the recent pi-mono migration plan references `tests/integration/pi_mono_tier1_composition.py` as a model — read it for fixture style if it has landed)

## Outputs

### New files

| Path | Purpose |
|---|---|
| `tests/integration/__init__.py` | Package marker (if not present) |
| `tests/integration/test_self_mod_mvp.py` | All acceptance scenarios |
| `tests/integration/_fixtures/` (dir) | Synthetic atom sources used by some scenarios (e.g. `_atom_with_syntax_error.py.txt`, `_atom_install_raises.py.txt`) — stored as `.txt` so discovery doesn't try to load them as real atoms |

## §1. Test inventory

The file is one `pytest` module with one test per scenario, named to match the scenario id verbatim for traceability:

```python
# tests/integration/test_self_mod_mvp.py

# --- self-modifiable-architecture §9 ---
async def test_S1_reload_tool_atom_takes_effect_next_turn(): ...
@pytest.mark.skip(reason="Phase 2: requires tier-2 deferral via propose_change")
async def test_S2_tier2_reload_deferred_pending_approval(): ...
async def test_S3_tool_edit_blocked_on_constitution_path(): ...
async def test_S4_syntax_error_rejected_no_write(): ...
async def test_S5_install_failure_rolls_back(): ...
async def test_S6_assert_active_raises_after_reload(): ...
async def test_S7_api_version_too_new_rejected(): ...
async def test_S8_tool_edit_blocked_on_harness_extension(): ...
@pytest.mark.skip(reason="Phase 2: requires propose_change author identification")
async def test_S9_tier_downgrade_blocked_for_agent(): ...
async def test_S10_manifest_change_moves_constitution_boundary(): ...

# --- evolution-substrate §9 ---
@pytest.mark.skip(reason="Phase 2: compare() not in MVP")
async def test_E1_compare_returns_numbers_with_ci(): ...
@pytest.mark.skip(reason="Phase 2: compare() not in MVP")
async def test_E2_compare_returns_inconclusive_on_small_n(): ...
@pytest.mark.skip(reason="Phase 2: find_best + regressed flag not in MVP")
async def test_E3_find_best_skips_regressed(): ...
async def test_E4_catalog_path_blocked(): ...
async def test_E5_rebuild_is_idempotent(): ...
@pytest.mark.skip(reason="Phase 3: experiment-mode lock deferred")
async def test_E6_experiment_mode_locks_other_atoms(): ...
@pytest.mark.skip(reason="Phase 2: decisions.jsonl + propose_change not in MVP")
async def test_E7_reactivating_regressed_blocked(): ...
async def test_E8_mid_session_reload_emits_marker(): ...
@pytest.mark.skip(reason="Phase 2: scenario-version dirs deferred")
async def test_E9_scenario_compare(): ...
@pytest.mark.skip(reason="Phase 2: find_best not in MVP")
async def test_E10_find_best_returns_none_when_no_winner(): ...

# --- additional MVP-only acceptance points ---
async def test_M1_freeze_idempotent(): ...        # plan §6 M1
async def test_M2_fingerprint_in_session_record(): ...
async def test_M3_list_versions_after_first_session(): ...
async def test_M4_per_atom_api_instances_distinct(): ...
async def test_M5_section_11_contract_still_green(): ...
```

Some Mx tests overlap with unit tests in earlier tasks — duplicate coverage at the integration level is intentional for the plan-final gate.

## §2. Test fixtures

A reusable pytest fixture that builds a temp-dir AgentM session with:

- A fake `StreamFn` that yields a single `MessageEnd(stop_reason="end_turn")`.
- The `observability` atom loaded so traces land at `<tmpdir>/.agentm/observability/<id>.jsonl`.
- One tool atom (`tool_read`) so reload is exercisable.
- The `tool_catalog` atom so M3 is runnable.
- A factory for synthesizing atom source strings (used by S1, S4, S5).

Key fixture name: `agentm_mvp_session` — yields an `AgentSession` and the temp dir.

## §3. Concrete shape — a few representative tests

### S1 — Reload takes effect next turn

```python
async def test_S1_reload_tool_atom_takes_effect_next_turn(agentm_mvp_session):
    session, root = agentm_mvp_session
    new_source = """
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI
from agentm.core.kernel import FunctionTool, TextContent, ToolResult

MANIFEST = ExtensionManifest(
    name="tool_read",
    description="MUTATED: returns a sentinel",
    registers=("tool:read",),
)

def install(api, config):
    api.register_tool(FunctionTool(
        name="read",
        description="mutated",
        parameters={"type":"object","properties":{}},
        execute=lambda args: ToolResult(content=[TextContent(type="text", text="MUTATED")], is_error=False),
    ))
"""
    api = session._apis["agentm.extensions.builtin.tool_read"]
    result = api.reload_atom("tool_read", new_source, agent_initiated=True)
    assert result.ok
    # Assert the registered tool's behavior changed
    read_tool = next(t for t in session.tools if t.name == "read")
    assert "MUTATED" in (await read_tool.execute({}, ...)).content[0].text
```

### S3 + S8 — Constitution path block

```python
@pytest.mark.parametrize("path", [
    "src/agentm/core/kernel/loop.py",     # S3
    "src/agentm/harness/extension.py",    # S8
    ".agentm/catalog/atoms/tool_read/abc/metrics.jsonl",  # E4
])
async def test_constitution_path_blocked(agentm_mvp_session, path):
    session, _ = agentm_mvp_session
    api = session._apis["agentm.extensions.builtin.tool_read"]
    assert api.is_constitution_path(path) is True
```

### S10 — Boundary moves with manifest

```python
async def test_S10_manifest_change_moves_constitution_boundary(tmp_path, monkeypatch):
    custom_manifest = tmp_path / "core-manifest.yaml"
    custom_manifest.write_text("""
version: 1
constitution:
  paths:
    - core-manifest.yaml
extension_api:
  current: 1
  semver_rules: {major: "x", minor: "x", patch: "x"}
  deprecation: {grace: 1}
reload:
  tier_2_atoms: []
""")
    monkeypatch.setenv("AGENTM_CORE_MANIFEST", str(custom_manifest))
    from agentm.core.catalog import manifest as cm
    cm.reload_manifest()
    assert cm.is_constitution_path("src/agentm/core/kernel/loop.py") is False
```

(The `AGENTM_CORE_MANIFEST` env var is a hook the `core-manifest` task adds for tests — it overrides the default repo-root path. Document it in `manifest.py`.)

### E5 — Rebuild idempotent

```python
async def test_E5_rebuild_is_idempotent(agentm_mvp_session):
    session, root = agentm_mvp_session
    await session.prompt("hello")
    await session.shutdown()
    metrics_dir = root / ".agentm" / "catalog" / "atoms"
    before = _capture_all_metrics(metrics_dir)  # dict: path -> [rows w/o indexed_at]

    # Wipe metrics.jsonl files
    for f in metrics_dir.rglob("metrics.jsonl"):
        f.unlink()

    subprocess.check_call([
        sys.executable, "-m", "agentm.core.catalog.indexer", "rebuild",
        "--root", str(root / ".agentm" / "catalog"),
        "--observability", str(root / ".agentm" / "observability"),
    ])
    after = _capture_all_metrics(metrics_dir)
    assert before == after
```

### E8 — Mid-session reload marker

```python
async def test_E8_mid_session_reload_emits_marker(agentm_mvp_session):
    session, root = agentm_mvp_session
    await session.prompt("hello")
    api = session._apis["agentm.extensions.builtin.tool_read"]
    api.reload_atom("tool_read", _IDENTITY_RELOAD_SOURCE, agent_initiated=True)
    await session.prompt("again")
    await session.shutdown()
    trace = next((root / ".agentm" / "observability").glob("*.jsonl"))
    records = [json.loads(l) for l in trace.read_text().splitlines()]
    reloads = [r for r in records if r["kind"] == "atom.reload"]
    assert len(reloads) == 1
    assert reloads[0]["attributes"]["name"] == "tool_read"
```

## §4. Acceptance Conditions

- [ ] `uv run pytest tests/integration/test_self_mod_mvp.py -v` shows the expected mix: MVP tests pass (`.`), Phase 2/3 tests skipped (`s`) with explicit reason
- [ ] Skip count: exactly **9** tests skipped (S2, S9, E1, E2, E3, E6, E7, E9, E10)
- [ ] All MVP tests pass (S1, S3, S4, S5, S6, S7, S8, S10, E4, E5, E8, M1-M5)
- [ ] No `xfail` or unexplained skips
- [ ] `uv run ruff check tests/integration/` clean
- [ ] `uv run mypy tests/integration/` (or whatever the project mypy config covers for tests) clean
- [ ] Test runtime ≤ 30 seconds total

## §5. Acceptance scenarios covered

All of S1-S10 and E1-E10 (the deferred ones via documented `pytest.skip`) plus M1-M5.

## §6. Notes

- **Skips are intentional progress markers** — Phase 2 entry is "remove these `pytest.skip` decorators and watch the tests turn green or red". Each skip's `reason` text names the missing capability so the Phase 2 plan can directly target it.
- **Fixture coupling**: this file imports `agentm.core.catalog`, `agentm.harness.session`, `agentm.harness.extension`, and `agentm.extensions.builtin.tool_catalog`. It does NOT import any other builtin atom directly — atoms enter the test session via the scenario YAML / explicit `extensions=[...]` list. Keep tests this way to validate the integration surface.
- **Internal access**: tests reach into `session._apis` to acquire a per-atom `_ExtensionAPIImpl`. This is intentional — the public `ExtensionAPI` is consumed by atoms, not by application code. Document this in the test file's docstring so a Phase 2 author doesn't try to "clean it up" by exposing `_apis` publicly.
- **Non-determinism**: hashing is deterministic, but observability writes JSONL via a daemon thread; tests must `session.shutdown()` (which flushes) before reading the file. The `agentm_mvp_session` fixture's teardown does this.
- **No modification to product code in this task** — if a test fails, the fix lands in the relevant earlier task. This separation keeps the gate clean.
- **CI gate**: `tests/integration/test_self_mod_mvp.py` is the single-file regression marker for this MVP. Future Phase 2 plans will reference it explicitly.
```

============ END FILE ============

---

## Summary of what I'm handing back

- **One plan**: `/home/ddq/AoyangSpace/AgentM/.claude/plans/2026-05-01-self-mod-mvp.md` — full content above between the first set of `============` delimiters.
- **Eight tasks** in `/home/ddq/AoyangSpace/AgentM/.claude/tasks/`:
  - `2026-05-01-core-manifest.md`
  - `2026-05-01-manifest-schema.md`
  - `2026-05-01-catalog-storage.md`
  - `2026-05-01-transactional-reload.md`
  - `2026-05-01-observability-fingerprint.md`
  - `2026-05-01-indexer-mvp.md`
  - `2026-05-01-tool-catalog-atom.md`
  - `2026-05-01-acceptance-tests.md`
- **Index update** specified in plan §9 — append `plans:` and `tasks:` lists to `self_modifiable_architecture` and `evolution_substrate` concepts in `/home/ddq/AoyangSpace/AgentM/.claude/index.yaml`.

## Structural decisions and tradeoffs (from the brief's §9)

I had to take positions on several things that affect plan structure. They are flagged in the risk register; the load-bearing ones:

1. **R1 + §3 of `transactional-reload`**: refactoring `_ExtensionAPIImpl` to per-atom instances. There is no other way to make `assert_active` a meaningful per-atom flag. I positioned this as in-scope for `transactional-reload` and noted it's the largest single change in the wave; if the resulting PR is too big, split the per-atom-API plumbing into its own task.

2. **R10 + §1 of `observability-fingerprint`**: the design says fingerprint is in `session.start`, but `session.start` writes immediately during `observability.install()` — at that moment no other extension has been installed. I chose to **emit a separate `session.fingerprint` record at `session_ready`** rather than have observability poke recipe internals at install time. Documented as a deviation; the indexer reads `session.fingerprint`.

3. **R11**: split `core-manifest` and `manifest-schema` into two sequential tasks rather than one. Rationale: the YAML+parser landing alone is a self-contained foundation; merging the atom-touching schema growth into the same PR would couple two failure surfaces (parser bugs vs. atom contract test breaks).

4. **R12**: cost is reported as `tokens_per_task` (no pricing) in MVP because no pricing module exists. Phase 2's `guard_metrics.yaml` (per evolution-substrate §11) is where this gets resolved.

5. **Genesis-version rule** in indexer §1.1: an atom that has never been reloaded has no catalog dir; the indexer lazily freezes it on first session shutdown. This was not stated in either design doc but is required for the indexer to attribute. I positioned it as the indexer's responsibility (constitution layer, not autonomy).

6. **Catalog public surface narrowing**: `tool_catalog` (autonomy) imports only from `agentm.core.catalog` (top level), never from `_layout`/`freeze`/`indexer` private modules. The atom-level tests include an AST grep to enforce this — same style as the existing layer-purity grep in `pi-mono-migration` plan.

The plan and tasks total roughly 1700 lines of markdown — comparable to the existing `2026-05-01-pi-mono-migration.md` plan's depth — and cover the MVP slice of both designs without leaking Phase 2 complexity into the work.