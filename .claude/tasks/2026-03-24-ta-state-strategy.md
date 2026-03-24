# Task 1a: State + Strategy

**Plan**: [2026-03-24-trajectory-analysis](../plans/2026-03-24-trajectory-analysis.md)
**Design**: [trajectory-analysis](../designs/trajectory-analysis.md)
**Status**: TODO

## Scope

Create `src/agentm/scenarios/trajectory_analysis/` with:

### 1. `state.py` — TrajectoryAnalysisState

```python
class TrajectoryAnalysisState(BaseExecutorState):
    source_trajectories: list[str]
    skill_name: str
    analysis_results: Annotated[list[dict], operator.add]
    structured_output: Optional[Any]
```

### 2. `strategy.py` — TrajectoryAnalysisStrategy

- `name` → `"trajectory_analysis"`
- `initial_state()` → creates `TrajectoryAnalysisState` with `current_phase="analyze"`
- `format_context()` → renders skill catalog XML or pre-loaded skill body + source trajectories
- `phase_definitions()` → two phases: `analyze → synthesize`
- `should_terminate()` → True when `current_phase == "synthesize"`
- `compress_state()` → no-op (pass through)
- `get_answer_schemas()` → `{"analyze": AnalyzeAnswer}`
- `get_output_schema()` → None (configured in scenario.yaml)
- `state_schema()` → `TrajectoryAnalysisState`
- `orchestrator_hooks()` → default `OrchestratorHooks()`
- `_discover_skills(vault)` → scans vault for SKILL.md frontmatter, builds catalog
- Config-driven mode: reads `skill` from config, pre-loads SKILL.md body

### 3. `data.py` — SkillCatalogEntry

```python
@dataclass
class SkillCatalogEntry:
    name: str
    description: str
    path: str
```

### 4. `__init__.py` — empty (register function in separate task)

## Files to Create

- `src/agentm/scenarios/trajectory_analysis/__init__.py`
- `src/agentm/scenarios/trajectory_analysis/state.py`
- `src/agentm/scenarios/trajectory_analysis/strategy.py`
- `src/agentm/scenarios/trajectory_analysis/data.py`

## Dependencies

- `agentm.models.state.BaseExecutorState`
- `agentm.models.data.PhaseDefinition, OrchestratorHooks`
- `agentm.tools.vault.store.MarkdownVault` (for skill discovery)
