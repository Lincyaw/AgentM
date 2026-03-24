# Task 1b: Schemas + Registry

**Plan**: [2026-03-24-trajectory-analysis](../plans/2026-03-24-trajectory-analysis.md)
**Design**: [trajectory-analysis](../designs/trajectory-analysis.md)
**Status**: TODO

## Scope

### 1. `answer_schemas.py` — Generic AnalyzeAnswer

```python
class AnalyzeAnswer(_BaseAnswer):
    """Generic trajectory analysis worker result."""
    leads: list[str] = Field(default_factory=list)
```

Single answer schema for the `analyze` task type. The `findings` field is inherited from `_BaseAnswer`.

### 2. `output.py` — AnalysisReport

```python
class AnalysisReport(BaseModel):
    skill: str
    source_count: int
    findings: list[dict]
    artifacts: list[str]
    summary: str
```

### 3. `__init__.py` — register() function

```python
def register() -> None:
    register_state("trajectory_analysis", TrajectoryAnalysisState)
    register_strategy("trajectory_analysis", TrajectoryAnalysisStrategy())
    ANSWER_SCHEMA.setdefault("analyze", AnalyzeAnswer)
    OUTPUT_SCHEMAS.setdefault("AnalysisReport", AnalysisReport)
```

### 4. Update `scenarios/__init__.py` — discover()

Replace `register_mem()` with `register_ta()`:
```python
from agentm.scenarios.trajectory_analysis import register as register_ta
register_ta()
```

## Files to Create/Modify

- Create: `src/agentm/scenarios/trajectory_analysis/answer_schemas.py`
- Create: `src/agentm/scenarios/trajectory_analysis/output.py`
- Modify: `src/agentm/scenarios/trajectory_analysis/__init__.py` (add register)
- Modify: `src/agentm/scenarios/__init__.py` (swap register_mem → register_ta)

## Note on ANSWER_SCHEMA collision

The old memory_extraction registered `analyze` → `AnalyzeAnswer` (memory-specific). The new one also uses `analyze` as the key. Since we're removing the old scenario, no collision. But if both exist during migration, `setdefault` means first-registered wins — ensure old register is removed before new one is added.
