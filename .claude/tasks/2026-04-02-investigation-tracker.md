# Task: InvestigationTracker and Data Models

**Plan**: [investigation-sanitizer](../plans/2026-04-02-investigation-sanitizer.md)
**Phase**: 1 — Foundation
**Design**: [investigation-sanitizer](../designs/investigation-sanitizer.md) §5, §6

## Scope

Build the data structures that CodeSanitizer and CriticSanitizer depend on.

## Deliverables

### models.py — `src/agentm/scenarios/rca/sanitizer/models.py`

```python
class Severity(str, Enum):
    BLOCK = "BLOCK"
    WARN = "WARN"
    INFO = "INFO"

@dataclass(frozen=True)
class SanitizerFinding:
    code: str               # "E1", "C2", "J3", etc.
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class InvestigationEvent:
    round: int
    event_type: str          # "dispatch", "task_complete", "hypothesis_change", "tool_call"
    data: dict[str, Any]

class Sanitizer(Protocol):
    def check(self, trigger: str, hypothesis_store: HypothesisStore,
              profile_store: ServiceProfileStore, tracker: InvestigationTracker,
              ctx: LoopContext) -> list[SanitizerFinding]: ...
```

### tracker.py — `src/agentm/scenarios/rca/sanitizer/tracker.py`

```python
class InvestigationTracker:
    """Append-only event log for investigation-level events."""

    def record(self, round: int, event_type: str, data: dict) -> None: ...
    def dispatches(self) -> list[InvestigationEvent]: ...
    def task_completions(self) -> list[InvestigationEvent]: ...
    def hypothesis_changes(self) -> list[InvestigationEvent]: ...
    def tool_calls_for(self, tool_name: str) -> list[InvestigationEvent]: ...
    def events_after(self, round: int, event_type: str | None = None) -> list[InvestigationEvent]: ...
```

- Thread-safe (reuse locking pattern from ThreadSafeStore)
- Events stored in insertion order
- Query methods return filtered, sorted copies

### Tests — `tests/unit/test_investigation_tracker.py`

- Record events, verify order preserved
- Query dispatches/task_completions/hypothesis_changes returns correct subset
- events_after filters by round correctly
- tool_calls_for filters by tool_name
- Thread safety: concurrent writes don't lose events
- SanitizerFinding creation and field access
