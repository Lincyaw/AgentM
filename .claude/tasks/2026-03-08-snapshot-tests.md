# Task 7: Layer 2 Snapshot Tests

**Status**: PENDING
**Depends on**: 5, 6
**Plan**: [plan](../plans/2026-03-08-stub-implementation.md)
**Design**: [testing-strategy.md](../designs/testing-strategy.md) § Layer 2: Pipeline Integration
**Assignee**: tdd

## Objective

Create Layer 2 test suite in `tests/snapshot/`. Uses snapshot injection + Mock LLM. Covers P1–P8.

## Files

- `tests/snapshot/__init__.py`
- `tests/snapshot/conftest.py` — snapshot builders, Mock LLM, TaskManager fixtures
- `tests/snapshot/test_tool_pipeline.py` — P1 (dispatch_agent), P2 (check_tasks)
- `tests/snapshot/test_error_recovery.py` — P3 (retry), P4 (inject rejected), P5 (abort)
- `tests/snapshot/test_compression_recall.py` — P6 (compression preserves recall)
- `tests/snapshot/test_hypothesis_lifecycle.py` — P7 (illegal transition), P8 (consistency)

## Test Cases

| ID | Scenario | Assert |
|----|----------|--------|
| P1 | dispatch_agent updates Notebook | exploration_history has new step; TaskManager has RUNNING task |
| P2 | check_tasks flows data into Notebook | collected_data contains result; new ExplorationStep |
| P3 | Sub-Agent failure → retry → FAILED | TaskManager retries, marks FAILED; error_summary present |
| P4 | inject for COMPLETED task | ValueError raised; instruction NOT queued |
| P5 | abort cancels running task | status FAILED; error_summary contains reason |
| P6 | Compression preserves recall data | recall_history returns pre-compression data |
| P7 | REJECTED → CONFIRMED rejected | ValueError or error Command; notebook NOT updated |
| P8 | set_confirmed on non-CONFIRMED | ValueError; confirmed_hypothesis NOT set |

## Notes

- Use `FakeListChatModel` or custom mock for LLM
- Use `MemorySaver` from `langgraph.checkpoint.memory` for checkpointer
- Add `pytest-asyncio` for async tests
- Tests must be independent (no shared state)
