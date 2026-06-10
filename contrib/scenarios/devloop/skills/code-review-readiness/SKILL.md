# Code Review Readiness

Write code that passes the AI code review stage, not just the tests.

## Rules

1. **Implement every acceptance criterion.** The reviewer checks each AC independently against source code. A passing test suite does not guarantee every AC is fully covered.

2. **Match spec signatures exactly.** Function names, parameter names, return types, and class names must match the spec. The reviewer checks source code, not test output.

3. **Handle implied edge cases.** If the spec says "handle empty input" or "validate X", implement it even if no test explicitly asserts it.

4. **No stubs or TODOs.** Every function body must contain real logic. Stubs (`pass`, `raise NotImplementedError`, `TODO`) will be flagged by review.

5. **Document non-obvious decisions.** Add inline comments for design choices that aren't obvious from the spec — the reviewer evaluates reasoning, not just correctness.

6. **Re-read the spec after fixing test failures.** Fixes that only target test output often drift from spec intent. Before declaring done, re-read the spec and verify every AC is addressed in source code.
