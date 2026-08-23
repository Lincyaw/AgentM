---
name: simplify
description: Simplify and refine recently modified code for clarity, consistency, and maintainability while preserving exact behavior. Use after implementing code changes, when reviewing current-session edits, or when the user asks to clean up, simplify, refactor lightly, or make touched code easier to read without changing functionality.
---

# Code Simplifier

## Purpose

Refine code that was recently modified or touched in the current session. Preserve behavior exactly, including public APIs, outputs, error semantics, side effects, styling, and tests.

## Workflow

1. Identify the relevant diff before editing:
   - Prefer `git diff --stat` and `git diff`.
   - Include untracked files only when they are part of the current task.
   - Focus on current-session or recently modified code unless the user asks for broader cleanup.

2. Read local project guidance before changing code:
   - Check `CLAUDE.md`, package scripts, lint configuration, formatter configuration, and nearby code.
   - Follow existing naming, module boundaries, import style, and test patterns.

3. Simplify only when it improves maintainability:
   - Reduce unnecessary nesting, duplication, temporary abstractions, and incidental complexity.
   - Improve names when doing so clarifies intent without broad churn.
   - Consolidate related logic when the result remains explicit and debuggable.
   - Remove comments that merely repeat obvious code.
   - Keep helpful comments that explain non-obvious constraints, data shape, or business rules.

4. Preserve the codebase's TypeScript and React conventions:
   - Use ES modules with sorted imports and explicit file extensions where the project requires them.
   - Prefer `function` declarations over arrow functions when that is the local standard.
   - Add explicit return types for top-level functions when expected by the project.
   - Use explicit `Props` types for React components.
   - Follow the existing error-handling approach; avoid adding `try/catch` when the surrounding code uses result types, guards, or propagated errors.

5. Avoid clever compaction:
   - Do not use nested ternary operators; prefer `switch` or `if`/`else` chains for multiple conditions.
   - Do not compress multiple concerns into dense one-liners.
   - Do not remove abstractions that clarify ownership or extension points.
   - Do not prioritize fewer lines over readability.

6. Verify after edits:
   - Run the smallest relevant formatter, linter, typecheck, or test command available.
   - If verification is not practical, inspect the diff carefully and state what was not run.

## Guardrails

- Do not change behavior to make the code look cleaner.
- Do not rewrite unrelated files or perform broad refactors.
- Do not rename exported symbols, routes, config keys, event names, or persisted data fields unless the user explicitly asked.
- Do not revert user changes.
- Keep the final report focused on significant simplifications and verification results.
