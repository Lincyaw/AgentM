---
name: review-boundary
description: >
  Review AgentM code for boundary isolation, pluggability-axis violations,
  atom contract breaches, and design-pattern intrusion. Use when asked to
  run a boundary review over a path, git range, current branch, or staged
  changes; also trigger on "review-boundary", "boundary reviewer",
  "boundary audit", "pluggability review", or "§11 contract review".
---

# review-boundary

Audit AgentM code for boundary isolation, pluggability-axis violations,
and design-pattern intrusion.

## Input

Treat the user's text after the skill name as the review selector:

- Empty / omitted: review `src/agentm/`.
- A path or glob, such as `src/agentm/gateway/` or
  `contrib/extensions/llmharness/`: review that subtree.
- A git range, such as `HEAD~5..HEAD` or `main..HEAD`: review files
  touched in that range.
- `--branch`: review files changed on the current branch vs `main`
  (`git diff --name-only main...HEAD`).
- `--staged`: review currently staged files.

If the selector is ambiguous, ask the user before reviewing.

## Review Procedure

1. Resolve the file list from the selector.
2. Capture git context: current branch, base ref when relevant, and
   whether there are uncommitted changes.
3. Read the relevant code and design context. Prefer:
   - `.Codex/designs/pluggable-architecture.md`
   - `AGENTS.md`
   - nearby design docs under `.Codex/designs/`
   - `CONTEXT.md` when terminology is unclear
4. Check for:
   - direct atom imports from `agentm.core.runtime.*`, `core._internal`,
     or other forbidden kernel internals;
   - atom-to-atom imports or scenario-specific logic leaking into
     `agentm.core`;
   - bypasses of `ExtensionAPI` services for stateful subsystems;
   - changes that weaken §11 extension validation;
   - pluggability-axis behavior embedded in the substrate instead of
     registered through atom protocols;
   - design docs or `.Codex/index.yaml` left stale after concept changes.
5. Report findings first, ordered by severity, with concrete file and
   line references.

## Output

Use code-review shape:

```markdown
## Findings
- [block|major|minor] path:line - Issue and consequence.

## Open Questions
- ...

## Summary
- Reviewed scope and residual risk.
```

If any finding is `block`, end by saying merging is not advised until the
blockers are resolved.
