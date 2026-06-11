---
name: no-subagent-for-small-changes
description: For small refactors, do the work directly instead of launching dev-worker subagents
metadata:
  type: feedback
---

For small-to-medium code changes (a few files, <200 lines of diff), do the work directly instead of launching dev-worker subagents.

**Why:** Subagents introduce overhead (worktree divergence from main, cherry-pick conflicts, unrelated changes leaking in) that outweighs the parallelism benefit for small tasks. The user prefers direct edits.

**How to apply:** Only use dev-worker subagents for truly independent, large-scope changes. For typical refactors touching 1-3 files, edit directly in the main working tree.
