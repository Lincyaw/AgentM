---
description: Run the boundary-reviewer agent over a path, diff range, or the current branch's changes vs main.
argument-hint: "[path | git-range | --branch]   (default: src/agentm/)"
---

Invoke the `boundary-reviewer` subagent to audit AgentM code for boundary
isolation, pluggability-axis violations, and design-pattern intrusion.

## Scope resolution

The argument `$ARGUMENTS` selects what to review:

- empty / omitted → `src/agentm/` (full SDK tree)
- a path or glob (e.g. `src/agentm/harness/`, `contrib/extensions/llmharness/`) → review that subtree
- a git range (e.g. `HEAD~5..HEAD`, `main..HEAD`) → review the files touched in that range
- `--branch` → review files changed on the current branch vs `main` (`git diff --name-only main...HEAD`)
- `--staged` → review currently staged files

If the argument is ambiguous, ask the user before launching the agent.

## Launch

Spawn the `boundary-reviewer` agent (subagent_type: `boundary-reviewer`) with a
self-contained prompt that includes:

1. The resolved file list.
2. The git context (current branch, base ref, whether there are uncommitted changes).
3. Any reviewer hints the user passed after the scope argument
   (e.g. `/review-boundary src/agentm/harness/ focus on H1 and S4`).

Pass the agent's Markdown report back to the user verbatim — do **not**
summarize or paraphrase. If the report has any `block` findings, end your
message with one sentence flagging that merging is not advised until they are
resolved.
