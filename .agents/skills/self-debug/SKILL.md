---
name: self-debug
description: >
  Inspect your own running session to debug yourself — read your own live
  trace and mine historical traces. Use when the user reports a bug in how
  you behaved, or asks you to figure out what went wrong in this or a past
  session.
---

# self-debug

You are running inside a session. Your behavior, your tool calls, and
the trajectory are all observable from where you sit. When something
looks wrong, look at the evidence before guessing.

## Principle

Two independent surfaces, each for a different question:

| Question | Surface |
|----------|---------|
| "What did *I* just do this session?" | `agentm trace … --latest` via bash |
| "What happened in some past session?" | `agentm trace sessions` → drill in |

Prefer evidence over speculation. The session trace is the source of
truth for your own actions; the conversation is a lossy view of it.

## Trace locations

| Data | Location | Query with |
|------|----------|-----------|
| Session trajectories | `.agentm/trajectory/` (JSONL per session) or Postgres (`AGENTM_TRAJECTORY_DSN`) | `agentm trace` subcommands |

For the full command reference and composition patterns, load the
`trace-analysis` skill.

## Diagnostic workflow

1. **Reproduce-from-evidence**: `agentm trace tools --latest` to see
   exactly what you did.
2. **Check turn structure**: `agentm trace turns --latest` to see error
   counts, token usage per turn, and outcome causes.
3. **Form a hypothesis** grounded in that evidence.
4. If the bug is in a past session: `agentm trace sessions` → find the
   session → drill in with `--session <id>`.
5. Only then propose or apply a fix.
