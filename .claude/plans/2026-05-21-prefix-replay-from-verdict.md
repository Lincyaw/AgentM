# Prefix replay from auditor verdict

Date: 2026-05-21
Author: dev-worker (worktree agent-ad6754d1a1a29f846)

## Goal

Given a recorded auditor `ReplayRecord` that surfaced a reminder at turn t,
materialize a new main-agent session that *starts* at the end of turn t with
that reminder seeded as the first injection. Skips re-running turns 0..t
during reminder / auditor iteration — same prefix, different reaction.

## Scope

- New §11 single-file atom: `llmharness.replay.reminder_seed` — one-shot
  injector that drains exactly one configured reminder on the first
  `DecideTurnActionEvent` after install and then no-ops.
- New CLI subcommand `agent-from-reminder` under the existing
  `llmharness-replay` typer app:
    * locates the auditor record at `--turn`
    * opens the source session
    * picks the leaf entry that ends turn t (assistant message #t on the
      branch — matches `adapters.agentm`'s `turn_index = len(messages) - 1`)
    * calls `SessionManager.create_branched_session(leaf_id)`
    * prints (and optionally writes) the `agentm --resume <new-sid>` command
      with the reminder text wired through `--extension llmharness.replay.reminder_seed:{...}`
- Tests: atom unit + CLI integration (synthetic session JSONL + audit replay
  sidecar). Fail-stop assertion: branch length of the new session equals the
  expected prefix length.
- Docs: new `contrib/extensions/llmharness/docs/07-prefix-replay.md`; README CLI
  table updated; project-index gets REQ-035.

## Non-scope / why no core change

`SessionManager.create_branched_session(leaf_id)` already does what we need
(persisted truncation + entry-id rewrite + writes `parent_session` link in the
new header). The task brief is explicit: **do not** add a fork helper to core
or extend `SessionStore`. Everything new lives under
`contrib/extensions/llmharness/`.

## Reminder-message contract

The atom must emit a message indistinguishable from what the live adapter
produces. To single-source the format, introduce
`llmharness/audit/_reminder_format.py` holding `REMINDER_PREAMBLE` plus a
`build_reminder_message(text, *, timestamp)` helper. Both
`adapters/agentm.py::_make_reminder_injector` and the new
`replay/reminder_seed.py` import from it. This is the smallest deltas-on-disk
way to guarantee train/inference message-shape parity — the brief's main risk
flag.

## CLI flag shape (verified against `src/agentm/cli.py`)

The agentm CLI does NOT have a `--extension-config` flag; it uses the unified
`--extension MODULE[:JSON_OBJECT]` form (`_parse_extensions` in
`src/agentm/cli.py:116`). The composed command therefore looks like:

```
agentm --cwd <orig> --resume <new-sid> \
  --extension 'llmharness.adapters.agentm:{"enable_reminders":false,"enable_auditor":true}' \
  --extension 'llmharness.replay.reminder_seed:{"text":"..."}'
```

The brief's example used `--extension-config`; we deviate and document it
here so the recipe doc matches reality.

## Adapter / seed combo

`enable_reminders:false` + `enable_auditor:true` keeps the live auditor
*observing* (verdicts get persisted, evidence keeps flowing) while preventing
it from injecting a second reminder that would mask the seed reminder's
effect. The seed atom is the only reminder source in the branched session.

## Deferred

- No long-running operationalisation: this is a developer-iteration tool, not
  a hot path. No retry / queue / batch.
- No automatic linkage from the branched session back into the live audit DB —
  surfaced via the on-disk `parent_session` header field that
  `create_branched_session` already writes.
- No re-anchoring of the reminder if the session was compacted between turn t
  and now. If the leaf entry has been compaction-collapsed, the branched
  session inherits the compaction marker exactly as `get_branch(leaf_id)`
  serves it. Out of scope for v1.

## Index propagation

This feature is contrib-side, no `.claude/designs/*` doc is added, so
`.claude/index.yaml` is not updated. `contrib/extensions/llmharness/project-index.yaml`
gets one new requirement (REQ-035).
