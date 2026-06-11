# Prefix replay from an auditor verdict

When you are iterating on auditor prompts or reminder shapes, you usually
do not need to re-run the entire trajectory up to the verdict-firing turn.
The cognitive-audit pipeline already wrote a full `ReplayRecord` to
`.agentm/audit_replay/<session_id>.jsonl` for every auditor firing —
including the reminder text it produced. ``llmharness-replay
agent-from-reminder`` lets you reuse that prefix and *only* re-run the
post-reminder portion of the trajectory.

## Mental model

```
turn 0 ─┬─ turn 1 ─ ... ─ turn t  ┐
        │                         │  auditor fires → surfaces reminder
        │ <───── reused prefix ───┘
        │
        └── branched session resumes here
                                   │
                                   └─ seeded reminder injected at turn t+1
                                      → model reacts → trajectory continues
```

The branched session is a real persisted session under
`.agentm/sessions/`; its header carries a ``parent_session`` field
pointing at the original JSONL so you can later trace which experiments
came from which source.

## Step-by-step

1. **Locate the audit-replay sidecar.** Live sessions write to
   `<cwd>/.agentm/audit_replay/<session_id>.jsonl`. Use
   ``llmharness-replay list --record <path> --phase auditor`` to see
   which turns surfaced a reminder.

2. **Pick a turn.** Note the ``turn`` column for the auditor record whose
   reminder you want to react to.

3. **Branch + emit the resume command:**

   ```bash
   llmharness-replay agent-from-reminder \
     --audit-replay <cwd>/.agentm/audit_replay/<sid>.jsonl \
     --turn <t> \
     --print-only          # omit to also write a runnable .sh script
   ```

   The command will:
   - open the source session JSONL under `<cwd>/.agentm/sessions/`
     (override with ``--session-dir``);
   - pick the leaf entry that ends turn `t` (the `t`-th `message` entry
     on the active branch, same indexing as ``adapters.agentm``);
   - call ``SessionManager.create_branched_session(leaf_id)`` — a new
     JSONL is materialised with the prefix entries (audit_event /
     audit_edge entries included; do *not* try to filter them);
   - print an ``agentm`` invocation that resumes the new session with
     ``llmharness.replay.reminder_seed`` mounted.

4. **Run the printed command.** The seed atom delivers the recorded
   reminder as the first injection of the next turn — byte-identical to
   what the live adapter would have produced for the same text — and
   then unsubscribes so it never fires again. Subsequent reminders, if
   any, will come from the still-active auditor in the branched session.

## Output shape

The printed command always looks like this (single-line wrap shown for
clarity):

```
agentm \
  --cwd <orig-session-cwd> \
  --resume <new-sid> \
  --extension 'llmharness.atom:{"enable_reminders":false,"enable_auditor":true}' \
  --extension 'llmharness.replay.reminder_seed:{"text":"..."}'
```

`enable_reminders: false` + `enable_auditor: true` keeps the live
auditor *observing* (verdicts still get persisted as evidence and the
audit graph keeps growing) while preventing a second live reminder from
masking the experimental seed. If you want a clean dataset-collection
run with no auditor at all, pass
``'llmharness.atom:{"enable_reminders":false,"enable_auditor":false}'``
instead.

## When NOT to use this

- The source session was compacted between turn `t` and now. The
  branched session inherits whatever ``get_branch(leaf_id)`` serves; if
  that includes a compaction marker, the seeded turn will see the
  compacted view, not the raw prefix. Use a fresh live run instead.
- You want to iterate on the *extractor*. The extractor runs at every
  ``TurnEndEvent`` independent of the verdict — use ``llmharness-replay
  extractor`` for that loop.

## Constraint: train/inference parity

The reminder message format is single-sourced as the `REMINDER_PREAMBLE`
constant in `atom.py` and used by both the live adapter and the seed atom. The test suite pins the format (see
``tests/test_reminder_seed_atom.py`` + ``tests/test_reminder_injector.py``);
do not duplicate the preamble string anywhere else.
