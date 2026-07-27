# Policy loop: what is built, what is verified, what is broken

State as of 2026-07-26. Companion to `docs/policy-devset-baseline.md`, which
holds the measurements this work is aimed at.

## What exists

`contrib/extensions/policy/src/policy_engine/loop/` — the evolution loop as
addressable stages. Three layers, and the top two know nothing about any
benchmark:

```
contracts.py     the records passed between stages, and the agents' output schemas
protocols.py     the two seams: CaseSource (read), ReplayBackend (write)
─────────────    below this line, no benchmark is named
collect  diagnose  abstract  compile_gate  replay  select  pipeline  cli
─────────────
benches/senior_swe.py    everything SSB + Harbor + ARL
environments/arl.py      restoring a sandbox; orthogonal to which bench uses it
```

Also `policy_engine/facts.py`: SQL fact queries over the running session, via
`{calls}` / `{bash}` / `{edits}` placeholders expanded against the session's own
schema. This exists because `policy.v_*` are defined against a hardcoded
`harbor_live`, so they cannot see any other run — the literal reason the data
plane had no runtime consumer.

Inner-loop changes that the candidates need in order to fire:

- `Gate.precondition` plus a `FactCheck` callable through
  `TriggerEngine.next_triggered`. An item with a precondition and no way to
  evaluate it stays silent rather than firing blind.
- Under `critic: "on"`, the finish reminder is sent unconditionally. It used to
  depend on a checklist item happening to match, which made the reviewer run by
  coincidence: three recorded sessions, one review.

Adding a benchmark means one file implementing `discover` / `open_environment`
and `rerun`. Orchestration, agents, records and the hard rules are shared.

## Five agents, three of them new

| Agent | Tools | Purpose |
|---|---|---|
| tagger | none | turn events to phase and tags (existed) |
| critic | scenario's | submitted work to accept/reject (existed) |
| diagnoser | shell, read, and the attempt's own sandbox when it can be restored | graded failure to root cause |
| abstractor | **none, deliberately** | root cause to a general check |
| compiler | none | plain-language conditions to trigger plus fact precondition |

The abstractor is blind by construction: it never sees the reference solution,
because with it the check names the specific defect, scores perfectly on the
case it came from, and is worth nothing anywhere else.

## Verified against real data

- **collect** — 21 graded failures from the 36-trial baseline batch, fields
  complete, evidence and backend reference present.
- **diagnoser** — independently reproduced the hand analysis on paperless
  (`wrong_implementation`; kept the aggregate over the document join rather than
  changing its shape) and did not fall into "it did not test enough", which the
  prompt explicitly refuses as a mechanism.
- **abstract** — 6 diagnoses to 5 candidates; posthog correctly skipped as
  `harness_nondeterminism`. The checks read generally and each names an
  observation. Two are as good as anything written by hand: plausible's "run the
  flow with an extra query parameter whose value is a different destination and
  show the final redirect" and prefect's "run a two-worker concurrency check
  against the backend the app uses in production, not the helper you changed".
- **facts** — preconditions execute against recorded trajectories. Used to
  confirm paperless ran zero timing commands across all three attempts.
- **check-gates** — earned its place immediately: see below.

## Known broken

**Compiled gates are inert.** All five preconditions hold for none of the 21
sessions. Two causes, both real:

1. The compiler wrote bare table names (`FROM edits`) instead of the
   placeholders (`FROM {edits}`), so nothing is substituted and the SQL fails
   with `relation "edits" does not exist`. `facts.documentation()` shows the
   braces but never says they must survive into the SQL.
2. The proposed predicates are the task restated —
   `optimization_reuses_broader_cached_or_indexed_data_for_narrower_selection`.
   A tagger cannot decide that, and it locks a general check to one task. The
   generalisation problem moved from the check to the gate.

`check-gates` caught both for free, before any replay was spent. Without it
these would have run to completion and reported every candidate as
`unchanged`, when the truth is they never fired.

**The leak detector is the wrong tool and should be deleted.** Token overlap
flags `from`, `that`, `must`, and cannot catch semantic leakage: "check whether
the count still aggregates through the large table" gives the answer away
without sharing one identifier. The real protection is measurement — replay a
candidate against cases it did *not* come from, which `replay` already supports
— plus the held-out 48.

**Diagnoser quality depends on the sandbox, which usually cannot be opened.**
Run without the repository, it got 2 of 6 hand-verified causes right, 1
partially, and 3 wrong — and the two consequential errors were retiring posthog
(whose check is the one with behavioural evidence) and keeping turborepo (which
has no lesson). Sandbox support is now wired, but ARL sources expire two hours
after a batch and the gateway has no persistent checkpoint store, so it only
works inside that window.

## Artifacts from the last run

`/tmp/.../scratchpad/loop/`: `cases.json` (21), `diagnoses.json` (6),
`candidates.json` (5), `compiled.json` (5, inert).

Baseline batch: `jobs/2026-07-26__11-31-03`, schema `harbor_base_0726`.

## Next

1. Make the placeholder contract explicit and validate it in `compile_gate`
   rather than discovering it at `check-gates`.
2. Constrain new predicates: decidable from events alone, and not a restatement
   of one task. Consider forbidding new predicates entirely at first, forcing
   reuse of the existing vocabulary.
3. Delete `leaks_answer` and the `audit` subcommand; replace with cross-case
   replay.
4. Re-run `diagnose` with a sandbox, inside a batch's window, and check whether
   turborepo and plausible are classified correctly.
5. Only then spend a replay.
