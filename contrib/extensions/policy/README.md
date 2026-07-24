# policy-engine

An evidence-driven failure-detection and intervention loop for coding
agents: mine failure patterns from recorded trajectories, compile them into
queries over a data plane, and intervene in live sessions when they fire.

```
① raw data        trajectory records: tool events, results, messages
② data plane      preprocess + correlate raw into ONE queryable fact schema
③ mining          find failure patterns BY QUERYING THE PLANE  → checklist
④ compile         each checklist item's `when` = a plane query (the DSL)
⑤ intervene       gate fires → inject a check into the working agent,
                  or spawn a reviewer subagent and inject its verdict
⑥ calibrate       every trigger change is replayed over the recorded
                  corpus; emissions are judged by their evidence
```

The ordering is load-bearing: **the plane is built before mining**, and the
miner explores by querying it. Whatever the miner can see, the runtime can
query — so mined patterns are compilable by construction. (When mining ran
on raw trajectories instead, it produced trigger prose referencing facts no
runtime could evaluate.)

## Layering contract

| Layer | Artifact | May contain | Must never contain |
|---|---|---|---|
| Raw | trajectory DB (psql), per-session SQLite (`policy_tool_events`) | append-only records | interpretation |
| **Data plane** | `plane.py` → `plane_*` tables, `v_*` views | **neutral facts and relations**: runs, edits, action↔file edges, claims, supersession, temporal order | **failure semantics.** A view named after a fault pattern (e.g. "blind edits") is a checklist concept that leaked downward — it belongs in the signal layer |
| Signals | `signals.yaml` | pattern definitions as SQL over the plane, with `:params`, evidence query + format, and calibration provenance | imperative code |
| Checklist | `checklist.yaml` | mined items: check text, `when: {signal, checkpoint, arm}`, `deliver: inject/critic/offline` | trigger logic (only references to signals) |
| Engine | `__init__.py`, `recording.py`, `plane.py`, `triggers.py`, `deliver.py` | mechanism: record → rebuild plane → evaluate gates → deliver | domain knowledge about specific failures |

Knowledge grows as data (signals + checklist items); the engine stays
fixed. The intended end state is that the miner itself emits new checklist
items *with* their signal SQL, and they ship after replay calibration with
no code change.

## Data plane

Derived bottom-up from what the raw layer actually records — a fact enters
the schema only if (a) it is derivable from raw data we have, and (b) some
checklist item's `when`/`check` references it. Schema changes are discussed
against that derivation before they are built.

| Table | Facts | Derived from |
|---|---|---|
| `plane_runs` (+`run_scopes`, `run_selectors`, `run_tokens`, `run_test_files`, `run_failures`) | executed commands: head, referenced scopes, narrowing selectors, failing test names | bash args + exit codes + result text |
| `plane_edits` | file mutations, test-file flag | write/edit tool args |
| `plane_action_files` | action↔file relations: `read` / `search_hit` / `write` / `edit` (correlation layer; IFG lineage) | read tool, bash read-shaped heads, grep/rg output the agent saw |
| `plane_claims` | assistant statements per turn, final flag | decide events (live) / trajectory DB (batch) |
| `plane_repo_files`, `plane_imports` | repository facts observed through the agent's own reads and writes | result text clips; an active repo-scan sensor can extend the same tables |
| `plane_superseded` | relation: green run covers red run (same head, equal-or-wider scopes, not narrower) | derived join |
| `v_validations` / `v_reds` / `v_greens` | neutral filters only | — |

`DataPlane.query()` is the single query entry, shared by the live watcher,
offline replay, the critic's evidence digest, and ad-hoc analysis:

```
python -m policy_engine query <session.db> "SELECT ... FROM plane_runs ..."
```

Live mode rebuilds facts in the session DB at every decision point (a
session is a few hundred rows; rebuild is idempotent). Snapshot mode
attaches archives read-only, so calibration never mutates the corpus.

## Signals: the trigger DSL

A signal is one SQL query over the plane; it is true when any row returns.

```yaml
under_validation:
  query: >-
    SELECT 1 WHERE (SELECT COUNT(*) FROM plane_edits) >= :min_mutations
      AND (SELECT COUNT(*) FROM v_validations)
          < (SELECT COUNT(*) FROM plane_edits) * :ratio
  params: { min_mutations: 8, ratio: 0.25 }
  evidence:
    query: "SELECT ..."
    format: "{0} file modifications vs {1} validation runs"
  provenance: derived from the 2026-07-23 batch (posthog 31/2, firezone 56/4)
```

Rules:

- every signal carries `provenance`: where the rule came from and how it
  was calibrated;
- thresholds live in `params` (overridable via atom config
  `signal_params`), never inlined;
- a new signal ships only after `replay` over the corpus, judged by
  per-emission evidence (see Calibration), not aggregate counts.

## Checklist items

```yaml
- id: evidence_adequacy_3
  deliver: inject            # inject | critic | offline
  when: { signal: narrow_only, checkpoint: continuous, arm: always }
  check: <the mined check question, quoted to the agent when it fires>
  when_note: <the miner's original trigger prose, kept as provenance>
```

- `deliver: inject` — reserved for signals with audited zero-noise
  emissions; fires once, quotes the agent's own commands as evidence.
- `deliver: critic` — the item queues for the stop decision; all gated-open
  items go to ONE delivery: either a self-check injection (the questions,
  agent reviews itself) or a reviewer subagent whose confirmed verdict is
  injected. Which form wins is an open experiment.
- `deliver: offline` — label-pipeline checks, never evaluated at runtime.
- `arm` names a task-side flag (perf_motivated, persistence_narrative, …)
  from a one-shot LLM classification of the instruction; until the
  classifier is wired, arming is permissive.

Interventions never carry task answers: messages are process facts plus
the check question. Budget: `max_injections` per session (default 3).

## Calibration workflow

```
python -m policy_engine replay <sessions-dir>
```

replays the recorded corpus through the exact live code path and prints
every emission with the evidence behind it. Acceptance is reading those
emissions: each firing must be correct *by its evidence*, not just
directionally. This process has caught, among others: failed-run blindness
(is_error filtering), `cat`/`gofmt`/`ls` counted as test runs, scope
matching that only worked for cargo, selector extraction that missed every
`go test -run`, and a config file defeating the self-authored-oracle rule.
Aggregate trigger-rate tables are a smell; emission-level evidence is the
standard.

## Known gaps (in priority order)

1. Batch plane construction from the trajectory DB (psql) — messages/claims
   axis for the ~12 items that reference the agent's own statements.
2. Re-point the miner at the plane (its tools become plane queries), so
   distilled items arrive with compilable `when` SQL.
3. Task classifier → arm flags (currently permissive).
4. Repository dependency sensor (downstream-consumer coverage items).
5. Delivery experiment: self-check vs subagent critic on a near-miss task.
