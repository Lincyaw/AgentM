# policy-engine

Evidence-driven failure detection and intervention for coding agents.

A coding agent's trajectory is a sequence of tool calls that read, write,
and execute against a codebase. This package detects structural failure
patterns as queries over the trajectory, recommends relevant checklist
items, and intervenes when those items are confirmed.

## The design

This is the intended shape. It is the reference we align on; where the code
currently departs from it, that is a defect to close, and the open ones are
listed under [Known departures](#known-departures).

**Within a session** — two observers feed one gate, the gate delivers at most
one check, and finishing goes through review:

```mermaid
flowchart LR
  AG([Agent loop])
  DP[1 Data plane]
  TG[2 Tagger]
  RT{3 Retrieval}
  DV[4 Delivery]
  AC[5 Acceptance]
  END([Ends])

  AG -->|tool events| DP
  AG -->|turn events| TG
  DP -->|facts| RT
  TG -->|tags| RT
  RT -->|both hold| DV
  DV -->|one check| AG
  AG -->|submit| AC
  AC -->|send back| AG
  AC -->|holds up| END

  style DP fill:#e8f0fe,stroke:#4285f4
  style TG fill:#e8f0fe,stroke:#4285f4
  style RT fill:#fef7e0,stroke:#f9ab00
  style DV fill:#e6f4ea,stroke:#34a853
  style AC fill:#e6f4ea,stroke:#34a853
```

| | Stage | What it is |
|---|---|---|
| 1 | Data plane | Tool events and the repository symbol index, as queryable facts. Neutral: no judgments. |
| 2 | Tagger | A side-car conversation, batched, emitting cumulative semantic tags. |
| 3 | Retrieval | An item fires only when its tag trigger **and** its fact precondition both hold. |
| 4 | Delivery | One check, as one user message, while the work is still open. |
| 5 | Acceptance | A reviewer with a shell at submit, running a differential over the task's claims. |

**Between sessions** — what fired and what came of it is the training signal:

```mermaid
flowchart LR
  TRAJ[(trajectories and outcomes)]
  OBS[(firing observations)]
  MINE[mine] --> COMP[compile] --> DEPL[deploy] --> EVAL[evaluate] --> SEL[select]
  SEL --> MINE
  TRAJ --> MINE
  OBS --> EVAL
  SEL -.->|revised| CK[checklist and vocabulary]

  style EVAL fill:#fef7e0,stroke:#f9ab00
  style OBS fill:#fce8e6,stroke:#ea4335
```

`trajectories and outcomes` come from ①; `firing observations` come from ④ and
⑤ — did the agent act on the check, did the reviewer find the concern real.
That is the edge which makes the checklist self-revising, and it is the one
currently missing.

Two properties are what make this a loop rather than a pile of parts:

**Detection is grounded in facts, not only in opinion.** A tag says a concern
*resembles* this session; a precondition query over the data plane says the
situation the concern presupposes has actually arrived. A check about narrowed
green test runs must not fire in a session that has not run a test yet.

**Fitness is computed, not supplied.** Every firing produces an observation —
did the agent act on it, did the reviewer find the concern real — and those
observations are what prune the checklist. A loop whose selection step needs a
human verdict is not a loop.

### Known departures

| # | Departure | Where |
|---|---|---|
| 1 | The data plane has no runtime consumer. Facts and symbols are written every turn and never read back; retrieval is tags only. | `__init__.py::_on_decide` |
| 2 | Triggers have no precondition, so an item fires as soon as its tags appear. Measured: 6 of 11 firings of the validation-scope item landed in sessions that had run no test at all. | `triggers.py::_matching` |
| 3 | `submit` is registered by this atom, so session termination depends on an optional extension being installed and enabled. | `__init__.py::install` |
| 4 | Acceptance is reachable only after a stop check fires; no checklist match means no review. | `__init__.py::_on_decide` |
| 5 | `evaluate` reports fire rates, not fitness, and `select` reads a hand-written JSON. | `__main__.py::cmd_select` |

## Data plane

The data plane provides **neutral facts** over the trajectory. No pattern
judgments — failure detection belongs in retrieval/signals, not here.

### Two inputs

**1. Agent tool events (from SDK trajectory, PG).**

Each tool event produces an **action → region** edge:
- `read(path, offset, limit)` → read edge to (path, [offset, offset+limit])
- `edit(path, old, new)` → edit edge to (path, file-level approximation)
- `write(path, content)` → write edge to (path, whole file)
- `bash(cmd)` → command text + exit code (no classification)

Exposed as PG views in the `policy` schema:
- `v_tool_calls` — all tool calls flattened
- `v_regions` — action → region (path, start_line, end_line, relation)
- `v_reads` / `v_edits` / `v_mutations` — convenience filters
- `v_bash_runs` — bash commands with exit_code
- `v_source_content` — source text the agent observed

**2. Repository symbol index (optional, degradable).**

AST-parsed symbol definitions and references, built lazily by
`RepositoryIndex` using `ast-grep` in the sandbox. When the agent
first reads or edits a file, that file + its 1-hop import neighbors
are parsed. The index expands as the agent's activity expands; edits
trigger re-parsing of the affected file.

Only the agent's activity subset is synced to PG — not the whole repo.
Stored in `policy.file_symbols`, exposed via `policy.v_file_deps`
(file → file dependencies via shared symbol names).

When no index is available, `file_symbols` is empty, `v_file_deps`
returns nothing, and signals that depend on symbol data silently
degrade.

### Failure detection = gap between the two

The action graph says what the agent touched. The symbol index says
what depends on what. A signal query joins the two to find gaps. The
data plane never encodes these gaps itself — signals do.

## Tagger

A side-car conversation that grows by one exchange per batch of turns, running
alongside the agent on the session's own provider.

It was originally one stateless call per turn, seeing a single step. That made
half the vocabulary unanswerable — predicates quantified over the session
("*all* executed test commands narrow scope", "*only* agent-authored tests",
"the same failure in *at least two* runs") cannot be decided from one step — so
it stayed silent on the discriminating ones and re-emitted the obvious ones
every turn. On one recorded session: 5 distinct tags, of which 3 checklist
items lit and 0 continuous ones.

### Shape

The system prompt and every prior batch stay byte-identical at the head, which
is what prompt caching rewards; only the tail moves. Each reply reports what
became true *since the last one*, so a tag is stated once and stays in force.

Steps arrive as **events, not contents**: files read, files edited with a line
count, commands with their exit status, whatever the agent said. File bodies,
diffs and stdout are what made the input large and decide no predicate. A turn
costs tens of tokens, so the whole history is cheaper than the old per-turn
call was.

Turns are batched (`tagger_interval`, default 5). Per-turn resolution buys
nothing: tags feed a session-cumulative set and triggers read the set as a
whole, never which turn a tag arrived on. The buffer is flushed early wherever
a decision needs the tags — when the agent wraps up, before an acceptance
review, and at run end.

Same session, three versions:

| | model calls | parse failures | distinct tags | items lit (stop/continuous) |
|---|---|---|---|---|
| per-turn, stateless | 48 | ~13 | 5 | 3 / 0 |
| per-turn, side-car | 48 | 0 | 18 | 10 / 6 |
| batched by 5 | **10** | **0** | **15** | **9 / 7** |

### Output

It answers by calling a `record` tool carrying **phase** (exploring /
diagnosing / implementing / validating / concluding) and **tags**. Arguments
arrive parsed, so there is no prose to get past — asking for JSON in free text
lost 19% of annotations on one run, and lost them disproportionately on the
turns where the agent had written the most, which are the turns worth reading.
The tags enum is generated from `vocabulary.yaml`, so a tag outside the
vocabulary is unrepresentable rather than merely discouraged.

Annotations land in `policy.turn_annotations`, keyed by session and the last
turn in the batch.

### Prompt generation

The tagger prompt is generated from `vocabulary.yaml`. When compile adds or
prunes predicates the prompt follows, and the schema with it. No manual prompt
maintenance — the vocabulary is the single source of truth for what tags exist.

## Retrieval

Two-stage cascade selecting which checklist items are relevant:

### Stage 1: Predicate matching (cheap, high recall)

Each checklist item declares a `trigger` expression — a boolean
combination (AND / OR / NOT) of predicates from the vocabulary:

```yaml
- id: belief_consistency_1
  when:
    trigger: validation_failure AND completion_claim
    checkpoint: stop
```

A predicate is true when the tagger has emitted that tag for any turn
in the session. Matching is deterministic boolean logic over the
tagger's cached output — no LLM call.

### Stage 2: Acceptance review (at submit, optional)

Matching is recall. Precision is a reviewer that looks at the work — a child
session with the scenario's own tools, so it can read the diff, build a case
and run it in the same sandbox. It judges one question: does this do what the
task asked?

It is given a **method**, not a rubric, because a rubric only says what a bad
submission looks like. Verifying a fix is a differential over the symptom the
task states: list every claim the task makes, name what output carries each and
what separates symptom-present from symptom-gone, reproduce the symptom on the
pre-change code, run the same probe now, and compare against the side the task
asked for. `agents/acceptance.yaml` holds the prompt.

`submit_verdict` requires an `evidence` field on acceptance as well as
rejection. Three reviews in a row accepted an inverted fix: one wrote a probe
and never ran it, one ran a probe that printed only startup noise and then read
a cached test result. Each believed it had checked. Quoting the output is what
separates looking from intending to look.

The reviewer never blocks the session. Any failure of its own — no verdict, a
broken spawn, an exception — accepts, and `max_review_rounds` bounds how many
times it can send the agent back. Off by default (`critic: "off"`).

## Delivery

| where | how it arrives | budget |
|---|---|---|
| Mid-work | user message, while the agent is still working | `max_injections` |
| Wrapping up | user message at the stop checkpoint, naming `submit` as the way out | `max_stop_checks` |
| At submit | the acceptance reviewer's verdict, as the tool's result | `max_review_rounds` |

### Wording

Checks arrive as user messages, so they read like the person who asked for the
work — one doubt, plainly put. Headed, bulleted audit blocks got answered in
kind: a written `Pass / Partial` verdict per point and no change to the work.
One item per message, since surfacing several lets the agent absorb the
relevant one among plausible neighbours and move on.

Finishing is the agent's own act, through `submit`. The loop used to infer it
from a turn carrying no tool call, which cannot tell "the work is done" from "I
just answered your note" — so every reply drew another check until the budget
ran out, and the agent could not end the session, only outlast the engine.
Every check now names `submit` as the exit, and submit never refuses.

## Running it

Everything below runs the harbor scenario against Senior SWE-Bench. The atom is
inert unless `AGENTM_CHECKLIST_WATCH_ENABLED=true`, so baseline batches stay
clean without editing the scenario.

### One batch

```bash
K=$(python3 -c "import tomllib;print(tomllib.load(open('.agentm/harbor-gpt55/config.toml','rb'))['models']['litellm-dsv4pro']['api_key'])")

AGENTM_CHECKLIST_WATCH_ENABLED=true \
AGENTM_TRAJECTORY_SCHEMA=<fresh-schema> \
ARK_BASE_URL=http://101.126.39.61:8088/v1 ARK_API_KEY=$K \
SSB_JUDGE_MODEL=openai/DeepSeek-V4-pro \
uv run .agentm/harbor-gpt55/run_better_auth_eval.py azure-gpt <task...> --n-concurrent 10
```

Always give a **fresh schema**. One 5-day schema accumulated runs from several
code revisions, and every statistic taken over it mixed them silently.

`critic: "on"` in `contrib/scenarios/harbor/scenario.yaml` enables the
acceptance review; `"off"` leaves checks as reminders only.

### The judge

Three model slots, all separately overridable, and `ALL_JUDGE_MODEL` does *not*
cover the other two:

| Slot | Variable | What it does |
|---|---|---|
| Judge | `SSB_OVERRIDE_ALL_JUDGE_MODEL` | rubric + taste + validation review |
| Classifier | `SSB_OVERRIDE_CLASSIFIER_MODEL` | behavioural vs cosmetic patch files |
| Validation agent | `SSB_OVERRIDE_VA_MODEL` | generates tests from user stories |

The validation agent is the one that matters: it gates `correctness`, and when
it dies the verifier writes an empty reward file and the trial fails with
`RewardFileEmptyError` — no score at all, which is not the same as zero.

`reward` is **not** just the functional tests. It requires the tests to pass
*and* every validation story to pass, so a task can show `verifier: 2/2` with
`correctness: 0.0`.

The Volcengine endpoint carries a weekly quota that, once spent, kills all
three slots at once. `http://101.126.39.61:8088/v1` (DeepSeek-V4-pro) is
reachable from inside the sandbox and is not on that quota.

### Re-running only the judge

The verifier derives everything from the workspace, so a no-op agent over a
restored workspace re-scores without spending agent tokens:

```bash
uv run --extra harbor python -m harbor.cli.main run -p <task-dir> \
  -a nop --env agentm_harbor:ArlEnvironment \
  --ek fork_from=<arl_session_id> --ek fork_step=<arl_step> \
  --ve SSB_OVERRIDE_ALL_JUDGE_MODEL=... --ve OPENAI_API_KEY=... [...]
```

`arl_session_id` and `arl_step` come from `agent_result.metadata` in the
trial's `result.json`. Verified faithful: `taste_patch_bloat` reproduced to
three decimals, meaning the restored patch is byte-identical.

`-a oracle` applies the task's own reference solution instead — the right way
to sanity-check that the verifier itself works, since the oracle should score
1.0.

### Re-running only the acceptance review

Two forks stacked: ARL restores the workspace, harbor restores the
conversation. The agent reissues submit and acceptance spawns for real — no
special harness, and the atom rebuilds its state from the trajectory.

```bash
AGENTM_FORK_FROM_SESSION=<agentm_session_id>   # from result.json metadata
AGENTM_FORK_TURN=<the turn before submit>
AGENTM_FORK_PROMPT="Call submit to finish."
# plus --ek fork_from/fork_step as above, and the same values via --ae
```

Look for `policy_engine: restored N turn(s), M concern(s), K tag(s)` — if the
counts are zero the fork carried nothing and the reviewer will read an empty
run. Cost is roughly 7 minutes against 15 for a full run, which is what makes
iterating on the acceptance prompt practical.

### Reading a run

```bash
D=postgresql://agentm:agentm@localhost:55432/agentm_test
uv run agentm trace --dsn $D --schema <schema> sessions
uv run agentm trace --dsn $D --schema <schema> tools -s <session-id>
uv run agentm trace --dsn $D --schema <schema> view  -s <session-id>
```

The acceptance reviewer is a child session with `purpose='acceptance'`; reading
its tool calls is how you tell whether it actually looked at anything.

Results live in `jobs/<timestamp>/<trial>/`: `result.json` for rewards and ARL
metadata, `verifier/agent_filtered.patch` for what was graded,
`verifier/reward_details.json` for per-test outcomes,
`verifier/run_judge_*.stderr` for a skipped or crashed judge.

### Things that have bitten

- **Disk.** A batch pulls images and leaves containers behind (`--no-delete`).
  Docker reached 1.5 TB with ~550 GB reclaimable; `docker container prune -f`
  is the cheap fix. When the root filesystem fills, harbor keeps writing trial
  results but tooling that needs `/tmp` stops working, and a `result.json` can
  land empty.
- **Concurrency.** Trials of the same task image share a warm pool. Different
  images are safe to run alongside each other; the same image is not.
- **Whiteouts.** `agent.patch` includes OverlayFS `.wh.*` entries. Read
  `agent_filtered.patch` for the real change set.


## Evolution

The checklist, predicate vocabulary, and tagger co-evolve through an
adaptive loop.

### The loop

```
① Mine       miner queries failing trajectories along D1–D7,
             produces candidate items with when_notes

② Compile    compiler agent decomposes each when_note into a trigger
             expression over the predicate vocabulary; proposes new
             predicates when needed

③ Deploy     tagger prompt regenerated from vocabulary;
             checklist updated with trigger expressions

④ Evaluate   replay over the corpus; measure per-item fitness
             (when it fires, does the critic confirm?)

⑤ Select     prune low-fitness items and unreferenced predicates

⑥ Diversify  ensure D1–D7 coverage; deduplicate overlapping items
```

### CLI

```bash
python -m policy_engine compile              # ② when_notes → triggers + vocabulary
python -m policy_engine tag <dsn>            # tag all sessions with tagger LLM
python -m policy_engine replay <dsn> <sid>   # ④ replay one session through the gates
python -m policy_engine evaluate <dsn>       # ④ evaluate across all sessions
python -m policy_engine select               # ⑤ prune by fitness
python -m policy_engine evolve <dsn>         # ②→④ in one command
```

### Predicate vocabulary lifecycle

| event | action |
|---|---|
| New item needs a new concept | Compile proposes predicate, added to vocabulary |
| Predicate unreferenced by any item | Pruned from vocabulary |
| Vocabulary too large | Merge near-synonymous predicates |

### Miner framework (D1–D7)

The miner works along seven dimensions derived from the representation
chain:

```
task → understanding → diagnosis → plan → change → evidence → belief → claim/stop
```

| Dimension | Comparand pair |
|---|---|
| D1 Interpretation fidelity | task ↔ understanding |
| D2 Requirement coverage | understanding → plan/change |
| D3 Implementation fidelity | claimed behavior ↔ actual code semantics |
| D4 Diagnosis validity | claimed cause ↔ collected evidence |
| D5 Evidence adequacy | change ↔ executed validation |
| D6 Belief-evidence consistency | evidence ↔ belief/claim |
| D7 Loop dynamics | the chain over time |

The full dimension model is in `docs/policy-feedback-dimensions.md`.

## Related work

- **AdaMAST** (arXiv 2607.16387): adaptive failure taxonomies, fixed
  axes + induced codes. Key finding: passive context injection
  outperforms forced mechanical auditing.
- **Act·onomy** (arXiv 2605.13625): behavioral taxonomy for agent
  runtime. Orthogonal — classifies what the agent is doing, not
  where the mismatch is.
- **ActPlane** (arXiv 2606.25189v1): OS-level policy enforcement via
  eBPF + IFC DSL. `after/since` gate model deferred in favor of
  predicate matching for simplicity.
