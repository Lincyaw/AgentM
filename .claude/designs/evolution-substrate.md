# Design: Evolution Substrate

**Status**: PROPOSED
**Created**: 2026-05-01
**Builds on**: [observability.md](observability.md), [pluggable-architecture.md](pluggable-architecture.md), [extension-as-scenario.md](extension-as-scenario.md)
**Sister doc**: [self-modifiable-architecture.md](self-modifiable-architecture.md)

---

## 1. First Principle

> **A version without its observations is a dead artifact. The substrate that lets an agent evolve must bind every (atom, scenario) version to all observation data generated under it, so that rollback, composition, and optimization are evidence-driven, not error-driven.**

`self-modifiable-architecture.md` establishes that change is *safe* (transactional, validated, atomic). This doc establishes that change is *informed*: an agent considering a rollback or a composition shift can answer "what did this version actually do, on what tasks, with what cost, with what failure modes" — by querying a catalog, not by guessing.

Without this layer, self-mod degenerates to "react to install errors". With it, self-mod becomes an MLOps loop: atoms = models, scenarios = compositions, traces = training data, catalog = experiment tracking + model registry.

**Concrete example.** Compaction prompts (`_SUMMARIZATION_PROMPT`, `_BRANCH_SUMMARY_PROMPT`) used to live as hard-coded constants in `core/compaction/`. After the constitution split (see `self-modifiable-architecture.md` §2.2), they live in the `llm_compaction` atom — a tier-1 autonomy module whose source is now versioned in `.agentm/catalog/atoms/llm_compaction/<hash>/`, paired with `metrics.jsonl` recording per-version compression ratio, downstream task completion, refusal rate, and cost. The agent can propose a tighter prompt, run it, compare against the previous hash via `tool_catalog.compare`, and roll back through `decisions.jsonl` if the guard metrics regress. Same machinery as for tool atoms — the substrate does not distinguish "tool" from "prompt" from "scheduler"; it only distinguishes versioned (atom, scenario) artifacts from observation streams.

---

## 2. Layer Position

The substrate is a constitution-owned write layer that the autonomy layer can only **read** from (and append a narrow class of decisions to). Re-stating the layering from `self-modifiable-architecture.md`:

```
constitution    core + validator + reload primitive + catalog write-protect
   ↑
evolution       catalog (versions × observations × decisions)        ← THIS DOC
   ↑
autonomy        atoms + scenarios
   ↑
discovery       fs scan + reload + assert_active
```

The substrate has three roles:

| Role | Owner | Direction |
|---|---|---|
| Producer | `observability` extension (autonomy, but pure subscriber) | writes raw JSONL |
| Indexer | catalog indexer (constitution-layer module: `agentm.core._internal.catalog`) | reads raw, writes aggregated metrics |
| Query | `tool_catalog` (autonomy, tier-1) | exposes read API to agent |

The agent never writes raw observability; the indexer never runs in the autonomy layer; the catalog data is write-protected from the agent (sister doc §3).

---

## 3. Catalog Data Structure

Lives at `.agentm/catalog/`. Constitution-layer write-protected.

```
.agentm/catalog/
├── core/
│   └── <core_hash>/
│       └── manifest.yaml                # tag, repo commit, build timestamp
├── atoms/
│   └── <atom_name>/
│       ├── current → <hash>             # symlink to active version
│       └── <content_hash>/
│           ├── source.py                # frozen, byte-identical to what loaded
│           ├── manifest.yaml            # api_version, affects, tier, parent_hash, author
│           ├── metrics.jsonl            # append-only; bucketed records
│           ├── runs/                    # one symlink per trace_id that used this version
│           │   └── <trace_id> → ../../../../observability/<trace_id>.jsonl
│           └── decisions.jsonl          # append-only; activate/deactivate/regress events
└── scenarios/
    └── <scenario_name>/
        ├── current → <hash>
        └── <yaml_content_hash>/
            ├── recipe.yaml              # frozen
            ├── manifest.yaml            # included atom_versions (dependency closure)
            ├── metrics.jsonl
            ├── runs/
            └── decisions.jsonl
```

### 3.1 `manifest.yaml` per atom version

```yaml
name: tool_read
content_hash: e5f6...
api_version: 1
parent_hash: a1b2...                       # null for the genesis version
author: "agent" | "human"
authored_at: 2026-05-01T12:34:56Z
tier: 1
affects:
  primary: ["read.success_rate"]
  secondary: ["context.tokens", "io.latency_ms"]
description: "Read tool with truncation + multi-edit detection"
```

`affects.primary` is what the agent claims this atom is *for*. `affects.secondary` is what it touches incidentally. The catalog tracks both. Universal guard metrics (cost, safety, refusal_rate, error_rate) are tracked on every atom, not declared per-atom.

### 3.2 `metrics.jsonl` per atom version

Append-only. One record per (task_type × scenario × time_bucket) the indexer aggregates. Records are computed, never authored:

```jsonc
{
  "indexed_at": "2026-05-02T03:00:00Z",
  "scenario": "general_purpose",
  "task_type": "rca",
  "n_runs": 47,
  "metrics": {
    "read.success_rate":     {"mean": 0.91, "ci95": [0.86, 0.95], "n": 312},
    "context.tokens":        {"mean": 14820, "ci95": [13900, 15740], "n": 47},
    "cost_per_task":         {"mean": 0.42, "ci95": [0.38, 0.46], "n": 47},
    "safety.refusal_rate":   {"mean": 0.00, "ci95": [0.00, 0.06], "n": 47},
    "task.completion_rate":  {"mean": 0.85, "ci95": [0.74, 0.93], "n": 47}
  },
  "regressed": false
}
```

### 3.3 `decisions.jsonl` per atom version

Append-only. Records every activation, deactivation, regress flag, rollback. **Rationale and evidence are part of the record** — the agent's memory of why it made a decision must persist.

```jsonc
{"at": "2026-05-01T...", "kind": "activate", "by": "agent", "rationale": "first version"}
{"at": "2026-05-03T...", "kind": "deactivate", "by": "agent",
 "rationale": "v2 (hash=...) showed 12% lift in read.success_rate on rca tasks (n=47, ci95 disjoint from 0)",
 "evidence": {"compare_id": "...", "successor_hash": "..."}}
{"at": "2026-05-08T...", "kind": "regressed", "by": "indexer",
 "reason": "safety.refusal_rate +18% from baseline, ci95 disjoint from 0"}
{"at": "2026-05-09T...", "kind": "rollback_blocked", "by": "tool_catalog",
 "rationale": "agent proposed reactivation but version is regressed=true on safety guard"}
```

---

## 4. Active-Set Fingerprint

The binding mechanism between observation and version. Every observability trace begins with a `session.start` record carrying the fingerprint of the *complete loaded set* of versions:

```jsonc
// First record of .agentm/observability/<trace_id>.jsonl
{
  "type": "session.start",
  "trace_id": "01JX...",
  "timestamp": "2026-05-01T...",
  "fingerprint": {
    "core":     "core@a1b2c3",
    "scenario": "general_purpose@d4e5f6",
    "atoms": {
      "tool_read":        "tool_read@e5f6",
      "tool_bash":        "tool_bash@7890",
      "tool_edit":        "tool_edit@1234",
      "system_prompt":    "system_prompt@5678",
      "micro_compact":    "micro_compact@9abc",
      "observability":    "observability@def0"
      // ... every loaded atom, including those not directly invoked
    }
  },
  "task_meta": {
    "type":       "rca",
    "difficulty": "hard",
    "external_id": "agentm-v12_7901"     // optional: ties to eval DB
  }
}
```

Producer is the `observability` extension — when it sees `SessionReadyEvent`, it computes the fingerprint from the active extension registry and writes the header. The fingerprint is **immutable for the life of the trace**; if a reload happens mid-session, that's a separate event in the stream (see §4.2).

### 4.1 Why every loaded atom, not just invoked

Causal honesty. A tool the agent never *calls* in this session can still affect outcomes — `micro_compact` runs autonomously, `permission` blocks operations, `cost_budget` halts mid-task. Pinning the full set lets the indexer attribute task-level metrics correctly.

### 4.2 Mid-session reload

When `reload_atom` succeeds during a live session, observability emits:

```jsonc
{
  "type": "atom.reload",
  "name": "tool_read",
  "old_hash": "e5f6...",
  "new_hash": "9999...",
  "trigger": "agent" | "human" | "propose_change_approved",
  "trace_id": "..."
}
```

Subsequent events in the same trace are attributed to the new fingerprint by the indexer — but the reload event itself is the boundary marker. Mid-session reloads are inherently confounded for that trace's metrics; the indexer flags them with `mid_session_reload: true` so `compare()` can exclude (default) or include (explicit override).

---

## 5. The Indexer

Lives at `agentm/core/_internal/catalog/indexer.py` (constitution layer). Triggered after each session ends (default), or by cron, or by explicit CLI command. **Idempotent**: catalog is fully derivable from raw observability + catalog/atoms/<hash>/source.py (which is the frozen artifact).

### 5.1 Pipeline

```
for each new trace_id in .agentm/observability/:
  1. parse session.start header → fingerprint, task_meta
  2. parse all events → derive metrics:
       - tool_call success/failure counts per tool
       - context.tokens at session.end
       - cost_per_task from llm.request.* events (tokens × pricing)
       - turn count, completion flag
       - any custom metric an atom declared in MANIFEST.affects
  3. for each (atom_name, atom_hash) in fingerprint.atoms:
       atomic-append to atoms/<name>/<hash>/metrics.jsonl
       symlink atoms/<name>/<hash>/runs/<trace_id> → raw trace
  4. for scenario in fingerprint.scenario:
       atomic-append to scenarios/<name>/<hash>/metrics.jsonl
  5. recompute regressed flag for the (atom_hash, scenario) buckets
       affected — see §7.3
```

### 5.2 Aggregation buckets

`metrics.jsonl` is bucketed by `(scenario, task_type, time_window=daily)`. Daily granularity is enough for evolution decisions and keeps the file size bounded. The indexer can rebuild from raw any time, so retention is flexible.

### 5.3 What the indexer does NOT do

- It does not make decisions. `regressed: true` is a marker, not an action — `tool_catalog.find_best` consumes it.
- It does not delete. Old buckets stay; agent can ask for full history.
- It does not run in the autonomy layer. The agent cannot influence what gets indexed.

---

## 6. `tool_catalog` — Query Surface

A new tier-1 atom: `extensions/builtin/tool_catalog.py`. Read API to the agent, plus one mediated write (`propose_change`).

### 6.1 API

```python
list_versions(atom: str) -> list[VersionRef]
    # all known versions of an atom, with hash, authored_at, author, current marker

get_manifest(atom: str, version: str) -> Manifest
    # full manifest.yaml for a specific version

compare(
    atom: str,
    v1: str, v2: str,
    metric: str,
    *,
    task_type: str | None = None,
    scenario: str | None = None,
    n_min: int = 20,
) -> ComparisonResult
    # returns: {
    #   metric, v1: {mean, ci95, n}, v2: {mean, ci95, n},
    #   effect_size, p_value, inconclusive: bool,
    #   experiment_isolated: bool,           # was confounding controlled?
    #   excluded_runs: int                   # mid-session reloads etc.
    # }

find_best(
    metric: str,
    *,
    task_type: str | None = None,
    scenario: str | None = None,
    constrain: dict[str, str] | None = None,
        # e.g. {"cost_per_task": "<=baseline+10%", "safety.refusal_rate": "<=baseline"}
    exclude_regressed: bool = True,
) -> VersionRef | None
    # None means "no version dominates"; agent must not assume any default

runs_for(fingerprint: dict | str) -> list[TraceRef]
    # exact-set fingerprint → list of trace_ids that ran with that set

decisions_for(atom: str, version: str | None = None) -> list[Decision]
    # full decision history; respects historical decisions when proposing new ones

propose_change(
    target: AtomRef | ScenarioRef,
    new_source: str | None = None,         # for atoms: full new source; for scenarios: YAML
    activate_existing_version: str | None = None,
        # alternative to new_source: switch back to a known version
    rationale: str,
    evidence: dict,                         # references to compare()/runs_for() results
) -> ChangeProposal
    # tier 1 → executes via reload (sister doc §5)
    # tier 2 → enqueued for approval; returns {status: "pending_approval", proposal_id}
    # writes to decisions.jsonl on success AND failure
```

### 6.2 `propose_change` is the only write path

Direct file writes to `extensions/builtin/<atom>.py` from the agent are blocked by `tool_edit` (sister doc §8). Atoms enter the catalog only via `propose_change`, which routes through `reload_atom`. This is the sole evolution channel.

### 6.3 Decision-respect contract

`propose_change` consults `decisions_for` before executing. If the proposal targets a version previously marked `regressed` for any reason, the proposal is **blocked** with a structured error including the prior decision record. The agent must explicitly `propose_change(..., override_prior_regression=True, rationale="...")` to proceed. This makes oscillation expensive, not impossible — there are legitimate reasons to revisit a regression (model upgrade, task distribution shift) and the override leaves an audit trail.

---

## 7. Hard Problems: Concrete Countermeasures

### 7.1 Statistical power

**Problem**: with N=5 runs per version, "v2 is 12% better" is noise.

**Countermeasure**:
- `compare()` always returns confidence intervals, never raw point estimates.
- If `n_samples < n_min` (default 20 per side, configurable per metric class) OR CI of the difference includes zero, `inconclusive: true` and the result string says so explicitly.
- `find_best` excludes versions whose lead over baseline is inconclusive — `None` is a legitimate answer.

This forces the agent to seek more evidence before deciding, instead of overfitting to small samples.

### 7.2 Confounding variables

**Problem**: agent edits `tool_read` and `micro_compact` simultaneously; metrics move; cause is unattributable.

**Countermeasure**:
- A constitution-layer flag in `core-manifest.yaml`: `experiment_mode_atom: <name> | null`.
- When set, `propose_change` for any *other* atom is blocked.
- Default is null (no experiment in progress) — agent can change anything, but `compare` records `experiment_isolated: false` so downstream analysis is honest about the confounding.
- `tool_catalog.start_experiment(atom: str, rationale: str)` and `end_experiment(...)` allow the agent to declare an isolated study (decision recorded). During an experiment, only the named atom can move.

### 7.3 Goodhart's law

**Problem**: optimizing one metric drifts other dimensions silently.

**Countermeasure**:
- Universal **guard metrics** tracked on every atom version regardless of `MANIFEST.affects`:
  ```
  cost_per_task
  safety.refusal_rate
  safety.harm_score    (later, when classifier exists)
  error.rate
  task.completion_rate
  ```
- Indexer flags `regressed: true` on a version if **any** guard metric degrades by more than X% from the baseline (parent_hash) with statistical significance.
- `find_best` excludes regressed versions by default; agent must explicitly opt in.
- `affects.primary` and `affects.secondary` declared by atoms are *additional* metrics the catalog tracks; they don't replace guards.

### 7.4 Replayability

**Problem**: deciding a new atom is better requires running tasks under it. Live A/B is slow and expensive.

**Countermeasure** (Phase 3, not MVP):
- A `replay_cache` extension at the LLM boundary. Hash `(model, messages, tools_schema, thinking_level)` → response. Stored at `.agentm/replay_cache/`.
- `tool_catalog.replay(trace_id, swap={atom_name: new_version})` re-runs a trace's user-message flow with one atom swapped, using cached LLM responses for unchanged turns.
- Caveat: replay is only valid when the swap doesn't change the message stream presented to the LLM (i.e., the atom is downstream of the LLM call — like a tool implementation or a result formatter, not a system-prompt mutator). Catalog flags ineligible replays.
- Replay results land in `metrics.jsonl` with `synthetic: true` so they don't pollute live-traffic statistics, but accelerate offline screening.

### 7.5 Catalog corruption

**Problem**: agent's first move when an atom regresses on a metric is to "fix the metric definition".

**Countermeasure**:
- `.agentm/catalog/**` is in the constitution-layer `paths:` list (sister doc §3).
- `tool_edit`, `tool_write`, and any future filesystem-mutating atom check `is_constitution_path` and reject.
- The indexer is the only constitution-layer process that writes `metrics.jsonl`. Indexer code is constitution-layer.
- `decisions.jsonl` is the one file the autonomy layer can append to — but only via the mediated `propose_change` API, which schema-validates the decision shape and rejects free-form bytes. The append happens server-side (in the constitution layer); the agent never gets a file handle.

---

## 8. Scenarios in the Catalog

Scenarios are first-class evolvable artifacts, not just atom containers. The catalog stores them with the same shape as atoms:

```
.agentm/catalog/scenarios/general_purpose/<yaml_content_hash>/
  recipe.yaml
  manifest.yaml
  metrics.jsonl
  decisions.jsonl
```

`manifest.yaml` for a scenario records the **dependency closure**:

```yaml
name: general_purpose
content_hash: d4e5f6...
recipe_yaml_hash: ...
atom_versions:
  tool_read: e5f6...
  tool_bash: 7890...
  ...
parent_hash: ...
author: agent | human
```

The closure hash makes scenarios A/B-able as a unit: "is `general_purpose@d4e5f6` (with `tool_read@e5f6`) better than `general_purpose@d999..` (with `tool_read@e2c1`) on RCA tasks?" — this is the same `compare()` call, but on a scenario.

`find_best_scenario(metric, task_type)` mines for compositions that dominate. Scenario-level optimization is where compositional gains live: an atom may be marginally worse, but its combination with another atom may unlock a new behavior.

---

## 9. Acceptance Scenarios

| # | Scenario | Expected |
|---|----------|----------|
| E1 | After 30 RCA traces under `tool_read@v1`, 25 under `@v2`, agent calls `compare("tool_read","v1","v2","root_cause_hit_rate","rca")` | Returns `{v1: {mean, ci}, v2: {mean, ci}, effect_size, inconclusive: false, experiment_isolated: bool}` with concrete numbers |
| E2 | After 5 traces under `@v2`, same compare call | Returns `inconclusive: true`, `n_samples: 5`, "insufficient data to decide" message |
| E3 | `find_best("context.tokens", task_type="rca")` while top candidate has `safety.refusal_rate` regressed | Candidate excluded; either next candidate or `None` |
| E4 | Agent calls `tool_edit` on `.agentm/catalog/atoms/tool_read/abc/metrics.jsonl` | Rejected by constitution path check |
| E5 | Operator runs `python -m agentm.core._internal.catalog.indexer rebuild` after wiping catalog | `metrics.jsonl` byte-identical to original (modulo timestamp ordering); `decisions.jsonl` not regenerated (it's not derivable) |
| E6 | Agent A in experiment mode for `tool_read`, agent B (or same agent) calls `propose_change("tool_bash", ...)` | Rejected; "another atom in experiment mode: tool_read" |
| E7 | Agent proposes reactivating a version previously regressed for safety | Blocked with prior decision record cited; agent must use explicit override + rationale |
| E8 | Active-set fingerprint changes mid-session via reload | New fingerprint recorded as `atom.reload` event; downstream events flagged `mid_session_reload: true`; default `compare()` excludes them |
| E9 | Scenario `general_purpose@d4e5f6` evaluated against `general_purpose@d999` on identical task distribution | `compare()` works at scenario granularity; can attribute differences to the dependency-closure delta |
| E10 | Agent calls `find_best` with no constraints on a metric where all known versions are inconclusive | Returns `None`; agent receives "no version dominates" — this is the correct answer, not an error |

---

## 10. Phasing

### MVP (ship first)

Goal: turn raw observability into a queryable atom registry. No decision-making yet.

- Active-set fingerprint in observability `session.start` headers (modify `observability` extension).
- Catalog directory layout (`atoms/<name>/<hash>/`, `scenarios/<name>/<hash>/`).
- Content-hash for atoms (constitution module: `agentm.core._internal.catalog.hashing`).
- `freeze_current` writes the source + manifest into the catalog at every reload (called from sister doc §5.2).
- Indexer (minimal): on session end, parse fingerprint, attribute basic counters (n_runs, completion, cost) to involved atom versions.
- `tool_catalog`: `list_versions`, `get_manifest`, `runs_for` (read-only browsing).

### Phase 2

Goal: enable evidence-driven decisions, not just browsing.

- `compare()` with confidence intervals (use `scipy.stats` or in-house bootstrap).
- `find_best` with constraints + Goodhart guard metrics.
- `decisions.jsonl` and `propose_change` (writes routed through reload API).
- Decision-respect contract for prior regressions.
- Scenario-level catalog and queries.

### Phase 3

Goal: accelerate the loop and reduce A/B cost.

- Replay cache at LLM boundary.
- `tool_catalog.replay` for offline screening.
- Experiment-mode lock (`start_experiment` / `end_experiment`).
- Web/TUI dashboard reading the catalog (presenter, not constitution).

---

## 11. Open Questions

- **Replay cache scope**: per-user vs per-org. Caching across users requires hashing only deterministic prompt content, not user data. Out of scope until SaaS.
- **Cross-task-type generalization**: an atom may be best on `rca` but worse on `general`. Catalog tracks per-task-type, but the agent needs heuristics for "which task is this current one closest to". Defer to a future `task_classifier` atom; out of scope here.
- **Observation cost**: storing every trace as raw JSONL grows unbounded. Retention policy (e.g., keep raw for 30 days, keep aggregates forever) is operator concern; the substrate must work with both.
- **Catalog of guard-metric definitions**: guard metrics themselves are versioned (cost formula changes with new pricing). Implementation: `guard_metrics.yaml` lives in constitution; changes are human-only PR.

---

## 12. References

### AgentM concepts touched

- `observability.md` — raw producer this layer indexes; `session.start` header is extended with the fingerprint field.
- `pluggable-architecture.md` — boundary contract; this doc adds a new layer between core and autonomy.
- `extension-as-scenario.md` §11 — atom manifest schema; gains `affects` and (already in sister doc) `tier`, `api_version`.
- `self-modifiable-architecture.md` — sister doc; defines what a "version" is (output of transactional reload) and reload's catalog hooks.

### What pi-mono does NOT have (we are inventing this)

`packages/coding-agent/src/core/telemetry.ts` — pi-mono ships a flat install-telemetry endpoint and per-session diagnostics, but no version registry, no per-version metrics, no compare API, no decision log. Self-modification is not their goal; an evolution substrate is the differentiator.

### Background

The substrate is an MLOps pattern recognizable from model serving:

| MLOps concept | AgentM mapping |
|---|---|
| Model registry | `catalog/atoms/`, `catalog/scenarios/` |
| Experiment tracking | observability + fingerprint |
| Feature store | active-set fingerprint binding |
| A/B testing | `compare()` + experiment-mode lock |
| Model card | `manifest.yaml` per version |
| Shadow / canary | replay cache + Phase 3 replay |
| Decision log | `decisions.jsonl` |
| Guard metrics | universal regressed-flag metrics |

The novelty is that the "model" being managed is *the agent's own components*, and the agent itself is a participant in the loop — both consumer of the registry and (mediated) author of the changes.
