# Policy Engine

**Status:** design draft
**Owner:** core / llmharness
**Last reviewed:** 2026-07-19

A deterministic, DSL-driven policy enforcement layer that detects
problematic patterns in agent trajectories and triggers targeted LLM
reasoning only when needed. Inspired by ActPlane (OS-level eBPF policy
enforcement for agent harnesses) but operating at the agent semantic
layer — tool calls, messages, session hierarchy — where the
interception surface is richer and the feedback mechanism more
expressive.

Related: [llmharness-cognitive-audit](llmharness-cognitive-audit.md),
[control-loop-harness](control-loop-harness.md),
[agent-loop](agent-loop.md).

---

## 1. Problem

The current llmharness fires an LLM auditor every k turns to check for
drift. This is:

- **Expensive** — LLM inference on every turn regardless of signal.
- **Unfocused** — the auditor reviews the entire trajectory without a
  specific hypothesis, reducing precision.
- **Opaque** — no declarative specification of what "going wrong" means;
  detection logic is baked into prompt prose.

Meanwhile, agents violate project policies (from CLAUDE.md, scenario
manifests, or task-specific instructions) through indirect paths that
simple tool-call guards miss — scripts that spawn subprocesses, multi-turn
sequences that individually look fine but collectively violate ordering
constraints, or data flowing from sensitive sources into unsafe sinks.

The ActPlane insight: separate **cheap deterministic detection** from
**expensive semantic reasoning**. Detect structurally, reason only when
triggered.

---

## 2. Design Principles

1. **Detection is deterministic, reasoning is optional.** The policy
   engine evaluates rules without LLM calls. LLM reasoning activates
   only when a rule fires (escalate effect).
2. **Agent-level enforcement is epistemic, not capability-based.**
   OS enforcement asks "can this process open that file?" Agent
   enforcement asks "does this agent have sufficient grounding to take
   this action?" and "is its trajectory converging?"
3. **Structured DSL, not prompt engineering.** Rules are data —
   parseable, testable, composable, versionable. The DSL rides YAML
   structure with Python expressions for predicates.
4. **Additive integration.** The policy engine is a standard atom. It
   uses existing EventBus channels, tool-call interception contracts,
   and diagnostic event emission. No kernel changes.
5. **Cost proportional to signal.** 90%+ turns cost zero LLM tokens.
   The <2ms deterministic evaluation budget is negligible against
   2–30s LLM inference turns.
6. **Measurement-first.** All rules ship from day one, classified as
   `enforce` or `observe`. Observe-mode rules evaluate and log
   triggers but produce no agent-visible effect. Effectiveness is
   determined empirically by comparing trigger logs against trajectory
   outcomes — rules graduate from observe to enforce based on data,
   not intuition.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Policy Files (.yaml)                       │
│   scenario manifest refs / ~/.agentm/policies/ / inline       │
└──────────────────────┬───────────────────────────────────────┘
                       │ compile (session start / hot reload)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                    PolicyEngine Atom                           │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │ Guard Index │  │   State     │  │  Rule Instances       │ │
│  │ (tool_name  │  │  (session/  │  │  - Query (L1)         │ │
│  │  → rules)   │  │   user/     │  │  - IFC (L2)           │ │
│  │             │  │   global)   │  │  - Meta (L3)          │ │
│  └─────────────┘  └─────────────┘  └──────────────────────┘ │
│  Subscriptions: ToolCallEvent, ToolResultEvent, TurnEnd,     │
│                 ChildSessionStartEvent, ContextEvent          │
└──────────────────────┬───────────────────────────────────────┘
                       │ on event: reactions → guard → eval → effect
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                      Effects                                  │
│                                                               │
│  notify   ──→ inject diagnostic into next turn context       │
│  block    ──→ return {"block": True, "reason": ...}          │
│  escalate ──→ inject rule + bound values + reason into       │
│               agent's next context (structured diagnostic)    │
│  abort    ──→ AgentAbortError (always enforced, never observe)│
└──────────────────────────────────────────────────────────────┘
```

Event name mapping to AgentM codebase:

| Doc name | AgentM event | Notes |
|---|---|---|
| ToolCallPre | `ToolCallEvent` | Fires before execution; supports `{"block": True}` return |
| ToolCallPost | `ToolResultEvent` | Fires after execution; carries result |
| TurnEnd | `TurnEndEvent` | End of a complete agent turn |
| SessionSpawn | `ChildSessionStartEvent` | Child session creation |
| MessagePre | `ContextEvent` | Before context sent to LLM |

---

## 4. DSL Specification

### 4.1 File Structure

```yaml
# <scenario>/policy.yaml or ~/.agentm/policies/<name>.yaml
version: 1

labels:            # §4.2 — IFC taint labels
rules:             # §4.5 — detection rules
reaction_config:   # §4.6 — extraction mode settings
```

Note: State tables (§4.6) and reactions are built-in to the engine,
not user-authored. The policy file declares only labels and rules.
Reactions are hardcoded recorders that populate the standard tables.

### 4.2 Labels (Information-Flow Control)

Labels track data provenance across tool boundaries. Phase 1 scope is
**minimal IFC**: literal substring tracking only, no lattice algebra.

```yaml
labels:
  secret:
    source: { tool: "read|bash", result_matches: "(?i)(api.?key|token|password)=\\S+" }
    clears_on: never
    min_length: 8           # ignore captures shorter than this
```

**Propagation mechanism (Phase 1 — literal substring match):**

1. **Source detection**: when a tool result matches a label's
   `result_matches` regex, the engine extracts the **actual captured
   values** (regex capture groups). These are the tainted literals.
2. **Taint store**: `{label_name: set[str]}` — a set of literal
   strings per label. Bounded: max 100 values per label per session
   (FIFO eviction on overflow).
3. **Propagation check**: on each `tool_call_pre`, for each arg value
   (string-coerced, recursive for dicts/lists), check if any tainted
   literal appears as a substring. If match → event carries that label.
4. **Minimum length**: captures shorter than `min_length` (default 8)
   are discarded — avoids false propagation on generic short strings.
5. **Clearing**: `clears_on: never` means the taint persists for the
   session. Future labels may support `clears_on` events.

**What this catches**: verbatim leakage — agent reads a secret, passes
it unchanged to a network tool. Covers the majority of real leaks.

**What this misses**: value transformations (base64, string
concatenation, variable embedding). These are deferred — measure the
miss rate via observe mode first. If needed, Phase 3 can add
structure-aware tracking (JSON path propagation).

**No lattice in Phase 1.** Labels are independent boolean tags on
events. No join/meet, no multi-label interaction. If the agent touches
a secret, the taint is "secret present in args: yes/no." Simple, fast,
no false-positive from lattice confusion.

**Performance**: substring search via Python `in` operator. With ≤100
tainted values of avg 20 chars, scanning a 10KB tool arg string costs
~0.1ms. Well within budget.

### 4.3 (Removed — state tables are built-in, see §4.6)

### 4.4 (Deferred — derived events)

Derived events (phase_transition, convergence_signal) are deferred
until data from observe-mode rules shows that L1/L2 query-based rules
cannot express needed trajectory-shape detection. If needed, they
would emit synthetic events from lightweight classifiers over recent
tool-call windows. Design on demand.

### 4.5 Rules

**Two rule families by evaluation path:**

| Family | Mechanism | Entity layer? | Example |
|---|---|---|---|
| **Deterministic** | event → query primitives → certain → effect | No | "block rm -rf", "test before commit" |
| **Semantic** | event → entity evidence → filter/signal → escalate with evidence | Yes | "writing to path with no grounding evidence" |

Deterministic rules produce hard effects (block/abort) because their
predicates are unambiguous. Semantic rules produce soft effects
(escalate/notify) because they rely on evidence strength, not
certainty — the agent sees the evidence and decides.

Four-layer rule hierarchy (orthogonal to the family distinction):

| Layer | Type | Compilation target | Example |
|---|---|---|---|
| L0 | Reaction | Inline state update | "record path to file_state on read" |
| L1 | Query | Compiled predicate closure | "block rm -rf", "test before commit since last edit" |
| L2 | IFC | Taint check on event | "secret data cannot reach network" |
| L3 | Meta | Queries effect_log | "abort after 2 escalations ignored" |

Note: temporal rules (using `since`/`last` window queries) compile to
the same predicate closure as stateless rules. The query primitives
provide implicit temporal memory via window parameters — no explicit
state machine needed. A rule like "test before commit since last edit"
is just a `tool_log.exists(..., since=tool_log.last(...).turn)` call.

#### `match` vs `when` — two-stage evaluation

| Stage | Purpose | Access | Cost |
|---|---|---|---|
| `match` | Fast-reject pre-filter | Event fields ONLY (tool name, arg patterns) | O(1) via Guard index |
| `when` | Full predicate | Event + all state tables + query primitives | O(N) over window |

`match` populates the compile-time Guard index (tool_name → rules).
It uses the where-clause pattern language on event fields. It CANNOT
access state tables — it answers "is this event structurally relevant
to this rule?" without any computation.

`when` evaluates only for candidates that pass `match`. It has full
access to query primitives and state tables. A rule without `match`
subscribes to all events on its channel (uses the wildcard index).

This separation is purely for performance: `match` reduces the
candidate set before any predicate evaluation.

#### `escalate_context` — evidence attachment

Rules with `effect: escalate` may declare an `escalate_context`
expression. It is evaluated at fire-time (after `when` passes) and
its result is serialized into the diagnostic message the agent sees.

```yaml
- name: weak-grounding-edit
  escalate_context: entity_evidence(event.args['path'])
```

**Evaluation**: same restricted expression language as `when`, same
namespace. Evaluated only when the rule fires (not on every event).

**Serialization**: the return value is rendered to a human-readable
string appended to the diagnostic:
- `EvidenceList` → markdown table of evidence records
- `Row` / dict → key-value block
- `list` → bulleted list
- scalar → inline text

If `escalate_context` is absent, the diagnostic contains only the
interpolated `reason` string (no evidence attachment).

#### Rule syntax — query-based

Rules evaluate predicates over state tables using query primitives.
The `when` expression has access to:
- `event` — the current event being evaluated
- `tool_log` — query interface to the tool call log
- `file_state` — query interface to the file entity map
- `session` — session metadata (turn_count, total_tokens, etc.)
- `labels` — IFC label state on the current event

**Query primitives** available in `when` expressions:

| Primitive | Semantics | Returns |
|---|---|---|
| `tool_log.count(where={...}, last=N)` | Count matching rows in last N entries | `int` (0 if empty) |
| `tool_log.count(where={...}, since=turn)` | Count matching rows since turn T (inclusive) | `int` (0 if empty) |
| `tool_log.count(where={...}, scope=user, ttl=str)` | Count across sessions within TTL | `int` |
| `tool_log.distinct(field, last=N)` | Count distinct values of field in last N | `int` |
| `tool_log.exists(where={...}, last=N)` | Any matching row in last N? | `bool` |
| `tool_log.exists(where={...}, since=turn)` | Any matching row since turn T (inclusive)? | `bool` |
| `tool_log.last(where={...})` | Most recent matching row | `Row` or `EMPTY` sentinel |
| `file_state.get(path)` | Lookup file entity | `Row` or `EMPTY` sentinel |
| `session.turn_count` | Current turn number | `int` |
| `session.total_tokens` | Tokens used so far | `int` |
| `labels.<name> in event.taint` | Check IFC taint on current event | `bool` |

**Window parameter semantics:**
- `last=N` — scan the N most recent entries (bounded, O(N)).
- `since=T` — scan entries where `turn >= T` (inclusive).
- Every query accepts exactly ONE of `last` or `since`, never both.

**Null-safety contract (EMPTY sentinel):**

`tool_log.last(...)` and `file_state.get(...)` never return Python
`None`. They return an `EMPTY` sentinel object whose attribute access
always returns a safe default: `EMPTY.turn = -1`,
`EMPTY.exit_code = -1`, `EMPTY.last_read_turn = -1`, etc. Boolean
test: `EMPTY` is falsy.

This means `tool_log.last(where={tool: "edit"}).turn` evaluates to
`-1` when no edit exists, not a null-pointer crash. Rules that need
to distinguish "never happened" from "happened at turn 0" can test:
`if tool_log.last(where={...}):` (falsy = never happened).

**Bounded evaluation:**

`tool_log` is capped at a rolling window of 500 entries per session.
Overflow evicts oldest. Cross-session queries (`scope=user`) are
indexed by `(rule_id, error_fingerprint)` in SQLite with a B-tree on
`ts` for TTL-bounded scans.

**`fingerprint()` definition:**

Deterministic hash of error messages: strip ANSI codes, normalize
whitespace, remove line numbers and timestamps, then SHA-256 truncated
to 16 hex chars. Pure string operation, no embedding.

**Additional query primitives:**

| Primitive | Signature | Semantics |
|---|---|---|
| `streak(where, last=N)` | → int | Consecutive matching events from most recent backward |
| `trend(field, where, last=N)` | → "increasing"\|"decreasing"\|"stable" | Linear slope classification over window |
| `ratio(count_a, count_b)` | → float | Safe division (0 if denominator is 0) |
| `sequence(steps, last=N)` | → bool | Ordered pattern match (linear scan, O(N*steps)) |
| `group(field, where, last=N, top=K)` | → list[(val, count)] | Group-by with top-K results |
| `diff(set_a, set_b)` | → set | Set difference between two distinct_values results |
| `lookup(table, key, field)` | → value \| EMPTY | Cross-table point lookup by key |
| `entity_evidence(name)` | → EvidenceList | Retrieve evidence records for an entity |
| `entity_evidence(name).has(type=str)` | → bool | Check if evidence of given type exists |
| `entity_evidence(name).strongest()` | → Evidence \| EMPTY | Highest-signal evidence record |

**`sequence` step format:**
```yaml
sequence:
  steps:
    - { where: {tool: "read"}, label: "read_step" }
    - { where: {tool: "bash", cmd: "rm*"}, gap_max: 5, absent: {tool: "ask_user"} }
  last: 30
```
Each step matches in order. `gap_max` limits events between steps.
`absent` rejects if that pattern appears between steps. Linear scan,
no backtracking — patterns needing Kleene star or NFA belong in LLM
escalation.

**Where clause matching syntax:**

The `where` parameter in query primitives uses a compact pattern
language (not full regex):

| Pattern | Meaning | Example |
|---|---|---|
| `"exact"` | Exact string match | `{tool: "bash"}` |
| `"a|b|c"` | OR over literals | `{tool: "read|edit|write"}` |
| `"prefix*"` | Prefix glob | `{cmd: "git commit*"}` |
| `"*suffix"` | Suffix glob | `{path: "*.py"}` |
| `"*mid*"` | Contains | `{cmd: "*--force*"}` |
| `"!=value"` | Not equal (string) | `{error_category: "!=timeout"}` |
| `42` (bare int) | Exact numeric match | `{exit_code: 0}` |
| `"!=0"` (string) | Not equal (coerced to field type) | `{exit_code: "!=0"}` |
| `null` | Field is null/absent | `{error: null}` |

Numeric fields accept bare integers for exact match. String
comparison operators (`!=`) are coerced to the field's declared type.
No full regex. Complex content-based matching belongs in LLM
escalation, not in the deterministic detector.

**`match` field syntax:** The `match` field on rules uses the same
pattern language as `where`. Dotted paths (e.g., `args.cmd`,
`result.error`) navigate into the event's nested structure.

**Default window behavior:**

If neither `last` nor `since` is provided in a query, the full
rolling window is scanned (all retained entries, up to 500 for
tool_log). This is explicit: "no window param = unbounded within
retention limits."

**Expression language boundary:**

The `when` field is a **restricted Python expression** — pure,
side-effect-free, evaluating to bool. Allowed constructs:

- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Boolean: `and`, `or`, `not`
- Arithmetic: `+`, `-`, `*`, `/`, `%`, `max()`, `min()`
- Containment: `in`, `not in`
- Comprehension predicates: `any(... for ... in ...)`, `all(...)`
- Attribute access: `event.args['path']`, `.get()` on dicts
- Query primitives (above): `tool_log.count(...)`, `file_state.get(...)`, etc.
- Built-in helpers: `fingerprint()`, `hash()`
- String operations: `startswith()`, `endswith()`, slicing

**Forbidden:** imports, assignment, function definitions, lambdas,
I/O, `eval`/`exec`, class definitions, `await`. The expression
compiler rejects any AST node not in the allowlist.

This means the "12 primitives" are a library available within a
restricted Python expression — not a closed language. The restriction
ensures determinism and safety while giving rules enough power to
compose primitives freely.

**What is NOT a query primitive (push to LLM):**
- Semantic similarity ("this edit contradicts that read")
- Content-based classification ("this bash command looks dangerous")
- Causal inference ("did X cause Y")
- Unbounded regex over event sequences
- Full cross-table joins (O(N*M))
- Historical baseline comparison ("is this session anomalous vs last 100")

```yaml
rules:
  # --- Edit without read ---
  - name: edit-without-read
    on: tool_call_pre
    match: { tool: "edit" }
    when: "not file_state.get(event.args['path']) or file_state.get(event.args['path']).last_read_turn == -1"
    effect: notify
    reason: "Editing {event.args['path']} without reading it first."

  # --- Test before commit (temporal: query-based) ---
  - name: test-before-commit
    on: tool_call_pre
    match: { tool: "bash", args.cmd: "git commit*" }
    when: |
      not tool_log.exists(
        where={tool: "bash", cmd: "pytest*|go test*|npm test*", exit_code: 0},
        since=tool_log.last(where={tool: "edit|write"}).turn
      )
    effect: block
    reason: "Tests haven't passed since last edit. Run tests first."

  # --- Stuck loop (aggregation query) ---
  - name: stuck-loop
    on: turn_end
    when: |
      tool_log.count(last=5, where={exit_code: "!=0"}) >= 3
      and tool_log.distinct("args_hash", last=5) < 2
    effect: escalate
    cooldown: 5 turns
    reason: "Agent stuck — repeated failures with no strategy change."

  # --- Weak grounding on edit (evidence-based, semantic rule) ---
  # Note: restricted to "edit" (modify existing file). "write" (create
  # new file) is exempt — new files have no prior evidence by design.
  - name: weak-grounding-edit
    on: tool_call_pre
    match: { tool: "edit" }
    when: "not entity_evidence(event.args['path']).has(type='tool_success')"
    effect: escalate
    escalate_context: entity_evidence(event.args['path'])
    reason: "Editing {event.args['path']} with no tool-confirmed evidence of existence."

  # --- Premature irreversible action ---
  - name: premature-irreversible
    on: tool_call_pre
    match: { tool: "bash", args.cmd: "git push*|rm -rf*" }
    when: |
      session.turn_count <= 4
      and tool_log.count(where={tool: "read"}) < 3
    effect: escalate
    reason: "Irreversible action at turn {session.turn_count} with insufficient reads."

  # --- IFC: secret leakage ---
  - name: no-secrets-to-network
    on: tool_call_pre
    match: { tool: "bash" }
    when: "labels.secret in event.taint"
    effect: abort
    reason: "Secret material flowing to network via bash command."

  # --- Cross-session repeated error ---
  - name: cross-session-repeat
    on: tool_call_post
    match: { result.error: "*" }
    when: |
      tool_log.count(
        scope=user,
        where={error_fingerprint: fingerprint(event.result['error'])},
        ttl="14d"
      ) >= 3
    effect: escalate
    reason: "This error pattern has appeared in 3+ previous sessions."

  # --- Exploration without action ---
  - name: exploration-without-action
    on: turn_end
    when: |
      session.turn_count > 10
      and tool_log.count(last=10, where={tool: "read|glob|find"})
          / max(tool_log.count(last=10), 1) > 0.8
      and tool_log.count(last=10, where={tool: "edit|write"}) == 0
    effect: notify
    reason: "10+ turns of mostly reading with no writes — consider taking action."

  # --- File churn ---
  - name: file-churn
    on: tool_call_post
    match: { tool: "edit|write" }
    when: "file_state.get(event.args['path']).write_count > 3"
    effect: notify
    reason: "{file_state.get(event.args['path']).write_count} modifications to {event.args['path']} — likely thrashing."

  # --- Dangerous bash (pattern-based, not content classification) ---
  - name: dangerous-bash
    on: tool_call_pre
    match: { tool: "bash" }
    when: |
      any(p in event.args.get('cmd', '') for p in [
        'rm -rf', 'git push --force', 'git reset --hard',
        'git clean -f', 'DROP TABLE', 'DROP DATABASE',
        'kubectl delete', '> /dev/', 'dd if=', 'mkfs'
      ])
    effect: block
    reason: "Blocked destructive command: {event.args['cmd'][:80]}"

  # --- Budget warning (graduated) ---
  - name: budget-warning
    on: turn_end
    when: "lookup('context_state', session.turn_count, 'context_usage_pct') > 80"
    effect: notify
    reason: "Context budget at {lookup('context_state', session.turn_count, 'context_usage_pct')}% — prioritize completing the task."

  # --- Spawn storm ---
  - name: spawn-storm
    on: session_spawn
    when: "lookup('session_tree', session.id, 'concurrent_siblings') > 5"
    effect: block
    reason: "Blocked: too many concurrent sub-agents. Complete existing ones first."

  # --- Meta: severity ladder (queries effect_log) ---
  - name: abort-on-persistent-stagnation
    on: turn_end
    when: |
      effect_log.count(where={rule_id: "stuck-loop", effect: "escalate"}, last=10) >= 2
    effect: abort
    reason: "Stagnation persists after two escalations. Aborting."
```

Note: `effect_log` is the 9th queryable state table — see §4.6.
It records all rule firings (both enforce and observe mode) and is
available to meta-rules via the same query primitives as other tables.

#### Evaluation semantics

For each incoming event:

```
1. Reactions execute first (state recording):
   - Match event → append to tool_log / upsert file_state / etc.
   - Pure recording, no effects, no queries.

2. Rules evaluate second (detection):
   a. Channel dispatch → candidate rules (O(1) subscription).
   b. Guard fast-reject → tool name index.
   c. For each candidate:
      - Evaluate `match` (pattern matching on event fields).
      - Evaluate `when` (query expressions over state tables).
      - Cooldown check.
      - If all pass → fire effect.

3. Effect resolution:
   - Priority: abort > block > escalate > notify.
   - notify and escalate stack (multiple can fire).
   - block picks highest-priority rule's reason.
   - abort terminates immediately.
```

The key ordering: **reactions before rules within the same event**.
This ensures a tool_call_post event first records itself into
tool_log, then rules querying tool_log see the just-completed call.

**Constraint:** Reactions that record **completed actions** (tool_log,
file_state, error_log) bind to post-events only. Reactions that record
**agent intent** (entity_registry — paths/symbols referenced in tool
args before execution) may bind to pre-events. This avoids the
paradox of recording an unexecuted action as completed while still
capturing what the agent intended to act upon.

For pre-events (tool_call_pre): the event hasn't happened yet, so
tool_log doesn't contain it. Rules on tool_call_pre query historical
state to decide whether to allow the pending action.

#### Rule Mode: enforce vs observe

Each rule carries a `mode` field:

```yaml
- name: stuck-loop
  mode: observe    # evaluate + log, no agent-visible effect
  ...

- name: edit-without-read
  mode: enforce    # evaluate + actually produce the declared effect
  ...
```

| Mode | Evaluate? | Log trigger? | Produce effect? |
|---|---|---|---|
| `enforce` | yes | yes (effect_log) | yes |
| `observe` | yes | yes (effect_log) | no |

**Exception:** Rules with `effect: abort` are ALWAYS enforced
regardless of mode setting. Safety-critical rules that would
terminate the session must not be silently observed — a detected
abort-worthy violation that is merely logged defeats the purpose.
abort-effect rules that aren't ready for enforcement should not be
shipped at all.

Both modes write to `effect_log` on trigger — the same structured
record (rule_id, event, bound values, would-be effect). The only
difference is whether the effect reaches the agent.

This enables:
- **Precision measurement**: compare trigger log against human-labeled
  trajectory outcomes (TP/FP/FN).
- **Gradual rollout**: new rules start as `observe`, graduate to
  `enforce` once precision is validated.
- **A/B comparison**: run the same session with/without enforcement,
  measure task-success difference.

The `effect_log` table:

```sql
CREATE TABLE effect_log (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    session_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    mode TEXT NOT NULL,           -- 'enforce' | 'observe'
    event_channel TEXT NOT NULL,
    event_summary TEXT,           -- compact representation of triggering event
    bound_values TEXT,            -- JSON of matched/bound variables
    effect TEXT NOT NULL,         -- 'notify' | 'block' | 'escalate' | 'abort'
    reason TEXT,
    turn INT NOT NULL
);
```

### 4.6 Reactions (State Recording)

Reactions are **low-level stable recorders** — they observe events and
append structured facts into state tables. They do NOT aggregate or
compute derived values. Aggregation is the responsibility of rules via
query primitives in their `when` expressions.

This separation means: adding a new rule never requires a new reaction.
The raw data is already there; the rule queries it as needed.

#### Extraction Modes

Some fields can be extracted either deterministically (heuristic) or
via lightweight LLM classification (semantic). Both modes are
supported and configurable:

```yaml
reaction_config:
  extraction_mode: heuristic   # heuristic | semantic | both
```

| Mode | Cost | Accuracy | Use |
|---|---|---|---|
| `heuristic` | zero (regex, hash, pattern) | Lower (misses paraphrases) | Default, production |
| `semantic` | ~50-100 tokens/event (LLM call) | Higher (catches intent) | Measurement runs |
| `both` | Both costs | A/B comparison | Calibration phase |

In `both` mode, both versions are recorded as separate fields
(e.g., `is_repeat_h` and `is_repeat_s`) enabling offline comparison
of detection quality. Once data shows which mode is sufficient per
field, the config locks it in.

`extraction_mode` is the ONLY user-configurable reaction setting.
Reactions themselves (what gets recorded, into which table) are
hardcoded. The comparison tooling for `both` mode is a Phase 2
deliverable — Phase 1 records both fields; analysis is manual.

#### State Tables

Reactions maintain 8 structured tables:

```yaml
state_tables:
  tool_log:
    # Append-only: one row per tool call completion
    # Rolling window: 500 entries per session
    source: ToolCallPost
    schema:
      turn: int
      tool: str
      args_hash: str              # deterministic hash for dedup
      path: str | null            # extracted file path if applicable
      cmd: str | null             # extracted command if bash
      exit_code: int | null
      error: str | null
      error_fingerprint: str | null   # normalized hash of error message
      error_category: str | null      # heuristic: not-found|permission|timeout|validation|runtime
      duration_ms: int
      result_hash: str | null     # hash of tool output (for regression detection)
      result_length: int          # bytes of output
      is_repeat: bool             # heuristic: same (tool, args_hash) seen before
      repeat_count: int           # times this (tool, args_hash) has appeared
      taint_labels: set[str]
      # semantic mode adds:
      # semantic_intent: str      # normalized action description
      # is_retry_semantic: bool   # LLM-judged: is this a retry of a prior failed attempt?
    retention: session | user(ttl=14d)

  file_state:
    # Entity map: one row per file path, updated on each file op
    source: ToolCallPost (file-related tools)
    schema:
      path: str                   # primary key
      content_hash: str | null    # latest known content hash
      first_read_turn: int | null
      last_read_turn: int | null
      last_write_turn: int | null
      read_count: int
      write_count: int
      reverts_to_prior_hash: bool # current hash matches any earlier hash
    retention: session

  turn_summary:
    # One row per turn: aggregated turn-level metrics
    source: TurnEnd
    schema:
      turn_index: int             # primary key
      tool_calls_count: int
      tool_names_set: set[str]    # unique tools used this turn
      error_count: int
      files_modified: set[str]    # paths changed this turn
      assistant_text_length: int  # chars of non-tool output
      context_growth_tokens: int  # tokens added to context this turn
      net_progress_signal: bool   # heuristic: files changed or new content hash
      duration_ms: int
    retention: session

  session_tree:
    # One row per session: parent/child relationships
    source: SessionSpawn + SessionEnd
    schema:
      session_id: str             # primary key
      parent_id: str | null
      spawn_depth: int
      purpose: str | null         # from spawn args
      budget_allocated: int | null
      concurrent_siblings: int    # alive at spawn time
      exit_reason: str | null     # completed|budget|error|aborted|timeout
      total_turns: int | null     # filled on SessionEnd
      total_cost: float | null
    retention: user(ttl=30d)

  error_log:
    # Append-only: classified errors
    source: ToolCallPost (when error) + ErrorEvent
    schema:
      error_id: str               # unique
      turn: int
      error_fingerprint: str
      error_category: str         # not-found|permission|timeout|validation|runtime
      source_component: str       # tool|provider|kernel|atom
      related_tool: str | null
      is_transient: bool          # heuristic: retryable?
      recovery_action: str | null # retry|abort|escalate|ignore (filled on next event)
    retention: session | user(ttl=14d)

  context_state:
    # One row per turn: context window health
    source: MessagePre + TurnEnd
    schema:
      turn_index: int             # primary key
      total_context_tokens: int
      context_usage_pct: float    # % of budget consumed
      compaction_happened: bool   # was context compressed this turn?
      turns_since_user_message: int
      injected_content_count: int # skills, memories, policy feedback injected
    retention: session

  # repetition_index — INTERNAL CACHE, not a declared state table.
  # Derivable from tool_log via group(). Implemented as an optimization
  # inside the query engine (memoize repeat_count lookups). Not exposed
  # to rule authors or the query namespace.

  entity_registry:
    # Evidence retrieval system for entities. Inspired by
    # trajectory_index's Symbol/Reference/Grounding model but purely
    # deterministic (no LLM extraction).
    #
    # == Purpose ==
    # The entity layer is an IR system that collects EVIDENCE about
    # entities the agent interacts with. It does NOT judge or score —
    # it provides evidence records that downstream rules consume:
    #   - Deterministic rules bypass this layer entirely (they match
    #     on event fields directly via query primitives).
    #   - Semantic rules use entity evidence as a FILTER: "does this
    #     entity have tool-confirmed evidence?" If not, escalate with
    #     the evidence set attached for the agent to see.
    #
    # == Extraction Pipeline (3 layers, no regex, no LLM) ==
    #
    # Layer 1 — Structural: tool schema field dispatch. If the tool
    #   schema declares args['path'] is a file path, extract it
    #   directly. JSON result keys named "path", "file", "url" are
    #   auto-extracted. Cost: ~0.02ms.
    #
    # Layer 2 — Lexical: character-class tokenization on free-text
    #   fields (bash output, error messages). Classify tokens by
    #   character-class signature: slash-separated = path candidate,
    #   camelCase/snake_case = symbol, :// prefix = URL.
    #   Cost: ~0.05ms.
    #
    # Layer 3 — Session Dictionary: a trie built from entities that
    #   have structural or tool-success evidence. Scans subsequent
    #   free text to detect references to known entities.
    #   Improves over session as dictionary fills. Cost: ~0.1ms.
    #
    # Key insight: tool calls are the SOURCE of entities; free text
    # is where entities are REFERENCED. Be strict on discovery,
    # generous on recall. Total: ~0.15ms/event.
    #
    # == Evidence Model ==
    # Each entity accumulates evidence records over the session.
    # An evidence record = (evidence_type, turn, detail).
    # Evidence types:
    #   - structural:     entity appeared in a typed tool schema field
    #   - tool_success:   tool operated on entity and succeeded
    #   - tool_failure:   tool operated on entity and failed (e.g., FileNotFoundError)
    #   - lexical_match:  extracted from free text by character-class analysis
    #   - dict_recall:    detected via session dictionary in subsequent text
    #   - user_provided:  appeared in user message / task description
    #   - agent_mention:  agent referenced in reasoning without tool interaction
    #
    # Rules query evidence via `entity_evidence(name)` which returns
    # the evidence list. Semantic rules filter on evidence presence:
    #   when: "not entity_evidence(path).has(type='tool_success')"
    # On escalation, the evidence set is attached to the diagnostic
    # so the agent sees concrete, actionable context.
    #
    source: ToolCallPre (intent) + ToolCallPost (result)
    schema:
      entity: str                 # path, symbol, or URL — primary key
      entity_type: str            # path|symbol|url|command
      evidence: list[Evidence]    # accumulated evidence records
      first_seen_turn: int
      last_seen_turn: int
      occurrence_count: int
    retention: session
    index: inverted              # entity → evidence list (O(1) lookup)

    # Evidence record structure:
    #   Evidence:
    #     type: str               # structural|tool_success|tool_failure|lexical_match|dict_recall|user_provided
    #     turn: int
    #     detail: str             # human-readable context ("read() returned 200 lines")
    #
    # Note: `agent_mention` type deferred — requires extraction from
    # assistant message text via ContextEvent + Layer 3 dictionary
    # scan. Add when dictionary recall is proven effective.
    #
    # Retention: per entity, per evidence type, keep first occurrence
    # + last occurrence + total count. Caps evidence list at ~14
    # records per entity (7 types × 2). Avoids unbounded growth for
    # frequently-referenced entities.
```

#### Reaction Definitions

```yaml
reactions:
  - on: tool_call_post
    record_to: tool_log
    fields:
      turn: session.turn_count
      tool: event.tool_name
      args_hash: hash(event.args)
      path: event.args.get('path')
      cmd: event.args.get('cmd') or event.args.get('command')
      exit_code: event.result.get('exit_code')
      error: event.result.get('error')
      error_fingerprint: fingerprint(event.result.get('error'))
      duration_ms: event.duration_ms
      taint_labels: event.taint

  - on: tool_call_post
    match: { tool: "read|edit|write|glob" }
    upsert_to: file_state
    key: event.args['path']
    fields:
      first_read_turn: "coalesce(existing.first_read_turn, session.turn_count) if event.tool_name == 'read' else existing.first_read_turn"
      last_read_turn: "session.turn_count if event.tool_name == 'read' else existing.last_read_turn"
      last_write_turn: "session.turn_count if event.tool_name in ('edit', 'write') else existing.last_write_turn"
      read_count: "existing.read_count + (1 if event.tool_name == 'read' else 0)"
      write_count: "existing.write_count + (1 if event.tool_name in ('edit', 'write') else 0)"
```

Rules then query these tables with aggregation primitives — see §4.5.

---

## 5. Compilation & Runtime

### 5.1 Compilation Target

| Rule layer | Compiled form |
|---|---|
| L0 Reaction | `(guard, update_fn)` — inline state mutation |
| L1 Query | `(guard, predicate_code)` — compiled Python code object |
| L2 IFC | `(guard, taint_check)` — event.taint set membership |
| L3 Meta | `(guard, predicate_code)` — same as L1, queries effect_log |

All predicate rules (L1, L2, L3) compile to the same form: a Guard
for fast-reject + a code object that evaluates to bool. L2 is just L1
with access to the taint namespace. L3 is just L1 querying a different
table. The unified compilation path keeps the runtime simple.

```python
@dataclass(slots=True)
class RuleInstance:
    rule_id: str
    layer: int                           # 0-3
    mode: str                            # "enforce" | "observe"
    schema_version: int                  # hash of rule structure
    guard: Guard                         # fast-reject filter
    predicate: CodeType | None           # compiled when expression
    effect: EffectSpec
    escalate_context_expr: CodeType | None  # compiled escalate_context
    cooldown_turns: int
    last_fired_turn: int = -1
```

### 5.2 Expression Compilation Strategy

**Approach: AST whitelist + compile() → code object.**

```
YAML string → ast.parse(mode='eval') → AST validation → compile() → code object
                                                                         ↓
                                                            eval(code_obj, namespace)
```

**Step 1: Parse.** `ast.parse(expr_str, mode='eval')` produces an
`Expression` AST node. Syntax errors fail at compile time (session
start), not at runtime.

**Step 2: AST validation.** Walk every node; reject if ANY node type
is not in the allowlist:

```python
ALLOWED_NODES = {
    # Expressions
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.IfExp, ast.Call, ast.Attribute, ast.Subscript, ast.Starred,
    # Containers
    ast.Dict, ast.List, ast.Tuple, ast.Set,
    # Comprehensions (for any()/all() only)
    ast.GeneratorExp, ast.ListComp, ast.comprehension,
    # Atoms
    ast.Name, ast.Constant, ast.JoinedStr, ast.FormattedValue,
    # Operators
    ast.And, ast.Or, ast.Not, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Mod, ast.FloorDiv, ast.Pow,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
    ast.UAdd, ast.USub,
    # Context
    ast.Load, ast.Store,  # Store only inside comprehension targets
}
```

**Rejected (compile-time error, rule fails to load):**
`Import`, `FunctionDef`, `AsyncFunctionDef`, `ClassDef`, `Assign`,
`AugAssign`, `Delete`, `For`, `While`, `If` (statement), `With`,
`Raise`, `Try`, `Assert`, `Global`, `Nonlocal`, `Return`, `Yield`,
`Lambda`, `Await`.

**Additional checks:**
- `ast.Call` targets must resolve to names in the allowed namespace
  (no `eval`, `exec`, `compile`, `__import__`, `open`, `getattr`,
  `setattr`, `delattr`, `type`, `vars`, `dir`, `globals`, `locals`).
- No `__dunder__` attribute access (reject `ast.Attribute` where
  `attr.startswith('__')`).
- `ast.Name` ids must be in the namespace or comprehension-bound.

**Step 3: Compile.** `compile(ast_tree, filename=f"<rule:{rule_id}>",
mode='eval')` produces a `CodeType` object. This is a one-time cost
at session start (~0.1ms per rule).

**Step 4: Runtime evaluation.**

```python
def _evaluate_predicate(self, rule: RuleInstance, event: Event) -> bool:
    namespace = self._build_namespace(event)
    try:
        return bool(eval(rule.predicate, {"__builtins__": {}}, namespace))
    except Exception:
        # Log diagnostic, do not crash the session
        return False
```

**Namespace injection:**

```python
def _build_namespace(self, event: Event) -> dict:
    return {
        # Event context
        "event": EventProxy(event),
        "session": self._session_state,
        "labels": self._labels,
        # Query interfaces (bound to state tables)
        "tool_log": self._tool_log_query,
        "file_state": self._file_state_query,
        "effect_log": self._effect_log_query,
        # Standalone query primitives
        "streak": self._streak,
        "trend": self._trend,
        "ratio": self._ratio,
        "sequence": self._sequence,
        "group": self._group,
        "diff": self._diff,
        "lookup": self._lookup,
        "entity_evidence": self._entity_evidence,
        "fingerprint": fingerprint,
        "hash": deterministic_hash,
        # Safe builtins
        "max": max, "min": min, "abs": abs, "len": len,
        "any": any, "all": all, "sorted": sorted,
        "True": True, "False": False, "None": None,
        "EMPTY": EMPTY,
    }
```

**How `where={}` works at runtime:**

In the expression `tool_log.count(where={tool: "read"}, last=10)`:
- `{tool: "read"}` is a normal Python dict literal in the AST.
- At eval time, it constructs a `{"tool": "read"}` dict.
- `tool_log.count()` receives it as a keyword argument.
- Inside `count()`, each key-value pair is matched against rows using
  the where-clause pattern language (§4.5).

No compile-time magic needed — the dict is constructed fresh each eval.

**Security boundary summary:**

| Layer | Blocks |
|---|---|
| AST allowlist | Imports, assignments, I/O, class/function defs |
| Name resolution | Only namespace-provided names are accessible |
| Dunder guard | No `__class__`, `__subclasses__`, `__globals__` escape |
| Empty builtins | `{"__builtins__": {}}` — no access to Python stdlib |
| Exception catch | Predicate errors → False + diagnostic, never crash |

**Compilation errors** (rule fails to load at session start):
- Syntax error in expression string
- Disallowed AST node
- Unknown name reference
- Call to forbidden function

These are reported via a compilation diagnostic (logged + surfaced in
`policy_explain` tool output). The session starts normally — only the
broken rule is skipped.

### 5.3 Event Routing

```python
class PolicyEngine:
    _by_channel: dict[str, list[RuleInstance]]
    _tool_index: dict[str, list[RuleInstance]]   # tool_name → rules
    _wildcard: dict[str, list[RuleInstance]]      # rules with no tool filter

    def _on_event(self, event: Event) -> list[Effect]:
        candidates = self._tool_index.get(event.tool_name, [])
        candidates += self._wildcard.get(event.channel, [])
        effects = []
        for rule in candidates:
            if rule.guard.reject(event):
                continue
            if rule.last_fired_turn >= 0 and \
               (session.turn - rule.last_fired_turn) < rule.cooldown_turns:
                continue
            if self._evaluate(rule, event):
                effects.append(rule.effect)
                rule.last_fired_turn = session.turn
        return self._resolve_effects(effects)
```

### 5.4 Performance Budget

| Operation | Budget | Implementation |
|---|---|---|
| Event routing + guard | < 100μs | dict lookup + frozenset check |
| Predicate evaluation (all matching) | < 500μs | compiled code objects + eval |
| IFC taint check | < 100μs | set membership + substring scan |
| Entity extraction (3 layers) | < 200μs | struct fields + char-class + trie |
| State persistence (batched) | < 1ms | SQLite WAL at TurnEnd |
| **Total per event** | **< 2ms** | negligible vs 2–30s inference |

### 5.5 Cross-Session Persistence

Storage: `~/.agentm/policy_state/<scope_key>.db` (SQLite WAL mode).

```sql
CREATE TABLE rule_state (
    rule_id TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    state_blob BLOB NOT NULL,       -- msgpack: registers + current_state
    fire_count INTEGER DEFAULT 0,
    last_fired_at REAL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (rule_id, scope_key)
);

CREATE TABLE event_log (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    session_id TEXT NOT NULL,
    rule_id TEXT,
    effect TEXT,
    context_json TEXT
);
```

Write strategy: batch all mutations at `TurnEnd`, single transaction.
On crash: state reverts to last committed turn boundary.

### 5.6 Hot Reload

Rules can be reloaded without restarting the session:

1. File watcher or `reload_policies` tool call triggers compilation.
2. New `CompiledRuleSet` prepared.
3. At next quiescent point (between events), generation swap:
   - Same `rule_id` + same `schema_version` → carry state forward.
   - Same `rule_id` + different schema → reset + log diagnostic.
   - New rule → initialize fresh.
   - Removed rule → drop from active set (state persists in DB for audit).
4. Rebuild indices.

No ring buffer needed (unlike ActPlane). Python GIL + sequential event
processing = natural quiescent points.

---

## 6. Escalation & Feedback Path

All effects use **direct context injection** — no separate LLM call.
When a rule fires, its `reason` template (with bound values
interpolated) is appended to the agent's next turn context as a
structured diagnostic block.

```
PolicyEngine fires rule "stuck-loop"
    │
    ▼
Construct diagnostic:
    "[Policy: stuck-loop] Agent stuck — 4 consecutive failures
     with identical args_hash. Consider a different approach."
    │
    ▼
Inject into agent's next context (via ContextEvent injection)
    │
    ▼
Agent sees diagnostic in its next reasoning step
```

**Effect-specific injection format:**

| Effect | Agent visibility | Mechanism |
|---|---|---|
| `notify` | Soft hint appended to context | DiagnosticEvent with `severity=info` |
| `escalate` | Prominent warning with context | DiagnosticEvent with `severity=warning` + bound values |
| `block` | Tool call rejected, reason shown | `{"block": True, "reason": str}` return on ToolCallEvent |
| `abort` | Session terminated | `AgentAbortError` raised immediately |

This is cheaper than a separate LLM reasoner (zero additional tokens)
and simpler. The agent's own next reasoning step decides how to
respond to the diagnostic. If data later shows high false-positive
rates on specific rules, a confirmation gate (lightweight LLM call
before injection) can be added per-rule as a Phase 3 enhancement.

---

## 7. Integration with AgentM

### 7.1 Atom Contract & Module Structure

Package atom at `src/agentm/extensions/builtin/policy/`. The §11
contract is satisfied: the package is one atom, exports `MANIFEST` +
`install()` from `__init__.py`, no cross-atom imports.

```
src/agentm/extensions/builtin/policy/
├── __init__.py        # MANIFEST + install() — atom entry point
├── types.py           # Shared dataclasses: RuleInstance, Guard, EffectSpec,
│                      #   EMPTY sentinel, Evidence, EventProxy (~200 lines)
├── compiler.py        # YAML parse → RuleInstance list. AST whitelist,
│                      #   code object compilation, guard construction (~400 lines)
├── evaluator.py       # Event dispatch, guard matching, predicate eval,
│                      #   cooldown check, effect resolution (~300 lines)
├── query.py           # 12 query primitives (count, distinct, exists, last,
│                      #   streak, trend, ratio, sequence, group, diff,
│                      #   lookup, entity_evidence) (~500 lines)
├── state.py           # State tables + reactions: tool_log, file_state,
│                      #   turn_summary, session_tree, error_log,
│                      #   context_state, effect_log (~400 lines)
├── entity.py          # entity_registry: 3-layer extraction pipeline
│                      #   (structural + lexical + session dictionary),
│                      #   evidence model, trie builder (~350 lines)
├── ifc.py             # IFC: taint store, source detection (regex on result),
│                      #   propagation check (substring match) (~150 lines)
├── persistence.py     # SQLite WAL: cross-session state, batch write at
│                      #   TurnEnd, schema migration on version change (~250 lines)
└── effects.py         # Effect types, diagnostic message construction,
                       #   context injection formatting (~150 lines)
```

Total: ~2700 lines. Each module has a single responsibility and can
be tested independently.

```python
# __init__.py
MANIFEST = {
    "name": "policy_engine",
    "version": "0.1.0",
    "description": "DSL-driven policy detection and enforcement",
    "config_schema": {
        "policy_files": {"type": "list", "default": []},
        "extraction_mode": {"type": "str", "enum": ["heuristic", "semantic", "both"], "default": "heuristic"},
    },
}

def install(api: ExtensionAPI, config: dict) -> None:
    engine = PolicyEngine(api, config)
    api.subscribe("tool_call", engine.on_tool_call, priority=EARLY)
    api.subscribe("tool_result", engine.on_tool_result, priority=EARLY)
    api.subscribe("turn_end", engine.on_turn_end, priority=EARLY)
    api.subscribe("child_session_start", engine.on_session_spawn, priority=EARLY)
    api.subscribe("context", engine.on_context, priority=EARLY)
    api.register_tool("reload_policies", engine.reload_policies)
    api.register_tool("policy_explain", engine.explain_rule)
    api.register_tool("policy_stats", engine.show_stats)
    api.register_service("policy_engine", engine)
```

### 7.2 Scenario Manifest Integration

```yaml
# contrib/scenarios/<name>/manifest.yaml
extensions:
  - policy_engine

atom_config:
  policy_engine:
    policy_files:
      - policies/base.yaml
      - policies/safety.yaml
      - policy.yaml          # scenario-local
    mode: enforce
```

### 7.3 Policy Composition & Authority

Inspired by ActPlane's hierarchical domains:

```
Platform policies (ship with AgentM, ~/.agentm/policies/base.yaml)
  └── Scenario policies (manifest-referenced)
       └── Task policies (dynamically added by orchestrator agent)
            └── Self-restriction (agent tightens its own child's rules)
```

**Inheritance rule**: child scope inherits all parent rules. Can add
rules (tighten) but cannot remove or weaken inherited rules.

**Override mechanism**: A scenario can `disable: [rule-name]` only for
rules it authored or rules explicitly marked `overridable: true`. Safety
rules from platform level are `overridable: false` by default.

### 7.4 Observability

- **Trace integration**: Policy evaluations emit spans into existing
  OTLP telemetry. Queryable via `agentm trace`.
- **Explain mode**: `AGENTM_POLICY_EXPLAIN=1` or tool call
  `policy_explain(rule="name")` → structured explanation of why a rule
  did/didn't fire on the last event.
- **Dry-run mode**: `mode: dry_run` evaluates all rules, emits trace
  events, but downgrades all effects to notify with `[DRY RUN]` prefix.
- **TUI panel**: Active rules, current states, fire counts, last-fired
  turn. Uses existing `BackgroundActivityEvent`.

---

## 8. Comparison with Existing Systems

| System | Interception layer | State model | Feedback | Coverage |
|---|---|---|---|---|
| ActPlane | OS kernel (syscalls) | 64-bit bitmask | Pre-authored string | All execution paths |
| FIDES/CaMeL | Tool-call boundary | Typed IFC in agent loop | Block + reason | Tool API only |
| llmharness v3 | Every k turns | Full trajectory graph | LLM-generated | Semantic (expensive) |
| **This design** | Agent events (tool/turn/session) | Query windows + taint labels | Deterministic detect → context inject | Tool + subprocess + cross-event |

Key advantage over each:
- vs ActPlane: richer semantics (epistemic state, convergence, phases).
- vs FIDES: covers indirect paths (script inside bash, cross-turn state).
- vs llmharness v3: 100x cheaper (LLM only on trigger), more precise
  (focused hypothesis vs broad review).

---

## 9. Implementation Plan

All rules ship from day one. New/unvalidated rules default to
`mode: observe`. Once precision data confirms they work, they
graduate to `mode: enforce`.

### Phase 1: Full Engine + All Rule Layers

- PolicyEngine package (see §7.1 module structure).
- Reactions: tool_log + file_state + entity_registry recorders.
- Expression compiler: AST whitelist + code object compilation.
- Query engine: count, distinct, exists, last, streak, trend, ratio,
  sequence, group, diff, lookup, entity_evidence (12 primitives).
- All four effect types: notify, block, escalate, abort.
- observe/enforce mode per rule + effect_log recording.
- EMPTY sentinel for null-safety across all primitives.
- Minimal IFC: literal substring taint propagation (§4.2).
- Session-scoped state + cross-session SQLite WAL.
- L1 (query-based), L2 (IFC taint check), L3 (meta over effect_log).
- tool_log capped at 500 entries rolling window.
- Entity registry with 3-layer extraction (structural + lexical +
  session dictionary).
- Ship with full rule set (all rules below), most in observe mode:

| Rule | Mode | Effect |
|---|---|---|
| edit-without-read | enforce | notify |
| dangerous-bash | enforce | block |
| test-before-commit | observe | block |
| stuck-loop | observe | escalate |
| hallucinated-path | observe | block |
| premature-irreversible | observe | escalate |
| exploration-without-action | observe | notify |
| file-churn | observe | notify |
| cross-session-repeat | observe | escalate |
| no-secrets-to-network | observe | abort |
| budget-warning | enforce | notify |
| spawn-storm | enforce | block |

### Phase 2: Measurement & Graduation

- Run engine on real llmharness / ARL sessions.
- Collect effect_log data, compute per-rule precision/recall.
- Graduate validated rules from observe → enforce.
- Tune thresholds (window sizes, count thresholds) based on data.
- Fix false-positive sources identified by precision analysis.
- **Effect utility measurement**: for escalate/notify rules, compare
  agent behavior in the 3 turns after diagnostic injection vs before.
  Signal: did the agent change tool-call pattern? Did the session
  succeed vs fail? Precision says "did the rule fire correctly";
  utility says "did the intervention help."
- **Block-retry analysis**: monitor how agents respond to block effects.
  If repeated blocked attempts are common, add per-rule retry cap →
  abort escalation.

### Phase 3: Advanced IFC & Derived Events

- Full IFC lattice if Phase 2 data shows literal substring misses
  significant leaks (structure-aware propagation, JSON path tracking).
- Derived events (phase_transition, convergence_signal) if L1 query
  rules prove insufficient for trajectory-shape detection.
- Per-rule confirmation gate (lightweight LLM call before injection)
  for high-FP escalation rules identified in Phase 2.

### Phase 4: Developer Experience

- `agentm policy lint <file>` — static expression validation.
- `agentm policy test <file> --event <json>` — unit test a rule.
- `agentm policy stats --session <id>` — per-rule trigger stats.
- Explain mode (why did/didn't a rule fire).
- JSON Schema for IDE autocomplete.

---

## 10. Resolved Decisions

| Question | Decision | Rationale |
|---|---|---|
| Expensive predicates (embedding) | Push to LLM reasoner | Detector stays purely deterministic |
| Agent-authored runtime rules | Not now | Pre-define all rules; add dynamic capability when needed |
| Interaction with llmharness v3 | Policy engine replaces cognitive audit | This is the evolution of that code; validate via ablation |
| Escalation mechanism | Context injection (no separate LLM call) | Simpler, cheaper; add confirmation gate later for high-FP rules |
| Null safety | EMPTY sentinel object | All attribute access returns safe defaults; falsy test = "never happened" |
| Reaction binding | Post-events for completed actions; pre-events for intent-tracking (entity_registry) | Avoids recording unexecuted actions as completed; still captures agent intent |
| Query window semantics | `since=T` is inclusive (turn >= T) | Matches natural reading |
| Rule priority tiebreaker | Layer number, then file order | Deterministic without explicit priority field |
| fingerprint() | Deterministic string hash (normalize + SHA-256[:16]) | No embedding, cheap |
| EFSM state machines | Removed — query primitives sufficient | All temporal rules expressible via `since`/`last` window queries; no shipped rule needs explicit states |
| IFC scope (Phase 1) | Minimal: literal substring match only | Source regex captures values; propagation = substring in args; no lattice, no transformation tracking |
| Expression compilation | AST whitelist + compile() → code object | Standard Python security pattern; eval with empty builtins + injected namespace |
| Atom structure | Package (multi-file) not single file | ~2700 lines total; modules have clear single-responsibility boundaries |

## 11. Known Gaps & Deferred Work

| Gap | Status | Resolution path |
|---|---|---|
| Cross-session file conflict (two agents editing same file) | Not expressible in v1 | Needs shared file_state view or cross-session file query scope; design when multi-agent scenarios exercise this |
| Scope creep detection | Correctly deferred | Fundamentally semantic — requires LLM to judge "is this action relevant to the task" |
| Forgetting earlier decisions | Weak signal only | Can detect context_usage_pct + regression; "forgetting a commitment" is semantic |
| repetition_index | Optimization cache | Derivable from tool_log via `group()`; treat as internal cache, not a declared user-facing table |
| IFC value transformations | Deferred to Phase 3 | base64, concatenation, variable embedding not tracked; measure miss rate first |
| Effect utility measurement | Phase 2 | Track "did the agent change behavior after escalation" — compare trajectory before/after diagnostic injection |
| Block-retry loop cap | Design on demand | If agent retries blocked tool N times → should escalate to abort? Monitor in Phase 2 data |

## 12. Resolved (batch 2)

| Question | Decision | Rationale |
|---|---|---|
| Taint across LLM boundary | Literal substring match of captured values | Most real flows are verbatim copy; no pattern match on sink side (patterns introduce inaccuracy) |
| Phase inference | No derived events; use inline query predicates | `tool_log.count(where={tool: "read"}, last=10)` expresses the same thing without premature abstraction |
| Cross-session visibility | User-scope sees committed data only | Cross-session rules are low-freq ("3+ occurrences historically"); session-scope handles within-session detection |
| effect_log retention | Persist, 30d TTL | Core measurement data; needed for precision/recall analysis across validation cycles |
