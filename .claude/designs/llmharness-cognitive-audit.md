# Cognitive Audit for llmharness

**Status:** active design
**Owners:** llmharness scenario
**Last reviewed:** 2026-05-10

A two-phase audit pipeline that supervises a running main agent for
semantic-level drift. Phase 1 reconstructs a typed event graph from
the trajectory by having an extractor LLM build it via tool calls
with witness-verified edges. Phase 2 judges drift over the graph
every `k` turns and emits an optional reminder.

The pipeline borrows a *static-analysis lens*: the trajectory is
"source code", the extractor is "lex+parse+type-check", the graph is
the IR (closer to a Program Dependence Graph than a CFG), and the
auditor is the verifier. The load-bearing trust axiom: the agent's
**thought** is testimony, **tool calls and tool results** are
evidence — extraction must anchor on evidence.

This is **v3**, replacing v2 outright. v2's `validator.py` 5-rule
chain, SDK-fixed hint signals, and `refs: list[int]` event-internal
edges are gone. The schema, `pyproject.toml` `version`, and downstream
rca-autorl integration all break.

---

## 1. Goals

- **Inferential extraction with verifiable witnesses.** Edges between
  events are produced by the extractor LLM via dedicated tool calls;
  every edge carries witness fields (`cited_entities` for data flow,
  `cited_quote` for textual reference) that are checked literally at
  tool-call time. No mechanical entity-extraction heuristics.
- **Mechanical event boundary.** Each event is exactly one ReAct cycle
  (§4.a). Boundary detection is protocol-level, not LLM-decided.
- **Trust asymmetry as an explicit axiom.** Both the extractor prompt
  and the auditor prompt state that thought is advisory and tool
  calls / tool results are evidence.
- **Non-gating advisory checks.** Graph-health observations are
  computed by scenario-registered Python checks invoked on the
  finalized graph; their output (`list[Finding]`) is fed to the
  auditor as advisory context, never used to block the firing.
- **Visible failure modes.** Extractor witness retries that exhaust
  the budget drop the offending edge but never fail-stop the firing.
  All extractor / auditor failures land as distinct entry types so
  outage is visible in the durable graph.

---

## 2. Eleven design decisions (from issue #134, v3 proposal)

| # | Decision | Rationale |
|---|---|---|
| 1 | **Event = one ReAct cycle.** | Mechanical boundary keeps extraction reproducible. |
| 2 | **Parallel `tool_call`s in one assistant turn collapse to one event.** | One thought drives the action batch; splitting breaks the Markov-step semantics. |
| 3 | **Edges are LLM-built tool calls, not regex / entity match.** | Entity matching does not generalize across tool surfaces; LLM + witness > brittle rules. |
| 4 | **Trust asymmetry lives in the extractor prompt.** | Extractor is an external observer; the prompt instructs it to anchor on `tool_call.args` + `tool_result` and treat thought as advisory. |
| 5 | **Edge witness is enforced at `add_edge` time.** | `data` requires `cited_entities` literally appearing in BOTH src and dst turn texts; `ref` requires `cited_quote` to be a verbatim substring of src and to appear in dst. The tool rejects mismatches. |
| 6 | **One `add_edge(kind=...)` tool, not one tool per kind.** | Smaller surface; per-kind required-field validation at runtime. |
| 7 | **`analyze(graph)` is non-gating.** | Returns `list[Finding]`; auditor decides whether findings matter. There is no `extractor_invalid` blocking entry in v3. |
| 8 | **Scenario-registered checks, not SDK-fixed.** | "Problem patterns" are domain-specific. Scenarios plug in via `api.audit.register_check(...)`. |
| 9 | **Auditor still produces J1: audit → reminder decision → reminder content.** | Single LLM, V2 `Verdict` shape preserved (§7.2). |
| 10 | **Auditor input degrades to summaries past N=30 events.** | Default: full graph as JSON. If `len(events) > 30`: send summaries + edges only, auditor pulls details on demand. |
| 11 | **Auditor tools: `get_turn(idx)`, `get_event_detail(ids)`, `submit_verdict`.** | Minimum viable surface, batched detail fetch. |

---

## 3. Mathematical sketch

```
Trajectory    T = ⟨t₀, …, tₙ₋₁⟩,  tᵢ ∈ Turn
Event         e = ⟨id, kind, summary, source_turns ⊆ [0,n)⟩
Edge          (src, dst, kind ∈ {data, ref}, witness)
Graph         G = (V, E)
Abstraction   α : T → G   (extractor LLM, inferential — no soundness proof)

Trust axiom:
  trust(K(tᵢ) ∩ {tool_call, tool_result, user})
    > trust(K(tᵢ) ∩ {thinking, text})
```

EventKind ∈ {`task`, `hyp`, `evid`, `act`, `dec`, `concl`}, classified
by **action signature**, not by what thought claims.

---

## 4. Resolutions for open items

### 4.a ReAct cycle boundary algorithm

**Rule.** Walk turns linearly. A cycle opens at the first
non-`tool_result` turn after the previous cycle closed (or at index 0).
A cycle **closes** when one of:

1. The next assistant turn carries `text` and/or `thinking` blocks
   **without** any `tool_call`. The closing turn belongs to the *next*
   cycle (it is the "next observation/thought" boundary), unless it is
   the final turn, in which case it is absorbed into the closing cycle
   as a `concl`-candidate.
2. Trajectory end is reached (session end / extractor firing window
   ends with an in-flight cycle).

**Parallel `tool_call`s** in a single assistant turn stay in the same
cycle (decision #2). All `tool_result` turns following an action batch
belong to that batch's cycle until a fresh assistant text/thinking
turn appears.

**Pathological streams.** If the extractor's firing window contains
only `tool_result` turns (no closing assistant message), the adapter
treats the window as **one open cycle** and fires the extractor on it
anyway. The extractor classifies it as a partial `evid` event with
`source_turns` covering the entire result span. The next firing
re-extracts the same window if the cursor was not advanced — see §6.

### 4.b `ref` edge near-paraphrase rule

**Rule (verbatim with case + whitespace normalization).** Both src
and dst turn texts and `cited_quote` are lowercased and have runs of
whitespace collapsed to a single space before substring comparison.
No edit-distance, no stemming.

**Why.** Verbatim-only is too brittle for plausible casing/quoting
variations the extractor will produce; bounded edit-distance is
non-deterministic to specify and review. Case+whitespace
normalization is reproducible without an LLM, easy to test, and
catches the realistic mismatches without inviting hallucinations.

### 4.c `api.audit.register_check` precise signature

**Surface.** Lives at `audit/registry.py` inside the `llmharness`
package (SDK side). Atom files under
`contrib/extensions/llmharness/src/llmharness/extensions/`
implement specific checks and call `api.audit.register_check` from
their `install(api, config)`. The §11 single-file contract is
preserved: each check ships in its own atom file; check
implementations are not co-located inside `audit/`.

```python
# Public Protocol (in audit/registry.py)
class CheckContext(Protocol):
    events: tuple[Event, ...]   # frozen snapshot, monotonic order
    edges:  tuple[Edge, ...]    # frozen snapshot

class Check(Protocol):
    name: str                              # human-readable, used in Finding.category
    def __call__(self, ctx: CheckContext) -> list[Finding]: ...

# Registration call atoms make from inside install(api, config):
api.audit.register_check(check)            # idempotent on (name, callable id)
```

**Semantics.**
- Registration is **per-session** (during atom install) and **idempotent**
  — registering the same `(name, callable)` twice is a no-op.
- The `audit/registry.py` registry is owned by the llmharness adapter;
  `compose_auditor_extensions()` reads `api.audit.registered_checks()`
  at firing time, runs each check on the frozen `CheckContext`, and
  concatenates the resulting `Finding` lists into the auditor prompt
  block.
- Per-firing semantics: every Phase 2 firing rebuilds the
  `CheckContext` from the current graph snapshot; checks are pure
  functions, no per-firing registration.
- **Zero registered checks** → empty `findings: []` block in the
  auditor prompt; the firing proceeds normally.
- A check that raises is logged with its `name` as a structured
  `extractor_check_error` payload on the verdict's metadata
  (non-fatal); other checks still run.

**SDK cleanliness.** `audit/registry.py` is mechanism only — it does
not embed any concrete check. Concrete checks (`check_repeated_actions`,
`check_open_branches`, `check_premature_conclusion`, …) ship as
separate atom files under `extensions/`. This satisfies the SDK vs.
scenario separation rule.

### 4.d Migration path for v2 entry types

| Entry type | v2 | v3 |
|---|---|---|
| `audit_event` | ✓ | ✓ (payload changes — see §7) |
| `audit_edge` | — | **new**: one entry per witnessed edge |
| `extractor_cursor` | ✓ | ✓ |
| `extractor_no_call` | ✓ | ✓ |
| `extractor_error` | ✓ | ✓ |
| `extractor_empty` | ✓ | ✓ |
| `extractor_invalid` | ✓ | **removed** (no gating in v3) |
| `extractor_partial` | — | **new**: emitted when ≥1 edge dropped after retry exhaustion; payload = `{dropped_edges: [{src, dst, kind, last_error}], turn_window: [a,b]}` |
| `verdict` | ✓ | ✓ (V2 shape preserved) |
| `audit_no_call` | ✓ | ✓ |
| `audit_error` | ✓ | ✓ |
| `reminder_delivered` | ✓ | ✓ |

**rca-autorl impact.** The dataset exporter must drop the
`extractor_invalid` branch and add `extractor_partial`. `Edge`
records become first-class — exporters that previously inferred edges
from `Event.refs` must read `audit_edge` entries.

### 4.e Design-doc / index propagation

Enumerated in `_v3_propagation_notes.md`. Implementer-phase docs
touched: `observability.md`, `extension-as-scenario.md`,
`per-task-evolution-loop.md`, `pluggable-architecture.md`, plus
`.claude/index.yaml` (add `audit_check_registry` concept; refresh
`related_concepts`).

### 4.f `add_edge` schema and witness algorithm

```python
add_edge(
    src_event_id: int,
    dst_event_id: int,
    kind: Literal["data", "ref"],
    reason: str,
    src_turns: list[int],
    dst_turns: list[int],
    cited_entities: list[str] = [],   # required non-empty if kind="data"
    cited_quote: str = "",            # required non-empty if kind="ref"
) -> {"ok": True} | {"error": "<detail>"}
```

**Validation order** (fail fast, return precise error string):

1. Both `src_event_id` and `dst_event_id` already registered in this
   firing's event set; `src ≠ dst`.
2. `src_turns ⊆ events[src].source_turns`,
   `dst_turns ⊆ events[dst].source_turns`.
3. Edge would not introduce a cycle in the existing data+ref edge
   set (DFS reachability).
4. Per-kind:
   - `kind="data"`: `cited_entities` is non-empty; for each entity
     `s`, `normalize(s)` is a substring of
     `normalize(concat(turn_text(t) for t in src_turns))` AND of
     `normalize(concat(turn_text(t) for t in dst_turns))`.
   - `kind="ref"`: `cited_quote` is non-empty; `normalize(cited_quote)`
     is a substring of `normalize(concat(src turns))` AND of
     `normalize(concat(dst turns))`. (The "appears in dst" check is
     where the case+whitespace rule from §4.b lands.)

`normalize(s)` ≜ lowercase + collapse whitespace.

**Turn text access for the extractor.** The full window of new turns
is embedded in the extractor system prompt as a JSON list keyed by
`turn_index`. The extractor does **not** need a `get_turn` tool in
v3 — at extraction time the entire window fits in context, and any
older-than-window text required for `ref` edges into prior events is
served by including the prior events' `summary` plus their
`source_turns` excerpts (≤ 800 chars each) inside the prompt header.
This avoids the round-trip of an in-extractor drill-down tool.

**Witness retry policy.** Each `add_edge` failure is reported back
to the extractor LLM as a tool-result error message, allowing the LLM
to retry with corrected fields. Per **(src, dst, kind)** tuple within
a single firing: `max_retry = 2`. After the third failure, the
adapter records the dropped edge into the `extractor_partial` payload
and rejects further attempts on the same tuple with a terminal error
string `"giving up on this edge"`. Other edges continue normally.

### 4.g Auditor degradation rule

**Rule (N = 30).** When `len(events) ≤ 30`: pass full
`Event.to_dict()` records plus all edges into the prompt. When
`len(events) > 30`: pass a degraded shape:

```json
{
  "events": [{"id": ..., "kind": ..., "summary": ..., "source_turns": [...]}],
  "edges":  [{"src": ..., "dst": ..., "kind": ..., "reason": ...}]
}
```

Witness fields (`cited_entities`, `cited_quote`) and any per-event
extra metadata are stripped. The auditor pulls them on demand via
`get_event_detail([ids])`. The `30` threshold is configurable via
`audit_summary_threshold`; default 30 chosen because at ~150 tokens
per full event-with-edges, 30 events ≈ 4.5k tokens — fits the audit
prompt budget without degradation.

---

## 5. Triggering

```
on TurnEndEvent:
    await run_phase_1()                        # always, advances cursor
    if (turn_count % k) == 0:
        await run_phase_2()                    # graph-level audit
```

`turn_count` counts `TurnEndEvent` firings, not message indices.
`k = audit_interval_turns`, default 3. Phase 2 runs even when Phase 1
emitted `extractor_partial` — the auditor explicitly tolerates a
partial graph (decision #7).

---

## 6. Persistence layout

| Entry type | Producer | Payload |
|---|---|---|
| `llmharness.audit_event` | Phase 1 success | `Event.to_dict()` |
| `llmharness.audit_edge` | Phase 1 success | `Edge.to_dict()` |
| `llmharness.extractor_cursor` | Phase 1 (every firing) | `{"last_turn_index": int, "extraction_run_id": str}` |
| `llmharness.extractor_no_call` | Phase 1 failure | `{"reason": str, "turn_window": [a,b]}` |
| `llmharness.extractor_error` | Phase 1 failure | `{"reason": str, "turn_window": [a,b]}` |
| `llmharness.extractor_empty` | Phase 1 (zero-event submit on non-trivial window) | `{"turn_window": [a,b]}` |
| `llmharness.extractor_partial` | Phase 1 (≥1 edge dropped after retry exhaustion) | `{"dropped_edges": [...], "turn_window": [a,b]}` |
| `llmharness.verdict` | Phase 2 success | `Verdict.to_dict()` |
| `llmharness.audit_no_call` | Phase 2 failure | `{"reason": str}` |
| `llmharness.audit_error` | Phase 2 failure | `{"reason": str}` |
| `llmharness.reminder_delivered` | Reminder injection | `{"text": str}` |

`extractor_cursor` advances on every firing that wrote at least one
event (including `extractor_partial`). On `extractor_no_call`,
`extractor_error`, `extractor_empty` the cursor does **not** advance —
the next firing re-attempts the same window.

---

## 7. Tool-call schemas

### 7.1 Extractor tools

```python
register_event(
    turn_indices: list[int],
    kind: Literal["task", "hyp", "evid", "act", "dec", "concl"],
    summary: str,                     # ≤ 30 words
) -> {"event_id": int}

add_edge(...)  # see §4.f

submit_extraction() -> ToolTerminate  # marks the firing complete
```

`submit_extraction` (replaces v2's `submit_events`) takes no
arguments — events and edges are already accumulated through the
prior tool calls. It exists solely to give the LLM a clean
"I'm done" terminator.

### 7.2 Auditor tools

```python
get_turn(idx: int) -> dict                       # serialized raw turn
get_event_detail(event_ids: list[int]) -> dict   # full Event + Edge records
submit_verdict(verdict: V2Verdict) -> ToolTerminate
```

`V2Verdict` shape (preserved from v2 §6.2):

```python
{
    "surface_reminder": bool,
    "reminder_text": str,                # non-empty iff surface_reminder
    "continuation_notes": list[str],     # auditor's free-text memory
    "matched_event_ids": list[int],
    "cited_cards": list[str],
}
```

### 7.3 Edge / Event records

```python
@dataclass(frozen=True)
class Edge:
    src: int
    dst: int
    kind: Literal["data", "ref"]
    reason: str
    src_turns: tuple[int, ...]
    dst_turns: tuple[int, ...]
    cited_entities: tuple[str, ...]   # empty for ref
    cited_quote: str                  # empty for data

@dataclass(frozen=True)
class Finding:
    category: str
    description: str
    related_event_ids: tuple[int, ...]
```

`Event` keeps its v2 fields **minus** `refs: list[int]` (now
materialized as `Edge` records).

---

## 8. Module layout

```
contrib/extensions/llmharness/src/llmharness/
├── audit/
│   ├── extractor/
│   │   ├── prompt.py                # v3 prompt with trust-asymmetry block
│   │   ├── tools.py                 # register_event + add_edge + submit_extraction
│   │   ├── witness.py               # normalize() + witness validation
│   │   ├── extensions.py            # compose_extractor_extensions()
│   │   └── output.py                # extractor turn-window serializer
│   ├── auditor/
│   │   ├── prompt.py                # v3 prompt: trust asymmetry + degradation handling
│   │   ├── submit_tool.py           # submit_verdict (V2 shape, unchanged)
│   │   ├── extensions.py            # compose_auditor_extensions(): reads registered checks
│   │   ├── get_turn_tool.py
│   │   ├── get_event_detail_tool.py
│   │   └── output.py
│   ├── registry.py                  # api.audit.register_check + Check Protocol
│   ├── _enum_schema.py              # EVENT_KIND_VALUES, EDGE_KIND_VALUES
│   ├── entry_types.py               # constants — extractor_invalid removed, extractor_partial added
│   └── __init__.py
├── extensions/                      # check atoms (one file each)
│   ├── check_repeated_actions.py
│   ├── check_open_branches.py
│   └── ...
└── adapters/
    └── agentm.py                    # orchestrates Phase 1 + Phase 2
```

**§11 contract:** every `extensions/check_*.py` is a single file with
`MANIFEST` + `install(api, config)`, no atom-to-atom imports, no
`core._internal`, no `harness.session`. Checks reach the registry
exclusively through `api.audit.register_check`.

---

## 9. Public contract

- `schema.py`: `Event`, `Edge`, `Verdict`, `Reminder`, `EventKind`,
  `Finding`. **Breaking from v2:**
  - `Event.refs` removed.
  - `Edge` is new (top-level, witnessed).
  - `DriftType` stays removed.
  - Verdict shape unchanged from v2.
- `api.audit` namespace is new public surface:
  `register_check(check)` and `registered_checks() -> tuple[Check, ...]`.
- `pyproject.toml` `version` bumps; downstream rca-autorl pinned
  upgrade required.
- `cards.py`, `cards_tools` atom, `inherit_provider`, `BeforeAgentStartEvent`
  injection — unchanged from v2.

---

## 10. Deferred (P2)

1. Per-kind edge tools (`add_data_edge` / `add_ref_edge`) — revisit if
   the LLM struggles with the branched schema.
2. `get_event_detail` for the extractor (drill into older-than-window
   prior events) — added when prompt-embedded summaries prove too lossy.
3. Cross-session graph reuse and long-running extractor daemon.
4. Bounded-edit-distance witness rule — only if §4.b proves too strict.
5. Auditor mid-`k`-window triggers fired by graph-pattern heuristics.
6. Fork/merge ("branch") quality as a first-class check
   — currently expressible as a registered check from rca / coding
   scenarios.

---

## 11. References

- [observability.md](observability.md) — training-data join works on
  the same `<trace>.jsonl`; entry-type delta in §4.d propagates here.
- [extension-as-scenario.md](extension-as-scenario.md) — §11
  single-file contract that check atoms follow.
- [pluggable-architecture.md](pluggable-architecture.md) — the new
  `api.audit` namespace is a non-Operations ExtensionAPI service.
- [per-task-evolution-loop.md](per-task-evolution-loop.md) — verdicts +
  graph remain the evolution-loop input.
- [GitHub issue #134 (Lincyaw/AgentM)](https://github.com/Lincyaw/AgentM/issues/134) —
  v3 design proposal.
