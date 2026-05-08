# Two-Phase Cognitive Audit for llmharness

**Status:** active design (V1; supersedes single-pass audit in `llmharness-cognitive-audit.md` §4.5)
**Owners:** llmharness scenario
**Last reviewed:** 2026-05-08
**Predecessor:** `designs/llmharness-cognitive-audit.md` (V0 single-pass realization)

---

## 1. Motivation

V0 wedges three jobs into one child session per `TurnEndEvent`:
trajectory ingestion, event extraction, and drift adjudication. That
shape has four problems baked in:

1. **O(N²) token cost.** Every `TurnEndEvent` re-extracts events from
   every prior turn; the audit prompt grows with the full trajectory.
2. **Conflated failure modes.** A child session that crashes mid-flight,
   declines to call `submit_audit`, or judges "no drift" all collapse
   to the same `([], Verdict(drift=False))` fallback. Extraction outage
   and clean trajectory become indistinguishable.
3. **Thinking blocks dropped.** The serializer hands the auditor only
   user-visible text, but agent thinking is where drift first surfaces
   (premature conclusion, ignored evidence). The single-pass shape
   leaves no room to put thinking in front of one consumer (extractor)
   without burning tokens on the other (auditor).
4. **No reusable mid-product.** The event graph never materializes as
   a first-class artifact — the data shape that downstream training,
   forensics, and human dashboards most want is not the verdict but
   the structured timeline that produced it.

The two-phase split addresses all four by giving extraction its own
incremental loop and reducing the auditor's input from the trajectory
to the graph.

---

## 2. Two phases

### Phase 1 — Extractor (per `TurnEndEvent`, incremental)

Input: the new turn(s) since the last extractor firing **plus** the
recent slice of the existing event graph (for `refs` resolution).
Output: a non-empty list of new `Event` records, ref-linked to prior
events and stamped with monotonic ids.

Realization (option **B** from design discussion, **no rules layer**):

- Stateless child AgentM session spawned per `TurnEndEvent`.
- State (the event graph) lives entirely in the parent session's entry
  tree under `llmharness.audit_event` entries. Each firing reads the
  tail and appends.
- Terminal tool: `submit_events(events=[...])` → `ToolTerminate`.
- The extractor receives **everything** from the new turns, including
  thinking blocks, tool calls, and tool results. Rejected from V0:
  the "pure-text moves only" filter.
- Failure of the extractor is **not silent**. Three failure modes are
  distinguished and recorded as entries the auditor can read:
  - `llmharness.extractor_no_call` — child exited without calling
    `submit_events`
  - `llmharness.extractor_error` — spawn or prompt raised
  - `llmharness.extractor_empty` — submitted but `events=[]` for a
    non-trivial input window
  These entries do not poison the graph (auditor filters them), but
  they make outage visible and become a primary diagnostic for runs
  that "should have" detected drift but didn't.

### Phase 2 — Auditor (every `k` turns over the graph)

Input: the event graph (full, or a configured tail) **plus** recent
verdicts. Notably **not** the raw trajectory — auditor reads structured
events, not free text.

Output: a `Verdict` plus optional `Reminder`. Same shape as V0 (so
`schema.py` public contract holds for rca-autorl).

Realization:

- Triggered on `TurnEndEvent` only when `(turn_count % k) == 0`.
  Default `k = 3`; configurable via `audit_interval_turns` knob.
- Stateless child session, like Phase 1.
- Terminal tool: `submit_verdict(verdict={...})` → `ToolTerminate`.
- Auditor MAY drill back into raw turns via an optional `fetch_turn(index)`
  tool if uncertain; the registration of that tool is gated by a config
  flag (default off in V1, on in V2 once we measure how often it would
  be used).
- Failure modes recorded the same way as Phase 1 (`audit_no_call`,
  `audit_error`).

---

## 3. Triggering

Single `TurnEndEvent` handler, two phase invocations:

```
on TurnEndEvent:
    await run_phase_1()                      # always
    if (turn_count % k) == 0:
        await run_phase_2()                   # graph-level audit
```

`turn_count` is the count of `TurnEndEvent` firings observed in the
session, not the absolute message count. Auditor sees a graph that
has at least `k` new turns of events extracted onto it.

V1 keeps "every k turns" as the only Phase 2 trigger. Graph-pattern
heuristics (e.g. "3 actions in a row with no reflection event") are
deferred to V2 — they require the graph schema to stabilize first.

---

## 4. Persistence layout

All state lives on the session entry tree (no new file artifacts).
Entry types (namespaced):

| Entry type | Producer | Payload |
|---|---|---|
| `llmharness.audit_event` | Phase 1 success | `Event.to_dict()` (existing schema) |
| `llmharness.extractor_cursor` | Phase 1 (every firing) | `{"last_turn_index": int, "extraction_run_id": str}` |
| `llmharness.extractor_no_call` | Phase 1 failure | `{"reason": str, "turn_window": [a, b]}` |
| `llmharness.extractor_error` | Phase 1 failure | `{"reason": str, "turn_window": [a, b]}` |
| `llmharness.extractor_empty` | Phase 1 (zero-event submit on non-trivial window) | `{"turn_window": [a, b]}` |
| `llmharness.verdict` | Phase 2 success | `Verdict.to_dict()` (existing schema) |
| `llmharness.audit_no_call` | Phase 2 failure | `{"reason": str}` |
| `llmharness.audit_error` | Phase 2 failure | `{"reason": str}` |

The `extractor_cursor` entry is the explicit "what's the last turn we
extracted from" pointer. Phase 1 reads it on entry, computes
`new_turns = messages[cursor.last_turn_index + 1 :]`, runs the LLM,
appends new events, and writes a fresh cursor. This is how
"persistent process" gets simulated without a long-running session.

---

## 5. Event graph rules

### 5.1 Immutability

Event entries are **append-only**. No `update_event`, no in-place
mutation. This is the stable target you asked about.

### 5.2 Invalidation via refs

When a later event materially contradicts or refines an earlier one,
the new event's `refs` list points at the earlier event's id. To make
the relation legible (without a preset edge-type enum, per
`feedback_no_preset_subjective_labels.md`), the new event's `summary`
text describes the relation in free form ("contradicts ev #4 by
showing the file does not contain what was inferred there"). The
auditor reads summaries; readers needing structured edges in V2 can
extract them from summaries with a small classifier.

### 5.3 Graph queries

V1 keeps querying the graph as branch-walks:

```python
branch = api.session.get_branch()
events = [Event.from_dict(e.payload) for e in branch
          if e.type == "llmharness.audit_event"]
```

V1 does **not** add a typed query API to `ReadonlySession`. If V2
auditing needs `events_referencing(id)` or `events_of_kind(kind)`,
promote those to `ReadonlySession` then. Today, one consumer.

---

## 6. Tool-call schemas

### 6.1 `submit_events`

Replaces the `events` half of V0's `submit_audit`.

```python
{
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                # same shape as V0 _EVENT_SCHEMA, generated from EventKind enum
                # rather than hand-listed; see §7.2.
            },
        },
    },
    "required": ["events"],
    "additionalProperties": False,
}
```

`events` MAY be empty. If empty for a non-trivial new-turn window,
adapter writes `extractor_empty` so the failure is visible.

### 6.2 `submit_verdict`

Replaces the `verdict` half of V0's `submit_audit`.

```python
{
    "type": "object",
    "properties": {
        "verdict": {
            # same shape as V0 _VERDICT_SCHEMA, with these tightenings:
            # - drift / type relation expressed via JSON Schema if/then so
            #   "drift=true ⇒ type required" is enforced at the provider edge,
            #   not silently dropped by the adapter.
            # - type / kind enums generated from DriftType / EventKind.
        },
    },
    "required": ["verdict"],
    "additionalProperties": False,
}
```

---

## 7. What changes vs. V0

### 7.1 Module structure

```
scenarios/llmharness/src/llmharness/
├── audit/
│   ├── extractor/                    ← new
│   │   ├── prompt.py                 EXTRACTOR_SYSTEM_PROMPT
│   │   ├── submit_tool.py            submit_events terminal tool
│   │   ├── extensions.py             compose_extractor_extensions()
│   │   └── output.py                 RawExtractorOutput coercion
│   ├── auditor/                      ← new
│   │   ├── prompt.py                 AUDITOR_SYSTEM_PROMPT (graph-only)
│   │   ├── submit_tool.py            submit_verdict terminal tool
│   │   ├── extensions.py             compose_auditor_extensions()
│   │   └── output.py                 RawVerdictOutput coercion
│   └── __init__.py                   re-exports
├── adapters/
│   └── agentm.py                     orchestrates Phase 1 + Phase 2
└── ...
```

V0's `audit/{prompt,submit_tool,output,extensions}.py` are deleted.
Their content splits across the two new subpackages.

### 7.2 Schema parity

JSON-Schema enums for `EventKind` / `DriftType` are **derived from
the enum classes**, not hand-listed. Single source of truth:

```python
EVENT_KIND_VALUES = [k.value for k in EventKind]
DRIFT_TYPE_VALUES = [t.value for t in DriftType] + [None]
```

This closes the V0 problem of two enum copies drifting silently.

### 7.3 Adapter

```python
async def _on_turn_end(event):
    nonlocal turn_count, pending
    turn_count += 1

    branch = api.session.get_branch()
    new_events = await _run_extractor(api, branch, extractor_extensions)

    if (turn_count % k) == 0:
        verdict = await _run_auditor(api, branch + new_events, auditor_extensions)
        if verdict and verdict.drift and verdict.reminder and verdict.type:
            pending = Reminder(...)
```

`_run_extractor` and `_run_auditor` each spawn their own child session
with the right `compose_*_extensions()` list. Both share the
`provider=None` + `inherit_provider` auto-wire pattern.

### 7.4 Trajectory serialization

V0's `_serialize_for_audit_prompt` becomes `_serialize_new_turns_for_extractor`:

- Slices `messages[cursor.last_turn_index + 1 :]` instead of the full
  list.
- **Keeps thinking blocks.** Block dropped only if the kernel produced
  no text content for it (e.g. internal-state-only blocks the kernel
  records but the SDK does not surface to extensions).
- Tool-result content kept structured (not flattened to one string)
  so the extractor can distinguish error vs success.

The auditor receives **no** trajectory serialization — it sees the
graph in `Event.to_dict()` form plus recent verdicts.

---

## 8. Failure semantics (concrete)

The V0 silent-fallback problem is fixed by recording failure as a
distinct entry type (§4) instead of a no-drift verdict. Concretely:

| Scenario | V0 behavior | V1 behavior |
|---|---|---|
| Child spawn raises | silent `Verdict(drift=False)` | `extractor_error` / `audit_error` entry |
| Child runs but never calls submit | silent `Verdict(drift=False)` | `extractor_no_call` / `audit_no_call` entry |
| Child submits empty events on real new turns | event list empty (indistinguishable from clean trajectory) | `extractor_empty` entry |
| Child submits parsed verdict drift=false | normal no-drift verdict | unchanged |
| Child submits drift=true with type=null | reminder silently dropped | provider-side schema rejects it (if/then) → no submit possible → `audit_no_call` |

Operators reading a session's entry stream can tell at a glance
whether the audit pipeline stayed alive.

---

## 9. What stays the same

- `schema.py` public contract: `Event`, `Verdict`, `Reminder`,
  `EventKind`, `DriftType` — all unchanged. rca-autorl break-free.
- `cards.py` and the AFC card YAMLs ship as package data, unchanged.
- `cards_tools` atom unchanged — both extractor and auditor can load
  it (see §7.1 `compose_*_extensions`).
- `inherit_provider` builtin unchanged — both phase children rely on
  it.
- `BeforeAgentStartEvent.system` injection mechanism unchanged.

---

## 10. Open questions / V2 deferred

1. **Auditor `fetch_turn` tool.** Defer until measurement shows the
   auditor genuinely needs to drill back. V1 graph-only.
2. **Graph-pattern triggers** (cheap heuristics that can fire the
   auditor mid-`k`-window). Deferred to V2; needs stable graph
   schema first.
3. **Cross-session graph reuse.** Today each session's graph is
   isolated. V2 may join across sessions for the same task / agent
   profile.
4. **Long-running extractor session.** Option A from design discussion
   — the extractor as a daemon child with streaming input — was
   rejected for V1 because AgentM lacks the long-lived child + stream
   primitives. Revisit when SDK gains them.
5. **Ref edge typing.** V1 keeps `refs: list[int]`, free-text relation
   in `summary`. V2 may add a small classifier that materializes typed
   edges for forensic queries.

---

## 11. Migration

V1 is a hard cut, not a parallel track:

- `audit/{prompt,submit_tool,output,extensions}.py` deleted.
- `audit/{extractor,auditor}/...` added per §7.1.
- `adapters/agentm.py` rewritten per §7.3.
- `compose_extensions` factory removed (split into two factories per
  §7.1).
- Tests rewritten: V0's smoke tests checked the V0 `compose_extensions`
  signature and the V0 `submit_audit` schema. Both go away.

A new stub-provider integration test pinned to the two-phase
trajectory becomes the V1 fail-stop:

> Spawn a session, drive a 4-turn dialog, assert (a) Phase 1 fired
> 4 times, (b) Phase 2 fired once at turn 3 (`k=3`), (c) the entry
> tree contains the expected mix of `audit_event` + `verdict` +
> at most one `extractor_*` failure entry per phase invocation.

---

## 12. References

- Predecessor: [llmharness-cognitive-audit.md](llmharness-cognitive-audit.md) §4.5
- Substrate: [observability.md](observability.md) (training-data join still works on the same `<trace>.jsonl`)
- Provider inheritance: [pluggable-architecture.md](pluggable-architecture.md) — V1 keeps the `inherit_provider` builtin path
