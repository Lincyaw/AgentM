# Cognitive Audit for llmharness

**Status:** active design
**Owners:** llmharness scenario
**Last reviewed:** 2026-05-08

A two-phase audit pipeline that supervises a running main agent for
semantic-level drift. Phase 1 extracts a typed event graph from each
new turn; Phase 2 judges drift over the graph every `k` turns and
emits an optional reminder back into the trajectory.

---

## 1. Goals

- **Incremental extraction.** Each `TurnEndEvent` only re-reads the
  new turns since the last firing — cost is roughly O(N) over the
  session, not O(N²).
- **Disjoint inputs per phase.** The extractor sees the trajectory
  (including thinking blocks); the auditor sees only the structured
  graph plus recent verdicts. Each consumer's prompt carries only
  what it needs.
- **Visible failure modes.** Spawn errors, no-call exits, and empty
  submissions are recorded as distinct entry types instead of
  collapsing to a silent "no drift" verdict.
- **Reusable mid-product.** The event graph is a first-class artifact
  on the session entry tree — readable by downstream training,
  forensics, and dashboards without re-deriving from raw turns.

---

## 2. Two phases

### Phase 1 — Extractor (per `TurnEndEvent`, incremental)

Input: the new turn(s) since the last extractor firing **plus** the
recent slice of the existing event graph (for `refs` resolution).
Output: a list of new `Event` records, ref-linked to prior events
and stamped with monotonic ids.

- Stateless child AgentM session spawned per `TurnEndEvent`.
- State (the event graph) lives entirely in the parent session's
  entry tree under `llmharness.audit_event` entries. Each firing
  reads the tail and appends.
- Terminal tool: `submit_events(events=[...])` → `ToolTerminate`.
- The extractor receives **everything** from the new turns,
  including thinking blocks, tool calls, and tool results.
- Failure of the extractor is **not silent** — see §4.

### Phase 2 — Auditor (every `k` turns over the graph)

Input: the event graph (full, or a configured tail) **plus** recent
verdicts. Notably **not** the raw trajectory — auditor reads
structured events, not free text.

Output: a `Verdict` plus optional `Reminder`.

- Triggered on `TurnEndEvent` when `(turn_count % k) == 0`.
  Default `k = 3`; configurable via `audit_interval_turns`.
- Stateless child session, like Phase 1.
- Terminal tool: `submit_verdict(verdict={...})` → `ToolTerminate`.
- Failure modes recorded as distinct entries (§4).

---

## 3. Triggering

Single `TurnEndEvent` handler, two phase invocations:

```
on TurnEndEvent:
    await run_phase_1()                      # always
    if (turn_count % k) == 0:
        await run_phase_2()                   # graph-level audit
```

`turn_count` is the count of `TurnEndEvent` firings observed in
the session, not the absolute message count. The auditor sees a
graph that has at least `k` new turns of events extracted onto it.

---

## 4. Persistence layout

All state lives on the session entry tree (no new file artifacts).
Entry types (namespaced):

| Entry type | Producer | Payload |
|---|---|---|
| `llmharness.audit_event` | Phase 1 success | `Event.to_dict()` |
| `llmharness.extractor_cursor` | Phase 1 (every firing) | `{"last_turn_index": int, "extraction_run_id": str}` |
| `llmharness.extractor_no_call` | Phase 1 failure | `{"reason": str, "turn_window": [a, b]}` |
| `llmharness.extractor_error` | Phase 1 failure | `{"reason": str, "turn_window": [a, b]}` |
| `llmharness.extractor_empty` | Phase 1 (zero-event submit on non-trivial window) | `{"turn_window": [a, b]}` |
| `llmharness.verdict` | Phase 2 success | `Verdict.to_dict()` |
| `llmharness.audit_no_call` | Phase 2 failure | `{"reason": str}` |
| `llmharness.audit_error` | Phase 2 failure | `{"reason": str}` |

The `extractor_cursor` entry is the explicit "what's the last turn
we extracted from" pointer. Phase 1 reads it on entry, computes
`new_turns = messages[cursor.last_turn_index + 1 :]`, runs the LLM,
appends new events, and writes a fresh cursor.

Failure entries do not poison the graph (auditor filters them), but
they make outage visible. Operators reading a session's entry stream
can tell at a glance whether the audit pipeline stayed alive:

| Scenario | Behavior |
|---|---|
| Child spawn raises | `extractor_error` / `audit_error` entry |
| Child runs but never calls submit | `extractor_no_call` / `audit_no_call` entry |
| Child submits empty events on real new turns | `extractor_empty` entry |
| Child submits parsed verdict drift=false | normal no-drift verdict |
| Child submits drift=true with type=null | provider-side schema rejects it (if/then) → no submit possible → `audit_no_call` |

---

## 5. Event graph rules

### 5.1 Immutability

Event entries are **append-only**. No `update_event`, no in-place
mutation.

### 5.2 Invalidation via refs

When a later event materially contradicts or refines an earlier
one, the new event's `refs` list points at the earlier event's id.
The relation is described in free-form `summary` text (per
`feedback_no_preset_subjective_labels.md`, no edge-type enum).
Readers needing structured edges can extract them from summaries
with a small classifier.

### 5.3 Graph queries

Querying the graph is a branch-walk:

```python
branch = api.session.get_branch()
events = [Event.from_dict(e.payload) for e in branch
          if e.type == "llmharness.audit_event"]
```

No typed query API on `ReadonlySession` — promote
`events_referencing(id)` / `events_of_kind(kind)` to the SDK only
when a second consumer needs them.

---

## 6. Tool-call schemas

### 6.1 `submit_events`

```python
{
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                # kind enum derived from EventKind (see §7.2)
            },
        },
    },
    "required": ["events"],
    "additionalProperties": False,
}
```

`events` MAY be empty. If empty for a non-trivial new-turn window,
the adapter writes `extractor_empty` so the failure is visible.

### 6.2 `submit_verdict`

```python
{
    "type": "object",
    "properties": {
        "verdict": {
            # drift / type relation expressed via JSON Schema if/then so
            # "drift=true ⇒ type required" is enforced at the provider edge,
            # not silently dropped by the adapter.
            # type / kind enums generated from DriftType / EventKind.
        },
    },
    "required": ["verdict"],
    "additionalProperties": False,
}
```

---

## 7. Implementation notes

### 7.1 Module layout

```
contrib/extensions/llmharness/src/llmharness/
├── audit/
│   ├── extractor/
│   │   ├── prompt.py                 EXTRACTOR_SYSTEM_PROMPT
│   │   ├── submit_tool.py            submit_events terminal tool
│   │   ├── extensions.py             compose_extractor_extensions()
│   │   └── output.py                 RawExtractorOutput coercion
│   ├── auditor/
│   │   ├── prompt.py                 AUDITOR_SYSTEM_PROMPT (graph-only)
│   │   ├── submit_tool.py            submit_verdict terminal tool
│   │   ├── extensions.py             compose_auditor_extensions()
│   │   └── output.py                 RawVerdictOutput coercion
│   ├── _compose.py                   shared extension-list composer
│   ├── _enum_schema.py               EVENT_KIND_VALUES / DRIFT_TYPE_VALUES
│   └── __init__.py
├── adapters/
│   └── agentm.py                     orchestrates Phase 1 + Phase 2
└── ...
```

### 7.2 Schema parity

JSON-Schema enums for `EventKind` / `DriftType` are derived from
the enum classes — single source of truth:

```python
EVENT_KIND_VALUES = [k.value for k in EventKind]
DRIFT_TYPE_VALUES = [t.value for t in DriftType] + [None]
```

### 7.3 Adapter

```python
async def _on_turn_end(event):
    nonlocal turn_count, pending
    turn_count += 1

    branch = api.session.get_branch()
    new_events = await _run_extractor(api, branch, extractor_extensions)

    if (turn_count % k) == 0:
        verdict = await _run_auditor(
            api, branch + new_events, auditor_extensions
        )
        if verdict and verdict.drift and verdict.reminder and verdict.type:
            pending = Reminder(...)
```

`_run_extractor` and `_run_auditor` each spawn their own child
session with the right `compose_*_extensions()` list. Both share
the `provider=None` + `inherit_provider` auto-wire pattern.

### 7.4 Trajectory serialization

`_serialize_new_turns_for_extractor` slices
`messages[cursor.last_turn_index + 1 :]` and keeps thinking blocks
plus structured tool-result content. The auditor receives no
trajectory serialization — it sees the graph in `Event.to_dict()`
form plus recent verdicts.

---

## 8. Public contract (unchanged)

- `schema.py`: `Event`, `Verdict`, `Reminder`, `EventKind`,
  `DriftType` — stable for rca-autorl.
- `cards.py` and the AFC card YAMLs ship as package data.
- `cards_tools` atom — both extractor and auditor can load it.
- `inherit_provider` builtin — both phase children rely on it.
- `BeforeAgentStartEvent.system` injection for reminders.

---

## 9. Deferred

1. **Auditor `fetch_turn` tool.** Drill back into raw turns when
   the graph is ambiguous. Defer until measurement shows it's
   needed.
2. **Graph-pattern triggers.** Cheap heuristics that fire the
   auditor mid-`k`-window. Needs the graph schema to stabilize.
3. **Cross-session graph reuse.** Each session's graph is
   isolated; joining across sessions for the same task / agent
   profile is future work.
4. **Long-running extractor session.** Today's stateless per-turn
   spawn could become a streaming daemon once AgentM gains
   long-lived child + stream primitives.
5. **Ref edge typing.** A small classifier that materializes
   typed edges from free-text `summary` relations.

---

## 10. References

- [observability.md](observability.md) — training-data join works
  on the same `<trace>.jsonl`.
- [pluggable-architecture.md](pluggable-architecture.md) — the
  `inherit_provider` builtin path used by both child sessions.
