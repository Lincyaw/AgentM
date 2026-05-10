# Cognitive Audit for llmharness

**Status:** active design
**Owners:** llmharness scenario
**Last reviewed:** 2026-05-10

A two-phase audit pipeline that supervises a running main agent for
semantic-level drift. Phase 1 extracts a typed event graph from each
new turn; Phase 2 judges drift over the graph every `k` turns and
emits an optional reminder back into the trajectory.

The auditor is the **primary judge** — deterministic signals are
advisory hints in the prompt, not a triage queue and not a mandatory
checklist. The verdict shape carries free-text continuation memory
(`continuation_notes`) rather than a preset enum of drift types (see
§8 for the DriftType rationale).

---

## 1. Goals

- **Incremental extraction.** Each `TurnEndEvent` only re-reads the
  new turns since the last firing — cost is roughly O(N) over the
  session, not O(N²).
- **Disjoint inputs per phase.** The extractor sees the trajectory
  (including thinking blocks); the auditor sees only the structured
  graph plus recent verdicts. Each consumer's prompt carries only
  what it needs.
- **Visible failure modes.** Spawn errors, no-call exits, empty
  submissions, and graph-validation rejections are recorded as
  distinct entry types instead of collapsing to a silent "no drift"
  verdict.
- **Reusable mid-product.** The event graph is a first-class artifact
  on the session entry tree — readable by downstream training,
  forensics, and dashboards without re-deriving from raw turns.
- **LLM-led judgment with advisory hints.** The auditor LLM is the
  primary judge. Concepts such as backward continuity, forward
  fulfillment, and content correctness are framing lenses the auditor
  looks through, not rules it must apply in sequence. Cheap
  deterministic signals (§2.2, §7.5) are fed as scaffolding in the
  prompt; the auditor may ignore them or flag concerns they missed.

---

## 2. Two phases

### 2.1 Phase 1 — Extractor (per `TurnEndEvent`, incremental)

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

After `submit_events` the adapter runs the **Phase 1 graph
validator** (§5.4) before committing events. Validation failure
→ `llmharness.extractor_invalid` entry; auditor skipped this
firing.

### 2.2 Phase 2 — Auditor (every `k` turns over the graph)

Input: the validated event graph (full, or a configured tail)
**plus** recent verdicts **plus** advisory hint block (§7.5).
Notably **not** the raw trajectory — the auditor reads structured
events plus computed signals, not free-text turns by default. The
auditor may pull individual turns on demand via `get_turn(idx)`
when an event's `source_turns` reference needs verification.

Output: a `Verdict` (V2 shape — §6.2) plus optional `Reminder`.

- Triggered on `TurnEndEvent` when `(turn_count % k) == 0`.
  Default `k = 3`; configurable via `audit_interval_turns`.
- Stateless child session, like Phase 1.
- Terminal tool: `submit_verdict(verdict={...})` → `ToolTerminate`.
- Auditor has access to `get_turn(idx)` tool (§6.3) to drill back
  into the raw conversation when event `source_turns` refs need
  verification.
- Failure modes recorded as distinct entries (§4).

**Auditor lenses (prompt framing, not rules):**
- *Backward continuity* — are the agent's current actions traceable
  to what was known at the task start?
- *Forward fulfillment* — is the current trajectory converging on
  the stated task?
- *Content correctness* — are evidence citations accurate?
- *Branch quality* — see §5.5.

The auditor walks these lenses informally. There is no obligation
to "answer" each lens; they exist to direct attention.

---

## 3. Triggering

Single `TurnEndEvent` handler, two phase invocations:

```
on TurnEndEvent:
    await run_phase_1()                       # always
    if phase_1_graph_valid and (turn_count % k) == 0:
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
| `llmharness.extractor_invalid` | Phase 1 (graph validation failed) | `{"violations": list[str], "turn_window": [a, b]}` |
| `llmharness.verdict` | Phase 2 success | `Verdict.to_dict()` |
| `llmharness.audit_no_call` | Phase 2 failure | `{"reason": str}` |
| `llmharness.audit_error` | Phase 2 failure | `{"reason": str}` |
| `llmharness.reminder_delivered` | Reminder injection | `{"text": str}` |

The `extractor_cursor` entry is the explicit "what's the last turn
we extracted from" pointer. Phase 1 reads it on entry, computes
`new_turns = messages[cursor.last_turn_index + 1 :]`, runs the LLM,
validates the graph (§5.4), appends new events on success, and
writes a fresh cursor.

Failure entries do not poison the graph (auditor filters them), but
they make outage visible:

| Scenario | Behavior |
|---|---|
| Child spawn raises | `extractor_error` / `audit_error` entry |
| Child runs but never calls submit | `extractor_no_call` / `audit_no_call` entry |
| Child submits empty events on real new turns | `extractor_empty` entry |
| Graph validator rejects submitted events | `extractor_invalid` entry; auditor skipped |
| Child submits valid verdict | normal `llmharness.verdict` entry |

Constants for these entry-type strings live in
`audit/entry_types.py` (single source of truth shared by the adapter
and the dataset exporter).

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

### 5.4 Phase 1 graph validator

A mechanical validation step runs in the adapter after each
`submit_events` call. It is **not** an LLM — it is pure Python
logic operating on the submitted event list plus the existing graph.

Checks performed:

1. **Ref resolution.** Every id in `event.refs` resolves to a prior
   event in the graph (existing + newly submitted, in monotonic
   order — a new event may ref a prior new event from the same
   batch but never a later one).
2. **No cycles.** The directed ref graph has no cycles.
3. **kind↔source_turns rules:**
   - `evidence` events: `source_turns` must include at least one
     turn whose role/content carries a `tool_result`, or is a
     `user` message, or is an `assistant` thinking block.
   - `action` events: `source_turns` must include at least one
     turn carrying a `tool_call`.
4. **Task reachability.** Every event traces back to a `task` event
   via the ref chain (modulo the `task` event itself).
5. **Conclusion reachability.** If a `conclusion` event is present,
   it is reachable from `task` via refs.

On any violation: no new events are committed; adapter writes
`llmharness.extractor_invalid` with a `violations` list; auditor
is skipped for this firing. Garbage-in is the worst failure mode
for an LLM auditor reading a graph — refusing to feed an invalid
graph forward is cheaper than asking the LLM to be robust to it.

### 5.5 Branches as first-class audit objects

Fork and merge moments receive dedicated scrutiny in the auditor
prompt (not a rule — a framing lens the auditor applies). Two
sub-moments:

**Fork (branch creation):**
- Did the agent consider the right alternatives at fork time?
- Was the chosen branch supported by evidence known *at that
  moment* (not retroactively justified by what came later)?
- Was a discarded branch dismissed prematurely without evidence
  that would justify ruling it out?

**Merge (branch convergence):**
- Does each contributing branch carry sufficient evidence?
- Is the merge premature — are some hypotheses still open?
- Is there a "ghost merge" — did the synthesis quietly skip a
  branch whose evidence was never collected?

The advisory hints module (§7.5) surfaces `open_branches(graph)`
and `multi_branch_syntheses(graph)` as scaffolding — the auditor
sees the candidate spots and decides whether fork/merge quality
is a concern. "Sufficient evidence" is irreducibly semantic; only
the LLM can weigh it.

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

V2 shape — breaking change from V1:

```python
{
    "type": "object",
    "properties": {
        "verdict": {
            "type": "object",
            "properties": {
                "surface_reminder": {
                    "type": "boolean",
                    # True → reminder_text is injected before next turn
                },
                "reminder_text": {
                    "type": "string",
                    # Non-empty when surface_reminder is True;
                    # empty string allowed when False.
                },
                "continuation_notes": {
                    "type": "array",
                    "items": {"type": "string"},
                    # Auditor's free-text memory across firings —
                    # what it asked itself to recheck next firing.
                    # Replaces V1's downstream_reaction.
                },
                "matched_event_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    # Empty array allowed when surface_reminder=false.
                },
                "cited_cards": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "surface_reminder",
                "reminder_text",
                "continuation_notes",
                "matched_event_ids",
                "cited_cards",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["verdict"],
    "additionalProperties": False,
}
```

**Rationale for dropping `DriftType`:** the V1 enum
`{task_drift, evidence_ignored, premature_conclusion, stuck_loop}`
was exactly the "preset enum for subjective dimensions" the project
forbids (saved feedback memory: `feedback_no_preset_subjective_labels.md`).
Reasonable interpretations of "drift type" differ; an LLM-decided
free-text judgment in `reminder_text` + `continuation_notes` is
both more accurate and more flexible. The V1 `if/then` JSON-Schema
clause that enforced `drift=true ⇒ type required` is removed along
with `drift` and `type` fields.

### 6.3 `get_turn(idx)` — auditor drill-down tool

Read access to the raw conversation turn at index `idx`. Used when
an event's `source_turns` reference needs verification against the
actual trajectory text.

```python
{
    "type": "object",
    "properties": {
        "idx": {"type": "integer"},
    },
    "required": ["idx"],
    "additionalProperties": False,
}
# Returns: serialized turn dict (role, content, thinking blocks).
# Out-of-range idx returns a structured tool-result error rather
# than crashing the auditor child.
```

This tool is in-scope for the V2 implementation (no longer
deferred). The adapter serializes turns on demand rather than
bulk-loading the full trajectory into the auditor context.

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
│   │   ├── output.py                 RawExtractorOutput coercion
│   │   └── validator.py              Phase 1 graph validator (pure Python)
│   ├── auditor/
│   │   ├── prompt.py                 AUDITOR_SYSTEM_PROMPT (graph-only, LLM-led)
│   │   ├── submit_tool.py            submit_verdict terminal tool (V2 schema)
│   │   ├── extensions.py             compose_auditor_extensions()
│   │   ├── output.py                 RawVerdictOutput coercion
│   │   └── get_turn_tool.py          get_turn(idx) drill-down tool
│   ├── hints.py                      advisory hint signals (pure functions on graph)
│   ├── _compose.py                   shared extension-list composer
│   ├── _enum_schema.py               EVENT_KIND_VALUES (DriftType removed)
│   ├── entry_types.py                shared entry-type string constants
│   └── __init__.py
├── adapters/
│   └── agentm.py                     orchestrates Phase 1 + Phase 2
└── ...
```

### 7.2 Schema parity

The JSON-Schema enum for `EventKind` is derived from the enum
class — single source of truth:

```python
EVENT_KIND_VALUES = [k.value for k in EventKind]
```

`DriftType` and `DRIFT_TYPE_VALUES` are removed in V2. The
extractor's `submit_events` schema uses `EVENT_KIND_VALUES`; the
auditor's `submit_verdict` schema no longer references a
drift-type enum.

### 7.3 Adapter

```python
async def _on_turn_end(event):
    nonlocal turn_count, pending, last_continuation_notes
    turn_count += 1

    branch = api.session.get_branch()
    new_events = await _run_extractor(api, branch, extractor_extensions)
    # _run_extractor returns None on validation failure or other
    # extractor-side failure; entry-type record is already written.

    if new_events is not None and (turn_count % k) == 0:
        verdict = await _run_auditor(
            api,
            graph=branch_events(branch) + new_events,
            recent_verdicts=recent_verdicts(branch),
            continuation_notes=last_continuation_notes,
            hints=audit.hints.compute(graph=...),
            extensions=auditor_extensions,
        )
        if verdict is not None:
            last_continuation_notes = verdict.continuation_notes
            if verdict.surface_reminder and verdict.reminder_text:
                pending = Reminder(text=verdict.reminder_text)
```

`_run_extractor` runs the LLM child, calls the graph validator,
writes the appropriate entry type, and returns the new events on
success or `None` on any failure. `_run_auditor` builds the hint
block via `audit.hints.compute(...)` and prepends it to the
auditor prompt context; it also forwards the prior firing's
`continuation_notes` so the auditor can pick up its own
recheck list.

### 7.4 Trajectory serialization

`_serialize_new_turns_for_extractor` slices
`messages[cursor.last_turn_index + 1 :]` and keeps thinking blocks
plus structured tool-result content. The auditor receives no
bulk trajectory — it sees the graph in `Event.to_dict()` form,
recent verdicts, the hint block, and can pull individual turns
on demand via `get_turn(idx)`.

### 7.5 Advisory hints module — `audit/hints.py`

Pure functions on the event graph. Output is rendered as a compact
block and prepended to the auditor's context prompt. The auditor
may ignore any signal.

| Function | Signal |
|---|---|
| `repeated_actions(graph)` | action events with identical `(tool_name, hash(canonical(args)))` appearing ≥ N times — points at potential stuck loops |
| `convergence_ratio(graph)` | fraction of unresolved hypotheses / open decisions vs total events; trend is a health signal |
| `reachability_gaps(graph)` | events not connected back to `task`, or conclusions without an evidence chain |
| `open_branches(graph)` | `decision` events whose discarded alternatives have no closing evidence downstream |
| `multi_branch_syntheses(graph)` | conclusion / synthesis events whose `refs` reach multiple independent root paths |

No new database, no async I/O — all inputs are the in-memory
`list[Event]` the adapter already has. The auditor prompt template
includes a `{hints}` placeholder filled by the adapter before the
child session starts. The hints block must be phrased as
"consider …" rather than "concern: …" to avoid anchoring the
auditor to a binary triage stance.

---

## 8. Public contract

- `schema.py`: `Event`, `Verdict`, `Reminder`, `EventKind` —
  stable public surface.
  **Breaking changes (V2):**
  - `DriftType` is removed.
  - `Verdict` shape changes to V2 (§6.2): replaces
    `{drift, type, reminder, matched_event_ids, cited_cards,
    downstream_reaction}` with
    `{surface_reminder, reminder_text, continuation_notes,
    matched_event_ids, cited_cards}`.
  - `Reminder` no longer carries a `type: DriftType`.
  rca-autorl backwards-compatibility is explicitly **not** a
  constraint for this change (confirmed with user 2026-05-10).
  Bump `version` in `pyproject.toml` accordingly.
- `cards.py` and the AFC card YAMLs ship as package data.
- `cards_tools` atom — both extractor and auditor can load it.
- `inherit_provider` builtin — both phase children rely on it.
- `BeforeAgentStartEvent.system` injection for reminders.

---

## 9. Deferred (P2)

These are deliberately deferred. Re-evaluate after PR 1–4 ship and
the V2 pipeline runs in production for at least a week.

1. **ReAct cycle-aligned extractor firing.** Today the extractor
   fires every `TurnEndEvent`. Firing on ReAct cycle boundaries
   (cycle close = next assistant text-only turn after a
   `tool_result`, or session end) would reduce noise and align
   each firing with one semantic unit of agent work. Larger
   refactor — needs a cycle-detector heuristic plus a fallback
   timer for pathological cases.
2. **Two-pass extraction.** Draft → self-review → final inside
   the extractor child. Approximately doubles extractor cost;
   measure value first.
3. **`get_event(id)` tool for extractor.** Pull older events with
   full `source_turns` when refs need verification inside Phase 1
   (the extractor counterpart of `get_turn`).
4. **Graph-pattern triggers.** Cheap heuristics that fire the
   auditor mid-`k`-window on graph anomalies. Needs the schema
   and hint module to stabilize first.
5. **Cross-session graph reuse.** Each session's graph is
   isolated; joining across sessions for the same task / agent
   profile is future work.
6. **Long-running extractor session.** Today's stateless per-turn
   spawn could become a streaming daemon once AgentM gains
   long-lived child + stream primitives.
7. **Ref edge typing.** A small classifier that materializes
   typed edges from free-text `summary` relations.

---

## 10. References

- [observability.md](observability.md) — training-data join works
  on the same `<trace>.jsonl`.
- [pluggable-architecture.md](pluggable-architecture.md) — the
  `inherit_provider` builtin path used by both child sessions.
- [extension-as-scenario.md](extension-as-scenario.md) — §11
  single-file extension contract that the audit child sessions
  follow.
- [GitHub issue #134 (Lincyaw/AgentM)](https://github.com/Lincyaw/AgentM/issues/134) —
  the design redirection this document reflects (2026-05-10).
