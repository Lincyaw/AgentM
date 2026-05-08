# Cognitive Audit for llmharness — agent-agnostic supervisory advisor

**Status:** active design (V0 not yet implemented)
**Owners:** llmharness scenario
**Last reviewed:** 2026-05-08
**Predecessor research:** `scenarios/llmharness/references/papers/cards/` (42 AFC cards),
`cards/DETECTION-FEASIBILITY.md` (T1–T4 tier classification),
`cards/GRAPH-FSM-MAPPING.md` (GoS state-machine analysis)

---

## 1. Motivation

`scenarios/llmharness/` already ships a P0 supervisory loop: PostToolUse / Stop
hooks fold transcript turns into an event log; `summarizer.py` extracts a
small set of `Event` records via regex; `detector.py` runs four hand-coded
rules (`stuck_loop`, `premature_conclusion`, `evidence_ignored`,
`task_drift`); a single-line NL reminder gets injected on UserPromptSubmit.

Two pressures push beyond P0:

1. **Coverage ceiling for rules.** `cards/DETECTION-FEASIBILITY.md`
   classifies the surveyed failure patterns into four tiers. T1 (pure
   rule) and T2 (structural) are reachable by rules. T3 (hybrid) and T4
   (pure semantic) — ~20 cards in total — are not. Patterns like *task
   specification deviation*, *goal drift*, *causal misattribution*,
   *over-simplified memory*, *unfaithful CoT*, *incorrect verification*
   require reading what the agent *meant*, not just what it emitted.

2. **Long-term distillation target.** The plugin description points at
   "commercial model first, distilled 8B later". A distilled small
   model needs training data of shape `(input trajectory, audit
   reasoning, verdict, downstream reaction)`. That data must be a
   side effect of normal V0 operation — no second pipeline.

This document specifies the semantic-judgment layer — the **cognitive
audit** — including its conceptual model, agent-agnostic interface,
V0 realization, and data flow into a future training pipeline.

---

## 2. Role and authority

The audit is a **soft NL advisor**, not a controller. The following
constraints are load-bearing — every later section assumes them.

| Dimension | Constraint |
|---|---|
| Range | Only T3+T4 (semantic-judgment) failures. T1+T2 are owned by a separate rule layer (currently P0's `detector.py`). |
| Output | Natural-language reminder injected into the main agent's next-prompt context. Never a control signal. |
| Authority | Main agent retains full decision authority. It MAY act on the reminder, MAY ignore it. The audit MAY NOT mutate trajectory, plan, tool list, or model. |
| Tone | Low coercion. Phrasing is observational ("you might consider"); never imperative. |
| Default | Silent. A non-anomalous trajectory produces zero reminders. |
| Suppression | Same-type reminders are suppressed across firings unless the audit judges that the previously addressed anomaly has *re-drifted* (LLM judgment, not a state machine). |
| Cards as RAG | The 42 AFC cards are a *reference vocabulary*, retrieved on demand by the audit through tool calls. They are not enumerated card-by-card and run as a checklist. |

The single most consequential implication: the audit's failure mode is
**reminder fatigue**, not **wrong decision**. A noisy advisor that
emits useless reminders will be ignored entirely. The design therefore
optimizes for **signal-to-noise over coverage** — better to miss real
drift than to break trust.

---

## 3. Three-axis audit model

The audit views a trajectory as a **thought graph**: nodes are agent
cognitive moves (plan, observe, tool_call, hypothesis, conclusion,
reflection); edges are logical relationships between moves. Defects show
up as breaks. Three independent axes detect three different kinds:

### 3.1 Axis 1 — Backward continuity (local-edge check)

Every node in a healthy graph has a *reason to exist*: it responds to a
failure, elaborates a plan, consults evidence, corrects a prior step.
A node whose connection to its ancestors cannot be made coherent —
given the trajectory context — is an orphan, signaling a continuity
break.

**Operationalization:** pure LLM judgment over each new node + its
ancestors. No pre-defined edge-type vocabulary; the LLM judges
coherence directly. (Why no vocabulary: §3.4.)

**Cards covered:** AFC-0019 / 0022 / 0028 / 0029 / 0030 / 0012 / 0013 /
0026 / 0037 (~10 cards).

### 3.2 Axis 2 — Forward fulfillment (local-window check)

The agent declares *intents* throughout a trajectory: the original
task, mid-trajectory subgoals, hypotheses to verify. Each intent
should be addressed by some downstream action chain within a finite
forward window. A stranded intent — declared then dropped — is an
alert candidate.

**Operationalization:** LLM judgment per declared-intent node. Where
to check is structurally determined (the forward window since
declaration); whether fulfillment occurred is the LLM call. This
collapses the audit budget from O(turns × cards) to O(intent_nodes).

**Cards covered:** AFC-0016 / 0018 / 0023 / 0025 / 0009 + others
(~6 cards).

### 3.3 Axis 3 — Content correctness (hyperedge / hypernode check)

Even with no broken edges and no stranded intents, a trajectory may
still be wrong. The check operates over **groups of nodes** that are
not necessarily directly connected — what the user called
"超点 / 超边". Examples:

- **Internal consistency**: do reflections at distant points contradict
  each other?
- **Goal alignment**: does the action set as a whole still serve the
  original task?
- **Reasonability**: does the current move make sense in the broad
  context, even if local edges look fine?
- **Hallucination**: are there fabricated evidence claims, false
  memories, or post-observation rewrites?

**Operationalization:** LLM judgment over **trajectory-internal**
material only (original task + events + raw turns when needed). No
external evidence retrieval, no fact-checking against the world — the
audit cares about coherence within what's already been said.

**Cards covered:** AFC-0001 / 0007 / 0024 / 0036 / 0003 / 0020 / 0002 /
0008 / 0039 (~6+ cards).

### 3.4 Why this shape — alternatives rejected

| Considered | Why rejected |
|---|---|
| GoS state-machine only (arXiv 2603.21250) | The belief-state graph is "what the agent believes and why", missing patterns that live in "what the agent did and why". `cards/GRAPH-FSM-MAPPING.md`: 6 of 20 T3+T4 cards resist FSM reduction. |
| Per-step rationale audit only | No forward window for stranded intents; no graph backbone for orphan detection. |
| Hard-enumerated intent lifecycle states (`declared → pending → fulfilled`) | Brittle: non-binary fulfillment ("addressed but sloppy") doesn't fit any state. User: `人手举的状态不一定可靠`. Replaced by free-text LLM annotations per firing. |
| Preset edge-type vocabulary (10 labels) for axis 1 | Real edges are not enumerable cleanly: many fit none of the labels, many fit multiple. Labels become scaffolding the LLM has to wedge into rather than express its judgment. Replaced by raw LLM coherence judgment. |
| External evidence verification for axis 3 | Mistakenly framed earlier as "fact-checking the world". The user clarified axis 3 is internal: consistency / goal alignment / hallucination — all material is already in the trajectory. No retrieval infrastructure needed. |
| Diagnostic agent as decision-maker | A wrong audit issuing a control signal pulls the main agent off-task with no defense. Advisor + ignore-able-by-design is the working mitigation for "audit is sometimes wrong". |

---

## 4. Architecture — agent-agnostic adapter pattern

The audit is structured so the main agent can be **any** agent that
emits a stream of turns and accepts an injected NL reminder. V0
realizes both sides as AgentM, but layer boundaries are designed so
swapping in another main-agent type requires only a new adapter.

### 4.1 Layers

```
┌──────────────────────────────────────────────────────────────┐
│                     Main Agent (any kind)                    │
│            emits turns ─────────► accepts reminders          │
└────────────┬─────────────────────────────────────▲───────────┘
             │                                     │
             ▼                                     │
┌──────────────────────────────────────────────────────────────┐
│         Adapter layer  (one impl per main-agent kind)        │
│   TrajectoryAdapter   ReminderInjector   AuditTrigger        │
└────────────┬─────────────────────────────────────▲───────────┘
             ▼                                     │
┌──────────────────────────────────────────────────────────────┐
│          Core layer  (agent-agnostic)                        │
│   schema.py  Turn / Event / Verdict / Reminder               │
│   store      Persists turns / events / verdicts / alerts     │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│         Diagnostic engine (AgentM scenario as child session) │
│   Three-axis audit + card retrieval tools                    │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Adapter interfaces

```python
class TrajectoryAdapter(Protocol):
    def session_id(self) -> str: ...
    def pull_new_turns(self) -> list[Turn]: ...

class ReminderInjector(Protocol):
    def inject(self, reminder: Reminder) -> None: ...

class AuditTrigger(Protocol):
    def should_audit(self) -> bool: ...
```

`TrajectoryAdapter` and `ReminderInjector` are typically the same
adapter object. `AuditTrigger` is externally injected so each adapter
can own its trigger semantics: hook-driven for Claude Code,
bus-event-driven for AgentM.

### 4.3 V0 scope: AgentM-on-AgentM only

| Adapter | V0 status |
|---|---|
| **AgentM** | New. Subscribes to `TurnEndEvent`; spawns child session per turn end; injects reminders via `BeforeAgentStartEvent.system` mutation. |
| **Claude Code** | **Stays on existing P0 rule-based detector.** Migration to the three-axis audit is V1+. Rationale: AgentM-on-AgentM is the cleanest path to validate that the three-axis audit actually works (observability auto-captures, child session isolation is native, no cross-process glue), so V0 invests there. |

### 4.4 Cards as tools (not as static prompt)

The 42 AFC cards live as YAML at `scenarios/llmharness/references/papers/cards/`.
The diagnostic child session **does not** receive them as prompt
content. Instead it gets two tools:

```python
def cards_list() -> list[CardSummary]:
    """Return every card's (id, name, axis_hint, one_line_mechanism).
    No parameters. Total ~2000 tokens for ~42 cards. Audit calls at
    most once per firing."""

def cards_get(card_id: str) -> CardFull:
    """Fetch one card's full YAML content (mechanism / activation /
    observable / downstream_effects / evidence). Audit calls when a
    summary looks promising and it wants to verify or cite."""
```

`axis_hint` is a curated annotation living in `cards.py` (a Python
dict mapping `card_id -> axis ∈ {1, 2, 3, "?"}`), **not** a YAML
schema field. This keeps the rca-autorl public contract intact and
lets the three-axis partition evolve cheaply.

Why tools instead of prompt-prepended text:
- Audit prompt stays small.
- Agent's *retrieval choices* become observable — high-quality
  training signal for the future 8B model.
- Card YAMLs remain single-source-of-truth; no preprocessing step.
- Adding a new card does not change the prompt — only the YAML and a
  one-line `_AXIS_HINT` entry in `cards.py` are needed.

### 4.5 V0 realization sketch (AgentM adapter)

Surface verified against AgentM 2026-05-08 (`scenarios/llmharness/src/llmharness/adapters/agentm.py`).
The diagnostic agent's prompt + extension list ships as a Python
subpackage (`llmharness.audit`); there is **no `scenarios/harness_monitor/`
directory**. The adapter composes the child session's extensions list in
Python and hands it to `AgentSessionConfig` directly.

```python
# llmharness/adapters/agentm.py — sketch
from .. import audit
from ..audit import RawAuditOutput

def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    store = HarnessStore(api.cwd / ".harness")
    child_extensions = audit.compose_extensions(
        prompt_override     = config.get("prompt_override"),
        cards_tools_config  = config.get("cards_tools_config", {}),
        observability_config= config.get("observability_config", {}),
    )
    reminder_prefix = config.get("reminder_prefix", "\n\n[harness] ")

    @api.on(TurnEndEvent.CHANNEL)
    async def on_turn_end(event: TurnEndEvent) -> None:
        # Two-stage audit input: full conversation + running event log
        # + recent verdicts. Stage A extracts events; stage B does
        # three-axis analysis. Child must see the user prompt, not
        # just event.message.
        full_messages = api.session.get_messages()  # list[AgentMessage]
        prior_events = store.read_events(api.session_id)[-K:]
        payload = json.dumps({
            "trajectory":     _messages_to_dicts(full_messages),
            "prior_events":   [e.to_dict() for e in prior_events],
            "recent_alerts":  [v.to_dict() for v in store.recent_verdicts(
                                 api.session_id, n=N)],
        })

        child = await api.spawn_child_session(AgentSessionConfig(
            extensions=child_extensions,
            provider=(__name__, {"_bridge_provider": api.provider}),
        ))
        messages = await child.prompt(payload)
        # Parse into typed RawAuditOutput.
        events, verdict = _audit_output_from_messages(messages, prior_events)
        store.append_events(api.session_id, events)
        store.append_verdict(api.session_id, verdict)
        if verdict.drift and verdict.reminder and verdict.type is not None:
            store.write_reminder(_reminder_from(verdict, api.session_id))

    @api.on(BeforeAgentStartEvent.CHANNEL)
    def inject_pending(event: BeforeAgentStartEvent) -> None:
        pending = store.pop_reminder(api.session_id)
        if pending is not None:
            event.system = (event.system or "") + reminder_prefix + pending.text
```

Surface facts confirmed against AgentM source:

- `AgentSessionConfig.extensions: list[tuple[str, dict[str, Any]]]` is
  a first-class field. Resolution precedence in `AgentSession.create`:
  `no_extensions → []`, then `explicit extensions=`, then
  `scenario=name → load_scenario(...)`. So passing `extensions=...`
  bypasses the YAML-lookup path entirely. **V0 takes this branch.**
- `api.session.get_messages()` returns `list[AgentMessage]` — the
  full discriminated union of `UserMessage` / `AssistantMessage` /
  `ToolResultMessage`. Use `isinstance` dispatch in the serializer.
  The latest `event.message` is already in this list by the time
  `TurnEndEvent` fires; do NOT also append `event.message` separately.
- `api.spawn_child_session(AgentSessionConfig)` returns an
  `AgentSession`. Drive via `child.prompt(json_payload)` which
  returns `list[AgentMessage]`; parse the **trailing JSON** from the
  last assistant message via `audit.extract_json` + `RawAuditOutput`.
- The bridge-provider trick — `provider=(__name__, {"_bridge_provider":
  api.provider})` — has the child load THIS adapter as its provider
  extension; the `install()` top branch detects `_bridge_provider`,
  re-publishes the parent's `ProviderConfig`, and returns without
  registering handlers. Avoids re-authenticating the LLM gateway AND
  prevents recursive audit spawn. The structural defense also relies
  on `audit.compose_extensions` not inserting this adapter into the
  child's extensions list — by construction, it never does.
- `BeforeAgentStartEvent` is a `@dataclass(slots=True)` (mutable);
  `event.system: str | None` is the documented mutation surface.
- Store root defaults to `.harness/`. Override via `root` config key.

Three V0 design choices baked into this sketch:

- **Trigger = every `TurnEndEvent`.** No throttling. The audit itself
  is responsible for staying silent on healthy trajectories.
- **Isolation = `spawn_child_session`.** Diagnostic agent runs in its
  own AgentM session. Three reasons: clean third-person perspective;
  AgentM observability automatically captures the child's full
  reasoning into its own trace file (§7); authority isolation.
- **Composition = Python, not YAML.** The diagnostic agent's prompt
  body and extension list live in `llmharness.audit` as Python.
  Replacing audit logic is a Python edit, not a YAML edit — and the
  output-schema contract (`AUDIT_SYSTEM_PROMPT` step 10 ↔
  `RawAuditOutput.from_dict`) becomes co-located, not split across
  two files in two directories.

---

## 5. Trigger and audit flow

### 5.1 Trigger

Every `TurnEndEvent` on the AgentM bus. No threshold, no per-N-batch.

### 5.2 Audit flow inside the child session

```
inputs:  new_turns, history_events_tail, recent_alerts (with downstream_reaction)
         + tools: cards_list, cards_get

1. self_monitor:
     read recent_alerts and their downstream_reaction;
     judge "are my recent reminders being heeded?"
     if NO → emit Verdict(has_drift=False, reason="self_silenced: ...")
              and exit. No three-axis check.

2. fold new_turns into the running thought graph (events list).

3. axis 1 (backward continuity):
     for each new node, judge whether it has a coherent ancestor
     relationship in context. Flag orphans.

4. axis 2 (forward fulfillment):
     for each open intent node, judge whether the forward window
     contains an action chain that addresses it.

5. axis 3 (content correctness):
     for verification / conclusion / summary nodes and for cross-cutting
     groups (all-evidence-vs-task, parallel reflections), judge
     consistency / goal alignment / reasonability / hallucination.

6. if anomalies were found:
     consult cards_list / cards_get if helpful;
     synthesize one best Verdict with NL reason
     (and optional cited_cards list).

7. silence gates (route B): emit only if able to articulate a
     concrete reason. Card citation is optional but encouraged.

8. same-type suppression:
     8a. structural pre-filter: if the current finding overlaps the
         most recent alert by axis or cited_cards, hand off to 8b.
     8b. LLM judgment over recent_alerts: is this a re-drift after
         the previous alert was addressed, or a duplicate of an
         unresolved one? Suppress duplicates; emit on re-drift.

9. emit final Verdict.
```

### 5.3 Reminder format

Half-structured with a fixed prefix, free-text body:

```
[harness] {audit's natural-language observation + reflection prompt}
```

The prefix tells the main agent the source is the supervisory layer;
the body is whatever the audit's LLM produced. Tone convention is
observational, not prescriptive.

### 5.4 Cost shape

One child session per main-agent turn. For a 50-turn task: 50
diagnostic LLM calls (plus internal `cards_get` calls when the audit
drills in). For research scale, acceptable. Production-scale
optimization (a structural pre-filter that skips obviously-silent
turn-ends) is V1.

---

## 6. Schema

### 6.1 Existing schema is already a thought graph

```python
class EventKind(str, Enum):
    TASK = "task"
    HYPOTHESIS = "hypothesis"
    EVIDENCE = "evidence"
    DECISION = "decision"
    ACTION = "action"
    REFLECTION = "reflection"
    CONCLUSION = "conclusion"

@dataclass(frozen=True)
class Event:
    id: int
    kind: EventKind
    summary: str
    refs: list[int]            # incoming edges (untyped — stays untyped)
    source_turns: list[int]
```

`refs` stays a plain `list[int]`. No edge-type vocabulary. Axis 1
operates by free-form LLM judgment; structuring edge types into the
schema would (a) violate the principle "free text > preset enum for
subjective dimensions", (b) require breaking the rca-autorl public
contract on `schema.py`.

### 6.2 V0 additions

Two opt-in fields on `Verdict`:

| Field | Type | Purpose |
|---|---|---|
| `cited_cards` | `list[str]` (default `[]`) | Which AFC IDs the audit chose to cite. Optional — not all reminders cite. Drives the future training-data pipeline's card-retrieval signal. |
| `downstream_reaction` | `str \| None` (default `None`) | Free-text note populated by the *next* audit firing: "did the main agent address the prior reminder?" Drives both the self-monitor mechanism (§5.2 step 1) and the training-data tuple (§8). |

Both fields are nullable / default-empty so no existing P0 consumer
breaks. Both follow the "free-text not enum" rule (cf. memory:
`feedback_no_preset_subjective_labels`).

### 6.3 Schema stability

`scenarios/llmharness/CLAUDE.md` makes `schema.py` a public contract
for rca-autorl. Both V0 additions are additive and opt-in; no version
bump required. If future fields turn out to need a non-additive
change, version bump per the same convention.

---

## 7. Persistence and observability

V0 writes **no new persistence code**. Two pre-existing channels do
all the work:

1. **`store.py`** — file-based JSONL (P0, already shipping). Owns:
   raw turns (`inbox/`), extracted events (`events/`), cursor state,
   pending reminders, and (V0 addition) verdicts with their
   `cited_cards` / `downstream_reaction` fields. Source of truth.
   Agent-portable — works for any adapter.

2. **AgentM observability** — auto-captures the diagnostic child
   session's full trajectory (its prompt, every `cards_get` call,
   the Verdict-producing completion) into
   `.agentm/observability/<child_id>.jsonl`. Zero adapter code.

The two share no record-id today and don't need to. When a future
training-data export script runs (V1), it joins them by main session
id + verdict timestamp. That script is the only new code; it does
not exist in V0.

---

## 8. Training data (V1 export)

The architecture produces training data as a side effect. Each audit
firing yields a tuple:

```
(graph_snapshot, alert_emitted, downstream_reaction)
```

- `graph_snapshot` — events at audit time + the child session's
  prompt + every `cards_get` call (recoverable by joining
  store.py with child observability).
- `alert_emitted` — the Verdict in store.py (`text`, `cited_cards`)
  plus the diagnostic reasoning trace in observability.
- `downstream_reaction` — populated by the next audit firing's
  self-monitor judgment (§5.2 step 1), persisted on the Verdict.

This is exactly the shape needed to distill an 8B-class diagnostic
model. V0 emits it; the export and training pipeline is V1.

---

## 9. Companion artifacts

Living artifacts under `scenarios/llmharness/references/papers/`:

| Path | Role |
|---|---|
| `cards/README.md` | The 42 AFC cards index, vocabulary fork from chaoschain. |
| `cards/<class>/<id>.yaml` | Per-card schema-formal description. |
| `cards/DETECTION-FEASIBILITY.md` | Tier classification driving §1's "T3+T4 = 20 cards" claim. |
| `cards/GRAPH-FSM-MAPPING.md` | Maps T3+T4 onto a GoS-style state machine; §3.4 cites the 6 cards that resist reduction. |

Living artifacts under `scenarios/llmharness/src/llmharness/`:

| Path | Role in V0 |
|---|---|
| `schema.py` | Stable contract; gains `Verdict.cited_cards`, `Verdict.downstream_reaction` (both opt-in). |
| `summarizer.py` | P0 rule-based regex; subsumed by the diagnostic child session in V0 (the child session itself does turns→events). May be retired. |
| `detector.py` | P0 four hand-coded rules; remains for the Claude Code adapter (§4.3). |
| `store.py` | File-based persistence, source of truth. No schema change for V0 beyond §6.2. |
| `cards.py` (new) | Loads `references/papers/cards/<class>/*.yaml`; exposes `cards_list` / `cards_get` as AgentM tools; carries the curated `axis_hint` mapping. |
| `adapters/agentm.py` (new) | The V0 adapter described in §4.5. |
| `adapters/claude_code.py` (move from `claude_code.py`) | P0 adapter; unchanged behavior in V0. |
| `audit/` (new subpackage) | Diagnostic engine packaged as Python: `prompt.py` (`AUDIT_SYSTEM_PROMPT`), `extensions.py` (`compose_extensions`), `output.py` (`RawAuditOutput`), `json_extract.py` (trailing-JSON parser). Replaces the deleted `scenarios/harness_monitor/`. |

---

## 10. Open questions

- **Pre-filter to amortize child-session cost.** V0 fires per
  `TurnEndEvent`; many turns are obvious continuations where no new
  drift signal is possible. A cheap structural pre-filter could
  short-circuit silent turns without invoking the LLM. Deferred to
  V1, after V0 measures actual cost.
- **Cross-session memory for the diagnostic agent.** Today each
  main session is independent. A diagnostic that remembered "I have
  warned this agent twice about over-simplification, it never
  listened" could be more useful, but enabling cross-session memory
  needs separate design work. Out of scope for V0.
- **Reminder-usefulness evaluation.** V0 emits training data
  including `downstream_reaction`, which is the raw input for an
  evaluation harness. The evaluation design itself belongs in a
  separate document.

---

## 11. Out of scope

- Hard control over the main agent (no compact / abort / interrupt;
  always advisor-shaped).
- Detection of T1/T2 (rule-solvable) failures (owned by P0 rule layer).
- Replacing the AFC card corpus with the thought graph (cards remain
  a reference vocabulary; the graph is a runtime structure).
- Production-scale cost optimization.
- Self-modification of the diagnostic agent's own atoms during a run.
- External-world fact verification for axis 3 (axis 3 is
  trajectory-internal only).

---

## 12. Cross-references

- `designs/extension-as-scenario.md` — the AgentM atom + scenario
  model the AgentM adapter rides on.
- `designs/observability.md` — the AgentM observability stream that
  captures diagnostic child-session traces.
- `designs/sub-agent-lifecycle.md` — child-session lifecycle.
- `designs/self-modifiable-architecture.md` — establishes the
  `api.spawn_child_session` boundary.
- `scenarios/llmharness/CLAUDE.md` — schema-stability constraint
  for rca-autorl.
- `scenarios/llmharness/references/papers/cards/` — the AFC corpus.
- Memory: `feedback_no_preset_subjective_labels.md` — general
  principle that shaped the schema decisions in §6.
