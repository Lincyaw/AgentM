# Context Index Proposal

Status: in progress; runtime defaults now use a derived `ContextIndex`,
`auditor_context_mode="index"`, `auditor_prompt="minimal_index"`, and an
index-builder extractor prompt.

This proposal changes the role of the llmharness graph from a reasoning object
into an LSP-style context index. The index should help the auditor locate
evidence, claims, candidates, and contract failures in a trajectory prefix. It
should not decide whether the agent is wrong, whether to intervene, or what the
correct answer is.

## Motivation

The current extractor prompt asks the child agent to maintain a "logic-flow
graph" of the main agent's reasoning. That makes the extractor responsible for
too much:

- compressing the trajectory,
- inferring reasoning dependencies,
- preserving evidence witnesses,
- implying causal structure,
- and creating a surface that the auditor treats as judgment context.

RCA reminder experiments showed that many recoveries do not come from a better
global reasoning graph. They come from making one local, visible gap salient:
root-vs-effect confusion, service-vs-link ambiguity, observed evidence not
reconciled, silent demotion of a candidate, over-specific fault naming, or a
malformed final-report graph. These are easier to detect when the trajectory is
indexed by entities, claims, evidence, and lifecycle events.

## Design Principle

The context graph is a context server, not a judge.

```text
Trajectory prefix
  -> context index
  -> intervention policy
  -> NOOP or reminder
  -> fork/evaluate outcome
```

The graph/index layer answers "what happened, where, and which entities are
nearby." The auditor/policy layer decides whether that context contains a
material, actionable flaw.

## Non-Goals

- Do not compute the RCA root cause in the index layer.
- Do not treat graph edges as ground-truth causal links.
- Do not generate reminders from the index layer.
- Do not require perfect entity extraction before the policy can run.
- Do not break existing replay records while the migration is in progress.

## Proposed Model

Keep the existing `Event`/`Edge`/`Phase` graph as a compatibility view, but add
a derived `ContextIndex` view for auditor consumption.

### ContextIndex

```python
@dataclass(frozen=True)
class ContextIndex:
    turns: list[TurnRef]
    entities: list[EntityRef]
    observations: list[ObservationRef]
    claims: list[ClaimRef]
    candidates: list[CandidateRef]
    obligations: list[ObligationRef]
    contract_events: list[ContractEventRef]
    links: list[IndexLink]
```

The index is allowed to be sparse. Missing an entity is acceptable; inventing an
entity or a causal fact is not.

### TurnRef

One stable pointer per visible trajectory turn.

Fields:

- `turn_index`
- `role`
- `kind`: `user | assistant | tool_call | tool_result | system | reminder`
- `summary`

### EntityRef

Concrete tokens the auditor may need to look up.

Fields:

- `id`
- `name`
- `type`: free text such as `service`, `endpoint`, `edge`, `metric`,
  `log_pattern`, `fault_kind`, `tool`, `schema_field`
- `turns`
- `aliases`

### ObservationRef

Tool-grounded evidence, not the agent's interpretation.

Fields:

- `id`
- `turns`
- `source`: tool name or environment source
- `summary`
- `entities`
- `values`: optional structured values, for example abnormal/normal counts or
  ratios when the tool output made them explicit
- `polarity`: `supports | weakens | neutral | unknown` relative to a nearby
  claim, if mechanically available; otherwise `unknown`

### ClaimRef

Agent-authored hypotheses, demotions, conclusions, and final-answer claims.

Fields:

- `id`
- `turns`
- `text`
- `kind`: `hypothesis | decision | demotion | conclusion | final_answer`
- `status`: `exploratory | tentative | committed | finalized`
- `entities`

### CandidateRef

Lifecycle for a service, edge, endpoint, or root-cause candidate.

Fields:

- `entity_id`
- `first_seen_turn`
- `last_seen_turn`
- `state`: `mentioned | investigated | retained | demoted | finalized`
- `state_turn`
- `reason_claim_id`: optional claim that explains the state change
- `evidence_ids`

This is the key replacement for "branches" in the old logic-flow graph. The
auditor can ask whether a candidate with strong observations was silently
demoted, but the index does not decide whether the demotion was wrong.

### ObligationRef

Checks the agent created for itself or that a loaded methodology makes required
at this stage.

Fields:

- `id`
- `turns`
- `source`: `agent_plan | methodology | tool_contract`
- `text`
- `entities`
- `state`: `open | satisfied | abandoned | unknown`

### ContractEventRef

Task/tool protocol failures.

Fields:

- `id`
- `turns`
- `tool`
- `status`: `rejected | empty | malformed | validation_failed | repaired`
- `summary`
- `entities`

These events should have high intervention priority, because they are often
actionable without any domain judgment.

### IndexLink

Weak references between index objects.

Fields:

- `src`
- `dst`
- `kind`: `mentions | cites | near | follows | same_entity | derives_from`
- `reason`

These are navigation links, like "find references" in an LSP. They are not
causal proof.

## Auditor Inputs

Replace the current primary prompt section:

```text
## GRAPH
events: [...]
edges: [...]
```

with:

```text
## CONTEXT_INDEX
entities: [...]
observations: [...]
claims: [...]
candidates: [...]
obligations: [...]
contract_events: [...]
links: [...]

## COMPAT_GRAPH
events: [...]
edges: [...]
```

During migration, keep `COMPAT_GRAPH` available but tell the auditor to use the
index as the primary navigation surface.

The auditor prompt should shift from "judge the graph" to:

1. choose candidate intervention operators,
2. gate them by visibility/materiality/actionability/novelty,
3. submit `surface_reminder=false` or a single reminder.

## Intervention Policy Shape

The policy action space should be explicit:

```text
NOOP
remind(reconcile_observed_anomaly, slots)
remind(challenge_assumption, slots)
remind(root_effect_audit, slots)
remind(service_vs_link_audit, slots)
remind(missing_required_comparison, slots)
remind(output_repair, slots)
remind(stale_loop_breaker, slots)
```

Each candidate action is scored by:

- `visible`: relies only on prefix-available information,
- `material`: could change the final answer or output validity,
- `actionable`: names a concrete reasoning repair,
- `novel`: is not a repeat of a previous reminder,
- `risk`: chance of distracting or over-correcting the agent.

The index provides slot-filling evidence. The auditor owns the final decision.

## Migration Plan

### Phase 0: Documentation and replay targets

- Add this proposal.
- Pick a small fixed set of RCA replay records from the 31-case run.
- Define review metrics:
  - false-reminder rate on successful prefixes,
  - recovery uplift on failed prefixes,
  - malformed/leaky reminder rate,
  - operator distribution.

### Phase 1: Derived index view

Implement `llmharness.context_index` as a pure module that derives a
`ContextIndex` from the existing trajectory snapshot plus folded graph.

Status: landed.

Why derived first:

- no new extractor behavior,
- no replay-record breaking change,
- easy offline A/B through `llmharness-replay auditor`.

Initial derivation can be simple:

- entities from `Event.summary`, `Edge.cited_entities`, tool names, and final
  report tool payloads,
- observations from `act` events and tool-result turns,
- claims from `hyp`, `dec`, `concl` events,
- candidates from entity mentions across claims and observations,
- contract events from rejected tool calls and final-report validation errors.

### Phase 2: Auditor context v2

Extend `AuditorContextConfig` with:

- `context_index: dict | None = None`
- `context_mode: "graph" | "index" | "both" = "index"`

Add `build_auditor_index_prompt(...)` and a prompt variant such as
`minimal_index.md`.

Status: landed with default runtime on `"index"`. Use `"both"` or `"graph"` only
for A/B and compatibility debugging.

### Phase 3: Index-oriented extractor prompt

Add a new extractor prompt, for example `index.md`, that asks for indexing
facts rather than maintaining a reasoning-flow graph.

Two implementation options:

1. Reuse existing graph tools and encode index entries as typed event summaries.
   This is fastest and keeps the wire schema unchanged.
2. Add explicit index tools after the derived view proves useful. This is
   cleaner but requires schema, replay, distill, and docs updates.

Status: option 1 landed. The default extractor prompt builds index records
through the existing node/edge tool surface.

### Phase 4: Intervention policy data

Update replay/distill outputs so each surfaced reminder has structured metadata:

- `operator`
- `target_ids`
- `evidence_ids`
- `gate_scores`
- `risk_notes`

This metadata can start inside `continuation_notes` or a new optional verdict
field. Do not make it required until enough replay data exists.

### Phase 5: Train or tune a scorer

Use fork outcomes as contextual-bandit labels:

```text
success: +1.0
partial: +0.3
same: 0.0
worse: -0.5
leaky_or_invalid: -1.0
```

Before model training, a rule/LLM hybrid scorer is enough. The important part is
logging candidate actions that were rejected, not just the action that fired.

## Compatibility Notes

- Existing `AUDIT_GRAPH_OP` entries remain valid.
- Existing replay sidecars remain valid.
- `ContextIndex` can be regenerated from existing sidecars.
- New fields should be optional until a version bump is justified.
- Scenario-specific RCA rules remain in skills/methodology, not in the generic
  auditor prompt.

## Open Questions

1. Should `ContextIndex` be persisted as a session entry, or always derived at
   auditor time?
   - Recommendation: derive at first, persist only after the shape stabilizes.
2. Should entity extraction be deterministic or LLM-assisted?
   - Recommendation: deterministic first for tool names, services, metrics, and
     final-report payloads; allow LLM extraction only inside the existing
     extractor child.
3. Should the auditor see the raw trajectory?
   - Recommendation: yes, but only as a fallback. The index should be the
     primary view; raw turns are witnesses.
4. Should the policy fire at fixed intervals or event triggers?
   - Recommendation: support both. Fixed intervals are simpler; event triggers
     should be added for final-report failure, pre-submit, strong new
     observation, candidate demotion, and repeated probe loops.

## First Implementation Slice

The smallest useful implementation is:

1. Add `context_index.py` with dataclasses and `build_context_index(...)`.
2. Feed the index into `auditor_context` in `"both"` mode.
3. Add `minimal_index.md` that treats `CONTEXT_INDEX` as primary.
4. Run offline auditor replay on a handful of RCA prefixes and compare:
   - old `minimal.md` over graph,
   - new `minimal_index.md` over index+graph.

This slice validates the modeling change without changing live reminder
delivery or extractor behavior.
