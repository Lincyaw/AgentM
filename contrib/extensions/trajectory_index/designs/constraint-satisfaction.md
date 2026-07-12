# Constraint Satisfaction Analysis — Design

Extends the trajectory grounding analysis (SCHEMA.md) with constraint-level
verification. The current system catches entity grounding errors (32% of
TELBench gold); this extension targets the 40% value/reasoning errors it
misses.

## Motivation

Current index tells you: "entity X is ungrounded" or "entity X's value was
contradicted." But it cannot tell you:

- **Constraint omission**: agent never verified constraint C (no evidence chain
  for it)
- **Constraint misinterpretation**: agent had the right value but judged
  "satisfies constraint C" incorrectly (e.g., "1765 is before the 18th century")
- **Constraint relaxation**: agent silently weakened a constraint to fit its
  answer (e.g., dropped "American" from "American actor born in 1960s")

These are reasoning errors that leave no grounding trace — every entity is
grounded, every value matches, but the conclusion is wrong.

## Design: Two New Passes

Following SCHEMA.md's principle: "the LLM makes local judgments; all global
traversal is code."

### Pass 0 — Constraint Extraction

```
Input:  question text only (no trajectory)
Method: 1 LLM call (lightweight, question is ~100 tokens)
Output: [{id, type, description}]
```

Types: temporal, identity, occupation, location, relationship, event,
attribute, quantity, format.

Example:
```json
[
  {"id": "c1", "type": "temporal",   "desc": "born in the 1960s"},
  {"id": "c2", "type": "occupation", "desc": "worked as a technician before acting"},
  {"id": "c3", "type": "event",      "desc": "first film role in 1988"},
  {"id": "c4", "type": "event",      "desc": "major breakthrough/recognition in 2001"}
]
```

Does NOT decompose multi-hop reasoning chains — those emerge from the
trajectory, not the question. Only extracts answer-level constraints.

### Pass 4 — Constraint Satisfaction Adjudication

After DATAFLOW (Pass 3) and compare_values (Pass 3.5), runs as a new
adjudication in `adjudicate.py`.

```
Input:  Constraint list + existing Dependency edges + get_turn tool
Method: for each (constraint, relevant dependency edge), LLM judges
        "does the grounded value at this edge satisfy this constraint?"
Output: satisfy / violate / unclear per (constraint, edge) pair
```

#### Step 4a — Constraint-Edge Mapping (code + LLM)

Link constraints to relevant dependency edges. For each constraint, find
candidate edges by:
1. Code: find entities in the final answer span (last step with references)
2. Code: for each final-answer entity, get its def-use chain from DATAFLOW
3. LLM (local): for each (constraint, def-use edge), ask "is this edge
   about this constraint?" — yes/no/unclear

This is N_constraints × N_edges judgments, but most are trivially "no"
(a birth-year constraint is irrelevant to an edge about film titles). In
practice ~5-15 relevant pairs per trajectory.

#### Step 4b — Satisfaction Check (LLM, local)

For each relevant (constraint, edge) pair from 4a:
- Read the grounded def value (from DATAFLOW's def_value field)
- Ask LLM: "does value V satisfy constraint C?"
- Returns: satisfy / violate / unclear + reason

This is a local judgment — compare one value against one constraint. No
trajectory context needed.

#### Step 4c — Aggregation (code, global)

For each constraint:
- If no edges mapped → **unverified** (constraint omission)
- If all mapped edges satisfy → **satisfied**
- If any edge violates → **violated** at the use_step of that edge
- If agent's final answer treats a violated constraint as satisfied →
  **misinterpreted** at the commitment step

Output: per-constraint status + span localization.

## Data Model Extension

```python
# New: constraint
@dataclass
class Constraint:
    id: str
    type: str
    description: str

# Extended: Dependency gets optional constraint_id
@dataclass
class Dependency:
    ...  # existing fields
    constraint_id: str | None = None
    satisfaction: Literal["satisfy", "violate", "unclear", None] = None

# New: per-trajectory constraint status
@dataclass
class ConstraintStatus:
    constraint_id: str
    status: Literal["satisfied", "violated", "unverified", "misinterpreted"]
    evidence_span: str | None  # span where satisfaction was determined
    error_span: str | None     # span where agent committed to wrong judgment
```

## Interaction with Existing Passes

```
Question ──→ [Pass 0: constraint extraction] ──→ Constraint list
                                                      │
Trajectory ──→ [Pass 1: PARSE]                        │
           ──→ [Pass 2: RESOLVE]                      │
           ──→ [Pass 3: DATAFLOW]                     │
           ──→ [Pass 3.5: compare_values] ──→ Entity table + Dependencies
                                                      │
                                              [Pass 4a: constraint-edge mapping]
                                              [Pass 4b: satisfaction check]
                                              [Pass 4c: aggregation]
                                                      │
                                              Constraint statuses + error spans
                                                      │
                                              [Auditor: verify + localize]
```

Pass 0 runs independently (only needs question text).
Passes 1-3.5 run as before (unchanged).
Pass 4 consumes outputs of both Pass 0 and Passes 1-3.5.

## What This Catches vs What It Doesn't

### Catches
- Constraint omission: no evidence chain for constraint C → unverified
- Constraint misinterpretation: value satisfies C but agent judged wrong
- Constraint relaxation: agent weakened C → violated but treated as satisfied
- Value mismatch: already caught by compare_values (Pass 3.5)

### Doesn't Catch
- Pure reasoning errors: all facts correct, logical inference wrong
- Causal reasoning: post-hoc-ergo-propter-hoc
- Question misreading: agent pursues wrong goal entirely
- Multi-hop intermediate constraint failures: constraints that emerge from
  the trajectory, not the question

### Off-by-one Localization

This extension improves DETECTION (which constraints failed) but does not
directly solve LOCALIZATION (which span to flag). The off-by-one problem
(gold marks span N, we mark N+1) requires a separate span attribution
policy — mapping error type to localization rule. Proposed separately.

## Cost

- Pass 0: 1 lightweight LLM call (question only)
- Pass 4a: ~5-15 local LLM judgments (constraint × relevant edge)
- Pass 4b: ~5-10 local LLM judgments (relevant pairs only)
- Total: ~2 additional LLM calls equivalent per trajectory

## Implementation Notes

- Pass 0: standalone function, can be a simple prompt + JSON parse
- Pass 4: new function in `adjudicate.py`, following `compare_values` pattern
- Constraint list stored on TrajectoryIndex (new field)
- ConstraintStatus stored alongside warnings
- Auditor receives constraint statuses via `list_attention_hints` or a
  new `list_constraint_statuses` tool

## Estimated Impact

Based on TELBench error type distribution:
- constraint_check_omission (114 spans): high detectability
- constraint_semantics_error (320 spans): medium detectability
- constraint_relaxation (113 spans): medium detectability
- Total: ~547/2552 (21%) of gold errors potentially catchable

Optimistic: +5pp F1 (0.53 → 0.58)
Conservative: +3pp F1 (0.53 → 0.56)

## Open Questions

1. How to handle multi-hop questions where constraints emerge from the
   trajectory? Current plan only extracts answer-level constraints from the
   question text.

2. Constraint-edge mapping (4a) requires understanding what each edge is
   "about." Can this be done reliably with local LLM judgments on
   (constraint_text, def_value, use_value) triples?

3. Should constraint statuses be surfaced to the auditor as attention_hints
   (passive) or as a structured tool (active)? Previous experience shows
   the auditor may ignore passive context.

## Prior Art

- DRIFT (NJU-LINK): 4-stage claim-centric pipeline. Their Claim Keeper
  tracks claim lifecycle (exploratory → consequential → finalized). Our
  constraint extraction is analogous but scoped to answer-level constraints,
  not trajectory-derived claims.
- AgentForesight: step-level online auditing. Their prompt iteration (v7-v16)
  showed that "restating vs introducing" distinction is key for multi-agent
  trajectories — orthogonal to constraint satisfaction.

## References

- SCHEMA.md: existing grounding analysis design
- adjudicate.py: compare_values (Pass 3.5) — pattern to follow
- index.py: TrajectoryIndex data model
- TELBench case studies: 0001, 0006, 0009, 0159, 0657 (in session notes)
