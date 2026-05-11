**Status**: HISTORICAL — describes the pre-v2 architecture removed in Phase 2.5 (2026-04-30).
The current architecture lives in [pluggable-architecture.md](../pluggable-architecture.md) and
[extension-as-scenario.md](../extension-as-scenario.md).

---

# Design: Prompt Engineering Patterns

**Status**: DRAFT
**Created**: 2026-03-31

---

## Overview

A catalog of production-proven prompt design patterns that all AgentM prompt templates (`.j2` files) should follow, expressed in AgentM's Jinja2 template system.

---

## Motivation

AgentM's current prompt templates have evolved organically. They lack a unified structural standard. The 12 patterns below systematically improve LLM compliance, reduce hallucination, and enforce behavioral boundaries.

Current templates already use some implicitly (role opening, structured output) but miss several high-impact ones (anti-rationalization checklists, deterrence statements, context isolation declarations, permission boundary markers).

A codified catalog ensures:
1. **Consistency** — All scenarios follow the same structural conventions.
2. **Completeness** — A checklist prevents omitting critical patterns.
3. **Transferability** — New scenarios bootstrap high-quality prompts by composing from the catalog.

---

## Pattern Catalog

### Pattern A: Role Anchor Opening

**When to use**: Every system prompt. The first sentence defines who the agent is.

**Mechanism**: LLMs attend strongly to early tokens. A crisp role statement anchors all subsequent instructions.

**Example**:
```jinja2
You are a {{ role_name }} — {{ one_sentence_responsibility }}.
```

**Guideline**: Name a specific function ("lead investigator", "verification agent"), not a generic capability ("helpful assistant").

**Current coverage**: Present in RCA and TA orchestrators. GP orchestrator is generic ("helpful, capable AI assistant") — acceptable since the scenario is intentionally broad.

---

### Pattern B: `=== CRITICAL ===` Permission Boundary

**When to use**: Absolute constraints that must never be violated — permission boundaries, data-only restrictions, forbidden actions.

**Mechanism**: Triple-equals uppercase marker creates visual separation from normal markdown headings. Establishes an "alert level" hierarchy: normal instructions use `##` headings; inviolable constraints use `=== ===`.

**Example**:
```jinja2
=== CRITICAL: DATA-ONLY MODE — NO CONCLUSIONS ===
You provide raw data and measurements. You do NOT:
- Suggest root causes
- Rank suspects
- Recommend next investigation steps
The orchestrator makes all analytical decisions.
```

**Current coverage**: **Not used** in any AgentM template. RCA workers use prose sentences ("Sub-agents return FACTS and EVIDENCE only") which lack visual and semantic weight.

---

### Pattern C: Numbered Sections

**When to use**: Long prompts (>500 tokens) with 4+ conceptual sections.

**Mechanism**: Numbered headings let the LLM refer back ("see Rule 3"), improving self-consistency.

**Example**:
```jinja2
## 1. Role
## 2. Diagnostic Philosophy
## 3. Mission
## 4. Rules
## 5. Output Format
```

**Current coverage**: Partial. RCA orchestrator uses unnamed XML blocks (`<rules>`, `<workflow>`). Numbering within blocks would improve self-referencing.

---

### Pattern D: Decision Table

**When to use**: When the LLM must choose between discrete actions based on conditions. Replaces prose if-then-else.

**Mechanism**: Tables compress branching logic into a scannable format. LLMs parse tables more reliably than nested prose conditionals.

**Example**:
```jinja2
| Situation | Action | Task Type |
|-----------|--------|-----------|
| No data yet (round 1) | Survey the incident | scout |
| Anomaly found, mechanism unknown | Trace the causal mechanism | deep_analyze |
| Hypothesis formed, needs adversarial test | Try to disprove | verify |
| Coverage gap on a service | Fill measurement gaps | scout (targeted) |
```

**Current coverage**: **Not used**. RCA orchestrator relies on prose workflow descriptions.

---

### Pattern E: Good/Bad Contrast Examples

**When to use**: When desired behavior is subtle and easily confused with a plausible-but-wrong alternative.

**Mechanism**: Anti-pattern examples ("BAD") are more instructive than positive examples alone. LLMs learn what NOT to do more reliably when shown the specific failure mode.

**Example**:
```jinja2
<example type="good">
**Finding**: `ts-order-service` avg latency 450ms vs 52ms baseline (8.7x) [TRACE]
**Why good**: Named service, quantified delta, cited evidence type.
</example>

<example type="bad" label="Anti-pattern">
**Finding**: The order service experienced significant latency degradation
**Why bad**: No backtick service name, no numbers, no evidence tag.
</example>
```

**Current coverage**: Partial. `verify.j2` has good examples but no bad counterparts.

---

### Pattern F: Anti-Rationalization Checklist

**When to use**: When the LLM has a known tendency to skip verification or take shortcuts. **Highest-impact missing pattern.**

**Mechanism**: Pre-empting specific excuses is dramatically more effective than generic instructions. By naming the exact rationalization, you short-circuit self-justification.

**RCA Scout example**:
```jinja2
=== RECOGNIZE YOUR OWN RATIONALIZATIONS ===
You will feel the urge to stop early. These are the exact excuses you reach for:
- "I've checked the main services" — did you check ALL services on the call chain?
- "The latency delta is small so it's healthy" — did you check error rate and call volume?
- "I'm running low on tool budget" — that's not your decision. Use remaining budget wisely.
- "Resource metrics aren't available" — did you actually try the query? Run it.
If you catch yourself composing an explanation instead of querying data, stop. Query the data.
```

**RCA Orchestrator example**:
```jinja2
=== RECOGNIZE YOUR OWN RATIONALIZATIONS ===
- "The scout said this service is healthy" — what dimensions did it actually measure?
- "The evidence mostly supports my hypothesis" — 'mostly' means contradictions exist. Investigate them.
- "I should finalize because I've used many rounds" — correctness has no round limit.
```

**Current coverage**: **Not used** in any template. The RCA orchestrator has prose warnings against confirmation bias but not the named-excuse checklist format.

---

### Pattern G: Deterrence Statement

**When to use**: When output quality is critical. Inform the agent its output will be verified.

**Mechanism**: The statement that output "will be checked" measurably improves LLM diligence, even without actual verification.

**Example**:
```jinja2
<verification_notice>
Your output will be validated by the orchestrator against raw data.
Unsubstantiated claims or missing evidence tags will be rejected and the task re-dispatched.
</verification_notice>
```

**Current coverage**: **Not used**. The RCA orchestrator has implicit deterrence but no explicit statement in worker prompts.

---

### Pattern H: Purpose Statement

**When to use**: When the orchestrator dispatches a task. Include WHY, not just WHAT.

**Mechanism**: Workers that understand purpose self-adjust depth and focus.

**Example**:
```jinja2
When dispatching workers, always include:
1. WHAT: the specific action
2. WHY: the investigation purpose
3. CONTEXT: relevant findings from prior rounds

Workers cannot see your conversation. Every dispatch must be self-contained.
```

**Current coverage**: Partially present. RCA orchestrator has dispatch quality checks but they're buried in the `<workflow>` block. Should be elevated to a standalone section.

---

### Pattern I: Forced Output Format

**When to use**: When output is consumed by code (JSON parsing, regex extraction).

**Mechanism**: Explicit format specification with labeled fields + BANNED list.

**Current coverage**: **STRONG**. Best-implemented pattern across all scenarios. `verify.j2`, `scout.j2`, `deep_analyze.j2` all have proper output blocks with BANNED lists.

**Guideline**: Maintain current quality. Ensure every new worker template includes an `<output>` block with BANNED list.

---

### Pattern J: Context Isolation Declaration

**When to use**: In every orchestrator prompt. Reminds the orchestrator that workers operate in separate context.

**Mechanism**: Without this, orchestrators produce vague dispatches ("investigate the issue we discussed").

**Example**:
```jinja2
<context_isolation>
Workers cannot see your conversation history. Every dispatch must be fully self-contained:
- Name target service(s) explicitly
- Include relevant data points from prior rounds (exact values)
- State the hypothesis being tested
- Specify what success/failure looks like

Brief the worker like a colleague who just walked into the room.
</context_isolation>
```

**Current coverage**: Partially implicit. No orchestrator explicitly states "workers cannot see your conversation."

---

### Pattern K: Three-Layer Permission Declaration

**When to use**: When an agent has restricted permissions.

**Mechanism**: Defense in depth — prompt + tool filtering + middleware.

```
Layer 1 (Prompt):     === CRITICAL: DATA-ONLY MODE ===
Layer 2 (Config):     disallowed_tools: [dispatch_agent, update_hypothesis]
Layer 3 (Middleware):  PermissionMiddleware rejects at runtime
```

**Current coverage**: Weak. Only Layer 2 (tool lists in `scenario.yaml`). No `=== CRITICAL ===` markers. No middleware reminder injection.

---

### Pattern L: Critical System Reminder (Per-Turn Injection)

**When to use**: Constraints that must survive long conversations where system prompt influence decays.

**Mechanism**: Re-inject critical constraints each turn via `DynamicContextMiddleware`.

**Example**:
```jinja2
<system_reminder>
{% for constraint in critical_constraints %}
- {{ constraint }}
{% endfor %}
Round {{ current_round }}/{{ max_rounds }}.
{% if current_round >= max_rounds - 2 %}
=== URGENCY: {{ max_rounds - current_round }} rounds remaining. ===
{% endif %}
</system_reminder>
```

**Current coverage**: Partial. `DynamicContextMiddleware` injects `<current_state>` with round counter and hypothesis state. But does not re-inject behavioral constraints. Infrastructure exists; content needs enrichment.

---

## Template Structure Standard

### Orchestrator System Prompt

```
1. Role Anchor Opening                          [Pattern A — REQUIRED]
2. Context Isolation Declaration                 [Pattern J — REQUIRED]
3. Sub-Agent Return Schema                       [scenario-specific]
4. Domain Philosophy / Principles                [numbered, Pattern C]
5. Decision Tables                               [Pattern D — if dispatch logic exists]
6. Rules                                         [numbered within block]
7. Anti-Rationalization Checklist                 [Pattern F — RECOMMENDED]
8. Workflow                                      [numbered phases]
9. Depth Checks / Finalization Criteria
10. Feature Gate Conditionals                     [Jinja2 {% if %} blocks]
```

### Worker System Prompt

```
1. Role Anchor Opening                          [Pattern A — REQUIRED]
2. Permission Boundary                           [Pattern B — if restricted]
3. Domain Philosophy                             [brief]
4. Mission                                       [numbered steps]
5. Anti-Rationalization Checklist                 [Pattern F — RECOMMENDED]
6. Deterrence Statement                          [Pattern G — RECOMMENDED]
7. Evidence Standards                             [if evidence-producing]
8. Output Format                                 [Pattern I — REQUIRED]
   - Structure specification
   - Good/Bad examples                          [Pattern E]
   - BANNED list
```

### Per-Turn Reminder (via DynamicContextMiddleware)

```
1. Current State Block                           [existing]
2. Critical Constraint Reminders                 [Pattern L — NEW]
3. Urgency Indicator                              [round-based, existing]
```

---

## Gap Analysis

### RCA Scenario

| Pattern | orchestrator | scout | deep_analyze | verify | compress |
|---------|:-:|:-:|:-:|:-:|:-:|
| A. Role Anchor | YES | YES | YES | YES | YES |
| B. `=== CRITICAL ===` | NO | NO | NO | NO | NO |
| C. Numbered Sections | PARTIAL | PARTIAL | PARTIAL | PARTIAL | YES |
| D. Decision Table | NO | N/A | N/A | N/A | N/A |
| E. Good/Bad Contrast | NO | NO | NO | PARTIAL | NO |
| F. Anti-Rationalization | NO | NO | NO | NO | NO |
| G. Deterrence | NO | NO | NO | NO | NO |
| H. Purpose Statement | PARTIAL | N/A | N/A | N/A | N/A |
| I. Forced Output | YES | YES | YES | YES | YES |
| J. Context Isolation | PARTIAL | N/A | N/A | N/A | N/A |
| K. Three-Layer Permission | NO | NO | NO | NO | N/A |
| L. Per-Turn Reminder | PARTIAL | N/A | N/A | N/A | N/A |

### GP and TA Scenarios

GP templates are minimal by design but should still include Context Isolation (J) and Deterrence (G). TA orchestrator is well-structured but missing Anti-Rationalization (F) and Context Isolation (J).

### Highest-Impact Gaps (Priority Order)

1. **F (Anti-Rationalization)** — Missing everywhere. Highest impact on compliance.
2. **B (Critical Boundary)** — Missing everywhere. Workers lack inviolable constraint markers.
3. **J (Context Isolation)** — Missing/implicit in all orchestrators. Causes vague dispatches.
4. **G (Deterrence)** — Missing everywhere. Low-cost, measurable improvement.
5. **E (Good/Bad Contrast)** — Mostly missing. `verify.j2` has good examples but no bad counterparts.
6. **D (Decision Table)** — Missing. Would clarify RCA dispatch logic.

---

## Impact

### Prompt Template Files

| File | Patterns to Add |
|------|----------------|
| `rca_hypothesis/prompts/orchestrator_system.j2` | B, D, F, G, J, L |
| `rca_hypothesis/prompts/task_types/scout.j2` | B, F, G, E |
| `rca_hypothesis/prompts/task_types/deep_analyze.j2` | B, F, G, E |
| `rca_hypothesis/prompts/task_types/verify.j2` | B, F, G, E (bad examples) |
| `general_purpose/prompts/orchestrator_system.j2` | J, F (light) |
| `general_purpose/prompts/task_types/execute.j2` | G, I |
| `trajectory_analysis/prompts/orchestrator_system.j2` | F, G, J |

### Design Documents

- [orchestrator.md](orchestrator.md) — Update DynamicContextMiddleware section for Pattern L
- [sub-agent.md](sub-agent.md) — Reference prompt pattern standards for worker prompts

### SDK Components (Future)

- `DynamicContextMiddleware` — Extend to support `critical_constraints` list injection (Pattern L)
- Scenario config — Optional `anti_rationalizations`, `permission_boundary`, `deterrence_statement` fields

---

## Constraints and Decisions

| Decision | Rationale |
|----------|-----------|
| Patterns are guidelines, not enforced by tooling | Templates are hand-authored Jinja2; automated linting is premature |
| Anti-rationalization checklists are scenario-specific | Generic checklists ("be thorough") are ineffective; only observed failure modes work |
| `=== CRITICAL ===` over `<critical>` XML tag | Visual impact in raw editing |
| Template structure is recommendation, not rigid | Different scenarios have legitimately different needs |

---

## Open Questions

- Should anti-rationalization entries be in `.j2` templates or `scenario.yaml` config?
- How should Pattern L interact with CompressionMiddleware?
- Should GP scenario adopt a fuller pattern set or remain minimal?
- Implementation priority across templates? Recommendation: F > B > J > G > E > D.

---

## Related Concepts

- [Orchestrator](orchestrator.md) — Prompt patterns affect orchestrator system prompt and DynamicContextMiddleware
- [Sub-Agent](sub-agent.md) — Worker prompts are primary consumers
- [Configuration System](system-design-overview.md) — Pattern parameters may be configured in scenario.yaml
- [Middleware System](sdk-consistency.md) — Pattern L implemented via middleware