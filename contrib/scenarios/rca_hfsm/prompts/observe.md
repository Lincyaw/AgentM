# OBSERVE state

The symptoms are recorded. Before you propose a hypothesis, gather enough
L1 facts that any hypothesis you propose has at least one falsifiable
prediction tied to evidence the trace can re-check.

Use `record_observation` to append citable facts — log snippets, query
result rows, file contents. Each observation should name its `source_tool_call`
(the tool that produced it) and link to the symptoms it speaks to via
`related_symptoms`. The observation cache will memoise idempotent tool
calls, so you can re-issue a probing query without paying for it twice.

Resist the temptation to jump to a hypothesis on the first plausible
correlation. A hypothesis proposed without grounding observations cannot
declare meaningful negative predictions, and the gate will reject it.

Move to HYPOTHESIZE by calling `propose_hypothesis` once you have enough
facts to articulate at least one prediction that, if observed, would
**rule the hypothesis out**.

Available tools: `record_observation`, `record_symptom`, `propose_hypothesis`.
