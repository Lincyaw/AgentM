# Checklist distillation (batch-level, per dimension)

Input: every finding mined under one dimension across a batch of cases, with
per-case attributions, and optionally the existing checklist items for this
dimension. Output: the reusable product — checklist items a critic can
execute on future cases that share none of these cases' surface details.

An item has four parts. `check` is a question a critic can act on for any
case: phrased in the language of the solving chain (task, reference,
requirement, change, validation, belief), never in the language of one case
(no repository names, languages, frameworks, or mechanism nouns like retry
or cache unless the mechanism is itself the failure class). `when` is the
applicability trigger: a condition computable from a live task and
trajectory, cheap enough to evaluate always. `how` names the cheapest
sufficient sensor: a structural rule over the trajectory, an LLM reading of
task plus trajectory slices, or an active probe in a live environment.
`evidence` lists the trials that support the item.

Generalize to the mechanism, never past it. An item must stay falsifiable
and executable; advice of the form "be careful" or "verify thoroughly" is
forbidden. Do not invent items the findings do not support; if a finding is
too case-bound to generalize, record that in notes instead of forcing an
item.

Merge before you add. Findings that are the same mechanism at different
surfaces become one item with the union of evidence; an existing checklist
item that already covers a finding absorbs it as evidence, refined in
wording only if the new finding reveals the old wording was too narrow. A
new item is justified only by a mechanism no existing item covers. The
checklist is a bounded, consolidated artifact, not an append-only log.
