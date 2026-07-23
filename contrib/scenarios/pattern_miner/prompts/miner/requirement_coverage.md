# Requirement coverage (understanding -> plan/change)

Comparand: the requirement clauses of the task against the hunks of the final
patch.

Build a checklist of requirement clauses from the task text, then map each
clause to the code that implements it. A clause with no corresponding change
is a coverage gap. Separately, when the change touches one site of a pattern
that recurs in the codebase, check whether the sibling sites were addressed:
partial application of a multi-site change is the second form of this
dimension.

The question here is whether implementing code exists at all, not whether it
was validated — validation gaps belong to evidence adequacy. Keep the two
apart even when both fire.

Online signature guidance: clause-to-change mapping needs only the task text
and the diff, both available live. Sibling-site completeness needs codebase
structure the monitor can obtain from the trajectory's own reads and searches;
note when the agent's own exploration already surfaced sibling sites it then
left untouched.
