# Loop dynamics (the chain over time)

The other dimensions check consistency of the chain at a moment; this one
checks convergence of the loop itself. Any chain state, healthy or not, can
exhibit bad dynamics.

Look for the time-domain shapes: stagnation windows where effort continues
without state change; oscillation where the agent alternates between states
it has already visited; repeated identical failures with no intervening
change to what is being tried; effort concentrated far past diminishing
returns while other obligations starve; and stopping — early or late —
relative to the signal still available. Anchor each shape in turn ranges, not
impressions: what repeated, over which turns, with what changing between
repetitions.

Attribute conservatively. Long exploration is not stagnation if each pass
adds information; repetition is not oscillation if the inputs differ. The
finding must name what stayed invariant across the repetitions, because that
invariant is what the agent failed to update.

Online signature guidance: this dimension is natively online — cycles,
stagnation windows, and unresolved-repeat counts are computable from the live
trajectory as it grows. The open question for each finding is discrimination:
state why the shape you found would not fire equally on messy-but-successful
sessions.
