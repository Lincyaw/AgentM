"""Policy engine: an atom that runs beside an agent, and a loop that studies it.

Two halves, and they face opposite ways.

``runtime`` is the atom. It installs into a session, watches what the agent
does, and sends a check when one of its checklist items fires. It runs during
the work, inside the agent's own process, and everything it needs has to be
cheap enough to sit on the hot path.

``loop`` is offline. It reads finished runs -- trajectories, patches, graded
results -- diagnoses what went wrong, proposes checks and repository notes, and
measures whether they change anything by replaying the attempt with and without
them. It runs for hours after the fact and answers to nothing but the numbers.

What passes between them is the artefacts: a checklist the runtime injects, the
per-repository notes a reviewer is given, and the critic manifest both read.
The loop writes the first two; the runtime delivers them. The loop's compile
stage additionally imports the runtime's vocabulary and fact tables -- by
design, since the gates it emits must be exactly what the runtime evaluates.

``shared`` is what both need and neither owns: the query source, the agent
manifest loader, the policy-file resolver.
"""

from __future__ import annotations

from policy_engine.runtime.atom import MANIFEST, PolicyEngineConfig, install

__all__ = ["MANIFEST", "PolicyEngineConfig", "install"]
