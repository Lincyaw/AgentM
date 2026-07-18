"""AgentM Core v2 — Turn-centric trajectory model.

The v2 model treats Turn as the atomic unit of a trajectory, not Message
or Entry.  A Trajectory is an append-only sequence of committed Turns;
fork/resume/replay are operations on this sequence.  All pluggability
flows through the EventBus and ContextPolicy chain — atoms never need to
know about the Turn internals.
"""
