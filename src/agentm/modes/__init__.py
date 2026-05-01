"""Presenter / mode layer.

Per ``.claude/designs/pluggable-architecture.md`` §5, modes are thin
consumers of ``AgentSession``. They never touch the harness or kernel
directly; they subscribe to the bus and translate events into I/O.
"""
