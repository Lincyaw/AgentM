"""Tests for ManagedTask data contracts.

Ref: designs/orchestrator.md § TaskManager — ManagedTask
Ref: designs/testing-strategy.md § P4, P5 — instruction injection and abort

ManagedTask tracks async Sub-Agent execution. These tests verify
that the data structure supports the instruction injection and
abort workflows described in the design.

Bug prevented: ManagedTask missing pending_instructions field →
inject_instruction has nowhere to queue messages → instructions silently lost.
"""

from __future__ import annotations

from agentm.models.data import ManagedTask
from agentm.models.enums import AgentRunStatus


class TestManagedTaskInstructionIsolation:
    """Ref: designs/orchestrator.md § Instruction Injection

    pending_instructions is a mutable list. Each ManagedTask instance
    must have its own list (via default_factory).

    Bug: shared mutable default → instruction injected into one task
    appears in another task's queue.
    """

    def test_pending_instructions_independent_between_instances(self):
        t1 = ManagedTask(task_id="t1", agent_id="db", instruction="check connections")
        t2 = ManagedTask(task_id="t2", agent_id="app", instruction="check latency")
        t1.pending_instructions.append("redirect to CPU check")
        assert len(t2.pending_instructions) == 0

    def test_events_buffer_independent_between_instances(self):
        """Same pattern — events_buffer must also be independent."""
        t1 = ManagedTask(task_id="t1", agent_id="db", instruction="i1")
        t2 = ManagedTask(task_id="t2", agent_id="app", instruction="i2")
        t1.events_buffer.append({"type": "tool_call"})
        assert len(t2.events_buffer) == 0


class TestManagedTaskRequiredFields:
    """Ref: designs/orchestrator.md § TaskManager — ManagedTask fields

    ManagedTask must have all fields needed by TaskManager workflows:
    status tracking, error reporting, instruction queueing, and
    checkpoint chain linking.

    Bug: missing field → TaskManager method raises AttributeError at runtime.
    """

    def test_has_all_workflow_fields(self):
        """Fields required by TaskManager submit/check/inject/abort workflows."""
        required_fields = {
            "task_id",
            "agent_id",
            "instruction",
            "metadata",
            "status",
            "result",
            "error_summary",
            "pending_instructions",
            "asyncio_task",
            "parent_thread_id",
            "parent_dispatch_step",
        }
        actual_fields = set(ManagedTask.__dataclass_fields__.keys())
        missing = required_fields - actual_fields
        assert not missing, f"ManagedTask missing workflow fields: {missing}"

    def test_default_status_is_running(self):
        """New tasks start as RUNNING — they're submitted for immediate execution.

        Bug: default status is COMPLETED → TaskManager skips execution.
        """
        t = ManagedTask(task_id="t1", agent_id="db", instruction="check")
        assert t.status == AgentRunStatus.RUNNING


class TestManagedTaskCheckpointLinking:
    """Ref: designs/orchestrator.md § Trajectory — TaskTraceRef

    ManagedTask carries parent_thread_id and parent_dispatch_step to link
    the Sub-Agent's checkpoint chain back to the Orchestrator's timeline.

    Bug: these fields missing → trajectory export cannot reconstruct the
    Orchestrator ↔ Sub-Agent relationship.
    """

    def test_parent_fields_default_to_none(self):
        """Parent linking is optional — set by TaskManager during submit()."""
        t = ManagedTask(task_id="t1", agent_id="db", instruction="check")
        assert t.parent_thread_id is None
        assert t.parent_dispatch_step is None
