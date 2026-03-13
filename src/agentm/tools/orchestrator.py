"""Orchestrator tools for task dispatch, monitoring, hypothesis management, and recall."""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any, Callable, Literal, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command

from agentm.models.enums import HypothesisStatus
from agentm.models.types import TaskType


def create_orchestrator_tools(
    task_manager: Any,
    agent_pool: Any,
    trajectory: Any | None = None,
    config: Any | None = None,
) -> dict[str, Callable[..., Any]]:
    """Factory that creates orchestrator tool functions with injected dependencies.

    Returns a dict mapping tool name to bound tool function. The task_manager and
    agent_pool are captured in closures — no module-level globals needed.

    Args:
        task_manager: TaskManager instance.
        agent_pool: AgentPool or NodeAgentPool instance.
        trajectory: Optional TrajectoryCollector.
        config: Optional OrchestratorConfig — used to read recall.model.
    """

    async def dispatch_agent(
        agent_id: str,
        task: str,
        task_type: TaskType = "scout",
        hypothesis_id: Optional[str] = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Launch a Sub-Agent. Auto-blocks when this is the only running task.

        Single-worker: waits for completion and returns result directly,
        saving an LLM roundtrip through check_tasks.
        Multi-worker: returns immediately with status "running".
        """
        task_id = str(uuid.uuid4())
        subgraph = agent_pool.create_worker(agent_id, task_type, task_id=task_id)
        # Each tool-call step uses ~3 graph nodes (pre_model_hook + call_model
        # + tools), plus a final generate_structured_response node.
        recursion_limit = agent_pool.worker_max_steps * 3 + 20
        task_id = await task_manager.submit(
            agent_id,
            task,
            task_type,
            hypothesis_id,
            subgraph=subgraph,
            config={"recursion_limit": recursion_limit},
            task_id=task_id,
            trajectory_self_reported=agent_pool.worker_self_reports_trajectory,
        )

        # Auto-block: if this is the only running task and it has a real
        # asyncio task, wait for completion via the completion event.
        managed = task_manager.get_task(task_id)
        if task_manager.get_running_count() == 1 and managed.asyncio_task is not None:
            await task_manager.wait_for_task(task_id)
            content = json.dumps(
                {
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "status": managed.status.value,
                    "result": managed.result,
                    "error_summary": managed.error_summary,
                    "duration_seconds": managed.duration_seconds,
                },
                default=str,
            )
        else:
            content = json.dumps(
                {
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "status": "running",
                }
            )

        return Command(
            update={
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
            }
        )

    async def check_tasks(
        request: str,
        wait_seconds: float = 10,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Check status of all dispatched tasks and collect completed results."""
        _ = request
        results = await task_manager.get_all_status(wait_seconds=wait_seconds)
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps(results, default=str),
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    async def inject_instruction(task_id: str, instruction: str) -> str:
        """Inject a new instruction into a running Sub-Agent."""
        await task_manager.inject(task_id, instruction)
        return f"Instruction injected into task {task_id}"

    async def abort_task(task_id: str, reason: str) -> str:
        """Abort a running Sub-Agent task."""
        await task_manager.abort(task_id, reason)
        return f"Task {task_id} aborted: {reason}"

    async def update_hypothesis(
        id: str,
        description: str,
        status: Literal[
            "formed",
            "investigating",
            "confirmed",
            "rejected",
            "refined",
            "inconclusive",
        ] = "formed",
        evidence_summary: Optional[str] = None,
        parent_id: Optional[str] = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Create or update a hypothesis in the DiagnosticNotebook."""
        HypothesisStatus(status)  # validate the value matches enum
        content = f"Hypothesis {id} updated: {status} — {description}"

        if trajectory is not None:
            await trajectory.record(
                event_type="hypothesis_update",
                agent_path=["orchestrator"],
                data={
                    "hypothesis_id": id,
                    "status": status,
                    "description": description,
                    "evidence_summary": evidence_summary,
                    "parent_id": parent_id,
                },
                hypothesis_id=id,
            )

        return Command(
            update={
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
            }
        )

    async def remove_hypothesis(
        id: str,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Remove a hypothesis from the DiagnosticNotebook."""
        if trajectory is not None:
            await trajectory.record(
                event_type="hypothesis_update",
                agent_path=["orchestrator"],
                data={
                    "hypothesis_id": id,
                    "status": "removed",
                    "description": "",
                },
                hypothesis_id=id,
            )

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Hypothesis {id} removed",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    # --- Mutable references for recall_history ---
    # Set by builder after graph compilation (chicken-and-egg: graph is created with tools).
    _graph_ref: list[Any] = [None]
    _config_ref: list[dict[str, Any]] = [{}]

    # Recall model: read from config.compression.recall.model if available,
    # otherwise fall back to a sensible default.
    _recall_model_name = "gpt-5.1-mini"
    if config is not None:
        _compression = getattr(config, "compression", None)
        if _compression is not None:
            _recall = getattr(_compression, "recall", None)
            if isinstance(_recall, dict):
                _recall_model_name = _recall.get("model", _recall_model_name)
            elif _recall is not None:
                _recall_model_name = getattr(_recall, "model", _recall_model_name)

    def recall_history(
        query: str,
        scope: Literal[
            "current_compression", "all_compressions"
        ] = "current_compression",
    ) -> str:
        """Search pre-compression history for detailed information.

        Use this when you need details that were lost during context compression,
        such as: raw metric breakdowns, tool call parameters, time-series data,
        or observations not included in the summary.

        Args:
            query: Natural language description of what you're looking for.
                Examples:
                - "What were the top 5 slow queries and their execution times?"
                - "What parameters were used when checking the connection pool?"
                - "Was there any disk inode data collected during infrastructure scan?"
            scope: Search range.
                - "current_compression": Only the most recent compressed range
                - "all_compressions": All compressed ranges in this task
        """
        graph = _graph_ref[0]
        graph_config = _config_ref[0]

        if graph is None or not graph_config:
            return "recall_history is not available — graph reference not set."

        # Read compression_refs from current state
        try:
            state = graph.get_state(graph_config)
            compression_refs = state.values.get("compression_refs", [])
        except Exception:
            compression_refs = []

        if not compression_refs:
            return (
                "No compression has occurred yet. Full history is already in your context. "
                "This tool is only useful after context compression."
            )

        # NOTE: scope selects which compression_refs to search.
        # Currently traverses all checkpoints; will be filtered by ref ranges
        # when drill_down_compressed_range is implemented.

        # Extract messages from checkpoint history
        all_messages: list[dict[str, Any]] = []
        try:
            for state_snapshot in graph.get_state_history(graph_config):
                messages = state_snapshot.values.get("messages", [])
                for msg in messages:
                    all_messages.append(
                        {
                            "type": getattr(msg, "type", "unknown"),
                            "content": getattr(msg, "content", ""),
                            "tool_calls": getattr(msg, "tool_calls", None),
                        }
                    )
                # Cap messages to avoid loading too many checkpoints
                if len(all_messages) > 200:
                    break
        except Exception as e:
            return f"Error accessing checkpoint history: {e}"

        if not all_messages:
            return "No messages found in checkpoint history."

        # Format messages for retrieval prompt
        formatted_msgs: list[str] = []
        for m in all_messages[:200]:
            if m["tool_calls"]:
                tool_info = ", ".join(tc.get("name", "?") for tc in m["tool_calls"])
                formatted_msgs.append(f"[{m['type']}] Tools: {tool_info}")
            elif m["content"]:
                preview = m["content"][:300]
                formatted_msgs.append(f"[{m['type']}] {preview}")

        retrieval_prompt = (
            f"You are searching through an agent's historical execution records.\n\n"
            f"Query: {query}\n\n"
            f"Records:\n" + "\n".join(formatted_msgs) + "\n\n"
            "Find and return information relevant to the query. "
            "Include specific data values, not just summaries."
        )

        # Use LLM to find relevant information
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage as LCHumanMessage

            llm = ChatOpenAI(model=_recall_model_name, temperature=0)
            result = llm.invoke([LCHumanMessage(content=retrieval_prompt)])
            content_val = result.content
            if isinstance(content_val, list):
                return " ".join(str(part) for part in content_val)
            return str(content_val)
        except Exception as e:
            return f"Error during recall: {e}"

    def _set_graph_ref(graph: Any, graph_config: dict[str, Any]) -> None:
        """Set the compiled graph and config references for recall_history."""
        _graph_ref[0] = graph
        _config_ref[0] = graph_config

    tools: dict[str, Callable[..., Any]] = {
        "dispatch_agent": dispatch_agent,
        "check_tasks": check_tasks,
        "inject_instruction": inject_instruction,
        "abort_task": abort_task,
        "update_hypothesis": update_hypothesis,
        "remove_hypothesis": remove_hypothesis,
        "recall_history": recall_history,
        "_set_graph_ref": _set_graph_ref,
    }
    return tools

    """Factory that creates orchestrator tool functions with injected dependencies.

    Returns a dict mapping tool name to bound tool function. The task_manager and
    agent_pool are captured in closures — no module-level globals needed.
    """

    async def dispatch_agent(
        agent_id: str,
        task: str,
        task_type: TaskType = "scout",
        hypothesis_id: Optional[str] = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Launch a Sub-Agent. Auto-blocks when this is the only running task.

        Single-worker: waits for completion and returns result directly,
        saving an LLM roundtrip through check_tasks.
        Multi-worker: returns immediately with status "running".
        """
        task_id = str(uuid.uuid4())
        subgraph = agent_pool.create_worker(agent_id, task_type, task_id=task_id)
        # Each tool-call step uses ~3 graph nodes (pre_model_hook + call_model
        # + tools), plus a final generate_structured_response node.
        recursion_limit = agent_pool.worker_max_steps * 3 + 20
        task_id = await task_manager.submit(
            agent_id,
            task,
            task_type,
            hypothesis_id,
            subgraph=subgraph,
            config={"recursion_limit": recursion_limit},
            task_id=task_id,
            trajectory_self_reported=agent_pool.worker_self_reports_trajectory,
        )

        # Auto-block: if this is the only running task and it has a real
        # asyncio task, wait for completion via the completion event.
        managed = task_manager.get_task(task_id)
        if task_manager.get_running_count() == 1 and managed.asyncio_task is not None:
            await task_manager.wait_for_task(task_id)
            content = json.dumps(
                {
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "status": managed.status.value,
                    "result": managed.result,
                    "error_summary": managed.error_summary,
                    "duration_seconds": managed.duration_seconds,
                },
                default=str,
            )
        else:
            content = json.dumps(
                {
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "status": "running",
                }
            )

        return Command(
            update={
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
            }
        )

    async def check_tasks(
        request: str,
        wait_seconds: float = 10,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Check status of all dispatched tasks and collect completed results."""
        _ = request
        results = await task_manager.get_all_status(wait_seconds=wait_seconds)
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps(results, default=str),
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    async def inject_instruction(task_id: str, instruction: str) -> str:
        """Inject a new instruction into a running Sub-Agent."""
        await task_manager.inject(task_id, instruction)
        return f"Instruction injected into task {task_id}"

    async def abort_task(task_id: str, reason: str) -> str:
        """Abort a running Sub-Agent task."""
        await task_manager.abort(task_id, reason)
        return f"Task {task_id} aborted: {reason}"

    async def update_hypothesis(
        id: str,
        description: str,
        status: Literal[
            "formed",
            "investigating",
            "confirmed",
            "rejected",
            "refined",
            "inconclusive",
        ] = "formed",
        evidence_summary: Optional[str] = None,
        parent_id: Optional[str] = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Create or update a hypothesis in the DiagnosticNotebook."""
        HypothesisStatus(status)  # validate the value matches enum
        content = f"Hypothesis {id} updated: {status} — {description}"

        if trajectory is not None:
            await trajectory.record(
                event_type="hypothesis_update",
                agent_path=["orchestrator"],
                data={
                    "hypothesis_id": id,
                    "status": status,
                    "description": description,
                    "evidence_summary": evidence_summary,
                    "parent_id": parent_id,
                },
                hypothesis_id=id,
            )

        return Command(
            update={
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
            }
        )

    async def remove_hypothesis(
        id: str,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Remove a hypothesis from the DiagnosticNotebook."""
        if trajectory is not None:
            await trajectory.record(
                event_type="hypothesis_update",
                agent_path=["orchestrator"],
                data={
                    "hypothesis_id": id,
                    "status": "removed",
                    "description": "",
                },
                hypothesis_id=id,
            )

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Hypothesis {id} removed",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    # --- Mutable references for recall_history ---
    # Set by builder after graph compilation (chicken-and-egg: graph is created with tools).
    _graph_ref: list[Any] = [None]
    _config_ref: list[dict[str, Any]] = [{}]

    def recall_history(
        query: str,
        scope: Literal[
            "current_compression", "all_compressions"
        ] = "current_compression",
    ) -> str:
        """Search pre-compression history for detailed information.

        Use this when you need details that were lost during context compression,
        such as: raw metric breakdowns, tool call parameters, time-series data,
        or observations not included in the summary.

        Args:
            query: Natural language description of what you're looking for.
                Examples:
                - "What were the top 5 slow queries and their execution times?"
                - "What parameters were used when checking the connection pool?"
                - "Was there any disk inode data collected during infrastructure scan?"
            scope: Search range.
                - "current_compression": Only the most recent compressed range
                - "all_compressions": All compressed ranges in this task
        """
        graph = _graph_ref[0]
        config = _config_ref[0]

        if graph is None or not config:
            return "recall_history is not available — graph reference not set."

        # Read compression_refs from current state
        try:
            state = graph.get_state(config)
            compression_refs = state.values.get("compression_refs", [])
        except Exception:
            compression_refs = []

        if not compression_refs:
            return (
                "No compression has occurred yet. Full history is already in your context. "
                "This tool is only useful after context compression."
            )

        # NOTE: scope selects which compression_refs to search.
        # Currently traverses all checkpoints; will be filtered by ref ranges
        # when drill_down_compressed_range is implemented.

        # Extract messages from checkpoint history
        all_messages: list[dict[str, Any]] = []
        try:
            for state_snapshot in graph.get_state_history(config):
                messages = state_snapshot.values.get("messages", [])
                for msg in messages:
                    all_messages.append(
                        {
                            "type": getattr(msg, "type", "unknown"),
                            "content": getattr(msg, "content", ""),
                            "tool_calls": getattr(msg, "tool_calls", None),
                        }
                    )
                # Cap messages to avoid loading too many checkpoints
                if len(all_messages) > 200:
                    break
        except Exception as e:
            return f"Error accessing checkpoint history: {e}"

        if not all_messages:
            return "No messages found in checkpoint history."

        # Format messages for retrieval prompt
        formatted_msgs: list[str] = []
        for m in all_messages[:200]:
            if m["tool_calls"]:
                tool_info = ", ".join(tc.get("name", "?") for tc in m["tool_calls"])
                formatted_msgs.append(f"[{m['type']}] Tools: {tool_info}")
            elif m["content"]:
                preview = m["content"][:300]
                formatted_msgs.append(f"[{m['type']}] {preview}")

        retrieval_prompt = (
            f"You are searching through an agent's historical execution records.\n\n"
            f"Query: {query}\n\n"
            f"Records:\n" + "\n".join(formatted_msgs) + "\n\n"
            "Find and return information relevant to the query. "
            "Include specific data values, not just summaries."
        )

        # Use LLM to find relevant information
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage as LCHumanMessage

            llm = ChatOpenAI(model="gpt-5.1-mini", temperature=0)
            result = llm.invoke([LCHumanMessage(content=retrieval_prompt)])
            content = result.content
            if isinstance(content, list):
                return " ".join(str(part) for part in content)
            return str(content)
        except Exception as e:
            return f"Error during recall: {e}"

    def _set_graph_ref(graph: Any, config: dict[str, Any]) -> None:
        """Set the compiled graph and config references for recall_history."""
        _graph_ref[0] = graph
        _config_ref[0] = config

    tools: dict[str, Callable[..., Any]] = {
        "dispatch_agent": dispatch_agent,
        "check_tasks": check_tasks,
        "inject_instruction": inject_instruction,
        "abort_task": abort_task,
        "update_hypothesis": update_hypothesis,
        "remove_hypothesis": remove_hypothesis,
        "recall_history": recall_history,
        "_set_graph_ref": _set_graph_ref,
    }
    return tools
