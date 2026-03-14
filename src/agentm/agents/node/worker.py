"""Node-based worker subgraph.

Architecture (4 nodes, self-controlled ReAct loop):

    START → dispatch → llm_call ↔ tool_node → collect_and_compress → END

Differences from react/sub_agent.py (create_react_agent):
- ``route_after_llm`` decides the next node — the framework never sees
  a bare AIMessage with no tool_calls as a terminal condition.
- ``collect_and_compress`` always runs at the end regardless of how the
  LLM exited the loop, guaranteeing a non-null structured result.
- Budget, compression, and trajectory are handled via ``NodePipeline``
  (same middleware classes as React mode, but invoked explicitly).
"""

from __future__ import annotations

from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agentm.agents.node.state import WorkerResult, WorkerState
from agentm.config.schema import AgentConfig, ModelConfig
from agentm.middleware import NodePipeline
from agentm.middleware.budget import BudgetMiddleware
from agentm.middleware.compression import CompressionMiddleware
from agentm.middleware.trajectory import TrajectoryMiddleware
from agentm.core.prompt import load_prompt_template
from agentm.core.tool_registry import ToolRegistry
from agentm.core.trajectory import TrajectoryCollector
from agentm.models.types import TaskType
from agentm.tools.think import think


# ---------------------------------------------------------------------------
# Answer schemas (lazy-loaded from consolidated registry)
# ---------------------------------------------------------------------------

from agentm.models.answer_schemas import ANSWER_SCHEMA  # noqa: E402


# ---------------------------------------------------------------------------
# Tool tip formatting (mirrors graph.py in reference implementation)
# ---------------------------------------------------------------------------


def _format_tool_tips(tips: list[dict[str, Any]]) -> str:
    """Format cross-worker error tips as a markdown section."""
    if not tips:
        return ""
    by_tool: dict[str, list[dict]] = {}
    for tip in tips:
        by_tool.setdefault(tip.get("tool_name", "unknown"), []).append(tip)

    lines = ["## Tool Usage Tips (from prior sub-agents)", ""]
    for tool_name, tool_tips in sorted(by_tool.items()):
        lines.append(f"### `{tool_name}`")
        for tip in tool_tips[-5:]:  # cap at 5 per tool
            args = tip.get("args_summary", "")
            error = tip.get("error", "")
            lines.append(f"- Error with args {{{args}}}: {error}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_worker_subgraph(
    *,
    agent_id: str,
    config: AgentConfig,
    tool_registry: ToolRegistry,
    task_type: TaskType = "scout",
    model_config: ModelConfig | None = None,
    trajectory: TrajectoryCollector | None = None,
    task_id: str | None = None,
    checkpointer: Any = None,
) -> Any:
    """Build and compile the node-based worker subgraph.

    All dependencies are injected via parameters and captured in node
    closures — no module-level globals, safe to compile once per dispatch.

    Returns a CompiledStateGraph with the same astream interface as the
    create_react_agent subgraph produced by react/sub_agent.py.
    """
    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    llm_kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
    }
    if model_config is not None:
        llm_kwargs["api_key"] = model_config.api_key
        if model_config.base_url:
            llm_kwargs["base_url"] = model_config.base_url

    model_plain = ChatOpenAI(**llm_kwargs)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    tools: list[BaseTool] = [
        tool_registry.get(name).create_with_config(**config.tool_settings.get(name, {}))
        for name in config.tools
    ]
    tools.append(think)
    tools_by_name = {t.name: t for t in tools}

    # Dedup wrapping (optional)
    if config.execution.dedup is not None and config.execution.dedup.enabled:
        from agentm.middleware.dedup import DedupTracker, wrap_tool_with_dedup

        tracker = DedupTracker(max_cache_size=config.execution.dedup.max_cache_size)
        tools = [wrap_tool_with_dedup(t, tracker) for t in tools]
        tools_by_name = {t.name: t for t in tools}

    model_with_tools = model_plain.bind_tools(tools)

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------
    tools_description = "\n".join(f"- `{t.name}`: {t.description}" for t in tools)
    template_context = {"agent_id": agent_id, "tools_description": tools_description}

    if config.prompt is None:
        base_prompt = ""
    else:
        base_prompt = load_prompt_template(
            config.prompt, base_dir=None, **template_context
        )

    if config.task_type_prompts and task_type in config.task_type_prompts:
        overlay = load_prompt_template(
            config.task_type_prompts[task_type], base_dir=None, **template_context
        )
        system_prompt = (
            (base_prompt + "\n\n" + overlay).strip() if base_prompt else overlay
        )
    else:
        system_prompt = base_prompt

    # ------------------------------------------------------------------
    # Middleware pipeline (replaces inline budget/compression/trajectory)
    # ------------------------------------------------------------------
    middlewares: list = [BudgetMiddleware(config.execution.max_steps)]
    if config.compression is not None:
        middlewares.append(CompressionMiddleware(config.compression))
    if trajectory is not None:
        middlewares.append(
            TrajectoryMiddleware(trajectory, ["orchestrator", agent_id], task_id=task_id)
        )
    pipeline = NodePipeline(middlewares)

    # ------------------------------------------------------------------
    # Node 1: dispatch
    # Build the initial message list from task state.
    # No LLM call — purely deterministic setup.
    # ------------------------------------------------------------------

    async def dispatch(state: WorkerState) -> dict[str, Any]:
        msgs: list[Any] = []

        # System prompt with optional tool tips
        sp_parts = [system_prompt] if system_prompt else []
        tips_section = _format_tool_tips(state.get("tool_tips", []))
        if tips_section:
            sp_parts.append(tips_section)
        if sp_parts:
            msgs.append(SystemMessage(content="\n\n".join(sp_parts)))

        # ``instruction`` may come from two sources:
        # 1. WorkerState.instruction (populated when caller builds full state)
        # 2. The first HumanMessage already in messages (TaskManager always
        #    calls _execute_agent with {"messages": [HumanMessage(instruction)]})
        instruction = state.get("instruction", "")
        if not instruction:
            for m in state.get("messages", []):
                if isinstance(m, HumanMessage):
                    instruction = str(m.content)
                    break

        if instruction:
            msgs.append(HumanMessage(content=instruction))
        return {"messages": msgs}

    # ------------------------------------------------------------------
    # Node 2: llm_call
    # Runs the middleware pipeline (budget, compression, trajectory)
    # then invokes the LLM with the prepared messages.
    # ------------------------------------------------------------------

    async def llm_call(state: WorkerState) -> dict[str, Any]:
        messages = list(state.get("messages", []))

        # Pre-model pipeline: budget → compression → trajectory(llm_start)
        prepared = pipeline.before({"messages": messages})
        llm_messages = prepared.get("llm_input_messages") or prepared.get(
            "messages", messages
        )

        response = await model_with_tools.ainvoke(llm_messages)

        # Post-model pipeline: trajectory(tool_call / llm_end)
        await pipeline.after({"messages": messages}, response)

        return {"messages": [response]}

    # ------------------------------------------------------------------
    # Node 3: tool_node
    # Execute all tool calls from the last AIMessage.
    # Errors are caught and returned as ToolMessages (Layer 1 recovery).
    # ------------------------------------------------------------------

    async def tool_node(state: WorkerState) -> dict[str, Any]:
        last = cast(AIMessage, state["messages"][-1])
        tool_msgs: list[ToolMessage] = []

        for tc in last.tool_calls:
            tc_id = tc.get("id") or ""
            tc_name = tc.get("name") or ""

            if not tc_name:
                tool_msgs.append(
                    ToolMessage(
                        content="(skipped: empty tool name)", tool_call_id=tc_id
                    )
                )
                continue

            tool = tools_by_name.get(tc_name)
            if tool is None:
                tool_msgs.append(
                    ToolMessage(
                        content=f"Tool '{tc_name}' not found. Available: {list(tools_by_name)}",
                        tool_call_id=tc_id,
                    )
                )
                continue

            try:
                obs = await tool.ainvoke(tc.get("args", {}))
                result_text = str(obs)
            except Exception as exc:
                result_text = f"Tool execution failed: {exc}"

            if trajectory is not None:
                trajectory.record_sync(
                    event_type="tool_result",
                    agent_path=["orchestrator", agent_id],
                    data={"tool_name": tc_name, "result": result_text[:500]},
                    task_id=task_id,
                )

            tool_msgs.append(ToolMessage(content=result_text, tool_call_id=tc_id))

        return {"messages": tool_msgs}

    # ------------------------------------------------------------------
    # Node 4: collect_and_compress
    # Runs a separate LLM call (no tools) to produce a structured result.
    # This node ALWAYS runs at loop exit, guaranteeing a non-null result.
    # ------------------------------------------------------------------

    from agentm.scenarios import discover as _discover_scenarios

    _discover_scenarios()
    answer_schema = ANSWER_SCHEMA[task_type]
    compress_model = model_plain.with_structured_output(answer_schema)

    _compress_system = (
        "You are synthesizing a sub-agent investigation into a structured report. "
        "Extract concrete findings from the conversation history. "
        "Be precise — use exact service names, metric values, and timestamps. "
        "Omit reasoning steps; report only findings and conclusions."
    )

    async def collect_and_compress(state: WorkerState) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]

        compress_input = [
            SystemMessage(content=_compress_system),
            *non_system,
            HumanMessage(
                content=(
                    f"Task: {state.get('instruction', '')}\n"
                    f"Task type: {state.get('task_type', task_type)}\n\n"
                    "Produce your structured report now."
                )
            ),
        ]

        try:
            result = await compress_model.ainvoke(compress_input)
            structured: WorkerResult = result.model_dump()  # type: ignore[union-attr]
        except Exception:
            # Fallback: extract last non-tool AI message as plain findings
            last_content = ""
            for m in reversed(non_system):
                if isinstance(m, AIMessage) and not m.tool_calls and m.content:
                    last_content = str(m.content)
                    break
            structured = WorkerResult(
                findings=last_content or "(no findings produced)",
                leads=[],
                verdict="",
            )

        return {"structured_response": structured}

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route_after_llm(
        state: WorkerState,
    ) -> Literal["tool_node", "collect_and_compress"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tool_node"
        # No tool calls — exit the ReAct loop and compress
        return "collect_and_compress"

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    builder = StateGraph(WorkerState)

    builder.add_node("dispatch", dispatch)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)
    builder.add_node("collect_and_compress", collect_and_compress)

    builder.add_edge(START, "dispatch")
    builder.add_edge("dispatch", "llm_call")
    builder.add_conditional_edges(
        "llm_call",
        route_after_llm,
        {"tool_node": "tool_node", "collect_and_compress": "collect_and_compress"},
    )
    builder.add_edge("tool_node", "llm_call")
    builder.add_edge("collect_and_compress", END)

    compile_kwargs: dict[str, Any] = {"name": f"worker_{task_type}_{agent_id}"}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    return builder.compile(**compile_kwargs)


# ---------------------------------------------------------------------------
# AgentPool equivalent for node-based workers
# ---------------------------------------------------------------------------


class NodeAgentPool:
    """Factory for node-based worker subgraphs.

    Drop-in replacement for react.sub_agent.AgentPool — the TaskManager
    calls ``create_worker`` in the same way regardless of which pool is used.
    """

    def __init__(
        self,
        scenario_config: Any,
        tool_registry: ToolRegistry,
        model_config: ModelConfig | None = None,
        trajectory: TrajectoryCollector | None = None,
        checkpointer: Any = None,
    ) -> None:
        self._worker_config = scenario_config.agents["worker"]
        self._tool_registry = tool_registry
        self._model_config = model_config
        self._trajectory = trajectory
        self._checkpointer = checkpointer

    @property
    def worker_max_steps(self) -> int:
        return self._worker_config.execution.max_steps

    @property
    def worker_self_reports_trajectory(self) -> bool:
        """Node workers record trajectory events themselves — always True."""
        return True

    def create_worker(
        self,
        agent_id: str,
        task_type: TaskType,
        task_id: str | None = None,
    ) -> Any:
        """Compile a fresh node-based worker subgraph per dispatch."""
        return build_worker_subgraph(
            agent_id=agent_id,
            config=self._worker_config,
            tool_registry=self._tool_registry,
            task_type=task_type,
            model_config=self._model_config,
            trajectory=self._trajectory,
            task_id=task_id,
            checkpointer=self._checkpointer,
        )
