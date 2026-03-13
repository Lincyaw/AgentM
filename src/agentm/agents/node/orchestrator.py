"""Node-based Orchestrator graph.

Architecture (3-node controlled loop with explicit decision routing):

    START → llm_call → route_after_llm
                          ├─ "tool_node"  (has tool_calls, decision != finalize)
                          │    └─ tool_node → llm_call
                          ├─ "llm_call"   (no tool_calls, decision == continue)
                          └─ "synthesize" (decision == finalize)
                               └─ synthesize → END

Key difference from react/orchestrator.py and earlier versions of this file:
Routing is driven by the LLM's explicit ``<decision>`` XML tag in its response,
NOT by the presence or absence of tool_calls.  This decouples control flow from
tool invocation — the LLM can output tool_calls AND signal finalize at the same
time, or output plain text AND signal continue.

``<decision>finalize</decision>`` → synthesize (investigation complete)
``<decision>dispatch</decision>`` or anything else → tool_node or llm_call

The prompt must instruct the LLM to always include a <decision> tag.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from agentm.config.schema import OrchestratorConfig
from agentm.middleware.compression import build_compression_hook
from agentm.core.prompt import load_prompt_template
from agentm.core.trajectory import TrajectoryCollector
from agentm.models.output import get_output_schema
from agentm.models.state import BaseExecutorState

logger = logging.getLogger(__name__)

# Maximum number of retries when synthesize structured output fails validation.
# Total attempts = 1 (initial) + _SYNTH_MAX_RETRIES.
_SYNTH_MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_raw_from_error(exc: Exception) -> str:
    """Extract the raw LLM JSON text from a structured-output failure.

    LangChain's ``OutputParserException`` carries an ``llm_output`` attribute
    with the text the model actually produced.  Walk the exception chain to
    find it; fall back to ``str(exc)`` so the feedback message is still useful.
    """
    for e in (exc, getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
        if e is not None and hasattr(e, "llm_output") and e.llm_output:
            return str(e.llm_output)
    return str(exc)


def _parse_decision(text: str) -> str:
    """Extract the value of <decision>...</decision> from LLM output.

    Returns the lowercased content, or "dispatch" if the tag is absent
    (preserves backward compatibility with prompts that don't yet emit it).
    """
    match = re.search(r"<decision>(.*?)</decision>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return "dispatch"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def create_node_orchestrator(
    config: OrchestratorConfig,
    tools: list[Any],
    checkpointer: Any,
    store: Any,
    state_schema: type = BaseExecutorState,
    format_context: Callable[[dict], str] | None = None,
    model_config: Any | None = None,
    trajectory: TrajectoryCollector | None = None,
) -> Any:
    """Build the node-based Orchestrator as a CompiledStateGraph.

    Drop-in replacement for react/orchestrator.py::create_orchestrator —
    same signature, same return type, works with the existing builder.py
    wiring unchanged.

    Args:
        config: OrchestratorConfig from the scenario YAML.
        tools: List of tools available to the orchestrator.
        checkpointer: LangGraph checkpointer (or None).
        store: LangGraph store backend (or None).
        state_schema: TypedDict class for the StateGraph. Defaults to
            HypothesisDrivenState (backward-compatible).
        format_context: Callable ``(state: dict) -> str`` that produces the
            working-memory section of the system prompt. Defaults to
            ``format_rca_context`` (notebook-based, backward-compatible).
        model_config: Optional model configuration (API key, base_url).
        trajectory: Optional TrajectoryCollector for event recording.
    """
    # Default formatter: plain state description (no domain coupling)
    if format_context is not None:
        _format_context: Callable[[dict], str] = format_context
    else:
        def _format_context(state: dict) -> str:
            return f"Phase: {state.get('current_phase', 'unknown')}"

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    llm_kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
    }
    if model_config is not None:
        if getattr(model_config, "api_key", None):
            llm_kwargs["api_key"] = model_config.api_key
        if getattr(model_config, "base_url", None):
            llm_kwargs["base_url"] = model_config.base_url

    model_plain = ChatOpenAI(**llm_kwargs)
    model_with_tools = model_plain.bind_tools(tools)

    # ------------------------------------------------------------------
    # Config values captured by closures
    # ------------------------------------------------------------------
    system_prompt_template: str = config.prompts.get("system", "")
    max_rounds: int = config.max_rounds
    compression_cfg = config.compression

    # Structured output schema for synthesize node (optional)
    output_schema = None
    output_prompt_text = ""
    if config.output is not None:
        output_schema = get_output_schema(config.output.schema_name)
        output_prompt_text = load_prompt_template(config.output.prompt)

    # ------------------------------------------------------------------
    # Message preparation helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(state: dict, round_num: int) -> str:
        """Rebuild system prompt with current working memory and round context."""
        context_text = _format_context(state)

        if system_prompt_template:
            # Pass notebook= for backward compat with RCA prompt templates that
            # use {{ notebook }}, and context= for new templates.
            base = load_prompt_template(
                system_prompt_template,
                notebook=context_text,
                context=context_text,
            )
        else:
            base = f"You are an agent orchestrator.\n\n{context_text}"

        # Inject round context and finalize signal
        round_block = f"\n\n<round_context>\nRound: {round_num}/{max_rounds}\n"
        if round_num >= max_rounds:
            round_block += (
                "\n⚠️ LAST ROUND — you MUST output "
                "<decision>finalize</decision> now. "
                "Do NOT dispatch any more workers.\n"
            )
        elif round_num >= max_rounds - 1:
            round_block += (
                f"\n⚠️ Round {round_num}/{max_rounds} — 1 round remaining. "
                "Consider finalizing if evidence is sufficient.\n"
            )
        round_block += "</round_context>"

        return base + round_block

    compression_hook = (
        build_compression_hook(compression_cfg)
        if compression_cfg is not None and compression_cfg.enabled
        else None
    )

    def _prepare_messages(state: dict, round_num: int) -> list[Any]:
        """Build message list: system prompt + history, with optional compression."""
        system_msg = SystemMessage(content=_build_system_prompt(state, round_num))
        history = list(state.get("messages", []))
        messages = [system_msg, *history]

        if compression_hook is not None:
            hook_out = compression_hook({"messages": messages})
            return hook_out.get("llm_input_messages") or hook_out.get(
                "messages", messages
            )

        return messages

    # ------------------------------------------------------------------
    # Node 1: llm_call
    # ------------------------------------------------------------------

    async def llm_call(state: dict) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        round_num = sum(1 for m in messages if isinstance(m, AIMessage)) + 1

        # Hard cap: force finalize by injecting a no-tool AIMessage
        if round_num > max_rounds:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"[Max rounds ({max_rounds}) reached — finalizing.] "
                            "<decision>finalize</decision>"
                        )
                    )
                ]
            }

        llm_messages = _prepare_messages(state, round_num)

        # Record llm_start with full message context for dashboard Messages view
        if trajectory is not None:

            def _full_msg(msg: Any) -> dict[str, Any]:
                role = getattr(msg, "type", "unknown")
                content = getattr(msg, "content", "")
                entry: dict[str, Any] = {"role": role, "content": content}
                tc = getattr(msg, "tool_calls", None)
                if tc:
                    entry["tool_calls"] = [
                        {
                            "id": c.get("id", ""),
                            "name": c.get("name", ""),
                            "args": c.get("args", {}),
                        }
                        for c in tc
                    ]
                if role == "tool":
                    entry["name"] = getattr(msg, "name", "")
                    tcid = getattr(msg, "tool_call_id", None)
                    if tcid:
                        entry["tool_call_id"] = tcid
                return entry

            trajectory.record_sync(
                event_type="llm_start",
                agent_path=["orchestrator"],
                data={
                    "message_count": len(llm_messages),
                    "full_messages": [_full_msg(m) for m in llm_messages],
                },
            )

        response = await model_with_tools.ainvoke(llm_messages)
        return {"messages": [response]}

    # ------------------------------------------------------------------
    # Node 2: tool_node
    # ------------------------------------------------------------------

    _inner_tool_node = ToolNode(tools)

    async def tool_node(state: dict, config: RunnableConfig) -> Any:
        result = await _inner_tool_node.ainvoke(state, config)
        # ToolNode returns dict {"messages": [...]} for regular tools but
        # list [Command(...)] when tools return Command objects (e.g. dispatch_agent).
        # Return as-is in both cases — builder._record_node_event handles trajectory
        # recording for dict results; Command lists are handled by LangGraph natively.
        return result

    # ------------------------------------------------------------------
    # Node 3: synthesize
    # Runs exactly once when the LLM emits <decision>finalize</decision>.
    # Produces structured output (CausalGraph etc.) if schema is configured,
    # otherwise passes the last AI response through unchanged.
    # ------------------------------------------------------------------

    synthesize_model = (
        model_plain.with_structured_output(output_schema, method="json_mode")
        if output_schema is not None
        else None
    )

    async def synthesize(state: dict) -> dict[str, Any]:
        if synthesize_model is None:
            return {}

        messages = list(state.get("messages", []))
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]

        # Build a field-level example from the schema so the model sees
        # field names + types, NOT the raw JSON Schema meta-object.
        import json as _json

        def _schema_to_example(schema_cls: type[BaseModel]) -> dict:
            """Create a placeholder dict from Pydantic schema fields."""
            example: dict = {}
            for name, field_info in schema_cls.model_fields.items():
                ann = field_info.annotation
                if ann is int:
                    example[name] = 0
                elif ann is float:
                    example[name] = 0.0
                elif ann is str:
                    example[name] = f"<{field_info.description or name}>"
                elif ann is bool:
                    example[name] = False
                elif hasattr(ann, "__origin__") and ann.__origin__ is list:
                    example[name] = []
                else:
                    example[name] = f"<{field_info.description or name}>"
            return example

        example_obj = _schema_to_example(output_schema) if output_schema else {}
        json_instruction = (
            "\n\nYou MUST respond with a single valid JSON object with these fields "
            "(fill in actual values, not placeholders):\n"
            f"{_json.dumps(example_obj, indent=2)}\n"
            "Do NOT include any explanation or markdown — output raw JSON only."
        )

        attempt_messages = [
            SystemMessage(content=output_prompt_text + json_instruction),
            *non_system,
            HumanMessage(
                content="Produce your final structured report now. Output raw JSON only."
            ),
        ]

        structured: dict[str, Any] | None = None
        for attempt in range(1 + _SYNTH_MAX_RETRIES):
            try:
                result = await synthesize_model.ainvoke(attempt_messages)
                structured = (
                    result.model_dump() if hasattr(result, "model_dump") else result
                )
                break  # success
            except Exception as exc:
                if attempt < _SYNTH_MAX_RETRIES:
                    logger.info(
                        "synthesize attempt %d failed, retrying with feedback: %s",
                        attempt + 1,
                        exc,
                    )
                    raw_json = _extract_raw_from_error(exc)
                    attempt_messages.append(AIMessage(content=raw_json))
                    attempt_messages.append(
                        HumanMessage(
                            content=(
                                f"Your JSON output had validation errors:\n{exc}\n\n"
                                f"Your previous output:\n{raw_json}\n\n"
                                "Fix the errors and output valid JSON matching the schema exactly."
                            )
                        )
                    )
                else:
                    logger.warning(
                        "synthesize structured output failed after %d attempts (%s), "
                        "falling back to plain LLM",
                        attempt + 1,
                        exc,
                    )
                    try:
                        raw = await model_plain.ainvoke(attempt_messages)
                        raw_text = str(getattr(raw, "content", raw))
                    except Exception as exc2:
                        logger.warning("synthesize fallback also failed: %s", exc2)
                        raw_text = ""
                    structured = {"raw_text": raw_text}

        return {"structured_response": structured}

    # ------------------------------------------------------------------
    # Routing — driven by <decision> tag, NOT by tool_calls presence
    # ------------------------------------------------------------------

    def route_after_llm(
        state: dict,
    ) -> Literal["tool_node", "llm_call", "synthesize"]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return "llm_call"

        # Parse the explicit decision from LLM output
        content = str(last.content) if last.content else ""
        decision = _parse_decision(content)

        if decision == "finalize":
            return "synthesize"

        # LLM wants to continue — run tool_calls if any, else loop back
        if last.tool_calls:
            return "tool_node"

        # No tool_calls and not finalizing: LLM is mid-thought (rare).
        # Loop back so it can emit a proper tool call next round.
        return "llm_call"

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    builder = StateGraph(state_schema)

    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)
    builder.add_node("synthesize", synthesize)

    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call",
        route_after_llm,
        {"tool_node": "tool_node", "llm_call": "llm_call", "synthesize": "synthesize"},
    )
    builder.add_edge("tool_node", "llm_call")
    builder.add_edge("synthesize", END)

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if store is not None:
        compile_kwargs["store"] = store

    return builder.compile(name="node_orchestrator", **compile_kwargs)

