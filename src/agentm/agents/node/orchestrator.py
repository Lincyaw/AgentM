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
import os
import re
from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from agentm.config.schema import OrchestratorConfig
from agentm.config.schema import create_chat_model
from agentm.middleware import NodePipeline
from agentm.middleware.trajectory import TrajectoryMiddleware
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

_MSG_PREVIEW_LEN = int(os.environ.get("AGENTM_MSG_PREVIEW_LEN", "0"))


def _debug_log_messages(messages: list[Any], round_num: int) -> None:
    """Log the full message list sent to the LLM for debugging.

    Enable with DEBUG log level:
        AGENTM_LOG_LEVEL=DEBUG uv run agentm run ...

    Control content truncation:
        AGENTM_MSG_PREVIEW_LEN=500  — truncate each message to 500 chars
        AGENTM_MSG_PREVIEW_LEN=0    — no truncation (default)
    """
    limit = _MSG_PREVIEW_LEN
    parts: list[str] = [f"\n{'=' * 60} LLM INPUT (round {round_num}) {'=' * 60}"]
    for i, msg in enumerate(messages):
        role = getattr(msg, "type", type(msg).__name__)
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", None)

        header = f"[{i}] {role}"
        if tool_calls:
            tool_names = ", ".join(tc.get("name", "?") for tc in tool_calls)
            header += f"  tools=[{tool_names}]"

        parts.append(f"\n--- {header} ---")
        if isinstance(content, str) and content:
            if limit and len(content) > limit:
                parts.append(content[:limit] + f"\n... ({len(content)} chars total)")
            else:
                parts.append(content)
        elif content:
            text = str(content)
            if limit and len(text) > limit:
                parts.append(text[:limit] + f"\n... ({len(text)} chars total)")
            else:
                parts.append(text)
    parts.append(f"\n{'=' * 60} END LLM INPUT ({len(messages)} messages) {'=' * 60}")
    logger.debug("\n".join(parts))


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
    format_context: Callable[..., str] | None = None,
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
    model_plain = create_chat_model(
        model=config.model,
        temperature=config.temperature,
        model_config=model_config,
    )
    model_with_tools = model_plain.bind_tools(tools)

    # ------------------------------------------------------------------
    # Config values captured by closures
    # ------------------------------------------------------------------
    system_prompt_template: str = config.prompts.get("system", "")
    max_rounds: int = config.max_rounds
    compression_cfg = config.compression
    compression_hook = None
    if compression_cfg is not None and compression_cfg.enabled:
        from agentm.middleware.compression import build_compression_hook

        compression_hook = build_compression_hook(compression_cfg, model_config=model_config)

    # Structured output schema for synthesize node (optional)
    output_schema = None
    output_prompt_text = ""
    if config.output is not None:
        output_schema = get_output_schema(config.output.schema_name)
        output_prompt_text = load_prompt_template(config.output.prompt)

    # ------------------------------------------------------------------
    # Middleware pipeline (trajectory only — compression is handled
    # inside _prepare_messages because it needs the dynamic system prompt)
    # ------------------------------------------------------------------
    middlewares: list = []
    if trajectory is not None:
        middlewares.append(TrajectoryMiddleware(trajectory, ["orchestrator"]))
    pipeline = NodePipeline(middlewares) if middlewares else None

    # ------------------------------------------------------------------
    # Message preparation helpers
    # ------------------------------------------------------------------

    # Build the static system prompt once — it never changes across rounds.
    # Dynamic content (notebook/context, round info) goes into a user message
    # prepended to the history, enabling prompt caching on the system message.
    if system_prompt_template:
        _static_system_prompt: str = load_prompt_template(
            system_prompt_template,
            # Pass empty strings so templates render without errors.
            # The actual content is injected as a user message.
            notebook="",
            context="",
        )
    else:
        _static_system_prompt = "You are an agent orchestrator."

    _system_msg = SystemMessage(content=_static_system_prompt)

    def _build_context_message(state: dict, round_num: int) -> HumanMessage:
        """Build a user message with dynamic context: notebook/state + round info."""
        context_text = _format_context(state)

        parts: list[str] = []
        if context_text:
            parts.append(f"<current_state>\n{context_text}\n</current_state>")

        round_block = f"<round_context>\nRound: {round_num}/{max_rounds}\n"
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
        parts.append(round_block)

        # Nudge: prompt the LLM to evaluate progress and decide next action
        parts.append(
            "<next_action>\n"
            "Review the current state above. Then decide:\n"
            "1. Is there enough cross-validated evidence to confirm a root cause? "
            "If yes, finalize.\n"
            "2. Are there blind spots, unexplored leads, or untested hypotheses? "
            "If yes, dispatch the right agent type.\n"
            "3. Is the current hypothesis a SYMPTOM rather than a root cause? "
            "If yes, dig deeper with deep_analyze.\n"
            "</next_action>"
        )

        return HumanMessage(content="\n\n".join(parts))

    def _prepare_messages(state: dict, round_num: int) -> list[Any]:
        """Build message list for the LLM call.

        Layout (optimised for prompt caching and recency bias):
            [0]   SystemMessage   — static prompt (cacheable prefix)
            [1..N] history        — recent conversation messages
            [N+1] HumanMessage    — notebook + service profiles + round context

        The state snapshot is placed LAST so it sits closest to the
        generation cursor, maximising the LLM's attention on current
        investigation state.
        """
        context_msg = _build_context_message(state, round_num)
        history = list(state.get("messages", []))
        messages = [_system_msg, *history, context_msg]

        if compression_hook is not None:
            hook_out = compression_hook({"messages": messages})
            messages = hook_out.get("llm_input_messages") or hook_out.get(
                "messages", messages
            )

        if logger.isEnabledFor(logging.DEBUG):
            _debug_log_messages(messages, round_num)

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

        # Pre-model pipeline: trajectory(llm_start)
        if pipeline is not None:
            pipeline.before({"messages": llm_messages})

        response = await model_with_tools.ainvoke(llm_messages)

        # Post-model pipeline: trajectory(tool_call / llm_end)
        if pipeline is not None:
            await pipeline.after({"messages": messages}, response)

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
            """Create a placeholder dict from Pydantic schema fields.

            Recursively expands nested BaseModel types so the LLM sees
            the full structure (e.g. ``list[ComponentMapping]`` becomes
            ``[{component_name: ..., service_name: ...}]``).
            """
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
                elif (
                    ann is not None
                    and hasattr(ann, "__origin__")
                    and ann.__origin__ is list
                ):
                    args = getattr(ann, "__args__", ())
                    if (
                        args
                        and isinstance(args[0], type)
                        and issubclass(args[0], BaseModel)
                    ):
                        example[name] = [_schema_to_example(args[0])]
                    else:
                        example[name] = []
                elif isinstance(ann, type) and issubclass(ann, BaseModel):
                    example[name] = _schema_to_example(ann)
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
                logger.info(
                    "synthesize attempt %d/%d succeeded, keys=%s",
                    attempt + 1,
                    1 + _SYNTH_MAX_RETRIES,
                    list(structured.keys()) if isinstance(structured, dict) else type(structured).__name__,
                )
                break  # success
            except Exception as exc:
                raw_json = _extract_raw_from_error(exc)
                if attempt < _SYNTH_MAX_RETRIES:
                    logger.warning(
                        "synthesize attempt %d/%d failed (%s), "
                        "retrying with error feedback. raw_json_preview=%.300s",
                        attempt + 1,
                        1 + _SYNTH_MAX_RETRIES,
                        exc,
                        raw_json,
                    )
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
                    logger.error(
                        "synthesize FAILED all %d attempts, last error: %s. "
                        "Falling back to plain LLM. raw_json_preview=%.500s",
                        attempt + 1,
                        exc,
                        raw_json,
                    )
                    try:
                        raw = await model_plain.ainvoke(attempt_messages)
                        raw_text = str(getattr(raw, "content", raw))
                        logger.warning(
                            "synthesize fallback produced raw_text (len=%d), "
                            "preview=%.300s",
                            len(raw_text),
                            raw_text,
                        )
                    except Exception as exc2:
                        logger.error(
                            "synthesize fallback also FAILED: %s", exc2,
                        )
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

    builder: StateGraph = StateGraph(state_schema)

    builder.add_node("llm_call", llm_call)  # type: ignore[type-var]
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
