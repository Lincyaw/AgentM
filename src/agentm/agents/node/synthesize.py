"""Synthesize node factory for the node-based orchestrator.

Produces structured output (e.g. CausalGraph) when the orchestrator
emits ``<decision>finalize</decision>``.  If no output schema is
configured, the node is a no-op.

Extracted from ``orchestrator.py`` for separation of concerns.
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)


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


def create_synthesize_node(
    output_schema: type[BaseModel] | None,
    output_prompt_text: str,
    model_plain: Any,
    max_retries: int = 2,
) -> Callable[[dict], Awaitable[dict[str, Any]]]:
    """Build the synthesize node as an async function for StateGraph.

    Args:
        output_schema: Pydantic model class for structured output, or None
            to skip structured synthesis.
        output_prompt_text: System prompt text for the synthesize call.
        model_plain: Chat model instance (without tools bound).
        max_retries: Maximum retries on validation failure.

    Returns:
        An async function ``(state: dict) -> dict[str, Any]`` suitable for
        use as a LangGraph node.
    """
    synthesize_model = (
        model_plain.with_structured_output(output_schema)
        if output_schema is not None
        else None
    )

    async def synthesize(state: dict) -> dict[str, Any]:
        if synthesize_model is None:
            return {}

        messages = list(state.get("messages", []))
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]

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
        for attempt in range(1 + max_retries):
            try:
                result = await synthesize_model.ainvoke(attempt_messages)
                structured = (
                    result.model_dump() if hasattr(result, "model_dump") else result
                )
                logger.info(
                    "synthesize attempt %d/%d succeeded, keys=%s",
                    attempt + 1,
                    1 + max_retries,
                    list(structured.keys())
                    if isinstance(structured, dict)
                    else type(structured).__name__,
                )
                break  # success
            except Exception as exc:
                raw_json = _extract_raw_from_error(exc)
                if attempt < max_retries:
                    logger.warning(
                        "synthesize attempt %d/%d failed (%s), "
                        "retrying with error feedback. raw_json_preview=%.300s",
                        attempt + 1,
                        1 + max_retries,
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
                            "synthesize fallback also FAILED: %s",
                            exc2,
                        )
                        raw_text = ""
                    structured = {"raw_text": raw_text}

        return {"structured_response": structured}

    return synthesize
