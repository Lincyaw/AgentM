"""Internal helper: ``@harness_tool`` — Pydantic-backed FunctionTool factory.

Why this exists. Hand-written JSON schema constants drifted from the runtime
validation that followed them (the auditor's V2 ``submit_verdict`` schema
duplicated every field description across the constant and the
``RawVerdictOutput`` coercer). Pydantic v2's ``model_json_schema()`` is the
single source of truth for both — the decorator pipes the model's JSON schema
into ``FunctionTool.parameters`` after normalising it (inline ``$defs``,
strip ``title``).

Schema normalisation is delegated to
:func:`agentm.core.lib.pydantic_to_tool_schema`. Provider-specific
constraints (e.g. OpenAI strict mode: ``additionalProperties: false``
on every object, every property in ``required``) are applied by the
provider adapter layer, not at schema-generation time.

Two-axis contract preserved:

* schema parity — the parameters dict emitted here is OpenAI strict-mode
  ready, drift-free against the runtime ``model_validate`` coercer.
* termination sentinel — terminal tools (the auditor's ``submit_verdict``)
  carry ``metadata={"terminates": True}`` on the returned :class:`FunctionTool`.
  The merged ``atom.py`` reads this to compute the termination-reason mapping
  without a separate registry.

Leading underscore = internal; not exported via ``llmharness.__init__`` and
not part of any §11 atom file (atoms call this from their module body but
the helper itself is not an atom).
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.tool import ToolOutcome
from agentm.core.lib import pydantic_to_tool_schema
from pydantic import BaseModel, ValidationError

ArgsT = TypeVar("ArgsT", bound=BaseModel)
_HandlerFn = Callable[[ArgsT, Any], Awaitable[ToolResult | ToolOutcome]]


def harness_tool(
    name: str,
    *,
    terminates: bool = False,
    description: str | None = None,
) -> Callable[[_HandlerFn[ArgsT]], FunctionTool]:
    """Decorator factory: wrap a Pydantic-typed handler into a ``FunctionTool``.

    The decorated function must have the signature
    ``async def fn(args: SomeArgsModel, ctx) -> ToolResult | ToolTerminate``
    where ``SomeArgsModel`` is a :class:`pydantic.BaseModel` with
    ``model_config = ConfigDict(extra="forbid")``.

    **Docstring-first description.** Following the Pydantic-AI / LangChain
    ``@tool`` convention, the tool description comes from the handler's
    docstring by default. Pass ``description=`` only as an escape hatch.
    Resolution order:

    1. ``description=`` kwarg if given;
    2. else ``inspect.cleandoc(fn.__doc__)`` if non-empty;
    3. else :class:`ValueError` at decoration time — never silently register
       a tool with empty description (providers reject those anyway).

    On invocation the wrapper:

    * calls ``Model.model_validate(args)`` on the raw ``args`` dict;
    * on :class:`pydantic.ValidationError` returns ``ToolResult(is_error=True,
      ...)`` carrying the pydantic error message — the auditor LLM sees a
      visible retry prompt rather than a kernel-level crash;
    * on success forwards the parsed model into the user handler.

    ``terminates=True`` is reflected in the returned tool's ``metadata`` so
    the merged ``atom.py`` can map tool name → termination reason without a
    separate side-table.
    """

    def _decorate(handler: _HandlerFn[ArgsT]) -> FunctionTool:
        resolved_description = description
        if resolved_description is None:
            raw_doc = handler.__doc__
            if raw_doc and raw_doc.strip():
                resolved_description = inspect.cleandoc(raw_doc)
        if not resolved_description:
            raise ValueError(
                f"harness_tool({name!r}): provide description= or a docstring "
                "on the handler — empty descriptions are rejected by LLM providers"
            )
        # Resolve forward-ref annotations (handlers use ``from __future__ import
        # annotations`` so ``handler.__annotations__`` is a dict of strings).
        try:
            hints = typing.get_type_hints(handler)
        except Exception:  # pragma: no cover — fall back to raw annotations
            hints = dict(handler.__annotations__)
        # The first non-return annotation is the args model.
        model_cls: type[BaseModel] | None = None
        for key, val in hints.items():
            if key == "return":
                continue
            if isinstance(val, type) and issubclass(val, BaseModel):
                model_cls = val
                break
        if model_cls is None:
            raise TypeError(
                f"@harness_tool({name!r}): handler must annotate its first "
                "argument with a pydantic.BaseModel subclass"
            )

        parameters = pydantic_to_tool_schema(model_cls)

        async def _wrapped(args: dict[str, Any]) -> ToolResult | ToolOutcome:
            try:
                parsed = model_cls.model_validate(args)
            except ValidationError as exc:
                return ToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"{name} rejected: {exc}",
                        )
                    ],
                    is_error=True,
                )
            # ``ctx`` is reserved for future use (e.g. cancellation, span);
            # today we pass ``None`` so handler signatures already match.
            return await handler(parsed, None)  # type: ignore[arg-type]

        return FunctionTool(
            name=name,
            description=resolved_description,
            parameters=parameters,
            fn=_wrapped,
            metadata={"terminates": True} if terminates else {},
        )

    return _decorate


__all__ = ["harness_tool"]
