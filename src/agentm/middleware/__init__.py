"""AgentM middleware system.

Provides a uniform base class for pre-/post-model hooks and a
``compose_middleware`` function that chains multiple middleware
instances into a single ``pre_model_hook`` compatible with
``create_react_agent``.

Existing closure-based hooks (compression, budget, dedup, trajectory,
instruction injection) are re-exported from their respective
submodules as middleware classes.
"""

from __future__ import annotations

from typing import Any, Callable


class AgentMMiddleware:
    """Base class for AgentM middleware.

    Subclasses override one or more of the hook methods.  All methods
    receive the current state dict and return a (possibly modified)
    state dict, or ``None`` to signal "no change".

    The ``to_pre_model_hook()`` helper converts the middleware into a
    plain callable matching ``create_react_agent``'s ``pre_model_hook``
    signature.
    """

    def before_model(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """Called before each LLM invocation.

        Return a new state dict to modify the input, or ``None`` to
        pass through unchanged.
        """
        return None

    async def abefore_model(
        self, state: dict[str, Any], runtime: Any = None
    ) -> dict[str, Any] | None:
        """Async variant of ``before_model``.

        Default delegates to the sync version.
        """
        return self.before_model(state)

    async def aafter_model(
        self, state: dict[str, Any], runtime: Any = None
    ) -> dict[str, Any] | None:
        """Called after each LLM invocation.

        Not wired into ``create_react_agent`` by default — available
        for custom graph implementations.
        """
        return None

    async def awrap_tool_call(
        self, request: Any, handler: Callable[..., Any]
    ) -> Any:
        """Wrap a tool invocation.

        Default calls the handler unchanged.
        """
        return await handler(request)

    def to_pre_model_hook(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Convert this middleware to a ``pre_model_hook`` callable.

        The returned function calls ``before_model`` and returns
        the modified state, or passes through if the hook returns None.
        """

        def hook(state: dict[str, Any]) -> dict[str, Any]:
            result = self.before_model(state)
            if result is not None:
                return result
            # Pass through — preserve llm_input_messages if present
            if "llm_input_messages" in state:
                return {"llm_input_messages": state["llm_input_messages"]}
            return {"messages": state.get("messages", [])}

        return hook


def compose_middleware(
    middlewares: list[AgentMMiddleware],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Chain multiple middleware instances into a single ``pre_model_hook``.

    Middlewares are applied in order: the output of one feeds the input
    of the next.  This is the recommended way to build a hook pipeline
    for ``create_react_agent``.
    """
    hooks = [m.to_pre_model_hook() for m in middlewares]

    def chained(state: dict[str, Any]) -> dict[str, Any]:
        result = state
        for hook in hooks:
            result = hook(result)
        return result

    return chained
