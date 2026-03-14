"""AgentM middleware system.

Provides a uniform base class for pre-/post-model hooks and a
``compose_middleware`` function that chains multiple middleware
instances into a single ``pre_model_hook`` compatible with
``create_react_agent``.

For Node-mode graphs that control the LLM call directly,
``NodePipeline`` wraps the same middleware list and exposes
``before()`` / ``after()`` for explicit invocation inside the
``llm_call`` node.
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

        ``state`` contains the original messages plus a ``response`` key
        holding the LLM's ``AIMessage``.  Implementations should treat
        this as read-only observation (e.g. trajectory recording).

        Not wired into ``create_react_agent`` by default — used by
        ``NodePipeline`` for Node-mode graphs.
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


class NodePipeline:
    """Complete before + after middleware pipeline for Node-mode graphs.

    React-mode graphs use ``compose_middleware`` which returns a plain
    callable for the ``pre_model_hook`` parameter.  Node-mode graphs
    control the LLM call directly, so they need explicit ``before()``
    and ``after()`` entry points.

    Usage inside a Node ``llm_call`` node::

        pipeline = NodePipeline(middlewares)

        async def llm_call(state):
            prepared = pipeline.before(state)
            llm_msgs = prepared.get("llm_input_messages") or prepared["messages"]
            response = await model.ainvoke(llm_msgs)
            await pipeline.after(state, response)
            return {"messages": [response]}
    """

    def __init__(self, middlewares: list[AgentMMiddleware]) -> None:
        self._middlewares = middlewares
        self._pre_hook = compose_middleware(middlewares)

    def before(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run all ``before_model`` hooks in order.

        Returns the prepared state dict (may contain ``messages`` or
        ``llm_input_messages``).
        """
        return self._pre_hook(state)

    async def after(self, state: dict[str, Any], response: Any = None) -> None:
        """Run all ``aafter_model`` hooks in order.

        The ``response`` (typically an ``AIMessage``) is merged into
        the state dict under the ``response`` key so that middleware
        can inspect the LLM output.
        """
        merged = {**state}
        if response is not None:
            merged["response"] = response
        for mw in self._middlewares:
            await mw.aafter_model(merged)

