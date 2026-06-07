"""Acceptance #5 — ObservationLog memoisation (design §3.2).

Register a stub tool with ``metadata['idempotent'] = True`` whose
``execute`` increments a module-level counter on every REAL invocation.
Fire ``AgentStartEvent`` to trigger the cache atom's wrap, then call the
wrapped tool twice with identical args. Assert:

* The counter ended at 1 (second call hit the cache).
* The cached result text equals the first call's text.
* A third call with different args advances the counter to 2 (cache key
  is sensitive to args).
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentm.core.abi import AgentStartEvent, FunctionTool, ToolResult
from agentm.core.abi.messages import TextContent

from tests.hfsm._gate_fixtures import install_full_stack


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


class Counter:
    """Module-counter-shaped object so the stub tool's closure can mutate
    it without falling foul of the §11 mutable-globals rule (the test file
    is not an atom, so the rule does not apply — but a counter object is
    cleaner anyway).
    """

    value: int = 0


def _register_stub(api: Any, *, idempotent: bool, counter: Counter) -> str:
    async def _execute(args: dict[str, Any]) -> ToolResult:
        counter.value += 1
        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"real result for query={args.get('q', '')}, n={counter.value}",
                )
            ]
        )

    api.register_tool(
        FunctionTool(
            name="stub_idempotent" if idempotent else "stub_side_effect",
            description="test stub",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            fn=_execute,
            metadata={"idempotent": idempotent},
        )
    )
    return "stub_idempotent" if idempotent else "stub_side_effect"


def _fire_agent_start(api: Any) -> None:
    api.events.fire_handlers(
        AgentStartEvent.CHANNEL, AgentStartEvent(messages=[])
    )


def _get_tool(api: Any, name: str) -> Any:
    for t in api.tools:
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not registered")


def test_idempotent_tool_call_is_memoised() -> None:
    api, _, _ = install_full_stack()
    counter = Counter()
    name = _register_stub(api, idempotent=True, counter=counter)
    _fire_agent_start(api)
    tool = _get_tool(api, name)

    first = _run(tool.execute({"q": "alpha"}))
    second = _run(tool.execute({"q": "alpha"}))

    # The real tool ran exactly once; the second call returned the cached
    # observation rather than re-executing.
    assert counter.value == 1
    first_text = first.content[0].text  # type: ignore[attr-defined]
    second_text = second.content[0].text  # type: ignore[attr-defined]
    assert first_text == second_text


def test_cache_key_is_args_sensitive() -> None:
    api, _, _ = install_full_stack()
    counter = Counter()
    name = _register_stub(api, idempotent=True, counter=counter)
    _fire_agent_start(api)
    tool = _get_tool(api, name)

    _run(tool.execute({"q": "alpha"}))
    _run(tool.execute({"q": "alpha"}))  # cache hit
    _run(tool.execute({"q": "beta"}))  # different args — real call

    assert counter.value == 2


def test_canonicalisation_makes_key_order_independent() -> None:
    api, _, _ = install_full_stack()
    counter = Counter()
    # A stub whose params accept two keys so we can shuffle them.
    async def _execute(args: dict[str, Any]) -> ToolResult:
        counter.value += 1
        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"q={args.get('q')}/r={args.get('r')}",
                )
            ]
        )

    api.register_tool(
        FunctionTool(
            name="stub_pair",
            description="test",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}, "r": {"type": "string"}},
            },
            fn=_execute,
            metadata={"idempotent": True},
        )
    )
    _fire_agent_start(api)
    tool = _get_tool(api, "stub_pair")

    _run(tool.execute({"q": "alpha", "r": "beta"}))
    # Key order swapped — canonical_json sorts, so the signature matches.
    _run(tool.execute({"r": "beta", "q": "alpha"}))

    assert counter.value == 1
