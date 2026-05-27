"""``system_prompt`` builtin atom — inline + file-backed prompt sources.

Pins the contract orchestrators (e.g. workbuddy) rely on: the system prompt
can be supplied inline (``prompt``) or staged on disk (``prompt_file``), the
file source wins when both are present, and an empty/unconfigured atom
contributes nothing (registers no handler) rather than prepending stray
separators.
"""

from __future__ import annotations

from typing import Any

from agentm.core.abi.events import BeforeAgentStartEvent
from agentm.extensions.builtin import system_prompt as sp


class _FakeAPI:
    def __init__(self) -> None:
        self.handlers: dict[str, list[Any]] = {}

    def on(self, channel: str, fn: Any, **_: Any) -> Any:
        self.handlers.setdefault(channel, []).append(fn)
        return lambda: None


class _Evt:
    def __init__(self, system: str = "") -> None:
        self.system = system


def test_resolve_prompt_inline() -> None:
    assert sp._resolve_prompt({"prompt": "hi"}) == "hi"


def test_resolve_prompt_from_file(tmp_path) -> None:
    p = tmp_path / "sp.md"
    p.write_text("FILE PROMPT", encoding="utf-8")
    assert sp._resolve_prompt({"prompt_file": str(p)}) == "FILE PROMPT"


def test_resolve_prompt_file_wins_over_inline(tmp_path) -> None:
    p = tmp_path / "sp.md"
    p.write_text("FROM FILE", encoding="utf-8")
    assert sp._resolve_prompt({"prompt": "inline", "prompt_file": str(p)}) == "FROM FILE"


def test_resolve_prompt_empty() -> None:
    assert sp._resolve_prompt({}) == ""


def test_install_empty_registers_no_handler() -> None:
    api = _FakeAPI()
    sp.install(api, {})  # type: ignore[arg-type]
    assert api.handlers == {}


def test_install_inline_prepends() -> None:
    api = _FakeAPI()
    sp.install(api, {"prompt": "ROLE"})  # type: ignore[arg-type]
    handlers = api.handlers[BeforeAgentStartEvent.CHANNEL]
    assert len(handlers) == 1
    evt = _Evt(system="BASE")
    handlers[0](evt)
    assert evt.system == "ROLE\n\nBASE"


def test_install_from_file_prepends(tmp_path) -> None:
    p = tmp_path / "sp.md"
    p.write_text("FILEROLE", encoding="utf-8")
    api = _FakeAPI()
    sp.install(api, {"prompt_file": str(p)})  # type: ignore[arg-type]
    evt = _Evt(system="")
    api.handlers[BeforeAgentStartEvent.CHANNEL][0](evt)
    assert evt.system == "FILEROLE"
