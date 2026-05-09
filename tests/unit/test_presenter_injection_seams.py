from __future__ import annotations

from pathlib import Path

from textual.binding import Binding

from agentm.harness import AgentSessionConfig
from agentm.modes.textual_app import AgentMApp


class _FakeSession:
    model = None
    tool_renderers: dict[str, object] = {}

    def find_tool(self, name: str) -> None:
        del name
        return None

    def get_service(self, name: str) -> None:
        del name
        return None


def test_textual_app_accepts_css_path_and_keymap(tmp_path: Path) -> None:
    css_path = tmp_path / "custom.tcss"
    css_path.write_text("Screen { align: center middle; }\n")
    keymap = [Binding("ctrl+x", "force_quit", "Quit", show=True)]

    app = AgentMApp(
        AgentSessionConfig(cwd=str(tmp_path), provider=("fake", {})),
        session=_FakeSession(),
        css_path=css_path,
        keymap=keymap,
    )

    assert app.css_path == [css_path]
    assert [binding.action for _, binding in app._bindings] == ["force_quit"]
