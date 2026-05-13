"""Unit tests for the install-time selection rule in ``operations_agent_env``.

Verifies that the atom picks the right ARL session class based on which of
``image`` / ``pool_ref`` is configured, without touching a live cluster.

The test mocks ``arl.ManagedSession`` and ``arl.SandboxSession`` and asserts:

- ``image`` only        → ``ManagedSession`` (managed-pool path, the default).
- ``pool_ref`` only     → ``SandboxSession`` (legacy pre-created pool path).
- both ``image`` + ``pool_ref`` → ``ManagedSession`` wins.
- neither               → ``RuntimeError`` with a clear message.

A stub ``arl`` module is installed into ``sys.modules`` so the deferred import
inside ``install`` resolves to our fakes regardless of whether the real
``arl-env`` SDK is present in the environment.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest


# --- shared fake harness ---------------------------------------------------


class _FakeSession:
    """Captures construction args and records lifecycle calls."""

    def __init__(self, kind: str, **kwargs: Any) -> None:
        self.kind = kind
        self.kwargs = kwargs
        self.created = False
        self.deleted = False
        self.closed = False

    def create_sandbox(self) -> None:
        self.created = True

    def execute(self, _steps: Any) -> Any:  # pragma: no cover - not exercised
        raise AssertionError("execute should not be called during install")

    def delete_sandbox(self) -> None:
        self.deleted = True

    def close(self) -> None:
        self.closed = True


def _install_fake_arl(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[_FakeSession]]:
    """Install a fake ``arl`` module exposing ManagedSession + SandboxSession.

    Returns a dict whose lists are appended to every time the corresponding
    fake class is instantiated, so tests can introspect what install() chose.
    """
    instances: dict[str, list[_FakeSession]] = {"managed": [], "sandbox": []}

    def _make_managed(**kwargs: Any) -> _FakeSession:
        s = _FakeSession("managed", **kwargs)
        instances["managed"].append(s)
        return s

    def _make_sandbox(**kwargs: Any) -> _FakeSession:
        s = _FakeSession("sandbox", **kwargs)
        instances["sandbox"].append(s)
        return s

    fake_arl = types.ModuleType("arl")
    fake_arl.ManagedSession = _make_managed  # type: ignore[attr-defined]
    fake_arl.SandboxSession = _make_sandbox  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "arl", fake_arl)
    return instances


class _FakeAPI:
    """Minimal ExtensionAPI stand-in: records register/on calls."""

    def __init__(self) -> None:
        self.ops: dict[str, Any] | None = None
        self.writer: Any = None
        self.handlers: list[tuple[str, Any]] = []

    def register_operations(self, *, file: Any, bash: Any) -> None:
        self.ops = {"file": file, "bash": bash}

    def register_resource_writer(self, writer: Any) -> None:
        self.writer = writer

    def on(self, channel: str, handler: Any) -> None:
        self.handlers.append((channel, handler))


def _load_atom() -> Any:
    """Import the atom module fresh so the test sees current source."""
    # The contrib/extensions package is on sys.path via the project's pytest
    # config; importing by dotted name keeps mypy/ruff happy.
    return importlib.import_module(
        "contrib.extensions.operations_agent_env"
    )


# Drop env vars that could otherwise leak into _resolve(). Applied to every
# test so we don't accidentally pick up the operator's shell defaults.
_ATOM_ENV = (
    "AGENTM_AGENT_ENV_IMAGE",
    "AGENTM_AGENT_ENV_EXPERIMENT_ID",
    "AGENTM_AGENT_ENV_POOL_REF",
    "AGENTM_AGENT_ENV_GATEWAY_URL",
    "AGENTM_AGENT_ENV_NAMESPACE",
)


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ATOM_ENV:
        monkeypatch.delenv(name, raising=False)


# --- tests -----------------------------------------------------------------


def test_image_only_picks_managed_session(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake_arl(monkeypatch)
    atom = _load_atom()
    api = _FakeAPI()

    atom.install(
        api,
        {
            "image": "arl-executor-agent:latest",
            "experiment_id": "test-exp",
            "namespace": "ns-x",
            "gateway_url": "http://gw.example:8080",
            "work_dir": "/work",
        },
    )

    assert len(instances["managed"]) == 1
    assert len(instances["sandbox"]) == 0
    session = instances["managed"][0]
    assert session.created is True
    assert session.kwargs["image"] == "arl-executor-agent:latest"
    assert session.kwargs["experiment_id"] == "test-exp"
    assert session.kwargs["namespace"] == "ns-x"
    assert session.kwargs["gateway_url"] == "http://gw.example:8080"
    assert session.kwargs["workspace_dir"] == "/work"
    assert api.ops is not None
    assert api.writer is not None


def test_image_only_defaults_experiment_id(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake_arl(monkeypatch)
    atom = _load_atom()
    api = _FakeAPI()

    atom.install(api, {"image": "img:1"})

    session = instances["managed"][0]
    assert session.kwargs["experiment_id"] == "agentm-default"
    assert session.kwargs["namespace"] == "default"
    assert session.kwargs["gateway_url"] == "http://localhost:8080"
    assert session.kwargs["workspace_dir"] == "/workspace"


def test_pool_ref_only_picks_sandbox_session(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake_arl(monkeypatch)
    atom = _load_atom()
    api = _FakeAPI()

    atom.install(
        api,
        {
            "pool_ref": "python-39-std",
            "namespace": "ns-y",
            "idle_timeout_seconds": 300,
        },
    )

    assert len(instances["sandbox"]) == 1
    assert len(instances["managed"]) == 0
    session = instances["sandbox"][0]
    assert session.created is True
    assert session.kwargs["pool_ref"] == "python-39-std"
    assert session.kwargs["namespace"] == "ns-y"
    assert session.kwargs["idle_timeout_seconds"] == 300
    # legacy path always asks the gateway not to keep the pod warm post-run
    assert session.kwargs["keep_alive"] is False


def test_image_wins_over_pool_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake_arl(monkeypatch)
    atom = _load_atom()
    api = _FakeAPI()

    atom.install(
        api,
        {"image": "img:1", "pool_ref": "legacy-pool"},
    )

    assert len(instances["managed"]) == 1
    assert len(instances["sandbox"]) == 0
    # pool_ref is silently ignored; managed session args do not carry it
    assert "pool_ref" not in instances["managed"][0].kwargs


def test_neither_image_nor_pool_ref_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_arl(monkeypatch)
    atom = _load_atom()
    api = _FakeAPI()

    with pytest.raises(RuntimeError) as excinfo:
        atom.install(api, {})

    msg = str(excinfo.value)
    assert "image" in msg
    assert "pool_ref" in msg


def test_env_vars_drive_managed_default(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake_arl(monkeypatch)
    monkeypatch.setenv("AGENTM_AGENT_ENV_IMAGE", "img-from-env:2")
    monkeypatch.setenv("AGENTM_AGENT_ENV_EXPERIMENT_ID", "exp-from-env")
    atom = _load_atom()
    api = _FakeAPI()

    atom.install(api, {})

    assert len(instances["managed"]) == 1
    session = instances["managed"][0]
    assert session.kwargs["image"] == "img-from-env:2"
    assert session.kwargs["experiment_id"] == "exp-from-env"
