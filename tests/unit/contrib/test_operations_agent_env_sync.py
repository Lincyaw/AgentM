"""Unit tests for ``operations_agent_env``'s opt-in cwd↔sandbox workspace sync.

These pin the host-side half of the contract that lets a dispatcher (workbuddy)
hand the agent a real repo and recover its diff WITHOUT putting VCS credentials
in the sandbox:

  - ``_seed_sandbox_from_host``: host git HEAD → tar.gz → upload → pod baseline.
  - ``_sync_sandbox_to_host``: pod ``git diff --binary`` (base64) → ``git apply``
    onto the host work tree; empty diff is a no-op.

The ARL session and the gateway upload are faked; the real ``git archive`` /
``git apply`` / base64 plumbing runs against throwaway repos.
"""

from __future__ import annotations

import base64
import importlib.util
import subprocess
from pathlib import Path

import pytest

_MOD_PATH = (
    Path(__file__).resolve().parents[3]
    / "contrib"
    / "extensions"
    / "operations_agent_env.py"
)


def _load_atom():
    spec = importlib.util.spec_from_file_location("_oae_under_test", _MOD_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


oae = _load_atom()


class _Output:
    def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class _Result:
    def __init__(self, output: _Output) -> None:
        self.output = output


class _Response:
    def __init__(self, results: list[_Result]) -> None:
        self.results = results


class _FakeSession:
    """Records execute() calls; returns a scripted Output per call."""

    def __init__(self, session_id: str = "sess-1", outputs=None) -> None:
        self.session_id = session_id
        self.calls: list[str] = []
        self._outputs = list(outputs or [])

    def execute(self, steps):
        cmd = steps[0]["command"]
        self.calls.append(" ".join(cmd))
        out = self._outputs.pop(0) if self._outputs else _Output()
        return _Response([_Result(out)])


def _git(repo: Path, *args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], input=stdin, capture_output=True, check=True
    )


def _init_repo(path: Path, files: dict[str, str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@t")
    _git(path, "config", "user.name", "t")
    for rel, content in files.items():
        (path / rel).write_text(content)
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "base")


def _pod_patch_from_changes(tmp: Path, base: dict[str, str], changes: dict[str, str | None]) -> bytes:
    """Build a realistic ``git diff --cached --binary`` patch vs an identical base.

    ``changes`` maps path -> new content, or ``None`` to delete.
    """
    sim = tmp / "podsim"
    _init_repo(sim, base)
    for rel, content in changes.items():
        if content is None:
            (sim / rel).unlink()
        else:
            (sim / rel).write_text(content)
    _git(sim, "add", "-A")
    out = subprocess.run(
        ["git", "-C", str(sim), "diff", "--cached", "--binary"],
        capture_output=True, check=True,
    )
    return out.stdout


def test_seed_uploads_archive_and_baselines(tmp_path, monkeypatch):
    host = tmp_path / "host"
    _init_repo(host, {"a.txt": "hello\n"})

    uploaded: dict[str, bytes] = {}

    def _fake_upload(gateway_url, session_id, rel_path, payload):
        uploaded["url"] = gateway_url
        uploaded["sid"] = session_id
        uploaded["path"] = rel_path
        uploaded["payload"] = payload

    monkeypatch.setattr(oae, "_upload_to_pod", _fake_upload)

    sess = _FakeSession(outputs=[_Output(exit_code=0)])
    oae._seed_sandbox_from_host(sess, "http://gw:8080", str(host), "/workspace")

    # Archive was produced (non-empty gzip) and uploaded to the seed path.
    assert uploaded["path"] == oae._SEED_ARCHIVE_NAME
    assert uploaded["payload"][:2] == b"\x1f\x8b"  # gzip magic
    assert uploaded["sid"] == "sess-1"
    # The pod baseline command was issued and tags the baseline.
    assert any("git init" in c and "wb-baseline" in c and "tag" in c for c in sess.calls)
    # AgentM's host-side droppings are excluded from the dispatcher's commit.
    exclude = (host / ".git" / "info" / "exclude").read_text()
    assert ".agentm/" in exclude


def test_seed_requires_git_worktree(tmp_path):
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    with pytest.raises(RuntimeError, match="git work tree"):
        oae._seed_sandbox_from_host(_FakeSession(), "http://gw", str(not_a_repo), "/workspace")


def test_sync_back_applies_pod_diff(tmp_path):
    base = {"a.txt": "hello\n"}
    host = tmp_path / "host"
    _init_repo(host, base)

    # Pod produced: modify a.txt and add b.txt.
    patch = _pod_patch_from_changes(tmp_path, base, {"a.txt": "hello world\n", "b.txt": "new\n"})
    encoded = base64.b64encode(patch).decode("ascii")
    sess = _FakeSession(outputs=[_Output(stdout=encoded)])

    oae._sync_sandbox_to_host(sess, str(host), "/workspace")

    assert (host / "a.txt").read_text() == "hello world\n"
    assert (host / "b.txt").read_text() == "new\n"


def test_sync_back_empty_diff_is_noop(tmp_path):
    host = tmp_path / "host"
    _init_repo(host, {"a.txt": "hello\n"})
    sess = _FakeSession(outputs=[_Output(stdout="")])  # review agent: no changes

    oae._sync_sandbox_to_host(sess, str(host), "/workspace")

    assert (host / "a.txt").read_text() == "hello\n"
    # working tree clean (nothing applied)
    status = subprocess.run(
        ["git", "-C", str(host), "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    assert status.stdout.strip() == ""
