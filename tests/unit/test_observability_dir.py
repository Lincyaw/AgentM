"""Tests for AGENTM_OBSERVABILITY_DIR env var support."""

import os
from pathlib import Path
from unittest import mock

from agentm.core.lib.observability_dir import resolve_observability_dir


def test_resolve_default():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("AGENTM_OBSERVABILITY_DIR", None)
        result = resolve_observability_dir("/some/cwd")
        assert result == Path("/some/cwd/.agentm/observability")


def test_resolve_env_override():
    with mock.patch.dict(
        os.environ, {"AGENTM_OBSERVABILITY_DIR": "/pvc/traces/repo/issue-42"}
    ):
        result = resolve_observability_dir("/some/cwd")
        assert result == Path("/pvc/traces/repo/issue-42")


def test_resolve_env_override_ignores_cwd():
    with mock.patch.dict(os.environ, {"AGENTM_OBSERVABILITY_DIR": "/pvc/traces"}):
        result1 = resolve_observability_dir("/cwd1")
        result2 = resolve_observability_dir("/cwd2")
        assert result1 == result2 == Path("/pvc/traces")


def test_resolve_none_cwd_uses_process_cwd():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("AGENTM_OBSERVABILITY_DIR", None)
        result = resolve_observability_dir(None)
        assert result == Path.cwd() / ".agentm" / "observability"
