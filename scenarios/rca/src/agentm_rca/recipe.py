"""Recipe builder for the RCA scenario package."""

from __future__ import annotations

from typing import Any

from agentm_rca.stores import HypothesisStore, ServiceProfileStore

_RCA_SYSTEM_PROMPT = """You are an RCA scout.
Use the available observability and RCA tools to map the incident, record
hypotheses, and keep shared service profiles up to date. Prefer concise,
factual findings over speculation."""


def build_recipe(*, data_dir: str, task_type: str = "scout") -> list[tuple[str, dict[str, Any]]]:
    """Return the extension list for an RCA session.

    The returned list is ready to pass directly to ``AgentSessionConfig.extensions``.
    Shared stores are created once here and injected into the RCA-specific
    extensions so the scenario stays entirely within the workspace package.
    """

    hypothesis_store = HypothesisStore()
    profile_store = ServiceProfileStore()

    return [
        ("agentm.extensions.builtin.tool_read", {}),
        ("agentm.extensions.builtin.tool_grep", {}),
        ("agentm.extensions.builtin.tool_find", {}),
        ("agentm.extensions.builtin.tool_ls", {}),
        ("agentm.extensions.builtin.prompt_templates", {}),
        ("agentm.extensions.builtin.skill_loader", {}),
        ("agentm.extensions.builtin.dedup", {"window": 8}),
        ("agentm.extensions.builtin.trajectory", {"path": "rca_trajectory.jsonl"}),
        (
            "agentm.extensions.builtin.system_prompt",
            {"prompt": _RCA_SYSTEM_PROMPT},
        ),
        ("agentm.extensions.builtin.permission", {"deny": ["bash", "edit", "write"]}),
        ("agentm_rca.tools", {"data_dir": data_dir}),
        ("agentm_rca.tools.hypothesis_tools", {"store": hypothesis_store}),
        ("agentm_rca.tools.profile_tools", {"store": profile_store}),
        (
            "agentm_rca.extensions.dynamic_context",
            {
                "hypothesis_store": hypothesis_store,
                "profile_store": profile_store,
            },
        ),
        ("agentm_rca.extensions.answer_schema", {"task_type": task_type}),
    ]
