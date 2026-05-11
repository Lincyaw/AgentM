from __future__ import annotations

from agentm.extensions.builtin.sub_agent import _resolve_inherited_extensions


def test_sub_agent_inherits_parent_config_by_manifest_name() -> None:
    resolved = _resolve_inherited_extensions(
        ["duckdb_sql", "tool_read"],
        {
            "duckdb_sql": {"module": "agentm_rca.tools.duckdb_sql"},
            "tool_read": {"module": "agentm.extensions.builtin.tool_read"},
        },
        {
            "duckdb_sql": {
                "exclude": ["conclusion.parquet"],
                "row_limit": 200,
                "token_limit": 5000,
            },
            "tool_read": {},
        },
    )

    assert resolved == [
        (
            "agentm_rca.tools.duckdb_sql",
            {
                "exclude": ["conclusion.parquet"],
                "row_limit": 200,
                "token_limit": 5000,
            },
        ),
        ("agentm.extensions.builtin.tool_read", {}),
    ]


