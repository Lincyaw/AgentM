"""RCA scenario atoms — §11 compliant extensions for manifest composition.

Shared (used by baseline + multi-agent + harness):
    duckdb_sql          SQL query tool over parquet observability data
    finalize            submit_final_report orchestrator termination
    prompt_loader       loads prompt .md files into system prompt
    rcabench_contract   injects rcabench-platform agent contract

Multi-agent additions:
    hypothesis_tools    hypothesis CRUD via artifact_store (requires artifact_store)
    worker_finalize     return_response worker termination (also used by HFSM)

Overlays (orthogonal augmentations, enabled via manifest includes):
    runtime_context     surfaces AGENTM_RCA_DATA_DIR in system prompt
    worker_skills       contributes skills/ directory to skill_loader
"""
