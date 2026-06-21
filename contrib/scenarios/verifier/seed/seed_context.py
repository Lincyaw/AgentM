"""Build the user prompt for seed verification agents."""
from __future__ import annotations

_FAULT_CONTEXT_TEMPLATE = "Injected fault: {fault_kind} on {target}"


def build_seed_prompt(
    *,
    target: str,
    fault_kind: str,
    params: str,
    fault_doc: str,
    judge_context: str = "",
) -> str:
    """Build the user prompt for verifying an injection target."""
    sections: list[str] = []

    # -- Fault reference document --
    if fault_doc:
        sections.append(f"## Fault reference document: {fault_kind}\n{fault_doc}")
    else:
        sections.append(f"## Fault reference document: {fault_kind}\n(no reference document available)")

    # -- Current injection --
    line = _FAULT_CONTEXT_TEMPLATE.format(fault_kind=fault_kind, target=target)
    if params:
        line += f" ({params})"
    sections.append("## Current injection\n" + line)

    # -- Task --
    task_lines = [
        f"Target: **{target}**",
        "Verify whether this injected fault actually took effect by comparing normal vs abnormal traces, metrics, logs, and caller/link behavior for the target.",
    ]
    if target.startswith("link:") and "->" in target:
        left, right = target.removeprefix("link:").split("->", 1)
        task_lines.append(
            f"Link endpoints: **{left}** and **{right}**. Establish which direction is actually exercised with normal parent-span joins; if the configured direction is both or unclear, check both `{left} -> {right}` and `{right} -> {left}`. Also compare the rule-bearing/source service's own outbound/client spans to the peer; the peer's server span may stay healthy for link delay/loss/partition faults. In the abnormal window, missing child spans across a partitioned link can be the expected fault signature, so verify caller-owned timeout/error/latency symptoms instead of treating missing calls as no effect."
        )
    sections.append("## Task\n" + "\n".join(task_lines))

    if judge_context:
        sections.append(
            "## Judge's global context\n"
            + judge_context
            + "\n\nRe-check the seed against this whole-graph/entry-service context. "
            "Do not change the verdict unless fresh SQL/log/metric evidence supports it."
        )

    return "\n\n".join(sections)
