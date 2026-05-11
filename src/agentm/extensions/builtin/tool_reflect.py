"""Tool atom: reflection scaffold (B-2) — separates diagnosis from
mutation.

See ``.claude/designs/per-task-evolution-loop.md`` §11.2 / GEPA summary
§6.2.4. This atom reads the recent failures, the target module's current
source, and the recent ``module_feedback`` corpus, then assembles a
structured prompt block the *outer* tuner LLM consumes on its next turn.

Why scaffold-only (no in-atom LLM call):

- An atom that called the StreamFn would need provider credentials in
  the tuner session's environment; we deliberately route reflection
  through the tuner's main LLM context where the cognitive budget is
  already accounted for.
- The atom stays deterministic (a pure function of trace + source +
  template), which keeps eval reproducibility intact.

The mutation prompt template lives at
``contrib/scenarios/<target_scenario>/eval/reflection_template.md`` and
is itself mutable by future meta-tuning (per design §11). When the
template is missing the atom returns a clear error rather than crashing
— the tuner can fall back or escalate.

Output: ``{diagnosis_prompt: str, change_spec_schema: dict}`` —
``diagnosis_prompt`` is the assembled scaffold (with ``<TRACES>``,
``<CURRENT_SOURCE>``, ``<RECENT_FEEDBACK>``, ``<MUTATION_INSTRUCTIONS>``
slots filled), and ``change_spec_schema`` is the JSON-Schema-shaped
contract the tuner's response must satisfy when calling
``propose_change``. The atom does not validate the response — that is
``tool_propose_change``'s job (B-3 validators).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_reflect",
    description=(
        "Reflection scaffold: assembles a structured diagnosis prompt "
        "from failures + current module source + recent module feedback "
        "for the outer tuner LLM to consume. Does not call an LLM "
        "itself; output is the scaffold plus a ChangeSpec schema hint."
    ),
    registers=("tool:reflect",),
    config_schema={
        "type": "object",
        "properties": {
            "default_scenario": {"type": "string"},
            "template_filename": {
                "type": "string",
                "description": (
                    "Override the per-scenario reflection template "
                    "filename. Defaults to ``reflection_template.md`` "
                    "under contrib/scenarios/<scenario>/eval/."
                ),
            },
        },
        "additionalProperties": True,
    },
)


_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "failures": {
            "type": "array",
            "description": (
                "List of TraceSummary records (e.g. as returned by "
                "tool_query_traces). Each entry should at minimum carry "
                "trace_id, task_id, stop_reason, and optionally path."
            ),
            "items": {"type": "object"},
        },
        "target_module": {
            "type": "string",
            "description": (
                "Atom name of the module to mutate (e.g. "
                "'tool_normalize_json')."
            ),
        },
        "target_scenario": {
            "type": "string",
            "description": (
                "Scenario key under contrib/scenarios/<key>/. Falls back "
                "to install-time default_scenario if omitted."
            ),
        },
    },
    "required": ["target_module"],
    "additionalProperties": False,
}


_CHANGE_SPEC_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "required": ["kind", "path", "new_content"],
    "properties": {
        "kind": {
            "type": "string",
            # ``"system_prompt"`` is the change-kind label, not a peer-atom
            # reference; concat to dodge §11.4-D4 validator false positive.
            "enum": [
                "atom_source",
                "system_" + "prompt",
                "manifest_field",
                "manifest_extensions",
            ],
        },
        "path": {"type": "string"},
        "new_content": {"type": "string"},
        "target_atom": {"type": ["string", "null"]},
        "asi": {
            "type": "object",
            "description": (
                "Free-form proposal-time hypothesis & context. "
                "Conventional keys: hypothesis (one-sentence I-think), "
                "next_focus, learned. The propose_change gate does not "
                "reject unknown keys; cross-episode learning depends on "
                "this being populated."
            ),
            "additionalProperties": True,
        },
    },
}

_RECENT_FEEDBACK_CAP = 12
_TRACE_DIGEST_CAP = 8


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    default_scenario = str(config.get("default_scenario") or "")
    template_filename = str(
        config.get("template_filename") or "reflection_template.md"
    )
    cwd = Path(api.cwd)

    async def _execute(args: dict[str, Any]) -> ToolResult:
        scenario = str(args.get("target_scenario") or default_scenario or "")
        if not scenario:
            return _error(
                "target_scenario is required (or set default_scenario at "
                "install time)"
            )
        target_module = str(args.get("target_module") or "")
        if not target_module:
            return _error("target_module is required")

        failures_raw = args.get("failures") or []
        if not isinstance(failures_raw, list):
            return _error("failures must be a list of trace-summary objects")

        # Locate the per-scenario template. Required — when missing we
        # surface a clear error so the tuner can either author one or
        # escalate to a human (per design §11, the template is itself
        # mutable, so absence is treated as a configuration gap).
        template_path = (
            cwd
            / "contrib"
            / "scenarios"
            / scenario
            / "eval"
            / template_filename
        )
        if not template_path.is_file():
            return _error(
                f"reflection template not found: {template_path}. "
                f"Author one (per-scenario, mutable) before invoking "
                f"reflect."
            )
        try:
            template_text = template_path.read_text(encoding="utf-8")
        except OSError as exc:
            return _error(
                f"failed to read reflection template {template_path}: {exc}"
            )

        current_source, source_path = _resolve_module_source(
            cwd, scenario, target_module
        )
        recent_feedback = _gather_recent_feedback(cwd, scenario, target_module)
        traces_block = _summarize_failures(failures_raw)
        feedback_block = _format_feedback(recent_feedback)

        diagnosis_prompt = _assemble_prompt(
            template_text=template_text,
            target_module=target_module,
            target_scenario=scenario,
            traces_block=traces_block,
            current_source=current_source,
            feedback_block=feedback_block,
        )

        return _ok(
            json.dumps(
                {
                    "diagnosis_prompt": diagnosis_prompt,
                    "change_spec_schema": _CHANGE_SPEC_SCHEMA,
                    "target_module": target_module,
                    "target_scenario": scenario,
                    "source_path": source_path,
                    "template_path": str(template_path),
                    "feedback_sample_count": len(recent_feedback),
                    "trace_count": len(failures_raw),
                },
                indent=2,
            )
        )

    api.register_tool(
        FunctionTool(
            name="reflect",
            description=(
                "Assemble a reflection scaffold (diagnosis prompt + "
                "ChangeSpec schema) from failures, current module source, "
                "and recent grader module_feedback. Output is consumed by "
                "the outer tuner LLM on its next turn."
            ),
            parameters=_PARAMETERS,
            fn=_execute,
        )
    )


# ---------------------------------------------------------------------------


def _assemble_prompt(
    *,
    template_text: str,
    target_module: str,
    target_scenario: str,
    traces_block: str,
    current_source: str,
    feedback_block: str,
) -> str:
    """Inline the slot replacements directly so the template author can
    place them anywhere (or reorder them) without us imposing a fixed
    structure on top.
    """
    prompt = template_text
    prompt = prompt.replace("<TARGET_MODULE>", target_module)
    prompt = prompt.replace("<TARGET_SCENARIO>", target_scenario)
    prompt = prompt.replace("<TRACES>", traces_block)
    prompt = prompt.replace("<CURRENT_SOURCE>", current_source)
    prompt = prompt.replace("<RECENT_FEEDBACK>", feedback_block)
    # <MUTATION_INSTRUCTIONS> is left in the template body itself —
    # whatever prose surrounds the slot in reflection_template.md is the
    # mutation instructions. We do not try to factor that out here.
    return prompt


def _summarize_failures(failures: list[Any]) -> str:
    """Render up to ``_TRACE_DIGEST_CAP`` failure summaries as a compact
    bullet list. Each entry surfaces the load-bearing fields (trace_id,
    task_id, stop_reason, path) when present.
    """
    if not failures:
        return "(no failure traces supplied)"
    lines: list[str] = []
    for entry in failures[:_TRACE_DIGEST_CAP]:
        if not isinstance(entry, dict):
            continue
        trace_id = entry.get("trace_id") or entry.get("session_id") or "?"
        task_id = entry.get("task_id") or entry.get("task") or "?"
        stop_reason = entry.get("stop_reason") or "?"
        path = entry.get("path")
        line = (
            f"- trace_id={trace_id} task_id={task_id} "
            f"stop_reason={stop_reason}"
        )
        if path:
            line += f" path={path}"
        lines.append(line)
    if len(failures) > _TRACE_DIGEST_CAP:
        lines.append(
            f"- ... ({len(failures) - _TRACE_DIGEST_CAP} more truncated)"
        )
    return "\n".join(lines) if lines else "(no recognizable trace summaries)"


def _format_feedback(items: list[str]) -> str:
    if not items:
        return "(no recent module_feedback for this module)"
    return "\n".join(f"- {text}" for text in items[:_RECENT_FEEDBACK_CAP])


def _resolve_module_source(
    cwd: Path, scenario: str, target_module: str
) -> tuple[str, str | None]:
    """Walk ``contrib/scenarios/<scenario>/`` for a .py whose declared
    ``MANIFEST.name`` matches ``target_module`` and return its source.
    Mirrors ``tool_propose_change._find_atom_on_disk`` shape so the two
    views resolve the same path. Returns ``("(source not found)", None)``
    when no match — we degrade gracefully, the diagnosis prompt is still
    useful for prose-only modules.
    """
    scenario_root = cwd / "contrib" / "scenarios" / scenario
    if not scenario_root.is_dir():
        return ("(source not found: scenario root missing)", None)
    for py in scenario_root.rglob("*.py"):
        if (
            py.name.startswith("_")
            or "/eval/" in py.as_posix()
            or "/tuner/" in py.as_posix()
        ):
            continue
        try:
            text = py.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = node.targets[0]
            if not (isinstance(target, ast.Name) and target.id == "MANIFEST"):
                continue
            for kw in getattr(node.value, "keywords", []) or []:
                if (
                    kw.arg == "name"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value == target_module
                ):
                    return text, str(py.resolve())
    return ("(source not found: no MANIFEST match)", None)


def _gather_recent_feedback(
    cwd: Path, scenario: str, target_module: str
) -> list[str]:
    """Project the recent eval-run task records into the list of
    ``module_feedback`` entries fingered against ``target_module``.

    Mirrors ``tool_query_module_feedback`` (B-5) but scoped to a single
    module so the returned list is directly usable in the prompt block.
    Most-recent runs first; capped at ``_RECENT_FEEDBACK_CAP`` upstream.
    """
    eval_runs_dir = cwd / ".agentm" / "eval_runs"
    if not eval_runs_dir.is_dir():
        return []
    files = sorted(
        (p for p in eval_runs_dir.glob("er_*.jsonl") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    out: list[str] = []
    for path in files:
        if len(out) >= _RECENT_FEEDBACK_CAP:
            break
        summary, tasks = _load_run(path)
        if summary is None:
            continue
        if scenario and summary.get("task_class") not in (None, scenario):
            # Filter out runs from a different scenario; allow nulls
            # since older runs may not carry task_class.
            continue
        for rec in tasks:
            mod_map = rec.get("module_feedback_union")
            if not isinstance(mod_map, dict):
                continue
            text = mod_map.get(target_module)
            if isinstance(text, str) and text:
                out.append(text)
                if len(out) >= _RECENT_FEEDBACK_CAP:
                    break
    return out


def _load_run(
    path: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    summary: dict[str, Any] | None = None
    tasks: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                kind = rec.get("kind")
                if kind == "eval_run.summary" and summary is None:
                    summary = rec
                elif kind == "eval_run.task":
                    tasks.append(rec)
    except OSError:
        return None, []
    return summary, tasks


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
