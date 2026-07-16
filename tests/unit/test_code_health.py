from __future__ import annotations

from pathlib import Path

from agentm.code_health import check_file


def _issues_for(tmp_path: Path, source: str, *, relative: str = "sample.py"):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return check_file(path)


def test_silent_exception_rule_uses_ast_not_comments(tmp_path: Path) -> None:
    issues = _issues_for(
        tmp_path,
        """
def run():
    try:
        work()
    except Exception:
        note = "logger.warning would be misleading here"
        return note
""",
    )

    assert [issue.rule for issue in issues].count("AM001") == 1


def test_silent_exception_rule_ignores_nested_handler_logging(
    tmp_path: Path,
) -> None:
    issues = _issues_for(
        tmp_path,
        """
def run():
    try:
        work()
    except Exception:
        try:
            cleanup()
        except Exception:
            logger.exception("cleanup failed")
""",
    )

    assert [issue.rule for issue in issues].count("AM001") == 1


def test_silent_exception_rule_accepts_returned_exception(
    tmp_path: Path,
) -> None:
    issues = _issues_for(
        tmp_path,
        """
def run():
    try:
        work()
    except Exception as exc:
        return {"error": str(exc)}
""",
    )

    assert not [issue for issue in issues if issue.rule == "AM001"]


def test_silent_exception_rule_accepts_forwarded_exception(
    tmp_path: Path,
) -> None:
    issues = _issues_for(
        tmp_path,
        """
def run(errors):
    try:
        work()
    except BaseException as exc:
        errors.append(exc)
""",
    )

    assert not [issue for issue in issues if issue.rule == "AM001"]


def test_event_source_drift_flags_stale_preflight_local(
    tmp_path: Path,
) -> None:
    issues = _issues_for(
        tmp_path,
        """
async def run(bus, stream, messages, model, tools, system):
    event = BeforeSendToLlmEvent(
        messages=messages,
        model=model,
        tools=tools,
        system=system,
    )
    await bus.emit(event.CHANNEL, event)
    await stream(
        messages=event.messages,
        model=event.model,
        tools=event.tools,
        system=system,
    )
""",
    )

    drift = [issue for issue in issues if issue.rule == "AM015"]
    assert len(drift) == 1
    assert "'system'" in drift[0].message


def test_event_source_drift_accepts_explicit_rebind(tmp_path: Path) -> None:
    issues = _issues_for(
        tmp_path,
        """
async def run(bus, stream, messages, model, tools, system):
    event = BeforeSendToLlmEvent(
        messages=messages,
        model=model,
        tools=tools,
        system=system,
    )
    await bus.emit(event.CHANNEL, event)
    messages = event.messages
    model = event.model
    tools = event.tools
    system = event.system
    await stream(
        messages=messages,
        model=model,
        tools=tools,
        system=system,
    )
""",
    )

    assert not [issue for issue in issues if issue.rule == "AM015"]


def test_event_source_drift_applies_to_every_mutable_event(
    tmp_path: Path,
) -> None:
    issues = _issues_for(
        tmp_path,
        """
async def run(bus, messages):
    event = ContextEvent(messages=messages)
    await bus.emit(event.CHANNEL, event)
    consume(messages)
""",
    )

    drift = [issue for issue in issues if issue.rule == "AM015"]
    assert len(drift) == 1
    assert "event.messages" in drift[0].message


def test_event_source_drift_flags_pre_dispatch_derived_values(
    tmp_path: Path,
) -> None:
    issues = _issues_for(
        tmp_path,
        """
async def run(bus, messages):
    preparation = prepare(messages)
    event = BeforeCompactEvent(messages=messages, reason="manual")
    await bus.emit(event.CHANNEL, event)
    compact(preparation)
""",
    )

    drift = [issue for issue in issues if issue.rule == "AM015"]
    assert len(drift) == 1
    assert "'preparation'" in drift[0].message


def test_mutable_events_cannot_be_constructed_inline(tmp_path: Path) -> None:
    issues = _issues_for(
        tmp_path,
        """
async def run(bus, messages):
    await bus.emit(
        ContextEvent.CHANNEL,
        ContextEvent(messages=messages),
    )
""",
    )

    drift = [issue for issue in issues if issue.rule == "AM015"]
    assert len(drift) == 1
    assert "assigned to a local" in drift[0].message


def test_hook_mutation_contract_requires_machine_readable_fields(
    tmp_path: Path,
) -> None:
    issues = _issues_for(
        tmp_path,
        """
class BrokenEvent(Event):
    HOOK = HookContract(mutation_contract="event.value may change")
    value: str
""",
    )

    integrity = [issue for issue in issues if issue.rule == "AM016"]
    assert len(integrity) == 1
    assert "no mutable_fields" in integrity[0].message


def test_current_agent_loop_has_no_event_source_drift() -> None:
    issues = check_file(Path("src/agentm/core/abi/loop.py"))

    assert not [issue for issue in issues if issue.rule == "AM015"]


def test_current_mutable_event_emitters_have_no_source_drift() -> None:
    paths = [
        Path("src/agentm/core/runtime/session.py"),
        Path("src/agentm/core/runtime/atom_reloader.py"),
        Path("src/agentm/extensions/builtin/llm_compaction.py"),
    ]

    issues = [issue for path in paths for issue in check_file(path)]

    assert not [issue for issue in issues if issue.rule == "AM015"]


def test_current_hook_contracts_are_machine_readable() -> None:
    issues = check_file(Path("src/agentm/core/abi/events.py"))

    assert not [issue for issue in issues if issue.rule == "AM016"]
