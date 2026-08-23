from agentm.core.abi.context import turn_to_messages
from agentm.core.abi.messages import AssistantMessage, TextContent
from agentm.core.abi.termination import SignalAborted
from agentm.core.abi.trajectory import Outcome, Turn
from agentm.core.abi.trigger import TriggerMetadata, UserInput
from agentm.core.lib.trajectory_nodes import turn_to_nodes


def test_node_projection_matches_turn_replay_with_trigger_metadata() -> None:
    turn = Turn(
        index=0,
        id="interrupted-turn",
        run_id="run-1",
        run_step=0,
        trigger=UserInput(content=(TextContent(type="text", text="original task"),)),
        response=AssistantMessage(
            role="assistant",
            content=(TextContent(type="text", text="partial work"),),
            timestamp=1.0,
            stop_reason="end_turn",
        ),
        tool_results=(),
        outcome=Outcome(cause=SignalAborted(reason="submit_interrupt")),
        timestamp=2.0,
        trigger_metadata=TriggerMetadata(
            priority="now",
            origin="human",
            mode="interrupt",
            meta={"request_id": "interrupt-1"},
        ),
    )

    nodes = turn_to_nodes(turn, session_id="session-1")

    assert [node.message for node in nodes] == turn_to_messages(turn)
    assert {node.run_id for node in nodes} == {turn.run_id}
    assert {node.run_step for node in nodes} == {turn.run_step}
