"""
tests/test_graph.py
===================
Unit tests for ``src.agents.graph``.

All tests use a fully-mocked LLM and stub tools — no API keys, no
network. The fixtures here drive the LangGraph through every path the
production specialist will hit:

* Plain final answer (no tool calls).
* One tool call → result → final answer.
* Unknown tool name → graceful error message stays in the loop.
* Hard ``max_iterations`` cap (no infinite loops).
* Qwen3 ``<think>...</think>`` block stripping.
* Groq XML tool call recovery on LLM exception.
* ``messages_to_step_dicts`` emits the exact step dict shape ``app.py``
  reads.
"""
from __future__ import annotations

from typing import Any, List

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from src.agents.graph import (
    _strip_think,
    build_specialist_graph,
    messages_to_step_dicts,
)


# ---------------------------------------------------------------------------
# Stub tools (multi-tenant safe — no SKUs, no agent IDs hardcoded)
# ---------------------------------------------------------------------------


@tool
def fake_query(search_term: str) -> str:
    """Stub query tool returning a canned row."""
    return f"Found row: KEY={search_term}, value=42"


@tool
def fake_propose(
    row_key: str, target_column: str, delta: int, justification: str
) -> str:
    """Stub propose tool returning a canned proposal acknowledgement."""
    return f"PROPOSAL_OK row={row_key} col={target_column} delta={delta}"


# ---------------------------------------------------------------------------
# Mock LLM — scripts a list of AIMessage responses by call index
# ---------------------------------------------------------------------------


class FakeLLM:
    """Bind-tools-aware mock LLM. Returns scripted responses in order."""

    def __init__(self, scripted: List[AIMessage]):
        self._scripted = list(scripted)
        self._call_idx = 0
        self.invocations: List[List[BaseMessage]] = []

    def bind_tools(self, _tools):  # noqa: D401
        """``bind_tools`` is a no-op for the mock (returns self)."""
        return self

    def invoke(self, messages):
        # Capture so tests can assert the LLM was actually called.
        self.invocations.append(list(messages))
        if self._call_idx >= len(self._scripted):
            return AIMessage(content="default final")
        resp = self._scripted[self._call_idx]
        self._call_idx += 1
        return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_strip_think_blocks():
    """``<think>`` block stripping mirrors the legacy loop."""
    assert _strip_think("<think>internal</think>visible") == "visible"
    assert _strip_think("clean text") == "clean text"
    assert _strip_think("") == ""
    assert _strip_think(None) == ""
    # Multi-line + nested-ish
    assert _strip_think(
        "<think>line1\nline2</think>after"
    ) == "after"


def test_graph_terminates_on_final_message():
    """LLM returns plain text → graph stops after one ``agent`` step."""
    llm = FakeLLM([AIMessage(content="Done — no tools needed")])
    graph = build_specialist_graph(
        llm, [fake_query, fake_propose], max_iterations=5
    )
    final = graph.invoke({
        "messages": [HumanMessage(content="hello")],
        "iteration_count": 0,
        "max_iterations": 5,
        "agent_id": "maker",
    })
    assert final.get("final_answer") == "Done — no tools needed"
    assert final["iteration_count"] == 1


def test_graph_executes_tool_then_terminates():
    """One tool call followed by a final answer — full happy path."""
    llm = FakeLLM([
        AIMessage(
            content="Looking up.",
            tool_calls=[{
                "id": "t1",
                "name": "fake_query",
                "args": {"search_term": "WIDGET"},
            }],
        ),
        AIMessage(content="Got it. Done."),
    ])
    graph = build_specialist_graph(
        llm, [fake_query, fake_propose], max_iterations=5
    )
    final = graph.invoke({
        "messages": [HumanMessage(content="check WIDGET")],
        "iteration_count": 0,
        "max_iterations": 5,
        "agent_id": "maker",
    })
    msgs = final["messages"]
    # Expect at least: Human, AI(tool call), Tool, AI(final).
    assert any(
        isinstance(m, ToolMessage) and "Found row" in m.content
        for m in msgs
    ), f"ToolMessage not found in: {[type(m).__name__ for m in msgs]}"
    assert final.get("final_answer") == "Got it. Done."


def test_graph_handles_unknown_tool():
    """Unknown tool name surfaces as a graceful error in ToolMessage."""
    llm = FakeLLM([
        AIMessage(
            content="",
            tool_calls=[{
                "id": "t1",
                "name": "nope",
                "args": {},
            }],
        ),
        AIMessage(content="ok"),
    ])
    graph = build_specialist_graph(
        llm, [fake_query, fake_propose], max_iterations=5
    )
    final = graph.invoke({
        "messages": [HumanMessage(content="x")],
        "iteration_count": 0,
        "max_iterations": 5,
        "agent_id": "maker",
    })
    msgs = final["messages"]
    assert any(
        isinstance(m, ToolMessage) and "Unknown tool" in m.content
        for m in msgs
    )


def test_graph_respects_max_iterations():
    """Forever-tool-calling LLM is stopped by the iteration cap.

    Uses a custom LLM that emits unique ``tool_call_id`` per call so
    ``add_messages`` does not dedupe consecutive identical calls.
    """

    class ForeverLLM:
        def __init__(self):
            self._i = 0

        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            msg = AIMessage(
                content="",
                tool_calls=[{
                    "id": f"t{self._i}",
                    "name": "fake_query",
                    "args": {"search_term": "x"},
                }],
            )
            self._i += 1
            return msg

    llm = ForeverLLM()
    graph = build_specialist_graph(
        llm, [fake_query, fake_propose], max_iterations=3
    )
    # langgraph has its own recursion limit (default 25). Bump it so we
    # observe our own ``max_iterations`` safety-stop fire instead.
    final = graph.invoke(
        {
            "messages": [HumanMessage(content="x")],
            "iteration_count": 0,
            "max_iterations": 3,
            "agent_id": "maker",
        },
        config={"recursion_limit": 50},
    )
    # The agent_node should have stopped at iteration_count >= 3 (the
    # cap, plus one additional turn that emitted the stop message).
    assert final["iteration_count"] >= 3
    # And the safety stop message is the final assistant message.
    final_msg = [
        m for m in final["messages"] if isinstance(m, AIMessage)
    ][-1]
    assert "Maximum iterations reached" in (final_msg.content or "")


def test_groq_xml_recovery_kicks_in():
    """LLM raises XML-formatted error → graph parses and dispatches the tool."""

    class RaisingLLM:
        def __init__(self):
            self._calls = 0

        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            self._calls += 1
            if self._calls == 1:
                # Match the format parse_groq_xml_tool_call recognises:
                # <function=name>{json}</function>
                raise RuntimeError(
                    'Failed to parse: '
                    '<function=fake_query>{"search_term": "WIDGET"}</function>'
                )
            # Second call after recovery: terminate cleanly.
            return AIMessage(content="recovered & done")

    llm = RaisingLLM()
    graph = build_specialist_graph(
        llm, [fake_query, fake_propose], max_iterations=5
    )
    final = graph.invoke({
        "messages": [HumanMessage(content="x")],
        "iteration_count": 0,
        "max_iterations": 5,
        "agent_id": "maker",
    })
    msgs = final["messages"]
    # Recovery should have produced a ToolMessage with the canned result.
    assert any(
        isinstance(m, ToolMessage) and "Found row" in m.content
        for m in msgs
    )
    assert final.get("final_answer") == "recovered & done"


def test_messages_to_step_dicts_emits_expected_shape():
    """Diff translation matches ``app.py::_render_chat_bubble`` keys."""
    before: list[BaseMessage] = [HumanMessage(content="x")]
    after: list[BaseMessage] = before + [
        AIMessage(
            content="thinking",
            tool_calls=[{
                "id": "t1",
                "name": "fake_query",
                "args": {"search_term": "x"},
            }],
        ),
        ToolMessage(
            content="result",
            tool_call_id="t1",
            name="fake_query",
        ),
        AIMessage(content="final"),
    ]
    steps = messages_to_step_dicts(before, after, "maker", iteration=1)
    types = [s["type"] for s in steps]
    assert types == ["tool_call", "tool_result", "final_answer"]
    # tool_call shape
    assert steps[0]["tool"] == "fake_query"
    assert "search_term" in steps[0]["content"]
    # tool_result shape
    assert steps[1]["tool"] == "fake_query"
    assert steps[1]["content"] == "result"
    # final_answer shape — agent_id required by the renderer
    assert steps[2]["content"] == "final"
    assert steps[2]["agent_id"] == "maker"


def test_messages_to_step_dicts_strips_think_in_final_answer():
    """Final answer text has ``<think>`` blocks stripped."""
    before: list[BaseMessage] = [HumanMessage(content="x")]
    after: list[BaseMessage] = before + [
        AIMessage(content="<think>secret reasoning</think>visible answer"),
    ]
    steps = messages_to_step_dicts(before, after, "keeper", iteration=1)
    assert len(steps) == 1
    assert steps[0]["type"] == "final_answer"
    assert "secret reasoning" not in steps[0]["content"]
    assert "visible answer" in steps[0]["content"]


def test_messages_to_step_dicts_empty_diff():
    """No new messages → no new steps."""
    msgs: list[BaseMessage] = [HumanMessage(content="x")]
    assert messages_to_step_dicts(msgs, msgs, "mover", iteration=0) == []


def test_messages_to_step_dicts_multiple_tool_calls_in_one_message():
    """An AIMessage with N tool calls produces N tool_call step dicts."""
    before: list[BaseMessage] = [HumanMessage(content="x")]
    after: list[BaseMessage] = before + [
        AIMessage(
            content="",
            tool_calls=[
                {"id": "t1", "name": "fake_query", "args": {"search_term": "A"}},
                {"id": "t2", "name": "fake_query", "args": {"search_term": "B"}},
            ],
        ),
    ]
    steps = messages_to_step_dicts(before, after, "maker", iteration=0)
    assert [s["type"] for s in steps] == ["tool_call", "tool_call"]
    assert steps[0]["tool"] == "fake_query"
    assert steps[1]["tool"] == "fake_query"


def test_graph_streams_incrementally():
    """``.stream(stream_mode='values')`` yields snapshots after each node."""
    llm = FakeLLM([
        AIMessage(
            content="step1",
            tool_calls=[{
                "id": "t1",
                "name": "fake_query",
                "args": {"search_term": "x"},
            }],
        ),
        AIMessage(content="step2 final"),
    ])
    graph = build_specialist_graph(
        llm, [fake_query, fake_propose], max_iterations=5
    )
    snapshots = list(graph.stream(
        {
            "messages": [HumanMessage(content="x")],
            "iteration_count": 0,
            "max_iterations": 5,
            "agent_id": "maker",
        },
        stream_mode="values",
    ))
    # At least: input snapshot, after-agent-1, after-tools, after-agent-2.
    assert len(snapshots) >= 3
    final = snapshots[-1]
    assert final.get("final_answer") == "step2 final"
