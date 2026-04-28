"""
tests/test_graph_replay.py
==========================
F7 — checkpointer + scenario replay tests.

Covers:
* Backward compat: ``build_specialist_graph`` still compiles + runs when
  no ``checkpointer`` is passed.
* New behaviour: when a ``MemorySaver`` is passed, the compiled graph
  records state per-thread, and ``graph.get_state(config)`` returns the
  saved snapshot.
* Thread isolation: two thread IDs against the same checkpointer record
  independent message histories.

All tests use a fully-mocked LLM and stub tools — no Groq, no network.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from src.agents.graph import build_specialist_graph


# ---------------------------------------------------------------------------
# Fixtures: stub tool + stub LLM (multi-tenant — no SKUs, no agent IDs)
# ---------------------------------------------------------------------------


@tool
def fake_query(search_term: str) -> str:
    """Stub query tool — returns a canned row for any search term."""
    return f"row for {search_term}"


class FakeLLM:
    """Minimal LangChain-shaped chat model for graph tests.

    Returns a deterministic sequence of pre-baked ``AIMessage`` objects.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self._call = 0

    def bind_tools(self, tools):  # noqa: ARG002 — required by graph
        return self

    def invoke(self, messages):  # noqa: ARG002 — required by graph
        if self._call >= len(self._responses):
            return AIMessage(content="default")
        r = self._responses[self._call]
        self._call += 1
        return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_graph_compiles_with_checkpointer():
    """Passing a MemorySaver compiles successfully."""
    cp = MemorySaver()
    llm = FakeLLM([AIMessage(content="done")])
    g = build_specialist_graph(
        llm, [fake_query], max_iterations=3, checkpointer=cp
    )
    assert g is not None


def test_graph_runs_without_checkpointer_unchanged():
    """Existing call sites that don't pass checkpointer keep working."""
    llm = FakeLLM([AIMessage(content="done")])
    g = build_specialist_graph(llm, [fake_query], max_iterations=3)
    state = g.invoke({
        "messages": [HumanMessage(content="hi")],
        "iteration_count": 0,
        "max_iterations": 3,
        "agent_id": "test",
    })
    assert "final_answer" in state


def test_checkpointed_run_can_be_replayed_from_thread_id():
    """First run saves checkpoints; the saver retains the final state."""
    cp = MemorySaver()
    llm = FakeLLM([
        AIMessage(
            content="step1",
            tool_calls=[{
                "id": "t1",
                "name": "fake_query",
                "args": {"search_term": "X"},
            }],
        ),
        AIMessage(content="final"),
    ])
    g = build_specialist_graph(
        llm, [fake_query], max_iterations=5, checkpointer=cp
    )
    cfg = {"configurable": {"thread_id": "thread_1"}}

    final = g.invoke(
        {
            "messages": [HumanMessage(content="probe")],
            "iteration_count": 0,
            "max_iterations": 5,
            "agent_id": "maker",
        },
        config=cfg,
    )
    assert "final_answer" in final

    # Now retrieve the saved state via the checkpointer.
    state_after = g.get_state(cfg)
    assert state_after is not None
    msgs = state_after.values.get("messages", [])
    # checkpointer recorded: human probe + AI step1 + tool result + AI final
    assert len(msgs) >= 1


def test_checkpointer_isolates_threads():
    """Two thread IDs against the same saver keep independent histories."""
    cp = MemorySaver()
    llm1 = FakeLLM([AIMessage(content="thread1 final")])
    llm2 = FakeLLM([AIMessage(content="thread2 final")])
    g1 = build_specialist_graph(
        llm1, [fake_query], max_iterations=2, checkpointer=cp
    )
    g2 = build_specialist_graph(
        llm2, [fake_query], max_iterations=2, checkpointer=cp
    )

    g1.invoke(
        {
            "messages": [HumanMessage(content="run1")],
            "iteration_count": 0,
            "max_iterations": 2,
            "agent_id": "a",
        },
        config={"configurable": {"thread_id": "T1"}},
    )
    g2.invoke(
        {
            "messages": [HumanMessage(content="run2")],
            "iteration_count": 0,
            "max_iterations": 2,
            "agent_id": "a",
        },
        config={"configurable": {"thread_id": "T2"}},
    )

    s1 = g1.get_state({"configurable": {"thread_id": "T1"}})
    s2 = g2.get_state({"configurable": {"thread_id": "T2"}})
    # Both threads recorded their own runs.
    assert s1.values.get("agent_id") == "a"
    assert s2.values.get("agent_id") == "a"
    # But they hold different message histories — distinct human turns.
    assert s1.values["messages"][0].content == "run1"
    assert s2.values["messages"][0].content == "run2"


def test_unknown_thread_id_returns_empty_state():
    """A thread_id that was never run yields an empty / null state object.

    Documents the API for ``replay_crisis``: callers must guard against
    a missing checkpoint and surface a UI-friendly error.
    """
    cp = MemorySaver()
    llm = FakeLLM([AIMessage(content="done")])
    g = build_specialist_graph(
        llm, [fake_query], max_iterations=2, checkpointer=cp
    )
    state = g.get_state({"configurable": {"thread_id": "never-ran"}})
    # Either None, or a state with no messages — both indicate "no history".
    if state is not None:
        assert not state.values.get("messages")
