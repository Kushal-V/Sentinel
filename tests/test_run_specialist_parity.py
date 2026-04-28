"""
tests/test_run_specialist_parity.py
===================================
Phase 4b parity tests for the LangGraph migration.

Goal: drive ``AgentOrchestrator.run_specialist`` with both
``USE_LANGGRAPH=False`` (legacy manual ReAct loop) and ``USE_LANGGRAPH=True``
(new LangGraph ``StateGraph`` path) using IDENTICAL scripted LLM
responses, then assert that:

1. The TOOL CALL SEQUENCE matches (same tools invoked, in the same
   order, with the same args).
2. The FINAL ANSWER text matches.
3. The full normalised step sequence matches when stripped of
   incidental fields (``iteration``).

Phase 4a only asserted (1) and (2) — Phase 4b expands this to a
5-scenario suite covering the corner cases the production code is
expected to handle uniformly across both paths:

* Single tool call → final
* Two sequential tool calls → final
* Final answer with no tool calls
* Forever-tool-calling LLM hits ``max_iterations``
* Unknown tool name surfaces as a ``tool_result`` error in BOTH paths
"""
from __future__ import annotations

import importlib
import os
from typing import List

import pytest
from langchain_core.messages import AIMessage


# ---------------------------------------------------------------------------
# Shared fixtures: a scriptable LLM and a stub data manager
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """LLM mock with a fixed list of scripted responses.

    Mimics ``ChatGroq`` enough for both the legacy loop and the
    LangGraph path: ``bind_tools`` returns self, ``invoke(messages)``
    pops the next scripted ``AIMessage`` (or a default final once the
    script is exhausted, so a runaway loop terminates instead of
    raising).
    """

    def __init__(self, scripted: List[AIMessage]):
        self._scripted = list(scripted)
        self._idx = 0

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, _schema):  # noqa: D401
        return self

    def invoke(self, _messages):
        if self._idx >= len(self._scripted):
            return AIMessage(content="default final")
        resp = self._scripted[self._idx]
        self._idx += 1
        return resp


class ForeverToolLLM:
    """LLM mock that never produces a final answer — only tool calls.

    Each call yields a unique ``id`` so ``add_messages`` does not dedupe
    consecutive identical AIMessages on the LangGraph path.
    """

    def __init__(self, tool_name: str = "get_dataset_schema"):
        self._i = 0
        self._tool_name = tool_name

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, _schema):  # noqa: D401
        return self

    def invoke(self, _messages):
        msg = AIMessage(
            content="",
            tool_calls=[{
                "id": f"forever_{self._i}",
                "name": self._tool_name,
                "args": {},
            }],
        )
        self._i += 1
        return msg


def _build_orchestrator(workspaces_root):
    """Build an orchestrator backed by a tmp workspace.

    Args:
        workspaces_root: ``pathlib.Path`` to a fresh temp directory the
            ``FactoryDataManager`` will use as its workspaces root.

    Returns:
        ``(orchestrator, data_manager)``.
    """
    import src.core.config as cfg
    import src.core.state_manager as sm
    from src.core.state_manager import FactoryDataManager
    from src.tools.tool_registry import set_data_manager

    cfg.WORKSPACES_DIR = workspaces_root
    sm.WORKSPACES_DIR = workspaces_root

    dm = FactoryDataManager(workspace="parity_test")
    set_data_manager(dm)

    os.environ.setdefault("GROQ_API_KEY", "test-key")

    from src.agents.orchestrator import AgentOrchestrator

    orch = AgentOrchestrator(dm)
    return orch, dm


def _run_specialist_with_flag(
    use_graph: bool,
    llm,
    workspaces_root,
) -> List[dict]:
    """Run ``run_specialist`` with the flag set and return yielded steps."""
    os.environ["SENTINEL_USE_LANGGRAPH"] = "true" if use_graph else "false"
    from src.core import config as cfg
    importlib.reload(cfg)

    orch, _ = _build_orchestrator(workspaces_root)
    orch._specialist_llm = llm
    orch._agent_graphs = orch._build_agent_graphs()

    from src.agents.orchestrator import CrisisEvent, DispatchRoute

    route = DispatchRoute(
        selected_agent="maker",
        delegation_justification="parity test",
        urgency_tier="LOW",
        required_lookups=[],
    )
    crisis = CrisisEvent(
        event_id="EVT-PARITY",
        event_type="test",
        severity="LOW",
        description="parity smoke",
    )
    return list(orch.run_specialist(route, crisis, chat_history=[]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_call_sequence(steps: List[dict]) -> List[tuple]:
    """Project (tool_name, args_str) tuples for each tool_call step."""
    return [
        (s.get("tool"), s.get("content"))
        for s in steps
        if s.get("type") == "tool_call"
    ]


def _final_answer(steps: List[dict]) -> str:
    """Extract the last final_answer step's content (or '' if none)."""
    finals = [s for s in steps if s.get("type") == "final_answer"]
    return finals[-1]["content"] if finals else ""


def _normalised_signature(steps: List[dict]) -> List[tuple]:
    """Reduce a step list to its essence: (type, tool, content) tuples.

    Drops keys (``iteration``, ``agent_id`` on non-final steps) that are
    not user-facing and may differ in incidental ways between the two
    code paths. Preserves the parts the UI renders.
    """
    sig: List[tuple] = []
    for s in steps:
        kind = s.get("type")
        if kind == "final_answer":
            sig.append((kind, None, s.get("content"), s.get("agent_id")))
        elif kind in ("tool_call", "tool_result"):
            sig.append((kind, s.get("tool"), s.get("content"), None))
        else:  # error or other
            sig.append((kind, None, s.get("content"), None))
    return sig


def _assert_parity(legacy_steps, graph_steps, *, expect_final: bool = True):
    """Assert tool-call sequence + final answer match across both paths."""
    legacy_calls = _tool_call_sequence(legacy_steps)
    graph_calls = _tool_call_sequence(graph_steps)
    assert legacy_calls == graph_calls, (
        f"Tool call sequences diverged.\n"
        f"  Legacy: {legacy_calls}\n"
        f"  Graph:  {graph_calls}\n"
    )

    if expect_final:
        legacy_final = _final_answer(legacy_steps)
        graph_final = _final_answer(graph_steps)
        assert legacy_final == graph_final, (
            f"Final answers diverged.\n"
            f"  Legacy: {legacy_final!r}\n"
            f"  Graph:  {graph_final!r}\n"
        )


# ---------------------------------------------------------------------------
# Scripted scenarios
# ---------------------------------------------------------------------------


def _scenario_single_tool() -> List[AIMessage]:
    """One tool call, then final answer."""
    return [
        AIMessage(
            content="",
            tool_calls=[{
                "id": "t1",
                "name": "get_dataset_schema",
                "args": {},
            }],
        ),
        AIMessage(content="Final answer: schema captured."),
    ]


def _scenario_two_sequential_tools() -> List[AIMessage]:
    """Two sequential tool calls (schema → query) → final answer."""
    return [
        AIMessage(
            content="",
            tool_calls=[{
                "id": "t1",
                "name": "get_dataset_schema",
                "args": {},
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "id": "t2",
                "name": "query_data",
                "args": {"search_term": "WIDGET"},
            }],
        ),
        AIMessage(content="Schema and row both retrieved."),
    ]


def _scenario_immediate_final() -> List[AIMessage]:
    """LLM returns final answer immediately, no tool calls."""
    return [
        AIMessage(content="No tools needed. Final answer."),
    ]


def _scenario_unknown_tool() -> List[AIMessage]:
    """LLM tries an unknown tool then closes."""
    return [
        AIMessage(
            content="",
            tool_calls=[{
                "id": "t1",
                "name": "definitely_not_a_real_tool",
                "args": {"foo": "bar"},
            }],
        ),
        AIMessage(content="Closing after error."),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scripted_factory,name",
    [
        (_scenario_single_tool, "single_tool"),
        (_scenario_two_sequential_tools, "two_sequential_tools"),
        (_scenario_immediate_final, "immediate_final"),
        (_scenario_unknown_tool, "unknown_tool"),
    ],
)
def test_parity_tool_sequence_and_final_answer(scripted_factory, name, tmp_path):
    """Tool call sequences and final answers match across the two paths."""
    legacy_root = tmp_path / f"legacy_{name}"
    graph_root = tmp_path / f"graph_{name}"
    legacy_root.mkdir()
    graph_root.mkdir()

    legacy_steps = _run_specialist_with_flag(
        use_graph=False,
        llm=ScriptedLLM(scripted_factory()),
        workspaces_root=legacy_root,
    )
    graph_steps = _run_specialist_with_flag(
        use_graph=True,
        llm=ScriptedLLM(scripted_factory()),
        workspaces_root=graph_root,
    )

    _assert_parity(legacy_steps, graph_steps)


def test_parity_unknown_tool_emits_error_observation(tmp_path):
    """Unknown tool name produces a ``tool_result`` carrying an error in BOTH paths."""
    legacy_root = tmp_path / "legacy_unknown"
    graph_root = tmp_path / "graph_unknown"
    legacy_root.mkdir()
    graph_root.mkdir()

    legacy_steps = _run_specialist_with_flag(
        use_graph=False,
        llm=ScriptedLLM(_scenario_unknown_tool()),
        workspaces_root=legacy_root,
    )
    graph_steps = _run_specialist_with_flag(
        use_graph=True,
        llm=ScriptedLLM(_scenario_unknown_tool()),
        workspaces_root=graph_root,
    )

    def _has_unknown_tool_error(steps):
        for s in steps:
            if s.get("type") == "tool_result" and "Unknown tool" in str(s.get("content", "")):
                return True
        return False

    assert _has_unknown_tool_error(legacy_steps), (
        f"Legacy path did not surface unknown-tool error. Steps: {legacy_steps}"
    )
    assert _has_unknown_tool_error(graph_steps), (
        f"Graph path did not surface unknown-tool error. Steps: {graph_steps}"
    )


def test_parity_max_iterations_cap(tmp_path):
    """Forever-tool-calling LLM is stopped by the iteration cap on both paths.

    The user-visible contract is: a ``final_answer`` step is ALWAYS the
    last yield even if the LLM never produced a tool-call-free response.
    """
    legacy_root = tmp_path / "legacy_forever"
    graph_root = tmp_path / "graph_forever"
    legacy_root.mkdir()
    graph_root.mkdir()

    legacy_steps = _run_specialist_with_flag(
        use_graph=False,
        llm=ForeverToolLLM(tool_name="get_dataset_schema"),
        workspaces_root=legacy_root,
    )
    graph_steps = _run_specialist_with_flag(
        use_graph=True,
        llm=ForeverToolLLM(tool_name="get_dataset_schema"),
        workspaces_root=graph_root,
    )

    # Both paths must terminate (no infinite loop).
    assert legacy_steps, "legacy path produced no steps"
    assert graph_steps, "graph path produced no steps"

    # Both paths emit at least one final_answer to close out the conversation.
    legacy_finals = [s for s in legacy_steps if s.get("type") == "final_answer"]
    graph_finals = [s for s in graph_steps if s.get("type") == "final_answer"]
    assert legacy_finals, "legacy path never emitted final_answer"
    assert graph_finals, "graph path never emitted final_answer"


def test_parity_full_step_sequence_normalised(tmp_path):
    """Phase 4b: normalised step sequence matches across paths.

    Drops the ``iteration`` field that was the historical source of
    incidental divergence. The remaining tuple — ``(type, tool, content,
    agent_id_for_finals)`` — is exactly what the Streamlit UI consumes,
    so equality here means full user-visible parity.
    """
    legacy_root = tmp_path / "legacy_norm"
    graph_root = tmp_path / "graph_norm"
    legacy_root.mkdir()
    graph_root.mkdir()

    scripted = _scenario_two_sequential_tools()
    legacy_steps = _run_specialist_with_flag(
        use_graph=False,
        llm=ScriptedLLM(scripted),
        workspaces_root=legacy_root,
    )
    graph_steps = _run_specialist_with_flag(
        use_graph=True,
        llm=ScriptedLLM(_scenario_two_sequential_tools()),
        workspaces_root=graph_root,
    )

    legacy_sig = _normalised_signature(legacy_steps)
    graph_sig = _normalised_signature(graph_steps)
    assert legacy_sig == graph_sig, (
        f"Normalised step sequences diverged.\n"
        f"  Legacy: {legacy_sig}\n"
        f"  Graph:  {graph_sig}\n"
    )
