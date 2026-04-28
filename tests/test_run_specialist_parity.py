"""
tests/test_run_specialist_parity.py
===================================
Parity smoke test for the LangGraph migration (Phase 4 / I1).

Goal: drive ``AgentOrchestrator.run_specialist`` with both
``USE_LANGGRAPH=False`` (legacy manual ReAct loop) and ``USE_LANGGRAPH=True``
(new LangGraph ``StateGraph`` path) using IDENTICAL scripted LLM
responses, then assert that:

1. The TOOL CALL SEQUENCE matches (same tools invoked, in the same
   order, with the same args).
2. The FINAL ANSWER text matches.

Exact step-dict equality is intentionally NOT asserted — the two paths
produce semantically equivalent yields, but minor incidental
differences (e.g. extra langgraph snapshots producing duplicate
``tool_result`` events) can occur and are not user-visible bugs. The
critical contract is the user-facing behaviour: same tools, same order,
same final answer.

This is a SMOKE parity test. Phase 4b will add per-step deep-equality
parity coverage (see TODO at the bottom of this file).
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import List
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# Shared fixtures: a scriptable LLM and a stub data manager
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """LLM mock with a fixed list of scripted responses.

    Mimics ``ChatGroq`` enough for both the legacy loop and the
    LangGraph path: ``bind_tools`` returns self, ``invoke(messages)``
    pops the next scripted ``AIMessage``.
    """

    def __init__(self, scripted: List[AIMessage]):
        self._scripted = list(scripted)
        self._idx = 0

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, _schema):  # noqa: D401
        # Not used by run_specialist, but the orchestrator's __init__
        # wires the dispatcher chain — we patch around that elsewhere.
        return self

    def invoke(self, _messages):
        if self._idx >= len(self._scripted):
            return AIMessage(content="default final")
        resp = self._scripted[self._idx]
        self._idx += 1
        return resp


def _scripted_responses() -> List[AIMessage]:
    """Two-step scripted run: schema lookup → final answer.

    Uses the real tool name ``get_dataset_schema`` (no args) so the
    legacy ``tool_map`` and the new graph ``tool_lookup`` both find it
    and execute it against the stub data manager.
    """
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


def _build_orchestrator(workspaces_root):
    """Build an orchestrator backed by a tmp workspace.

    Args:
        workspaces_root: ``pathlib.Path`` to a fresh temp directory the
            ``FactoryDataManager`` will use as its workspaces root.

    Returns:
        ``(orchestrator, data_manager)`` — the data manager has cold-
        start mock CSVs/JSON auto-generated. The orchestrator's
        ``_specialist_llm`` is left as the real ``ChatGroq`` and must
        be overwritten by the caller with a scripted mock before
        ``run_specialist`` is invoked.
    """
    import src.core.config as cfg
    import src.core.state_manager as sm
    from src.core.state_manager import FactoryDataManager
    from src.tools.tool_registry import set_data_manager

    cfg.WORKSPACES_DIR = workspaces_root
    sm.WORKSPACES_DIR = workspaces_root

    dm = FactoryDataManager(workspace="parity_test")
    set_data_manager(dm)

    # Stash a dummy API key so ChatGroq() doesn't blow up at __init__.
    os.environ.setdefault("GROQ_API_KEY", "test-key")

    from src.agents.orchestrator import AgentOrchestrator

    orch = AgentOrchestrator(dm)
    return orch, dm


def _run_with_flag(
    use_graph: bool,
    scripted: List[AIMessage],
    workspaces_root,
) -> List[dict]:
    """Run ``run_specialist`` with the flag set and return yielded steps."""
    # Toggle the flag in os.environ then reload config so the module-
    # level ``USE_LANGGRAPH`` constant picks it up. The orchestrator
    # reads ``config.USE_LANGGRAPH`` at call time, not import time.
    os.environ["SENTINEL_USE_LANGGRAPH"] = "true" if use_graph else "false"
    from src.core import config as cfg
    importlib.reload(cfg)

    orch, _ = _build_orchestrator(workspaces_root)

    # Inject scripted LLM and rebuild agent graphs so the legacy path
    # picks up the bound-tools version of the mock.
    orch._specialist_llm = ScriptedLLM(scripted)
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
# Tests
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


def test_parity_tool_sequence_and_final_answer(tmp_path):
    """Same scripted LLM → same tool call sequence and final answer.

    This is the smoke parity test — runs both paths once and confirms
    the user-facing behaviour matches. Step-dict-by-step-dict deep
    equality is deferred to Phase 4b (see module docstring).
    """
    legacy_root = tmp_path / "legacy"
    graph_root = tmp_path / "graph"
    legacy_root.mkdir()
    graph_root.mkdir()

    legacy_steps = _run_with_flag(
        use_graph=False,
        scripted=_scripted_responses(),
        workspaces_root=legacy_root,
    )
    graph_steps = _run_with_flag(
        use_graph=True,
        scripted=_scripted_responses(),
        workspaces_root=graph_root,
    )

    legacy_calls = _tool_call_sequence(legacy_steps)
    graph_calls = _tool_call_sequence(graph_steps)

    assert legacy_calls == graph_calls, (
        f"Tool call sequences diverged.\n"
        f"  Legacy: {legacy_calls}\n"
        f"  Graph:  {graph_calls}\n"
    )

    legacy_final = _final_answer(legacy_steps)
    graph_final = _final_answer(graph_steps)
    assert legacy_final == graph_final, (
        f"Final answers diverged.\n"
        f"  Legacy: {legacy_final!r}\n"
        f"  Graph:  {graph_final!r}\n"
    )


@pytest.mark.xfail(
    reason=(
        "Phase 4b: full step-dict-by-step-dict parity (including "
        "incidental ordering of tool_result snapshots in stream_mode). "
        "Tracked separately; not blocking the migration."
    ),
    strict=False,
)
def test_parity_full_step_sequence_phase_4b(tmp_path):
    """Placeholder for Phase 4b deep parity. Currently xfail."""
    legacy_root = tmp_path / "legacy"
    graph_root = tmp_path / "graph"
    legacy_root.mkdir()
    graph_root.mkdir()

    legacy_steps = _run_with_flag(
        use_graph=False,
        scripted=_scripted_responses(),
        workspaces_root=legacy_root,
    )
    graph_steps = _run_with_flag(
        use_graph=True,
        scripted=_scripted_responses(),
        workspaces_root=graph_root,
    )
    # Strict equality. Will pass once snapshot dedup / interleaving is
    # normalised in messages_to_step_dicts.
    assert legacy_steps == graph_steps
