"""
src/agents/graph.py
===================
LangGraph-based specialist execution.

A ``StateGraph`` replacement for the manual ReAct loop in
``orchestrator.run_specialist``. Two nodes (``agent`` + ``tools``) wired
with conditional edges. Preserves all Sentinel guardrails — the
``ShadowSandbox``, the two-phase commit pipeline, and the trust-override
all live OUTSIDE this graph and are unaffected.

Why a graph?
------------
The legacy manual ReAct loop in ``orchestrator.run_specialist`` is a
single Python generator that:

* drives the LLM tool-calling loop,
* recovers from Groq's XML ``<function=...>`` tool-call bug,
* strips Qwen3 ``<think>...</think>`` reasoning blocks,
* yields step dicts to the Streamlit UI for live rendering.

That logic is also duplicated in ``answer_query`` and the
``ask_other_agent`` tool. A LangGraph ``StateGraph`` consolidates the
control flow, gives us native streaming, and unlocks future features
(checkpoints, parallel branches, sub-graphs).

Roll-out is gated behind ``config.USE_LANGGRAPH`` so production stays on
the manual loop until equivalence is proven in staging.

Public API
----------
``build_specialist_graph(llm, tools, max_iterations=10)``
    Compile a generic graph. Multi-tenant: any LLM, any tool list, any
    iteration cap. The graph state is a plain ``TypedDict``.

``messages_to_step_dicts(before, after, agent_id, iteration)``
    Diff two message lists and produce step dicts in the EXACT shape
    ``app.py`` already consumes from the legacy loop. Keys mirror the
    manual generator: ``type`` ∈ {tool_call, tool_result, final_answer,
    error}, ``tool``, ``content``, and ``agent_id`` for final answers.

The graph is invoked via ``.stream(state, stream_mode="values")`` to
match the generator-yielding contract that ``app.py`` expects from the
orchestrator.
"""
from __future__ import annotations

import logging
import re
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agents.groq_recovery import parse_groq_xml_tool_call

_LOG = logging.getLogger(__name__)

#: Strip Qwen3 internal reasoning blocks. Mirror the legacy loop exactly
#: so flag-on output matches flag-off output character-for-character on
#: the user-visible content stream.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class AgentState(TypedDict, total=False):
    """Generic specialist state.

    No tenant-specific fields. The ``agent_id`` and any other run-scoped
    metadata live here so the graph stays reusable across maker, mover,
    keeper, and any future specialist.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    iteration_count: int
    max_iterations: int
    agent_id: str
    final_answer: Optional[str]


def _strip_think(text: Any) -> str:
    """Remove ``<think>...</think>`` blocks the legacy loop strips.

    Accepts ``Any`` because some chat models return list-of-blocks
    content. Non-string inputs are coerced to ``str()`` before regex.
    """
    if not text:
        return "" if text is None else str(text)
    if not isinstance(text, str):
        return str(text)
    return _THINK_BLOCK_RE.sub("", text).strip()


def _extract_text_content(msg: AIMessage) -> str:
    """Extract user-visible text from an ``AIMessage``.

    Handles three shapes the legacy loop also handled:
    * plain ``str`` — strip ``<think>`` and return.
    * list of dict blocks — keep only ``type == "text"`` blocks.
    * list of strings — concatenate.
    """
    content = msg.content
    if isinstance(content, str):
        return _strip_think(content)
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    continue  # Drop internal reasoning blocks
                else:
                    parts.append(str(block))
            elif isinstance(block, str):
                parts.append(block)
        return _strip_think("\n".join(parts))
    return ""


def build_specialist_graph(
    llm: BaseChatModel,
    tools: List[BaseTool],
    max_iterations: int = 10,
):
    """Compile a generic specialist ``StateGraph``.

    Args:
        llm: A LangChain ``BaseChatModel`` (already configured with
            model + temperature). Must support ``.bind_tools(...)``.
        tools: List of ``@tool``-decorated callables. The graph builds a
            ``{name: tool}`` lookup once and reuses it for every node
            invocation.
        max_iterations: Hard cap on ``agent_node`` invocations. Matches
            the legacy loop's safety stop. Stored on the state so the
            agent node can short-circuit.

    Returns:
        A compiled ``langgraph`` graph. Use:

        * ``.invoke(state)`` for one-shot synchronous execution, or
        * ``.stream(state, stream_mode="values")`` for incremental
          state events the orchestrator translates back into step dicts.

    Multi-tenant: no SKU, agent ID, or column name is referenced here.
    The graph works for any specialist persona because the system prompt
    and tool list are caller-supplied.
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_lookup: Dict[str, BaseTool] = {t.name: t for t in tools}

    def agent_node(state: AgentState) -> Dict[str, Any]:
        """Invoke the LLM. Recover from Groq XML errors. Strip <think>."""
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", max_iterations)

        if iteration >= max_iter:
            stop_msg = AIMessage(
                content="Maximum iterations reached, stopping."
            )
            return {
                "messages": [stop_msg],
                "iteration_count": iteration + 1,
                "final_answer": stop_msg.content,
            }

        try:
            response = llm_with_tools.invoke(state["messages"])
        except Exception as exc:  # noqa: BLE001 — recovery layer
            recovered = parse_groq_xml_tool_call(str(exc))
            if recovered:
                # Synthesise an AIMessage carrying the recovered tool
                # call so the tools_node can dispatch it normally.
                response = AIMessage(
                    content="",
                    tool_calls=[{
                        "id": f"recovered_{iteration}",
                        "name": recovered["tool_name"],
                        "args": recovered.get("arguments", {}) or {},
                    }],
                )
                _LOG.warning(
                    "Recovered Groq XML tool call: %s",
                    recovered["tool_name"],
                )
            else:
                _LOG.exception(
                    "LLM invocation failed without recoverable tool call"
                )
                response = AIMessage(content=f"LLM error: {exc}")

        # Normalise <think> blocks on string content. List content is
        # left untouched (extraction happens at step-dict translation
        # time so the raw assistant message is preserved for the LLM's
        # next turn — same behaviour as the legacy loop).
        if isinstance(response, AIMessage) and isinstance(
            response.content, str
        ):
            response.content = _strip_think(response.content)

        update: Dict[str, Any] = {
            "messages": [response],
            "iteration_count": iteration + 1,
        }
        # Final answer = AIMessage with no tool calls.
        if isinstance(response, AIMessage) and not getattr(
            response, "tool_calls", None
        ):
            update["final_answer"] = _extract_text_content(response)
        return update

    def tools_node(state: AgentState) -> Dict[str, Any]:
        """Dispatch every tool call from the most recent AIMessage."""
        last = state["messages"][-1]
        if not (
            isinstance(last, AIMessage)
            and getattr(last, "tool_calls", None)
        ):
            return {"messages": []}

        agent_id = state.get("agent_id", "agent")
        tool_messages: List[ToolMessage] = []
        for call in last.tool_calls:
            name = call.get("name", "") or ""
            args = call.get("args", {}) or {}
            tool_id = call.get("id") or f"call_{len(tool_messages)}"
            tool = tool_lookup.get(name)
            if tool is None:
                content = (
                    f"TOOL ERROR: Unknown tool '{name}'. "
                    f"Available: {sorted(tool_lookup.keys())}"
                )
            else:
                try:
                    # Inject agent_id into tool config so the audit log
                    # captures who called what — same contract as the
                    # legacy loop.
                    content = tool.invoke(
                        input=args,
                        config={"configurable": {"agent_id": agent_id}},
                    )
                except Exception as exc:  # noqa: BLE001
                    _LOG.exception("Tool '%s' raised", name)
                    content = f"TOOL ERROR: {exc}"
            tool_messages.append(
                ToolMessage(
                    content=str(content),
                    tool_call_id=tool_id,
                    name=name,
                )
            )
        return {"messages": tool_messages}

    def should_continue(state: AgentState) -> str:
        """Route to ``tools`` if the last AIMessage has tool calls."""
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(
            last, "tool_calls", None
        ):
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


def messages_to_step_dicts(
    messages_before: List[BaseMessage],
    messages_after: List[BaseMessage],
    agent_id: str,
    iteration: int,
) -> List[Dict[str, Any]]:
    """Diff two message lists into step dicts in the legacy yield shape.

    The Streamlit UI in ``app.py::_render_chat_bubble`` reads exactly
    these keys (verified against the live renderer):

    * ``type``: ``"tool_call"`` | ``"tool_result"`` | ``"final_answer"``
    * ``tool``: tool name (only for tool_call / tool_result)
    * ``content``: ``str(args)`` for tool_call, ``str(result)`` for
      tool_result, the assistant's final text for final_answer
    * ``agent_id``: only set on final_answer

    Args:
        messages_before: Snapshot of ``state["messages"]`` at the start
            of the streaming step (or the previous step's ``after``).
        messages_after: Current snapshot from ``graph.stream(...)``.
        agent_id: The specialist persona running this turn — surfaced on
            ``final_answer`` so the UI can label the chat bubble.
        iteration: Current iteration count. Not exposed in the legacy
            step dict but accepted to keep the signature ready for
            future logging without a breaking change.

    Returns:
        A list of step dicts, in message order, that ``run_specialist``
        can ``yield`` directly.
    """
    new_msgs = messages_after[len(messages_before):]
    steps: List[Dict[str, Any]] = []
    del iteration  # Reserved for future per-step logging; not surfaced.

    for msg in new_msgs:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                # Mirror legacy: emit one ``tool_call`` step per call.
                for call in tool_calls:
                    steps.append({
                        "type": "tool_call",
                        "tool": call.get("name", "unknown_tool"),
                        "content": str(call.get("args", {})),
                    })
            else:
                text = _extract_text_content(msg)
                steps.append({
                    "type": "final_answer",
                    "content": text or "Agent produced no output.",
                    "agent_id": agent_id,
                })
        elif isinstance(msg, ToolMessage):
            steps.append({
                "type": "tool_result",
                "tool": getattr(msg, "name", "") or "tool",
                "content": str(msg.content),
            })
        # SystemMessage / HumanMessage are never produced by the loop
        # itself; ignore if they leak in via state.
    return steps
