"""
src/agents/orchestrator.py
==========================
The AgentOrchestrator — the central brain of the Sentinel Digital Twin.

This class is responsible for:
1. Holding three ``ChatGroq`` LLM clients (Dispatcher, Specialist, Analyst)
   configured for different tasks (structured routing, ReAct tool-calling,
   and trust-score analysis respectively).
2. Running the Dispatcher pipeline using ``.with_structured_output()`` to
   produce a guaranteed ``DispatchRoute`` Pydantic object — no free-text
   routing decisions allowed.
3. Enforcing **Novelty Claim B: Retrospective Weighting (Trust Override)**.
   After the LLM Dispatcher selects an agent, the orchestrator reads the
   live trust scores. If the chosen agent's score is below
   ``config.ROUTING_THRESHOLD``, the routing choice is overridden
   programmatically and the override is surfaced to the UI as a system event.
4. Executing the chosen specialist agent via a manual ReAct tool-calling
   loop (``run_specialist``) with Groq XML error recovery.
5. Streaming agent steps back to the Streamlit UI via a generator so the
   user sees tool calls and reasoning in real-time.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Generator, List, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.agents.prompts import (
    ANALYST_SYSTEM_PROMPT,
    DISPATCHER_SYSTEM_PROMPT,
    INFO_QUERY_SYSTEM_PROMPT,
    KEEPER_SYSTEM_PROMPT,
    MAKER_SYSTEM_PROMPT,
    MOVER_SYSTEM_PROMPT,
    SCANNER_SYSTEM_PROMPT,
)
from src.core import config
from src.core.state_manager import FactoryDataManager
from src.agents.groq_recovery import parse_groq_xml_tool_call
from src.observability.tracing import get_callbacks
from src.tools.tool_registry import INFO_TOOLS, SENTINEL_TOOLS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class DispatchRoute(BaseModel):
    """Structured output schema for the Dispatcher LLM call.

    The Dispatcher must output exactly this structure — no free-text responses
    are accepted.  ``with_structured_output`` enforces this contract at the
    LangChain layer.

    Attributes:
        selected_agent: The canonical agent ID chosen to handle the crisis.
        delegation_justification: One-sentence reason for the routing choice.
        urgency_tier: Controls specialist agent timeout and retry budget.
        required_lookups: Primary key values the specialist should query first.
        trust_override_applied: Set to True by the Orchestrator (not the LLM)
            if the routing was changed due to a low trust score.
        original_llm_choice: Populated by the Orchestrator if a trust override
            occurred; records what the LLM originally chose before the override.
    """
    selected_agent: Literal["maker", "mover", "keeper"] = Field(
        description="Canonical ID of the target specialist agent."
    )
    delegation_justification: str = Field(
        description="Concise reason for this routing decision."
    )
    urgency_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        description="Crisis urgency level controlling retry budget."
    )
    required_lookups: list[str] = Field(
        default_factory=list,
        description="Primary key values the specialist agent should query first.",
    )
    trust_override_applied: bool = Field(
        default=False,
        description="True if the orchestrator overrode the LLM routing due to low trust.",
    )
    original_llm_choice: str | None = Field(
        default=None,
        description="The agent the LLM originally selected before a trust override.",
    )


# ---------------------------------------------------------------------------
# Crisis Event Schema (matches event_simulator contract from architecture doc)
# ---------------------------------------------------------------------------

class CrisisEvent(BaseModel):
    """A fully-formed crisis event payload consumed by the Dispatcher.

    Attributes:
        event_id: Unique identifier for this crisis occurrence.
        event_type: Category of the crisis.
        severity: Severity label mirroring ``urgency_tier`` mapping.
        description: Natural-language description of what happened.
        affected_entities: Mapping of affected resource types to their IDs.
    """
    event_id: str
    event_type: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    description: str
    affected_entities: dict[str, list[str]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scanner Result Schema
# ---------------------------------------------------------------------------

class ScannerResult(BaseModel):
    """Structured output schema for the Scanner LLM."""
    crises: list[CrisisEvent] = Field(
        description="List of detected anomalies posing a risk to the supply chain.",
        default_factory=list,
    )


# ---------------------------------------------------------------------------
# Analyst Trust-Delta Schemas (replaces fragile regex parsing of free-text)
# ---------------------------------------------------------------------------

class TrustDeltaUpdate(BaseModel):
    """A single trust-score adjustment proposed by the Analyst LLM.

    The ``delta`` is bounded at the schema layer so a hallucinated extreme
    swing (e.g. -1.0) is rejected by Pydantic before it can be applied. The
    writer (``FactoryDataManager.update_trust_score``) clamps further to the
    global ``[MIN_TRUST_SCORE, MAX_TRUST_SCORE]`` window.

    Attributes:
        agent_id: Agent identifier matching keys in agent_trust_scores.json.
        delta: Trust score adjustment, clamped at writer side.
        reasoning: One-sentence justification for this adjustment.
    """
    agent_id: str = Field(
        description="Agent identifier matching keys in agent_trust_scores.json"
    )
    delta: float = Field(
        ge=-0.15,
        le=0.15,
        description="Trust score adjustment, clamped at writer side",
    )
    reasoning: str = Field(
        description="One-sentence justification for this adjustment"
    )


class AnalystReview(BaseModel):
    """Structured output schema for the Analyst LLM call.

    Replaces the fragile free-text regex parser. The Analyst returns a
    list of zero or more ``TrustDeltaUpdate`` objects plus a free-text
    summary intended for UI rendering.
    """
    updates: List[TrustDeltaUpdate] = Field(
        default_factory=list,
        description="Per-agent trust score adjustments to apply.",
    )
    summary: str = Field(
        description="Overall analyst observation"
    )


# ---------------------------------------------------------------------------
# AgentOrchestrator
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """Manages the full LLM orchestration lifecycle for Sentinel.

    One instance of this class should live in ``st.session_state`` for the
    lifetime of a Streamlit session.  It holds the LLM client and all
    agent executors so that tool bindings and model resources are not
    re-initialised on every Streamlit re-run.

    Attributes:
        _dispatcher_llm: ``ChatGroq`` client for structured routing.
        _specialist_llm: ``ChatGroq`` client for ReAct tool-calling agents.
        _analyst_llm: ``ChatGroq`` client for trust-score analysis.
        _dispatcher_chain: An LLM chain with structured output enforcing
            ``DispatchRoute``.
        _agent_graphs: Map of agent ID → manual ReAct config dict.
        _data_manager: Reference to the session ``FactoryDataManager``.
    """

    def __init__(self, data_manager: FactoryDataManager) -> None:
        """Initialise the orchestrator and bind all agents.

        Args:
            data_manager: The singleton ``FactoryDataManager`` instance.
                Passed in from ``st.session_state`` so all agents share the
                same live data state.

        Raises:
            RuntimeError: If the ``GROQ_API_KEY`` environment variable is
                missing when the ``ChatGroq`` client is created.
        """
        self._data_manager: FactoryDataManager = data_manager

        # Resolve the Langfuse callback list once. Empty list when tracing
        # is disabled — a fully supported LangChain configuration.
        _tracing_callbacks = get_callbacks()

        # ── 1. Dispatcher — small, fast, strict JSON (Groq / gemma2-9b-it) ─────────
        self._dispatcher_llm: ChatGroq = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model=config.DISPATCHER_MODEL,
            temperature=0,          # Zero temp for deterministic routing JSON
            max_retries=2,
            callbacks=_tracing_callbacks,
        )

        # ── 2. Specialists — fast + high-reasoning (Groq / Llama) ────────────
        self._specialist_llm: ChatGroq = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model=config.SPECIALIST_MODEL,
            temperature=0.2,        # Slight creativity for mitigation proposals
            max_retries=2,
            callbacks=_tracing_callbacks,
        )

        # ── 3. Analyst — large context window (Groq / Llama-3.3-70b) ─────────
        self._analyst_llm: ChatGroq = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model=config.ANALYST_MODEL,
            temperature=0,          # Zero temp for deterministic trust evaluation
            max_retries=2,
            callbacks=_tracing_callbacks,
        )

        self._dispatcher_chain = self._build_dispatcher_chain()
        self._agent_graphs: dict[str, Any] = self._build_agent_graphs()

        logger.info(
            "AgentOrchestrator initialised | dispatcher=%s | specialist=%s | analyst=%s",
            config.DISPATCHER_MODEL,
            config.SPECIALIST_MODEL,
            config.ANALYST_MODEL,
        )

    # ------------------------------------------------------------------
    # Private: Builder Methods
    # ------------------------------------------------------------------

    def _build_dispatcher_chain(self) -> Any:
        """Build the Dispatcher chain using Groq (gemma2-9b-it) structured output.

        ``ChatGroq`` is used here with ``.with_structured_output(DispatchRoute)``
        to enforce the routing JSON schema.  gemma2-9b-it is chosen because it
        is small (fast, free on Groq) yet strongly instruction-tuned for JSON.

        Returns:
            A LangChain runnable: ``prompt | llm.with_structured_output(DispatchRoute)``.
        """
        dispatcher_llm = self._dispatcher_llm.with_structured_output(DispatchRoute)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=DISPATCHER_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="trust_context"),
            ("human", "{crisis_description}"),
        ])
        return prompt | dispatcher_llm

    def _build_agent_graphs(self) -> dict[str, Any]:
        """Build agent configurations for each specialist.

        Instead of using ``create_react_agent`` (which breaks Groq's tool
        calling on langgraph-prebuilt 0.2.x), we store a dict containing
        the system prompt and a tool-bound LLM. The actual ReAct loop is
        implemented manually in ``run_specialist()``.

        Returns:
            A dict mapping agent ID strings to their config dicts.
        """
        specialist_configs: dict[str, str] = {
            "maker": MAKER_SYSTEM_PROMPT,
            "mover": MOVER_SYSTEM_PROMPT,
            "keeper": KEEPER_SYSTEM_PROMPT,
            "analyst": ANALYST_SYSTEM_PROMPT,
        }

        # Bind tools to the LLM once (shared across all agents)
        llm_with_tools = self._specialist_llm.bind_tools(SENTINEL_TOOLS)

        agents: dict[str, Any] = {}
        for agent_id, system_prompt_text in specialist_configs.items():
            agents[agent_id] = {
                "system_prompt": system_prompt_text,
                "llm": llm_with_tools,
            }
            logger.debug("Built manual agent config for '%s'.", agent_id)

        return agents


    # ------------------------------------------------------------------
    # Public: Dispatcher (with Trust Override — Claim B)
    # ------------------------------------------------------------------

    def dispatch(self, crisis: CrisisEvent) -> DispatchRoute:
        """Run the Dispatcher LLM and apply the Trust Override if necessary.

        This method implements **Novelty Claim B: Retrospective Weighting**.

        Steps:
            1. Read the live trust scores from ``FactoryDataManager``.
            2. Build a trust context message so the LLM is aware of current
               scores when forming its routing suggestion.
            3. Invoke the structured-output Dispatcher chain.
            4. Retrieve the LLM-chosen agent's trust score.
            5. If the score is below ``config.ROUTING_THRESHOLD``, override
               the LLM's choice with the agent's ``preferred_fallback`` and
               set ``trust_override_applied=True`` on the route object.

        Args:
            crisis: A fully-formed ``CrisisEvent`` payload.

        Returns:
            A ``DispatchRoute`` instance (potentially with an overridden
            ``selected_agent`` if the trust check triggered a reroute).

        Raises:
            RuntimeError: If the Dispatcher LLM fails to produce a valid
                ``DispatchRoute`` after retries.
        """
        trust_data: dict[str, Any] = self._data_manager.get_trust_scores()
        agents_trust: dict[str, Any] = trust_data.get("agents", {})
        threshold: float = config.ROUTING_THRESHOLD

        # Format trust scores into a concise context block for the LLM
        trust_lines = [
            f"  - {aid}: score={info.get('trust_score', 0.95):.2f} "
            f"(threshold={threshold:.2f}, "
            f"fallback={info.get('preferred_fallback', 'keeper')})"
            for aid, info in agents_trust.items()
        ]
        trust_context_text = (
            "CURRENT AGENT TRUST SCORES (use these when routing):\n"
            + "\n".join(trust_lines)
        )

        try:
            route: DispatchRoute = self._dispatcher_chain.invoke({
                "crisis_description": (
                    f"[EVENT ID: {crisis.event_id}] "
                    f"[SEVERITY: {crisis.severity}] "
                    f"{crisis.description}"
                ),
                "trust_context": [
                    HumanMessage(content=trust_context_text),
                    AIMessage(content="Understood. I will factor in trust scores."),
                ],
            })
        except Exception as exc:
            raise RuntimeError(
                f"Dispatcher LLM failed to produce a structured route: {exc}"
            ) from exc

        # ── Trust Override Check (Claim B) ─────────────────────────────────
        chosen_agent_id: str = route.selected_agent
        chosen_trust: float = float(
            agents_trust.get(chosen_agent_id, {}).get("trust_score", 1.0)
        )

        if chosen_trust < threshold:
            fallback_id: str = agents_trust.get(chosen_agent_id, {}).get(
                "preferred_fallback", config.AGENT_FALLBACKS.get(chosen_agent_id, "keeper")
            )
            logger.warning(
                "TRUST OVERRIDE: '%s' score=%.2f is below threshold=%.2f. "
                "Routing to fallback '%s'.",
                chosen_agent_id,
                chosen_trust,
                threshold,
                fallback_id,
            )
            # Pydantic models are immutable (frozen=False by default in v2),
            # so we rebuild the route with the override applied.
            route = DispatchRoute(
                selected_agent=fallback_id,  # type: ignore[arg-type]
                delegation_justification=(
                    f"[TRUST OVERRIDE] Original choice '{chosen_agent_id}' "
                    f"has trust score {chosen_trust:.2f} which is below the "
                    f"routing threshold of {threshold:.2f}. "
                    f"Rerouted to '{fallback_id}'. "
                    f"Original justification: {route.delegation_justification}"
                ),
                urgency_tier=route.urgency_tier,
                required_lookups=route.required_lookups,
                trust_override_applied=True,
                original_llm_choice=chosen_agent_id,
            )

        logger.info(
            "Dispatch complete | selected='%s' | urgency='%s' | override=%s",
            route.selected_agent,
            route.urgency_tier,
            route.trust_override_applied,
        )
        return route

    # ------------------------------------------------------------------
    # Public: Specialist Agent Execution
    # ------------------------------------------------------------------

    def run_specialist(
        self,
        route: DispatchRoute,
        crisis: CrisisEvent,
        chat_history: list[HumanMessage | AIMessage],
    ) -> Generator[dict[str, Any], None, None]:
        """Execute the routed specialist agent using a manual ReAct tool loop.

        Instead of relying on ``create_react_agent`` (which breaks Groq's tool
        calling on langgraph-prebuilt 0.2.x), this method implements a manual
        ReAct loop:

        1. Build a message list with SystemMessage + history + HumanMessage.
        2. Call ``llm_with_tools.invoke(messages)`` to get an AIMessage.
        3. If the AIMessage contains ``tool_calls``, execute each tool and
           append ``ToolMessage`` results back to the message list.
        4. Repeat until the model returns a plain text response (no tool calls)
           or a maximum iteration limit is reached.

        Args:
            route: The ``DispatchRoute`` from ``dispatch()``.
            crisis: The original ``CrisisEvent`` for context.
            chat_history: LangChain message history for multi-turn context.

        Yields:
            Dicts with keys ``"type"`` and ``"content"``.
        """
        agent_id = route.selected_agent
        if agent_id not in self._agent_graphs:
            yield {
                "type": "error",
                "content": (
                    f"No agent config found for '{agent_id}'. "
                    f"Available: {list(self._agent_graphs.keys())}"
                ),
            }
            return

        agent_cfg = self._agent_graphs[agent_id]
        llm_with_tools = agent_cfg["llm"]
        system_prompt = agent_cfg["system_prompt"]

        _lookups = ", ".join(route.required_lookups) if route.required_lookups else "Use get_dataset_schema to discover relevant rows."
        input_text = (
            "CRISIS EVENT [" + crisis.event_id + "] - " + crisis.description + "\n\n"
            "DISPATCHER NOTES: " + route.delegation_justification + "\n"
            "URGENCY: " + route.urgency_tier + "\n"
            "SUGGESTED LOOKUPS: " + _lookups + "\n\n"
            "Follow your mandatory tool call order. Produce a full Mitigation Proposal."
        )

        # Phase 4 — LangGraph migration. When the feature flag is on,
        # delegate to the StateGraph path. The flag-off path below
        # (manual ReAct loop) is the proven production behaviour and is
        # NOT modified by this commit.
        if config.USE_LANGGRAPH:
            yield from self._run_specialist_graph(
                agent_id=agent_id,
                system_prompt=system_prompt,
                input_text=input_text,
                chat_history=chat_history,
            )
            return

        # Build message list: System + history + human turn
        messages: list[Any] = [
            SystemMessage(content=system_prompt),
        ] + list(chat_history) + [
            HumanMessage(content=input_text),
        ]

        # Build a tool lookup map for execution
        tool_map: dict[str, Any] = {t.name: t for t in SENTINEL_TOOLS}

        MAX_ITERATIONS = 10
        final_text: str = ""

        try:
            for iteration in range(MAX_ITERATIONS):
                # Call the LLM
                try:
                    ai_msg = llm_with_tools.invoke(messages)
                except Exception as e:
                    # Recover from Groq/Llama XML <function=...> bug
                    parsed = parse_groq_xml_tool_call(str(e))
                    if parsed:
                        t_name = parsed["tool_name"]
                        t_args = parsed["arguments"]

                        yield {
                            "type": "tool_call",
                            "tool": t_name,
                            "content": str(t_args),
                        }

                        if t_name in tool_map:
                            try:
                                res = tool_map[t_name].invoke(input=t_args, config={"configurable": {"agent_id": agent_id}})
                            except Exception as tool_exc:
                                res = f"TOOL ERROR: {tool_exc}"
                        else:
                            res = f"Unknown tool: {t_name}"

                        yield {
                            "type": "tool_result",
                            "tool": t_name,
                            "content": str(res),
                        }

                        # Inject a mock AI tool call and the Tool message so the agent sees the result
                        mock_tool_call = {"name": t_name, "args": t_args, "id": f"call_{len(messages)}"}
                        messages.append(AIMessage(content="", tool_calls=[mock_tool_call]))
                        messages.append(ToolMessage(content=str(res), tool_call_id=mock_tool_call["id"], name=t_name))
                        continue
                    else:
                        yield {
                            "type": "error",
                            "content": f"LLM API Error: {e}",
                        }
                        break

                messages.append(ai_msg)

                # Debug: log the raw content type and structure
                logger.debug(
                    "Agent '%s' iteration %d — content type: %s, "
                    "tool_calls: %d, content preview: %.200s",
                    agent_id,
                    iteration,
                    type(ai_msg.content).__name__,
                    len(ai_msg.tool_calls) if ai_msg.tool_calls else 0,
                    str(ai_msg.content)[:200] if ai_msg.content else "(empty)",
                )

                # Extract text from the AI message.
                # Qwen3 wraps reasoning in <think>...</think> tags as a plain
                # string: "<think>reasoning</think>\n\nActual answer here"
                # We must strip the thinking block to get the real answer.
                msg_text = ""
                if isinstance(ai_msg.content, str):
                    # Strip <think>...</think> blocks (Qwen3 thinking mode)
                    cleaned = re.sub(
                        r"<think>.*?</think>",
                        "",
                        ai_msg.content,
                        flags=re.DOTALL,
                    ).strip()
                    msg_text = cleaned
                elif isinstance(ai_msg.content, list):
                    # Some models return structured content blocks
                    parts = []
                    for block in ai_msg.content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                parts.append(block.get("text", ""))
                            elif block.get("type") == "thinking":
                                pass  # Skip internal reasoning
                            else:
                                parts.append(str(block))
                        elif isinstance(block, str):
                            parts.append(block)
                    msg_text = "\n".join(parts)

                # If no tool calls, this is the final answer
                if not ai_msg.tool_calls:
                    final_text = msg_text or final_text
                    break

                # If there IS text alongside tool calls, accumulate it
                if msg_text.strip():
                    final_text = msg_text

                # Process each tool call
                for tc in ai_msg.tool_calls:
                    tool_name = tc.get("name", "unknown_tool")
                    tool_args = tc.get("args", {})
                    tool_call_id = tc.get("id", "")

                    yield {
                        "type": "tool_call",
                        "tool": tool_name,
                        "content": str(tool_args),
                    }

                    # Execute the tool
                    if tool_name in tool_map:
                        try:
                            # Inject agent_id into the tool config for audit logging
                            tool_result = tool_map[tool_name].invoke(
                                input=tool_args,
                                config={"configurable": {"agent_id": agent_id}},
                            )
                        except Exception as tool_exc:
                            tool_result = f"TOOL ERROR: {tool_exc}"
                    else:
                        tool_result = f"TOOL ERROR: Unknown tool '{tool_name}'"

                    # Append ToolMessage for the LLM to see
                    messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call_id,
                            name=tool_name,
                        )
                    )

                    yield {
                        "type": "tool_result",
                        "tool": tool_name,
                        "content": str(tool_result),
                    }

                logger.debug(
                    "ReAct iteration %d/%d for agent '%s' completed.",
                    iteration + 1,
                    MAX_ITERATIONS,
                    agent_id,
                )

        except Exception as exc:
            yield {"type": "error", "content": f"Agent execution failed: {exc}"}
            return

        yield {
            "type": "final_answer",
            "content": final_text or "Agent produced no output.",
            "agent_id": agent_id,
        }

    # ------------------------------------------------------------------
    # Internal: LangGraph specialist execution (Phase 4, behind flag)
    # ------------------------------------------------------------------

    def _run_specialist_graph(
        self,
        agent_id: str,
        system_prompt: str,
        input_text: str,
        chat_history: list[HumanMessage | AIMessage],
    ) -> Generator[dict[str, Any], None, None]:
        """Execute the specialist via the LangGraph ``StateGraph``.

        Mirrors ``run_specialist``'s yielded step-dict contract so the
        Streamlit UI in ``app.py`` continues to render unchanged. Only
        called when ``config.USE_LANGGRAPH`` is True.

        Args:
            agent_id: Specialist persona ID (e.g. ``"maker"``).
            system_prompt: The persona's system prompt text.
            input_text: The dispatcher-formatted human turn text.
            chat_history: Multi-turn LangChain message history.

        Yields:
            Step dicts in the exact shape ``app.py::_render_chat_bubble``
            consumes from the legacy manual ReAct loop.
        """
        # Local imports keep the legacy path free of any LangGraph
        # imports. If a user flips the flag without ``langgraph``
        # installed, the failure surfaces here, not at module load.
        from src.agents.graph import (
            build_specialist_graph,
            messages_to_step_dicts,
        )

        max_iterations = 10
        graph = build_specialist_graph(
            llm=self._specialist_llm,
            tools=SENTINEL_TOOLS,
            max_iterations=max_iterations,
        )

        initial_messages: list[Any] = [
            SystemMessage(content=system_prompt),
        ] + list(chat_history) + [
            HumanMessage(content=input_text),
        ]
        initial_state: dict[str, Any] = {
            "messages": initial_messages,
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "agent_id": agent_id,
        }

        prev_messages: list[Any] = list(initial_messages)
        final_emitted = False
        try:
            for state in graph.stream(
                initial_state, stream_mode="values"
            ):
                current_messages = state.get("messages", [])
                new_steps = messages_to_step_dicts(
                    prev_messages,
                    current_messages,
                    agent_id=agent_id,
                    iteration=state.get("iteration_count", 0),
                )
                for step in new_steps:
                    if step.get("type") == "final_answer":
                        final_emitted = True
                    yield step
                prev_messages = list(current_messages)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LangGraph specialist execution failed")
            yield {
                "type": "error",
                "content": f"Agent execution failed: {exc}",
            }
            return

        # Safety net: if the graph terminated without ever emitting a
        # final answer (e.g. max_iterations capped before the LLM
        # produced a tool-call-free response), surface a stub so the UI
        # always closes the conversation cleanly.
        if not final_emitted:
            yield {
                "type": "final_answer",
                "content": "Agent produced no output.",
                "agent_id": agent_id,
            }

    # ------------------------------------------------------------------
    # Public: Informational Query (no crisis fabrication)
    # ------------------------------------------------------------------

    def answer_query(
        self,
        query: str,
        chat_history: list[HumanMessage | AIMessage],
    ) -> Generator[dict[str, Any], None, None]:
        """Answer an informational user query using tools but without crisis framing.

        Uses the same ReAct tool loop as ``run_specialist`` but with a neutral
        prompt that instructs the LLM to retrieve and present data — not to
        fabricate crises or propose state changes.

        Args:
            query: The user's natural-language question.
            chat_history: LangChain message history for context.

        Yields:
            Dicts with keys ``"type"`` and ``"content"``.
        """
        # Phase 4b — LangGraph migration. When the feature flag is on,
        # delegate to the StateGraph path (mirrors run_specialist).
        if config.USE_LANGGRAPH:
            yield from self._answer_query_graph(
                agent_id="info",
                system_prompt=INFO_QUERY_SYSTEM_PROMPT,
                query_text=query,
                chat_history=chat_history,
            )
            return

        llm_with_tools = self._specialist_llm.bind_tools(INFO_TOOLS)
        tool_map: dict[str, Any] = {t.name: t for t in INFO_TOOLS}

        messages: list[Any] = [
            SystemMessage(content=INFO_QUERY_SYSTEM_PROMPT),
        ] + list(chat_history) + [
            HumanMessage(content=query),
        ]

        MAX_ITERATIONS = 10
        final_text: str = ""

        try:
            for iteration in range(MAX_ITERATIONS):
                try:
                    ai_msg = llm_with_tools.invoke(messages)
                except Exception as e:
                    parsed = parse_groq_xml_tool_call(str(e))
                    if parsed:
                        t_name = parsed["tool_name"]
                        t_args = parsed["arguments"]
                        yield {"type": "tool_call", "tool": t_name, "content": str(t_args)}
                        if t_name in tool_map:
                            try:
                                res = tool_map[t_name].invoke(input=t_args, config={"configurable": {"agent_id": "info"}})
                            except Exception as tool_exc:
                                res = f"TOOL ERROR: {tool_exc}"
                        else:
                            res = f"Unknown tool: {t_name}"
                        yield {"type": "tool_result", "tool": t_name, "content": str(res)}
                        mock_tc = {"name": t_name, "args": t_args, "id": f"call_{len(messages)}"}
                        messages.append(AIMessage(content="", tool_calls=[mock_tc]))
                        messages.append(ToolMessage(content=str(res), tool_call_id=mock_tc["id"], name=t_name))
                        continue
                    else:
                        yield {"type": "error", "content": f"LLM API Error: {e}"}
                        break

                messages.append(ai_msg)

                msg_text = ""
                if isinstance(ai_msg.content, str):
                    msg_text = re.sub(r"<think>.*?</think>", "", ai_msg.content, flags=re.DOTALL).strip()
                elif isinstance(ai_msg.content, list):
                    parts = []
                    for block in ai_msg.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            parts.append(block)
                    msg_text = "\n".join(parts)

                if not ai_msg.tool_calls:
                    final_text = msg_text or final_text
                    break

                if msg_text.strip():
                    final_text = msg_text

                for tc in ai_msg.tool_calls:
                    tool_name = tc.get("name", "unknown_tool")
                    tool_args = tc.get("args", {})
                    tool_call_id = tc.get("id", "")
                    yield {"type": "tool_call", "tool": tool_name, "content": str(tool_args)}
                    if tool_name in tool_map:
                        try:
                            tool_result = tool_map[tool_name].invoke(
                                input=tool_args,
                                config={"configurable": {"agent_id": "info"}},
                            )
                        except Exception as tool_exc:
                            tool_result = f"TOOL ERROR: {tool_exc}"
                    else:
                        tool_result = f"TOOL ERROR: Unknown tool '{tool_name}'"
                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call_id, name=tool_name))
                    yield {"type": "tool_result", "tool": tool_name, "content": str(tool_result)}

        except Exception as exc:
            yield {"type": "error", "content": f"Query execution failed: {exc}"}
            return

        yield {
            "type": "final_answer",
            "content": final_text or "No relevant data found for your query.",
            "agent_id": "info",
        }

    # ------------------------------------------------------------------
    # Internal: LangGraph informational-query execution (Phase 4b, behind flag)
    # ------------------------------------------------------------------

    def _answer_query_graph(
        self,
        agent_id: str,
        system_prompt: str,
        query_text: str,
        chat_history: list[HumanMessage | AIMessage],
    ) -> Generator[dict[str, Any], None, None]:
        """Execute an informational query via the LangGraph ``StateGraph``.

        Mirrors ``_run_specialist_graph``'s shape but binds the read-only
        ``INFO_TOOLS`` set (no ``propose_state_change``) and uses the
        neutral ``INFO_QUERY_SYSTEM_PROMPT``. Only invoked when
        ``config.USE_LANGGRAPH`` is True.

        Args:
            agent_id: Logical id surfaced on ``final_answer`` step dicts
                (defaults to ``"info"`` for the chat assistant path).
            system_prompt: The neutral system prompt to inject.
            query_text: The user's question — becomes the human turn.
            chat_history: Multi-turn LangChain message history.

        Yields:
            Step dicts in the exact shape ``app.py`` consumes from the
            legacy manual loop.
        """
        # Local imports keep the legacy path free of any LangGraph
        # dependency at module load time.
        from src.agents.graph import (
            build_specialist_graph,
            messages_to_step_dicts,
        )

        max_iterations = 10
        graph = build_specialist_graph(
            llm=self._specialist_llm,
            tools=INFO_TOOLS,
            max_iterations=max_iterations,
        )

        initial_messages: list[Any] = [
            SystemMessage(content=system_prompt),
        ] + list(chat_history) + [
            HumanMessage(content=query_text),
        ]
        initial_state: dict[str, Any] = {
            "messages": initial_messages,
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "agent_id": agent_id,
        }

        prev_messages: list[Any] = list(initial_messages)
        final_emitted = False
        try:
            for state in graph.stream(
                initial_state, stream_mode="values"
            ):
                current_messages = state.get("messages", [])
                new_steps = messages_to_step_dicts(
                    prev_messages,
                    current_messages,
                    agent_id=agent_id,
                    iteration=state.get("iteration_count", 0),
                )
                for step in new_steps:
                    if step.get("type") == "final_answer":
                        final_emitted = True
                    yield step
                prev_messages = list(current_messages)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LangGraph informational-query execution failed")
            yield {
                "type": "error",
                "content": f"Query execution failed: {exc}",
            }
            return

        # Safety net: ensure the UI always sees a closing final_answer.
        if not final_emitted:
            yield {
                "type": "final_answer",
                "content": "No relevant data found for your query.",
                "agent_id": agent_id,
            }

    # ------------------------------------------------------------------
    # Public: Analyst (Trust Score Updater — Claim B engine)
    # ------------------------------------------------------------------

    def run_analyst(
        self,
        chat_history: list[HumanMessage | AIMessage],
    ) -> str:
        """Run the Analyst agent using Groq for trust score evaluation.

        The Analyst does NOT need tools — it reads the full transaction ledger
        passed directly in the prompt.  The large context window of llama-3.3-70b
        is ideal for processing the entire CSV log in a single pass.

        This implementation uses ``with_structured_output(AnalystReview)`` to
        force the LLM to emit a strict Pydantic schema.  The fragile regex
        parser of the previous implementation is gone — if the LLM drifts
        (one decimal, missing colon, "N/A" output), Pydantic surfaces an
        exception immediately rather than silently freezing trust scores.

        Multi-tenancy: the set of valid agent IDs is read live from the data
        manager (``get_trust_scores().keys()``).  Agent IDs are NEVER
        hardcoded so a custom ``AGENT_IDS`` roster works transparently.

        Args:
            chat_history: Current session message history for context.

        Returns:
            The Analyst's summary string for display in the UI.
        """
        log_df = self._data_manager.get_transaction_log()
        if log_df.empty:
            return "No transactions in the ledger yet. The Analyst has nothing to evaluate."

        log_summary = log_df.to_string(index=False, max_rows=50)
        input_text = (
            f"Review the following transaction ledger and produce per-agent "
            f"trust score updates where warranted.\n\n"
            f"TRANSACTION LOG:\n{log_summary}"
        )

        try:
            # Bind structured output: the LLM is forced to return AnalystReview.
            analyst_chain = self._analyst_llm.with_structured_output(AnalystReview)
            messages: list[Any] = [
                SystemMessage(content=ANALYST_SYSTEM_PROMPT),
                *list(chat_history),
                HumanMessage(content=input_text),
            ]
            review: AnalystReview = analyst_chain.invoke(messages)

            # Discover valid agent IDs dynamically from the live trust roster
            # (multi-tenant: never hardcode).  Falls back to the top-level keys
            # of get_trust_scores() if the canonical "agents" subdict is absent.
            trust_data = self._data_manager.get_trust_scores()
            agents_section = trust_data.get("agents", trust_data) if isinstance(trust_data, dict) else {}
            valid_agents: set[str] = set(agents_section.keys()) if isinstance(agents_section, dict) else set()

            for update in review.updates:
                agent_id_clean = update.agent_id.strip()
                if agent_id_clean not in valid_agents:
                    logger.warning(
                        "Analyst proposed delta for unknown agent_id '%s'; "
                        "skipping. Valid agents: %s",
                        update.agent_id,
                        sorted(valid_agents),
                    )
                    continue
                try:
                    updated = self._data_manager.update_trust_score(
                        agent_id=agent_id_clean,
                        score_delta=update.delta,
                    )
                    logger.info(
                        "Analyst applied trust delta | agent=%s | delta=%+.4f | new=%.4f | reason=%s",
                        agent_id_clean,
                        update.delta,
                        updated["trust_score"],
                        update.reasoning,
                    )
                except (ValueError, KeyError) as exc:
                    logger.warning(
                        "Could not apply trust delta for '%s': %s",
                        agent_id_clean,
                        exc,
                    )

            return review.summary

        except Exception as exc:
            error_msg = f"Analyst execution failed: {exc}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    # ------------------------------------------------------------------
    # Public: Scanner (CSV Auto-Scan Feature)
    # ------------------------------------------------------------------

    def scan_for_crises(self) -> list[CrisisEvent]:
        """Run the Risk Scanner agent using structured output to identify crises.

        Returns:
            A list of detected CrisisEvent objects based on the current inventory.
        """
        inv_df = self._data_manager.get_inventory()
        if inv_df.empty:
            return []

        # Convert simple profile into text, plus some sample rows
        from src.core.schema_engine import DynamicSchemaInferencer
        schema_profile = DynamicSchemaInferencer(inv_df).infer()
        schema_summary = schema_profile.as_agent_summary()
        sample_data = inv_df.to_string(index=False, max_rows=100)
        
        agent_personas = {
            "Maker": "Focus on production bottlenecks, zero stock on critical Work-in-Progress (WIP) materials, and manufacturing issues.",
            "Mover": "Focus on shipping hurdles, route anomalies, transport disruptions, and abnormal stock levels indicating transit failures.",
            "Keeper": "Focus on warehouse overflow, inventory holding capacities alarms, and instances where current stock hits or exceeds warning levels."
        }
        
        all_crises: list[CrisisEvent] = []
        scanner_chain = self._analyst_llm.with_structured_output(ScannerResult)
        
        import uuid
        
        for agent_role, agent_focus in agent_personas.items():
            # Build a list of real product identifiers the scanner can reference
            pk_col = schema_profile.primary_key_column
            real_ids = inv_df[pk_col].astype(str).tolist() if pk_col in inv_df.columns else []
            real_ids_str = ", ".join(real_ids[:50])  # Cap at 50 to avoid token overflow

            # Gather unique values from categorical columns for grounding
            categorical_context = ""
            for col in inv_df.select_dtypes(include=["object", "category"]).columns:
                if col != pk_col:
                    uniques = inv_df[col].dropna().unique()[:15]
                    if len(uniques) > 0:
                        categorical_context += f"  {col}: {', '.join(str(v) for v in uniques)}\n"

            input_text = (
                f"Review the dataset schema and the data sample below to identify "
                f"potential supply chain crises.\n\n"
                f"SCHEMA RULES:\n{schema_summary}\n\n"
                f"REAL PRODUCT/ITEM IDs (you may ONLY reference these):\n{real_ids_str}\n\n"
            )
            if categorical_context:
                input_text += f"KNOWN CATEGORICAL VALUES:\n{categorical_context}\n"
            input_text += f"DATA SAMPLE:\n{sample_data}"
            
            prompt_content = SCANNER_SYSTEM_PROMPT.format(agent_role=agent_role, agent_focus=agent_focus)

            try:
                messages = [
                    SystemMessage(content=prompt_content),
                    HumanMessage(content=input_text)
                ]
                result: ScannerResult = scanner_chain.invoke(messages)
                
                # Tag crisis with the originating agent role mentally
                for c in result.crises:
                    if not c.event_id or c.event_id == "unknown":
                        c.event_id = f"EVT-{agent_role[:3].upper()}-{uuid.uuid4().hex[:4].upper()}"
                    all_crises.append(c)
                    
            except Exception as exc:
                logger.error(f"Scanner execution failed for {agent_role}: {exc}", exc_info=True)
                
        return all_crises

