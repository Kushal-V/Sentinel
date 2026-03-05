"""
src/tools/tool_registry.py
==========================
LangChain Tool Registry for the Sentinel Supply Chain Digital Twin.

This module defines every ``@tool``-decorated function that agents are allowed
to call.  These tools are the ONLY authorised interface between the stochastic
LLM agents and the deterministic ``FactoryDataManager`` / ``ShadowSandbox``
layers.

Paradigm
--------
* **Agents compute intent.**  They decide *what* should happen.
* **Tools gate execution.**  They validate *whether* it is physically possible
  before allowing any state mutation.
* **The Sandbox validates first.**  ``propose_state_change`` always routes
  through ``ShadowSandbox.evaluate_proposal`` before staging the change.

Tool Call Order (enforced via docstrings)
-----------------------------------------
1. ``get_dataset_schema()``   — MUST be called first so the agent learns what
   columns exist in the currently loaded dataset.
2. ``query_data()``           — Fetch the current values for a specific row
   before proposing a change.
3. ``propose_state_change()`` — Only after the agent has inspected the schema
   and the current row values.

Singleton Pattern
-----------------
The module maintains a settable ``_data_manager`` reference via
``get_data_manager()`` / ``set_data_manager()``.  In the Streamlit app,
``app.py`` calls ``set_data_manager(ss.manager)`` on every rerun so that
tools always operate on the same per-session ``FactoryDataManager`` instance
stored in ``st.session_state``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import pandas as pd

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.core.sandbox import SandboxResult, ShadowSandbox
from src.core.state_manager import FactoryDataManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settable Data Manager Singleton
# ---------------------------------------------------------------------------

_data_manager: FactoryDataManager | None = None


def set_data_manager(dm: FactoryDataManager) -> None:
    """Inject the session-scoped ``FactoryDataManager`` instance.

    Called by ``app.py`` on every Streamlit rerun so that all tools operate
    on the same in-memory state as the UI and orchestrator.  This ensures
    per-session isolation: each Streamlit session's ``st.session_state.manager``
    is the single source of truth.

    Args:
        dm: The ``FactoryDataManager`` instance from ``st.session_state``.
    """
    global _data_manager
    _data_manager = dm
    logger.debug("Tool registry data manager set to %s.", id(dm))


def get_data_manager() -> FactoryDataManager:
    """Return the active ``FactoryDataManager`` instance.

    If ``set_data_manager`` has not been called yet (e.g. during tests or
    standalone tool usage), a fresh instance is created as a fallback.

    Returns:
        The active ``FactoryDataManager``.
    """
    global _data_manager
    if _data_manager is None:
        logger.info("No data manager injected — creating standalone instance.")
        _data_manager = FactoryDataManager()
    return _data_manager


# ---------------------------------------------------------------------------
# Two-Phase Commit — Staging Queue for HITL Approval
# ---------------------------------------------------------------------------

def commit_pending_changes() -> list[dict[str, Any]]:
    """Commit all staged changes to the live FactoryDataManager.

    Called by the HITL approval flow in ``app.py``.  Each staged change is
    **re-validated** against the current inventory state via the Shadow Sandbox
    before being applied.  This guards against stale deltas: if the data changed
    between staging and approval, the sandbox will reject the now-invalid change.

    Returns:
        List of successfully committed change dicts for display.
    """
    dm = get_data_manager()
    committed: list[dict[str, Any]] = []

    for change in dm.pending_changes:
        # Re-validate against the CURRENT state (not the stale staging-time state)
        live_df = dm.get_inventory()
        sandbox = ShadowSandbox(live_df)
        result: SandboxResult = sandbox.evaluate_proposal(
            row_primary_key=change["row_key"],
            target_column=change["target_column"],
            delta=change["delta"],
        )

        if result.status == "REJECTED":
            logger.warning(
                "HITL COMMIT SKIPPED (stale) | row=%s | col=%s | delta=%+.2f | reason=%s",
                change["row_key"],
                change["target_column"],
                change["delta"],
                result.rejection_reason,
            )
            continue

        try:
            dm.update_inventory(
                item_id=change["row_key"],
                target_column=change["target_column"],
                quantity_change=change["delta"],
            )
            dm.log_transaction(
                event_id="AGENT_ACTION",
                agent_id=change.get("agent_id", "unknown"),
                action_schema={
                    "row_key": change["row_key"],
                    "target_column": change["target_column"],
                    "delta": change["delta"],
                    "justification": change["justification"],
                    "old_value": change["old_value"],
                    "new_value": change["new_value"],
                },
                financial_impact=change.get("financial_impact", 0.0),
                sandbox_approved=True,
            )
            committed.append(change)
            logger.info(
                "HITL COMMIT | row=%s | col=%s | delta=%+.2f",
                change["row_key"], change["target_column"], change["delta"],
            )
        except Exception as exc:
            logger.error("Failed to commit staged change: %s", exc, exc_info=True)

    # Only clear committed changes; keep failed ones for retry
    committed_keys = {(c["row_key"], c["target_column"]) for c in committed}
    dm.pending_changes = [
        c for c in dm.pending_changes
        if (c["row_key"], c["target_column"]) not in committed_keys
    ]
    return committed


def discard_pending_changes() -> int:
    """Discard all staged changes (called on user rejection).

    Returns the count of changes discarded.
    """
    dm = get_data_manager()
    count = len(dm.pending_changes)
    dm.pending_changes = []
    logger.info("Discarded %d staged changes.", count)
    return count


# ---------------------------------------------------------------------------
# Groq XML Tool-Call Recovery Helper
# ---------------------------------------------------------------------------

def parse_groq_xml_tool_call(error_str: str) -> tuple[str, dict[str, Any]] | None:
    """Parse a Groq XML tool call from an API error message.

    Groq/Llama models sometimes emit ``<function=name>args</function>`` XML
    instead of proper JSON tool calls, causing a 400 error.  This function
    extracts the tool name and arguments from the error string so the
    orchestrator can execute the intended tool call.

    Args:
        error_str: The string representation of the Groq API error.

    Returns:
        A ``(tool_name, args_dict)`` tuple, or ``None`` if no XML tool call
        was found in the error string.
    """
    match = re.search(
        r"<function=(\w+)[^>]*>(.*?)</function>"
        r"|<function=(\w+)(.*?)</function>",
        error_str,
    )
    if not match:
        return None

    tool_name = match.group(1) or match.group(3)
    args_str = (match.group(2) or match.group(4) or "").strip()

    try:
        args = json.loads(args_str)
    except (json.JSONDecodeError, ValueError):
        # Fallback heuristic for kwarg-style strings like 'search_term="SFT-HD"'
        val = re.sub(
            r'^.*?search_term\s*=\s*["\'](.*)["\'].*$', r"\1", args_str
        ).strip()
        if val == args_str:
            val = args_str.replace('"', "").replace("'", "").strip()
        args = {"search_term": val}

    # If propose_state_change args failed to parse, return empty dict
    # so the tool returns a helpful error instead of crashing.
    if tool_name == "propose_state_change" and not isinstance(args, dict):
        args = {}

    return (tool_name, args)


# ---------------------------------------------------------------------------
# Tool 1 — Schema Discovery
# ---------------------------------------------------------------------------

@tool
def get_dataset_schema() -> str:
    """Retrieve the full schema of the currently loaded dataset.

    **YOU MUST CALL THIS TOOL FIRST** before calling ``query_data`` or
    ``propose_state_change``.  This tool is your only way of knowing:

    * What columns exist in the dataset (column names vary by tenant —
      a toy factory uses ``current_stock``; a hospital uses ``beds_occupied``).
    * Which column is the Primary Key (needed to look up specific rows).
    * Which columns have detected upper-bound constraints (e.g., ``max_capacity``,
      ``max_beds``) and what those constraint relationships are.
    * The data type of each column (only numeric columns may be modified).

    Without calling this tool first, you will not know the correct column
    names and any subsequent ``propose_state_change`` call will be rejected
    by the Sandbox because you will reference non-existent column names.

    Returns:
        A multi-line plain-text schema summary listing:
        - The primary key column name.
        - All columns with their data types.
        - All detected constraint pairs (mutable_column -> limit_column).
        - Any unconstrained numeric columns.

    Example agent usage:
        Agent calls: ``get_dataset_schema()``
        Agent reads the output to learn: primary key is ``item_id``,
        the column to modify is ``current_stock``, and its upper bound is
        tracked in ``max_capacity``.
    """
    try:
        dm = get_data_manager()
        profile = dm.get_schema_profile()
        summary = profile.as_agent_summary()

        # Append sample data so agents know what values exist to search for
        df = dm.get_inventory()
        pk = profile.primary_key_column
        extra_lines: list[str] = []

        # Primary key values (all if ≤60, else first 50)
        if pk in df.columns:
            pk_values = df[pk].dropna().unique().tolist()
            if len(pk_values) > 60:
                pk_values = pk_values[:50]
            extra_lines += [
                "",
                f"PRIMARY KEY VALUES ({len(pk_values)} shown):",
            ]
            extra_lines.append("  " + ", ".join(str(v) for v in pk_values))

        # Sample unique values for categorical (object) columns
        cat_cols = [
            c for c in df.select_dtypes(include=["object"]).columns
            if c != pk
        ]
        if cat_cols:
            extra_lines += ["", "CATEGORICAL COLUMN SAMPLES:"]
            for col in cat_cols:
                uniques = df[col].dropna().unique().tolist()
                shown = uniques[:15]
                extra_lines.append(
                    f"  • {col}: {', '.join(str(v) for v in shown)}"
                    + (f" … (+{len(uniques) - 15} more)" if len(uniques) > 15 else "")
                )

        # A few sample rows for context
        sample = df.head(3).to_string(index=False)
        extra_lines += ["", "SAMPLE ROWS (first 3):", sample]

        summary += "\n".join(extra_lines)
        logger.info("get_dataset_schema called — returning %d-char summary.", len(summary))
        return summary
    except Exception as exc:  # noqa: BLE001
        error_msg = f"TOOL ERROR — get_dataset_schema failed: {exc}"
        logger.error(error_msg, exc_info=True)
        return error_msg


# ---------------------------------------------------------------------------
# Tool 2 — Row Query
# ---------------------------------------------------------------------------

@tool
def query_data(search_term: str) -> str:
    """Search the dataset for rows matching a keyword across all text columns.

    Use this tool AFTER calling ``get_dataset_schema`` so you know the column
    names and primary key values.

    Call this tool to inspect the CURRENT state of a row before proposing
    any changes — you need to know the live ``current_stock`` (or equivalent
    column) to calculate a meaningful delta.

    The search is case-insensitive and matches partial substrings across
    ALL string/object columns (not just the primary key). For example,
    searching ``"brake"`` will match a row where *any* text column contains
    ``"brake"`` — whether that's the primary key, a description, or a category.

    Args:
        search_term: A keyword or partial value to search for.
            Case-insensitive.  Examples: ``"AUTO-BRK-PAD-F"``, ``"brake"``,
            ``"COLD"``, ``"Insulin"``.

    Returns:
        A JSON-formatted string representing the matched rows (list of dicts).
        Each dict contains all column values for that row.  If no rows match,
        returns a descriptive message listing available primary key values.

    Example agent usage:
        Agent calls: ``query_data("brake")``
        Returns: ``[{"part_number": "AUTO-BRK-PAD-F", "description": "Front Brake Pad Set", ...}]``
    """
    try:
        dm = get_data_manager()
        df = dm.get_inventory()
        pk_col = dm.get_schema_profile().primary_key_column

        # Search across all string/object columns for broader matching
        str_cols = df.select_dtypes(include=["object"]).columns

        def _search(term: str) -> pd.Series:
            m = pd.Series(False, index=df.index)
            for col in str_cols:
                m = m | df[col].astype(str).str.contains(
                    term, case=False, na=False, regex=False
                )
            return m

        mask = _search(search_term)

        # Fallback: if full phrase has no matches, try individual words
        if not mask.any():
            words = search_term.split()
            if len(words) > 1:
                for word in words:
                    if len(word) >= 3:  # skip very short words
                        mask = mask | _search(word)

        matched_df = df[mask]

        if matched_df.empty:
            available = df[pk_col].tolist()
            return (
                f"QUERY RESULT — No Matches: "
                f"No rows found containing '{search_term}' in any text column. "
                f"Available primary key values are: {available}"
            )

        result: list[dict[str, Any]] = matched_df.to_dict(orient="records")
        logger.info(
            "query_data('%s') — found %d matching rows.", search_term, len(result)
        )
        return json.dumps(result, indent=2, default=str)

    except Exception as exc:  # noqa: BLE001
        error_msg = f"TOOL ERROR — query_data failed: {exc}"
        logger.error(error_msg, exc_info=True)
        return error_msg


# ---------------------------------------------------------------------------
# Tool 3 — Propose State Change (Sandbox-Gated)
# ---------------------------------------------------------------------------

@tool
def propose_state_change(
    row_key: str,
    target_column: str,
    delta: float,
    justification: str,
    config: RunnableConfig,
) -> str:
    """Propose a numeric change to a specific cell; validates it through the Shadow Sandbox first.

    This is the ONLY tool that mutates live data.  It acts as a strict,
    four-stage pipeline:

    1. **Schema Check** — Confirms ``target_column`` exists and is numeric.
    2. **Shadow Sandbox Validation** — Deepcopies the live DataFrame into an
       isolated "Imagination Room" and applies your proposed ``delta`` in
       simulation.  If the result violates any constraint (e.g., stock exceeds
       capacity, or goes below zero), the proposal is **REJECTED** and the
       exact mathematical failure reason is returned to you.  You must revise
       your plan and try again.
    3. **Stage for HITL Approval** — Only if the Sandbox returns ``SAFE``,
       the change is STAGED (not yet committed). The human operator must
       click "Approve & Execute" in the UI to commit it to the Master Clipboard.
    4. **Audit Logging** — Rejections are recorded in the transaction ledger
       (``transaction_log.csv``) with your agent ID and sandbox rejection
       status for the Analyst's Retrospective Weighting.

    IMPORTANT RULES:
    - You MUST call ``get_dataset_schema`` first so you know the correct
      column names.
    - You MUST call ``query_data`` first so you know the current value and
      can calculate a meaningful delta.
    - A positive ``delta`` INCREASES the quantity (e.g., a delivery arriving).
    - A negative ``delta`` DECREASES the quantity (e.g., consuming raw materials).
    - If the Sandbox REJECTS your proposal, read the rejection reason carefully.
      It tells you the exact constraint that was violated and the maximum
      allowable change, so you can correct your delta.

    Args:
        row_key: The exact primary key value of the row to modify
            (e.g., ``"ITEM-PLASTIC-01"``).  Must be an exact match.
        target_column: The name of the numeric column to modify
            (e.g., ``"current_stock"``).  Must match exactly as returned by
            ``get_dataset_schema``.
        delta: The signed numeric change to apply.  Example: ``delta=-500``
            removes 500 units; ``delta=1000`` adds 1000 units.
        justification: A brief explanation of why this change is being made.
            Recorded in the transaction log for audit and Analyst review.
            Example: ``"Consuming plastic resin to maintain production rate
            during port strike delay."``

    Returns:
        On **SAFE**: A JSON string summarising the staged change::

            {
                "status": "STAGED",
                "row_key": "ITEM-PLASTIC-01",
                "column": "current_stock",
                "old_value": 8500.0,
                "new_value": 8000.0,
                "delta": -500.0,
                "limit_value": 15000.0,
                "justification": "...",
                "note": "Change validated by Sandbox. Awaiting human approval before commit."
            }

        On **REJECTED**: A plain-text rejection string from the Sandbox
        explaining the exact constraint violation, including the column names,
        current values, proposed values, and any limit values.  READ THIS
        CAREFULLY and revise your delta before trying again.

    Example agent usage:
        # After schema + query checks:
        Agent calls: ``propose_state_change("ITEM-PLASTIC-01", "current_stock", -500, "Emergency buffer reduction")``
        Returns: ``{"status": "STAGED", ...}``
    """
    try:
        dm = get_data_manager()
        live_df = dm.get_inventory()

        # -- Stage 1: Column validation -----------------------------------
        if target_column not in live_df.columns:
            available = live_df.columns.tolist()
            return (
                f"PROPOSAL REJECTED — Invalid Column: "
                f"'{target_column}' does not exist in the dataset. "
                f"Call get_dataset_schema() to see valid column names. "
                f"Available columns: {available}"
            )

        if not pd.api.types.is_numeric_dtype(live_df[target_column]):
            return (
                f"PROPOSAL REJECTED — Non-Numeric Column: "
                f"Column '{target_column}' is not numeric and cannot be "
                f"modified by an agent. Choose a numeric column."
            )

        # -- Stage 2: Shadow Sandbox evaluation ---------------------------
        sandbox = ShadowSandbox(live_df)
        result: SandboxResult = sandbox.evaluate_proposal(
            row_primary_key=row_key,
            target_column=target_column,
            delta=delta,
        )

        if result.status == "REJECTED":
            logger.warning(
                "Sandbox REJECTED proposal | row=%s | col=%s | delta=%+.2f",
                row_key,
                target_column,
                delta,
            )
            # Log the rejection in the ledger for the Analyst to review
            dm.log_transaction(
                event_id="SANDBOX_REJECTION",
                agent_id=_infer_calling_agent(config),
                action_schema={
                    "row_key": row_key,
                    "target_column": target_column,
                    "delta": delta,
                    "justification": justification,
                    "rejection_reason": result.rejection_reason,
                },
                financial_impact=0.0,
                sandbox_approved=False,
            )
            return result.rejection_reason  # type: ignore[return-value]

        # -- Stage 3: STAGE for HITL approval (do NOT commit yet) ----------
        financial_impact = _estimate_financial_impact(live_df, row_key, delta)

        staged_change = {
            "row_key": row_key,
            "target_column": target_column,
            "delta": delta,
            "justification": justification,
            "old_value": result.current_value,
            "new_value": result.proposed_value,
            "limit_value": result.limit_value,
            "financial_impact": financial_impact,
            "agent_id": _infer_calling_agent(config),
        }
        dm.pending_changes.append(staged_change)

        staged_response: dict[str, Any] = {
            "status": "STAGED",
            "row_key": row_key,
            "column": target_column,
            "old_value": result.current_value,
            "new_value": result.proposed_value,
            "delta": delta,
            "limit_value": result.limit_value,
            "justification": justification,
            "financial_impact_usd": financial_impact,
            "note": "Change validated by Sandbox. Awaiting human approval before commit.",
        }

        logger.info(
            "State change STAGED | row=%s | col=%s | delta=%+.2f | new_val=%.2f",
            row_key,
            target_column,
            delta,
            result.proposed_value,
        )
        return json.dumps(staged_response, indent=2)

    except Exception as exc:  # noqa: BLE001
        error_msg = f"TOOL ERROR — propose_state_change failed unexpectedly: {exc}"
        logger.error(error_msg, exc_info=True)
        return error_msg


# ---------------------------------------------------------------------------
# Tool 4 — Inter-Agent Communication
# ---------------------------------------------------------------------------

@tool
def ask_other_agent(target_agent: str, question: str, config: RunnableConfig) -> str:
    """Ask a question to another specialist agent (Maker, Mover, Keeper).

    Use this tool when your proposed action crosses domain boundaries and you
    need permission or insight from the agent responsible for that domain.
    For example:
    - If you are the Keeper and want to increase stock, you MUST ask the Maker
      if production can be increased.
    - If you are the Maker and want to increase production, you MUST ask the Keeper
      if there is warehouse capacity.

    Args:
        target_agent: The name of the agent to consult ('Maker', 'Mover', 'Keeper').
        question: The specific question, including context about your planned action.

    Returns:
        The text response from the consulted agent.
    """
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )
        import os
        import src.agents.prompts as prompts

        target_clean = target_agent.lower().strip()

        # Select the correct prompt
        if target_clean == "maker":
            sys_prompt = prompts.MAKER_SYSTEM_PROMPT
        elif target_clean == "mover":
            sys_prompt = prompts.MOVER_SYSTEM_PROMPT
        elif target_clean == "keeper":
            sys_prompt = prompts.KEEPER_SYSTEM_PROMPT
        else:
            return f"COMMUNICATION ERR: Unknown agent '{target_agent}'. Options: Maker, Mover, Keeper."

        # Redefine the core duty so the sub-agent doesn't try to use propose_state_change
        sys_prompt += (
            "\n\n=======================================================\n"
            "CRITICAL OVERRIDE FOR CURRENT TASK:\n"
            "You are currently acting in a CONSULTATIVE role answering another agent's question.\n"
            "DO NOT attempt to use the propose_state_change tool.\n"
            "Your ONLY goal is to evaluate the question, use query_data if needed, and reply with text."
        )

        # Instantiate a temporary LLM for the sub-agent
        llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_retries=2,
        )

        # Bind lookup tools only (don't let sub-agents propose state changes themselves)
        sub_tools = [get_dataset_schema, query_data]
        llm_with_tools = llm.bind_tools(sub_tools)

        caller_id = _infer_calling_agent(config)
        prompt_context = (
            f"You are being consulted by the {caller_id.upper()} agent.\n"
            f"QUESTION: {question}\n\n"
            f"Use your lookup tools to check the current state if necessary, "
            f"then provide a clear 'Yes' or 'No' recommendation with brief justification."
        )

        logger.info("Agent '%s' is asking '%s': %s", caller_id, target_clean, question)

        # Run a short ReAct loop for the sub-agent
        messages: list[Any] = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=prompt_context),
        ]
        final_answer = ""
        tool_map = {t.name: t for t in sub_tools}

        for _ in range(5):
            try:
                ai_msg = llm_with_tools.invoke(messages)
            except Exception as e:
                parsed = parse_groq_xml_tool_call(str(e))
                if parsed:
                    t_name, t_args = parsed
                    if t_name in tool_map:
                        try:
                            res = tool_map[t_name].invoke(
                                input=t_args,
                                config={"configurable": {"agent_id": f"sub_{target_clean}"}},
                            )
                        except Exception as tool_exc:
                            res = str(tool_exc)
                    else:
                        res = f"Unknown tool: {t_name}"

                    mock_tool_call = {"name": t_name, "args": t_args, "id": f"call_{len(messages)}"}
                    messages.append(AIMessage(content="", tool_calls=[mock_tool_call]))
                    messages.append(ToolMessage(content=str(res), tool_call_id=mock_tool_call["id"], name=t_name))
                    continue
                else:
                    return f"Sub-agent failed (API Error): {e}"

            messages.append(ai_msg)

            if not ai_msg.tool_calls:
                final_answer = str(ai_msg.content)
                break

            for tc in ai_msg.tool_calls:
                t_name = tc["name"]
                t_args = tc["args"]
                if t_name in tool_map:
                    try:
                        res = tool_map[t_name].invoke(
                            input=t_args,
                            config={"configurable": {"agent_id": f"sub_{target_clean}"}},
                        )
                    except Exception as e:
                        res = str(e)
                else:
                    res = f"Unknown tool: {t_name}"
                messages.append(ToolMessage(content=str(res), tool_call_id=tc["id"], name=t_name))

        return final_answer or "Sub-agent failed to respond."

    except Exception as exc:  # noqa: BLE001
        logger.error("ask_other_agent failed: %s", exc, exc_info=True)
        return f"TOOL ERROR — ask_other_agent failed: {exc}"


# ---------------------------------------------------------------------------
# Helper utilities (private, not exposed as LangChain tools)
# ---------------------------------------------------------------------------

def _estimate_financial_impact(
    df: "pd.DataFrame",  # noqa: F821
    row_key: str,
    delta: float,
) -> float:
    """Approximate the financial impact of a delta using the unit cost column.

    Searches for a ``unit_cost`` style column in the DataFrame.  If found,
    computes ``delta * unit_cost``.  Returns 0.0 if no cost column is present.

    Args:
        df: The live inventory DataFrame.
        row_key: The primary key of the affected row.
        delta: The signed quantity change.

    Returns:
        Estimated USD financial impact (negative = cost, positive = revenue/saving).
    """
    # Match cost-like columns but NOT retail/sale price columns
    cost_pattern = re.compile(
        r"(unit.?cost|cost.?per.?unit|unit.?price|cost.?price|wholesale.?price)",
        re.IGNORECASE,
    )
    cost_col = next(
        (col for col in df.columns if cost_pattern.search(col)), None
    )
    if cost_col is None:
        return 0.0

    try:
        dm = get_data_manager()
        pk_col = dm.get_schema_profile().primary_key_column
        mask = df[pk_col].astype(str) == str(row_key)
        if mask.any():
            unit_cost = float(df.loc[mask, cost_col].iloc[0])
            return delta * unit_cost
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _infer_calling_agent(config: RunnableConfig) -> str:
    """Return the agent ID injected by the Orchestrator.

    The calling agent injects its ID through the LangChain
    Runnable config (``configurable={"agent_id": "mover"}``).

    Returns:
        The string agent_id or ``"unknown"`` as a fallback.
    """
    return config.get("configurable", {}).get("agent_id", "unknown")


# ---------------------------------------------------------------------------
# Tool registry export
# ---------------------------------------------------------------------------

#: Ordered list of all tools available to Sentinel agents.  Import this list
#: when binding tools to a LangChain agent executor.
SENTINEL_TOOLS: list = [
    get_dataset_schema,
    query_data,
    propose_state_change,
    ask_other_agent,
]

#: Read-only tools for informational queries — no state mutation allowed.
INFO_TOOLS: list = [
    get_dataset_schema,
    query_data,
]
