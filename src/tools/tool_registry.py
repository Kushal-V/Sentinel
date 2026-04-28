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

Per-Context Data Manager (F6 — Multi-Tenant Session Isolation)
---------------------------------------------------------------
The module maintains a per-context ``FactoryDataManager`` reference via
``get_data_manager()`` / ``set_data_manager()`` backed by a
``contextvars.ContextVar``.  Each Streamlit session thread, asyncio task,
or future FastAPI request gets its own copy — there is no longer a shared
module-level singleton that one session can clobber for another.

In the Streamlit app, ``app.py`` calls ``set_data_manager(ss.manager)``
on every rerun.  Streamlit runs each session's script on a per-session
script-runner thread, and ``ContextVar`` state is per-thread (and
asyncio-task aware), so tools operate on the same per-session
``FactoryDataManager`` instance stored in ``st.session_state`` without
race conditions across concurrent sessions.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

if TYPE_CHECKING:
    from langchain_groq import ChatGroq

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.agents.groq_recovery import parse_groq_xml_tool_call
from src.core import config as sentinel_config
from src.core.sandbox import SandboxResult, ShadowSandbox
from src.core.state_manager import FactoryDataManager
from src.observability.tracing import get_callbacks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-Context Data Manager (F6 — Multi-Tenant Session Isolation)
# ---------------------------------------------------------------------------
# A ``contextvars.ContextVar`` isolates the active ``FactoryDataManager``
# across:
#   - Streamlit session threads (each session has its own script-runner
#     thread + context).
#   - asyncio tasks (each task copies the current context — future
#     FastAPI / MCP server deployments stay isolated by default).
#   - Manual ``threading.Thread`` work, when callers use
#     ``contextvars.copy_context().run(...)`` (the idiomatic pattern).
#
# Default sentinel: ``None`` means "no manager bound for this context".
# ``get_data_manager()`` lazily creates a standalone ``FactoryDataManager``
# in that case so unit-tests and standalone tool usage keep working.

_data_manager_var: "contextvars.ContextVar[Optional[FactoryDataManager]]" = (
    contextvars.ContextVar("sentinel_data_manager", default=None)
)

# Tombstone for backwards compatibility: a few legacy fixtures
# (e.g. ``tests/test_mcp_server.py``) reset state by assigning
# ``tool_registry._data_manager = None`` between tests.  After the
# ContextVar refactor that assignment is functionally a no-op (the
# real state lives in ``_data_manager_var``), but keeping the bare
# name on the module prevents an ``AttributeError`` and lets those
# fixtures continue to run unchanged.  Per-test isolation is now
# achieved by each test calling ``set_data_manager(...)`` itself,
# which writes into the ContextVar.
_data_manager: Optional["FactoryDataManager"] = None


def set_data_manager(
    dm: Optional["FactoryDataManager"],
) -> "contextvars.Token[Optional[FactoryDataManager]]":
    """Bind a ``FactoryDataManager`` to the current context.

    Called by ``app.py`` on every Streamlit rerun so that all tools in
    that session's thread operate on the same in-memory state as the UI
    and orchestrator.  Because the binding lives on a ``ContextVar``,
    concurrent Streamlit sessions (each on its own script-runner thread)
    and concurrent asyncio tasks each see their own manager — no
    cross-tenant bleed.

    Args:
        dm: The ``FactoryDataManager`` instance to bind, or ``None`` to
            clear the binding for this context.

    Returns:
        A ``contextvars.Token`` which can be passed to
        :func:`reset_data_manager` to restore the previous value.  Most
        callers (Streamlit reruns) can ignore the return value — the
        previous public API was ``-> None`` and a returned ``Token`` is
        an additive, backward-compatible enrichment.
    """
    token = _data_manager_var.set(dm)
    logger.debug("Tool registry data manager set to %s.", id(dm) if dm is not None else None)
    return token


def get_data_manager() -> "FactoryDataManager":
    """Return the active ``FactoryDataManager`` for the current context.

    If ``set_data_manager`` has not been called for this context (e.g.
    during direct unit-test invocation or standalone tool usage), a fresh
    instance is created as a fallback and bound to this context — so
    repeated calls within the same context see the same instance, but no
    state leaks back to the parent context that spawned us.

    Returns:
        The active ``FactoryDataManager``.
    """
    dm = _data_manager_var.get()
    if dm is None:
        logger.info("No data manager injected — creating standalone instance.")
        dm = FactoryDataManager()
        _data_manager_var.set(dm)
    return dm


def reset_data_manager(
    token: "contextvars.Token[Optional[FactoryDataManager]]",
) -> None:
    """Restore the previous data manager value using a ``Token``.

    The token must be one previously returned by :func:`set_data_manager`
    in the same context.  Useful for tests and nested scopes that want to
    bind a temporary manager and then revert.

    Args:
        token: Token returned by an earlier ``set_data_manager`` call.
    """
    _data_manager_var.reset(token)


# ---------------------------------------------------------------------------
# Cached ChatGroq client for ask_other_agent (I9)
# ---------------------------------------------------------------------------
# ``ask_other_agent`` previously instantiated a fresh ``ChatGroq`` client on
# every call. Each instantiation runs the LangChain validation pipeline and
# resets HTTP keep-alive, so high-frequency consultation between specialists
# wasted both CPU and connection setup time. We hoist it to a module-level
# lazy-initialised singleton — the model and credentials never change at
# runtime, so caching is safe.

_ask_other_agent_client: Optional["ChatGroq"] = None


def _get_ask_other_agent_client() -> "ChatGroq":
    """Return a cached ``ChatGroq`` client for the ``ask_other_agent`` tool.

    The first call constructs the client using the same kwargs the inline
    instantiation used (model ``llama-3.1-8b-instant``, ``temperature=0.1``,
    ``max_retries=2``). Subsequent calls return the same instance.

    Returns:
        A configured ``ChatGroq`` client suitable for binding sub-agent
        lookup tools.
    """
    global _ask_other_agent_client
    if _ask_other_agent_client is None:
        from langchain_groq import ChatGroq
        import os
        _ask_other_agent_client = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_retries=2,
            callbacks=get_callbacks(),
        )
        logger.debug("Initialised cached ChatGroq client for ask_other_agent.")
    return _ask_other_agent_client


# ---------------------------------------------------------------------------
# Two-Phase Commit — Staging Queue for HITL Approval
# ---------------------------------------------------------------------------

def auto_commit_eligible(
    change: dict[str, Any],
    manager: "FactoryDataManager",
) -> tuple[bool, str]:
    """Decide whether a staged change is eligible for autonomous commit.

    A change is eligible iff ALL of the following hold:

    1. ``config.AUTO_COMMIT_ENABLED`` is True (master switch).
    2. ``change["confidence"] >= AUTO_COMMIT_CONFIDENCE_THRESHOLD``.
    3. The proposing agent's trust score is
       ``>= AUTO_COMMIT_TRUST_THRESHOLD``.
    4. ``|delta| / |current_value| <= AUTO_COMMIT_MAX_DELTA_FRACTION``.
       When the current value is 0 (cannot compute a ratio) or the
       row/column cannot be located, the change is treated as
       ineligible — the safe default.

    The check is column-name agnostic: it reads the primary-key column
    and current row value via the cached ``SchemaProfile`` and the live
    DataFrame, so it works on any tenant's schema without modification.

    Args:
        change: A staged change dict (as appended by ``propose_state_change``).
        manager: The active ``FactoryDataManager`` whose inventory and
            trust scores should be consulted.

    Returns:
        ``(eligible, reason)`` — ``reason`` is a short human-readable
        explanation that surfaces the failing predicate for audit/UI.
    """
    if not sentinel_config.AUTO_COMMIT_ENABLED:
        return False, "AUTO_COMMIT disabled (master switch off)"

    # 1) Confidence threshold ---------------------------------------------
    confidence = float(change.get("confidence", 0.0))
    if confidence < sentinel_config.AUTO_COMMIT_CONFIDENCE_THRESHOLD:
        return False, (
            f"confidence {confidence:.2f} < threshold "
            f"{sentinel_config.AUTO_COMMIT_CONFIDENCE_THRESHOLD:.2f}"
        )

    # 2) Trust score threshold --------------------------------------------
    # ``get_trust_scores`` returns a dict shaped ``{"agents": {agent_id:
    # {"trust_score": float, ...}}, "global_metrics": {...}}``. Older
    # callers/tests sometimes pass a flat ``{agent_id: float}`` shape, so
    # we accept both defensively.
    agent_id = str(change.get("agent_id", ""))
    raw_scores: Any = manager.get_trust_scores()
    agents_block: Any = (
        raw_scores.get("agents", raw_scores)
        if isinstance(raw_scores, dict)
        else {}
    )
    entry: Any = agents_block.get(agent_id) if isinstance(agents_block, dict) else None
    if isinstance(entry, dict):
        trust = float(entry.get("trust_score", 0.0))
    elif isinstance(entry, (int, float)):
        trust = float(entry)
    else:
        trust = 0.0
    if trust < sentinel_config.AUTO_COMMIT_TRUST_THRESHOLD:
        return False, (
            f"trust {trust:.2f} < threshold "
            f"{sentinel_config.AUTO_COMMIT_TRUST_THRESHOLD:.2f}"
        )

    # 3) Delta-magnitude check (column-name agnostic) ---------------------
    try:
        inv = manager.get_inventory()
        profile = manager.get_schema_profile()
        pk_col = profile.primary_key_column
        row_key = change.get("row_key")
        target_col = change.get("target_column")
        delta = float(change.get("delta", 0.0))

        if (
            not pk_col
            or row_key is None
            or target_col not in inv.columns
        ):
            return False, "row/column not resolvable for delta check"

        match = inv[inv[pk_col].astype(str) == str(row_key)]
        if match.empty:
            return False, f"row_key '{row_key}' not found in '{pk_col}'"

        current = float(match.iloc[0][target_col])
        if current == 0:
            # 0% baseline — any non-zero delta is "infinite" by ratio.
            # Defensive default: require human review.
            return False, "current value is 0 — relative delta undefined"

        rel = abs(delta) / abs(current)
        if rel > sentinel_config.AUTO_COMMIT_MAX_DELTA_FRACTION:
            return False, (
                f"delta {rel:.1%} > max "
                f"{sentinel_config.AUTO_COMMIT_MAX_DELTA_FRACTION:.0%}"
            )
    except Exception as exc:  # noqa: BLE001
        # Never let a probe error trigger a silent auto-commit.
        logger.warning(
            "auto_commit_eligible delta-check raised: %s — defaulting to ineligible",
            exc,
        )
        return False, f"eligibility check failed defensively: {exc}"

    return True, "all thresholds met"


def _commit_one(
    dm: FactoryDataManager,
    change: dict[str, Any],
    *,
    auto: bool,
) -> bool:
    """Commit a single staged change with sandbox re-validation.

    Shared core for both the HITL ("Approve & Execute") path and the F4
    auto-commit fast-path. Returns True on commit, False on stale-state
    rejection or update failure (in which case the change is re-staged).

    Args:
        dm: The active ``FactoryDataManager``.
        change: The staged change dict.
        auto: True iff this is the auto-commit fast-path (records
            ``auto_committed=True`` on the transaction log row).
    """
    # Re-validate against the CURRENT state (not the stale staging-time state).
    # Sandbox re-validation is INVARIANT — we run it for both HITL and
    # auto-commit so the safety guarantee never depends on which gate
    # the change passed through.
    live_df = dm.get_inventory()
    sandbox = ShadowSandbox(live_df)
    result: SandboxResult = sandbox.evaluate_proposal(
        row_primary_key=change["row_key"],
        target_column=change["target_column"],
        delta=change["delta"],
    )

    if result.status == "REJECTED":
        logger.warning(
            "%s COMMIT SKIPPED (stale) | row=%s | col=%s | delta=%+.2f | reason=%s",
            "AUTO" if auto else "HITL",
            change["row_key"],
            change["target_column"],
            change["delta"],
            result.rejection_reason,
        )
        # Re-stage the rejected change so the user/system can retry.
        dm.add_pending_change(change)
        return False

    try:
        confidence_val = change.get("confidence")
        dm.update_inventory(
            item_id=change["row_key"],
            target_column=change["target_column"],
            quantity_change=change["delta"],
            auto_committed=auto,
            confidence=confidence_val,
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
            auto_committed=auto,
            confidence=confidence_val,
        )
        logger.info(
            "%s COMMIT | row=%s | col=%s | delta=%+.2f",
            "AUTO" if auto else "HITL",
            change["row_key"], change["target_column"], change["delta"],
        )
        return True
    except Exception as exc:
        logger.error("Failed to commit staged change: %s", exc, exc_info=True)
        # Re-stage on update failure so the change isn't silently lost.
        dm.add_pending_change(change)
        return False


def commit_pending_changes(
    manager: Optional["FactoryDataManager"] = None,
    auto_only: bool = False,
) -> Any:
    """Commit staged changes to the live FactoryDataManager.

    Two modes:

    * ``auto_only=False`` (default — HITL path) — commit ALL staged
      changes. Returns ``list[dict]`` of the committed changes for
      backwards compatibility with the HITL approval UI.
    * ``auto_only=True`` (F4 fast-path) — commit ONLY changes that pass
      ``auto_commit_eligible``; leave ineligible changes staged for
      human review. Returns a summary dict ``{"auto_committed_count":
      int, "deferred_count": int, "auto_committed": [...],
      "deferred": [...]}``.

    Each change — auto or HITL — is **re-validated** against the current
    inventory state via the Shadow Sandbox before being applied. The
    sandbox guarantee is invariant; F4 only bypasses the human gate, not
    the safety check.

    Args:
        manager: Optional explicit ``FactoryDataManager``. Defaults to the
            module-level singleton resolved by ``get_data_manager``. The
            explicit parameter is provided so tests and ``app.py`` (which
            already holds ``ss.manager``) can inject without depending on
            the singleton.
        auto_only: When True, only auto-eligible changes are committed.

    Returns:
        ``list[dict]`` in HITL mode, ``dict`` summary in auto mode.
    """
    dm: FactoryDataManager = manager if manager is not None else get_data_manager()

    if not auto_only:
        # ── HITL path: drain everything and commit all ────────────────
        drained = dm.clear_pending_changes()
        committed: list[dict[str, Any]] = []
        for change in drained:
            if _commit_one(dm, change, auto=False):
                committed.append(change)
        return committed

    # ── Auto path: partition before committing ────────────────────────
    drained = dm.clear_pending_changes()
    auto_committed: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []

    for change in drained:
        eligible, reason = auto_commit_eligible(change, dm)
        if not eligible:
            # Re-stage with the eligibility reason attached so the UI
            # can surface it ("not auto-committed because: <reason>").
            change_with_reason = dict(change)
            change_with_reason["_auto_commit_skip_reason"] = reason
            dm.add_pending_change(change_with_reason)
            deferred.append(change_with_reason)
            continue

        if _commit_one(dm, change, auto=True):
            auto_committed.append(change)
        else:
            # _commit_one re-staged the change on stale-state rejection.
            # Track it as deferred so the caller's count is accurate.
            deferred.append(change)

    return {
        "auto_committed_count": len(auto_committed),
        "deferred_count": len(deferred),
        "auto_committed": auto_committed,
        "deferred": deferred,
    }


def discard_pending_changes() -> int:
    """Discard all staged changes (called on user rejection).

    Returns the count of changes discarded.
    """
    dm = get_data_manager()
    drained = dm.clear_pending_changes()
    count = len(drained)
    logger.info("Discarded %d staged changes.", count)
    return count


# ---------------------------------------------------------------------------
# Groq XML Tool-Call Recovery Helper
# ---------------------------------------------------------------------------
# The parse_groq_xml_tool_call helper has moved to ``src.agents.groq_recovery``.
# It is imported above so legacy import paths (``from src.tools.tool_registry
# import parse_groq_xml_tool_call``) continue to resolve.


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
    confidence: float = 0.5,
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
        confidence: Float in ``[0.0, 1.0]``. Your self-rated certainty that
            this proposal is correct given the data you've seen. Use:

            * ``0.9–1.0`` — "I am certain — schema clear, data unambiguous,
              no conflicts with constraint pairs."
            * ``0.7–0.9`` — "Likely correct, minor ambiguity."
            * ``0.4–0.7`` — "Reasonable but uncertain — could be wrong."
            * ``0.0–0.4`` — "Speculative — recommend human review."

            Set honestly. The system uses ``confidence`` (combined with
            your trust score and the relative size of ``delta``) to decide
            whether to auto-commit small low-risk changes without
            requiring human approval. Defaults to ``0.5`` so existing
            agents that omit the parameter never trip the auto-commit
            fast-path.

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

        # -- F4: clamp self-rated confidence into [0.0, 1.0] -------------
        # Defensive: agents may emit out-of-range floats or non-numeric
        # strings via JSON tool calls. Clamp + coerce so downstream
        # auto-commit logic always receives a well-formed float.
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.5
        confidence_value = max(0.0, min(1.0, confidence_value))

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
            # F4 — agent self-rated confidence; consumed by auto_commit_eligible.
            "confidence": confidence_value,
        }
        dm.add_pending_change(staged_change)

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
            "confidence": confidence_value,  # F4 — echoed back for transparency
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

def _resolve_consultation_prompt(target_agent: str) -> Optional[str]:
    """Look up a specialist persona's system prompt by canonical id.

    Multi-tenant note: the roster is read from ``config.AGENT_IDS`` and
    the actual prompt strings are pulled by name from ``src.agents.prompts``
    (e.g. ``MAKER_SYSTEM_PROMPT``). No agent ids are hardcoded here, so
    the consultation channel scales transparently to any roster.

    Args:
        target_agent: Free-form agent identifier as supplied by the
            calling LLM (case-insensitive, may include whitespace).

    Returns:
        The persona's system prompt text augmented with the consultative-
        role override, or ``None`` if the id is not in
        ``config.AGENT_IDS``.
    """
    import src.agents.prompts as prompts

    target_clean = target_agent.lower().strip()
    valid_ids = {aid.lower() for aid in sentinel_config.AGENT_IDS}
    if target_clean not in valid_ids:
        return None

    attr_name = f"{target_clean.upper()}_SYSTEM_PROMPT"
    base_prompt = getattr(prompts, attr_name, None)
    if not isinstance(base_prompt, str):
        return None

    # Redefine the core duty so the sub-agent doesn't try to use propose_state_change
    return base_prompt + (
        "\n\n=======================================================\n"
        "CRITICAL OVERRIDE FOR CURRENT TASK:\n"
        "You are currently acting in a CONSULTATIVE role answering another agent's question.\n"
        "DO NOT attempt to use the propose_state_change tool.\n"
        "Your ONLY goal is to evaluate the question, use query_data if needed, and reply with text."
    )


def _ask_other_agent_graph(
    target_agent: str, question: str, caller_id: str
) -> str:
    """Graph-backed inter-agent consultation.

    Builds a fresh ``build_specialist_graph`` bound to ``INFO_TOOLS``
    (the consulted agent reads but never proposes) and returns the
    final assistant text. Mirrors the legacy manual loop's contract:
    a single string. Multi-tenant — agent roster + prompts resolved
    dynamically via ``_resolve_consultation_prompt``.

    Args:
        target_agent: Free-form agent identifier the LLM supplied.
        question: The consultative question text.
        caller_id: The agent id that initiated the consultation; used
            to label the question so the consulted agent has context.

    Returns:
        The consulted agent's final answer text, or an error string
        starting with ``"COMMUNICATION ERR"`` on unknown roster ids.
    """
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
    )
    from src.agents.graph import build_specialist_graph

    sys_prompt = _resolve_consultation_prompt(target_agent)
    if sys_prompt is None:
        valid = ", ".join(
            sorted(aid.capitalize() for aid in sentinel_config.AGENT_IDS)
        )
        return (
            f"COMMUNICATION ERR: Unknown agent '{target_agent}'. "
            f"Options: {valid}."
        )

    target_clean = target_agent.lower().strip()
    prompt_context = (
        f"You are being consulted by the {caller_id.upper()} agent.\n"
        f"QUESTION: {question}\n\n"
        f"Use your lookup tools to check the current state if necessary, "
        f"then provide a clear 'Yes' or 'No' recommendation with brief justification."
    )

    llm = _get_ask_other_agent_client()
    # Read-only INFO_TOOLS only — the consulted agent must never propose
    # mutations (its caller stages those via the SENTINEL_TOOLS path).
    graph = build_specialist_graph(
        llm=llm,
        tools=INFO_TOOLS,
        max_iterations=6,
    )
    initial_state: dict[str, Any] = {
        "messages": [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=prompt_context),
        ],
        "iteration_count": 0,
        "max_iterations": 6,
        "agent_id": f"sub_{target_clean}",
    }

    try:
        final = graph.invoke(initial_state)
    except Exception as exc:  # noqa: BLE001
        logger.error("ask_other_agent graph invocation failed: %s", exc, exc_info=True)
        return f"Sub-agent failed (graph error): {exc}"

    answer = final.get("final_answer") or ""
    if not answer:
        for msg in reversed(final.get("messages", [])):
            if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
                answer = msg.content
                break
    return answer or "Sub-agent failed to respond."


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
    caller_id = _infer_calling_agent(config)
    logger.info(
        "Agent '%s' is asking '%s': %s",
        caller_id,
        target_agent.lower().strip(),
        question,
    )

    # Phase 4b — LangGraph migration. Behind the same flag as the
    # specialist + answer_query paths so the three migrate together.
    if sentinel_config.USE_LANGGRAPH:
        return _ask_other_agent_graph(target_agent, question, caller_id)

    try:
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        sys_prompt = _resolve_consultation_prompt(target_agent)
        if sys_prompt is None:
            valid = ", ".join(
                sorted(aid.capitalize() for aid in sentinel_config.AGENT_IDS)
            )
            return (
                f"COMMUNICATION ERR: Unknown agent '{target_agent}'. "
                f"Options: {valid}."
            )

        target_clean = target_agent.lower().strip()

        # Reuse the module-level cached LLM client (see I9 — avoids
        # paying the ChatGroq constructor cost on every consultation).
        llm = _get_ask_other_agent_client()

        # Bind lookup tools only (don't let sub-agents propose state changes themselves)
        sub_tools = [get_dataset_schema, query_data]
        llm_with_tools = llm.bind_tools(sub_tools)

        prompt_context = (
            f"You are being consulted by the {caller_id.upper()} agent.\n"
            f"QUESTION: {question}\n\n"
            f"Use your lookup tools to check the current state if necessary, "
            f"then provide a clear 'Yes' or 'No' recommendation with brief justification."
        )

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
                    t_name = parsed["tool_name"]
                    t_args = parsed["arguments"]
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
# Tool 5 — Past Incident Retrieval (RAG over transaction log) — F3
# ---------------------------------------------------------------------------

@tool
def query_past_incidents(crisis_summary: str, top_k: int = 5) -> str:
    """Retrieve past incidents semantically similar to the current crisis.

    Use this tool to find historical context — e.g., "supplier X had a
    similar shortage 3 months ago, was resolved by reordering Y."  The
    backing store is a per-workspace TF-IDF index over every committed
    transaction (and rejection), so it expands the analyst/specialist
    context window beyond the rolling 50-row CSV truncation in
    ``run_analyst``.

    Args:
        crisis_summary: A short description of the current crisis.
        top_k: Number of past incidents to return (clamped to 1-10).

    Returns:
        Formatted multi-line string of past incidents with similarity
        scores, or a short status message when no matches are found.
    """
    try:
        dm = get_data_manager()
        if dm is None:
            return "ERROR: No data manager configured."

        memory = dm.get_incident_memory()
        if memory is None:
            return "ERROR: Incident memory unavailable in this environment."
        if memory.count() == 0:
            return "No past incidents available."

        capped_k = max(1, min(int(top_k), 10))
        results = memory.query(crisis_summary, top_k=capped_k)
        if not results:
            return "No semantically similar past incidents found."

        lines = [f"Found {len(results)} similar past incidents:"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. [{r.get('timestamp', '?')}] "
                f"(similarity={r.get('similarity', 0.0):.2f}) "
                f"{r.get('description', '')}"
            )
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        error_msg = f"TOOL ERROR — query_past_incidents failed: {exc}"
        logger.error(error_msg, exc_info=True)
        return error_msg


# ---------------------------------------------------------------------------
# F9 — demand forecasting tool
# ---------------------------------------------------------------------------
@tool
def forecast_demand(target_columns: str = "", horizon: int = 7) -> str:
    """Forecast demand for inventory rows over the next ``horizon`` steps.

    Reads the workspace's transaction log + current inventory, builds per-row
    time series, and projects future values using regression or exponential
    smoothing. Identifies rows at risk of stock-out within the horizon.

    Args:
        target_columns: Comma-separated column names to forecast (e.g.,
            ``"current_stock,available_units"``). Empty string = auto-detect
            from schema constraint pairs.
        horizon: Number of future steps to project (default 7, max 30).

    Returns:
        Human-readable summary of fleet forecast and at-risk rows.
    """
    try:
        from src.core.forecaster import DemandForecaster

        capped_horizon = max(1, min(int(horizon), 30))

        dm = get_data_manager()
        if dm is None:
            return "ERROR: No data manager configured."

        cols = [c.strip() for c in target_columns.split(",") if c.strip()] or None

        # ``get_transaction_log`` is the public accessor on FactoryDataManager;
        # guard with ``hasattr`` so legacy/alternative managers without it
        # still produce a forecast (single-point series fallback).
        log_df = (
            dm.get_transaction_log()
            if hasattr(dm, "get_transaction_log")
            else None
        )

        forecaster = DemandForecaster(
            inventory=dm.get_inventory(),
            transaction_log=log_df,
            schema=dm.get_schema_profile(),
        )
        result = forecaster.forecast_fleet(
            horizon=capped_horizon, target_columns=cols
        )
        return result.summary()
    except Exception as exc:  # noqa: BLE001
        error_msg = f"TOOL ERROR — forecast_demand failed: {exc}"
        logger.error(error_msg, exc_info=True)
        return error_msg


# ---------------------------------------------------------------------------
# F2 — causal inference tool
# ---------------------------------------------------------------------------
@tool
def query_causal_impact(
    treatment_column: str,
    outcome_column: str,
    intervention_delta: float,
    candidate_features: str = "",
) -> str:
    """Estimate the causal effect of perturbing one column on another.

    Answers questions like: "If supplier_lead_time increased by 3 days, what
    happens to current_stock?" Uses linear regression at sample means as a
    counterfactual estimator. Lightweight — not a full causal-inference
    framework. Effect interpretation is correlational unless the dataset
    has true experimental structure.

    Args:
        treatment_column: Column to perturb (the "cause" candidate).
        outcome_column: Column to predict (the "effect" candidate).
        intervention_delta: Magnitude of the hypothetical change to
            ``treatment_column``. Positive or negative.
        candidate_features: Optional comma-separated list of additional
            features. Empty = use all numeric columns.

    Returns:
        Human-readable summary of the estimated effect, sample size,
        R², and confidence.
    """
    try:
        from src.core.causal import CausalAnalyzer

        dm = get_data_manager()
        if dm is None:
            return "ERROR: No data manager configured."

        feats = [c.strip() for c in candidate_features.split(",") if c.strip()] or None

        analyzer = CausalAnalyzer(
            inventory=dm.get_inventory(),
            schema=dm.get_schema_profile(),
        )
        effect = analyzer.estimate_effect(
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            intervention_delta=float(intervention_delta),
            candidate_features=feats,
        )
        return effect.summary()
    except Exception as exc:  # noqa: BLE001
        error_msg = f"TOOL ERROR — query_causal_impact failed: {exc}"
        logger.error(error_msg, exc_info=True)
        return error_msg


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
    query_past_incidents,
    forecast_demand,
    query_causal_impact,
]

#: Read-only tools for informational queries — no state mutation allowed.
INFO_TOOLS: list = [
    get_dataset_schema,
    query_data,
    query_past_incidents,
    forecast_demand,
    query_causal_impact,
]
