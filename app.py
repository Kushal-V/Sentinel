"""
app.py
======
Sentinel — Supply Chain Digital Twin
Multi-Tenant Streamlit Dashboard (Phase 3)

Entry point: ``streamlit run app.py``

Architecture notes:
- ``st.session_state`` is the single source of truth for all mutable UI state.
  The ``FactoryDataManager`` and ``AgentOrchestrator`` are instantiated ONCE
  and stored there to survive Streamlit's re-run model.
- A CSV upload replaces ``inventory.csv`` on disk and then rebuilds the
  ``FactoryDataManager`` in-place so the singleton LRU cache in
  ``tool_registry`` is also invalidated.
- All agent streaming steps are stored in ``st.session_state.agent_steps``
  which is rendered incrementally in the Crisis Console tab.
- Human-in-the-Loop: when the agent produces a SAFE Mitigation Proposal, an
  "Approve & Execute" button appears.  Clicking it calls the final
  ``FactoryDataManager.update_inventory`` calls described in the proposal's
  ACTIONS TAKEN block.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from streamlit import session_state as ss

# Ensure the project root is on sys.path when running via `streamlit run app.py`
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env BEFORE any LangChain/Gemini imports run their env-var checks
from dotenv import dotenv_values as _dotenv_values
_env_path = _PROJECT_ROOT / ".env"
if _env_path.exists():
    for k, v in _dotenv_values(_env_path).items():
        if v is not None:
            os.environ[k] = v

from src.agents.orchestrator import AgentOrchestrator, CrisisEvent, DispatchRoute
from src.core import config
from src.core.schema_engine import DynamicSchemaInferencer
from src.core.state_manager import FactoryDataManager
from src.observability.tracing import init_tracing, status_summary
from src.tools.tool_registry import (
    SENTINEL_TOOLS,
    get_data_manager,
    set_data_manager,
    commit_pending_changes,
    discard_pending_changes,
)

# Initialise Langfuse observability before any LLM client is constructed.
# Idempotent and side-effect-free when env vars are absent — the call site
# does not need to branch on availability.
init_tracing()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("sentinel.app")

# ---------------------------------------------------------------------------
# Page config (must be the very first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sentinel — Supply Chain Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — professional, dark-mode-friendly styling
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header gradient bar */
    .sentinel-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .sentinel-header h1 { margin: 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }
    .sentinel-header p  { margin: 0.25rem 0 0; font-size: 0.9rem; opacity: 0.7; }

    /* Trust score bar */
    .trust-bar-container { margin-top: 0.25rem; }
    .trust-bar {
        height: 8px; border-radius: 4px;
        background: linear-gradient(90deg, #e74c3c, #f39c12, #2ecc71);
        width: 100%;
    }
    .trust-fill {
        height: 100%; border-radius: 4px;
        background: #2ecc71;
        transition: width 0.4s ease;
    }
    .trust-fill.warn  { background: #f39c12; }
    .trust-fill.crit  { background: #e74c3c; }

    /* Message bubbles */
    .msg-human { background:#1a3a5c; border-radius:12px 12px 2px 12px; padding:0.75rem 1rem; margin:0.5rem 0; color:white; }
    .msg-agent { background:#1e3a2a; border-radius:12px 12px 12px 2px; padding:0.75rem 1rem; margin:0.5rem 0; color:#d4edda; }
    .msg-system { background:#2a2a40; border-radius:8px; padding:0.5rem 1rem; margin:0.5rem 0; color:#b0b0cc; font-size:0.85rem; border-left:3px solid #7070cc; }
    .msg-tool { background:#2a2020; border-radius:8px; padding:0.5rem 1rem; margin:0.25rem 0; color:#e8c4a0; font-size:0.82rem; border-left:3px solid #c0702a; }
    .msg-rejected { border-left:3px solid #e74c3c; background:#2a1a1a; color:#f5a0a0; border-radius:8px; padding:0.5rem 1rem; }
    .msg-safe { border-left:3px solid #2ecc71; background:#1a2a1a; color:#a0f5a0; border-radius:8px; padding:0.5rem 1rem; }
    .msg-override { border-left:3px solid #f39c12; background:#2a2210; color:#f5d580; border-radius:8px; padding:0.5rem 1rem; }

    /* Approve button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: #0a2a1a; font-weight: 700; border: none;
        padding: 0.75rem 2rem; border-radius: 8px; font-size: 1rem;
    }

    /* Section dividers */
    hr { border-color: #2a2a3a !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Mock Crisis Presets
# ---------------------------------------------------------------------------

MOCK_CRISES: dict[str, CrisisEvent] = {
    "Port Strike (CRITICAL)": CrisisEvent(
        event_id="EVT-0001-PORT-STRIKE",
        event_type="EXTERNAL_SHOCK",
        severity="CRITICAL",
        description=(
            "Major port strike at Long Beach affecting all ocean freight routes. "
            "Inbound plastic resin (ITEM-PLASTIC-01) and microchip (ITEM-MICROCHIP-01) "
            "shipments are delayed by 14 days. Air freight alternatives available "
            "at 4x standard cost."
        ),
        affected_entities={
            "transit_routes": ["ROUTE-OCEAN-PACZ"],
            "materials": ["ITEM-PLASTIC-01", "ITEM-MICROCHIP-01"],
        },
    ),
    "Machine Breakdown (HIGH)": CrisisEvent(
        event_id="EVT-0002-MACHINE-DOWN",
        event_type="INTERNAL_FAILURE",
        severity="HIGH",
        description=(
            "Injection moulding line #3 (Factory Floor 1) has failed. "
            "Work-in-progress body frame production halted. Estimated repair: 72 hours. "
            "Current WIP buffer (ITEM-WIP-BODYFRAME-01) must cover the repair window."
        ),
        affected_entities={
            "production_lines": ["FACTORY-FLOOR-1"],
            "materials": ["ITEM-WIP-BODYFRAME-01"],
        },
    ),
    "Warehouse Overflow Warning (MEDIUM)": CrisisEvent(
        event_id="EVT-0003-WH-OVERFLOW",
        event_type="CAPACITY_WARNING",
        severity="MEDIUM",
        description=(
            "Finished goods warehouse (WH-C-FG) is approaching 90% capacity. "
            "Incoming production output of deluxe toys (ITEM-FG-TOY-DELUXE-01) "
            "will exceed capacity within 48 hours if not dispatched urgently."
        ),
        affected_entities={
            "locations": ["WH-C-FG"],
            "materials": ["ITEM-FG-TOY-DELUXE-01"],
        },
    ),
    "Demand Spike — Emergency Order (HIGH)": CrisisEvent(
        event_id="EVT-0004-DEMAND-SPIKE",
        event_type="MARKET_SHOCK",
        severity="HIGH",
        description=(
            "Retail partner has submitted an emergency purchase order for 3,000 "
            "deluxe toy units (ITEM-FG-TOY-DELUXE-01) to be delivered within 5 days. "
            "Current finished-goods stock must be reserved and logistics rerouted."
        ),
        affected_entities={
            "customers": ["RETAIL-PARTNER-A"],
            "materials": ["ITEM-FG-TOY-DELUXE-01"],
        },
    ),
}

# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Bootstrap all required session state keys exactly once per session.

    The ``if key not in ss`` guard is the critical protection against the
    Streamlit Refresh Bug where objects are re-created on every widget
    interaction, losing in-memory state.

    Note: ``orchestrator`` is intentionally left as ``None`` here and built
    lazily on first use so that a missing API key does not crash the app
    before the user can even see the UI.
    """
    if "active_workspace" not in ss:
        ss.active_workspace = config.DEFAULT_WORKSPACE

    if "manager" not in ss:
        ss.manager = FactoryDataManager(workspace=ss.active_workspace)
        logger.info("FactoryDataManager created for workspace '%s'.", ss.active_workspace)

    if "orchestrator" not in ss:
        ss.orchestrator = None   # Lazy — built on first crisis trigger

    if "chat_history" not in ss:
        ss.chat_history = []   # list of LangChain HumanMessage / AIMessage

    if "agent_steps" not in ss:
        ss.agent_steps = []    # list of step dicts yielded by run_specialist

    if "pending_proposal" not in ss:
        ss.pending_proposal = None   # str | None — the Mitigation Proposal text

    if "pending_route" not in ss:
        ss.pending_route = None      # DispatchRoute | None

    if "pending_crisis" not in ss:
        ss.pending_crisis = None     # CrisisEvent | None

    if "schema_profile_summary" not in ss:
        ss.schema_profile_summary = None   # str | None — inferred schema text

    if "schema_profile" not in ss:
        ss.schema_profile = None   # SchemaProfile | None — full inferred object

    if "suggested_crises" not in ss:
        ss.suggested_crises = []

    if "crisis_running" not in ss:
        ss.crisis_running = False


def _get_or_create_orchestrator() -> AgentOrchestrator | None:
    """Return the live ``AgentOrchestrator``, building it lazily on first use.

    Checks that GROQ_API_KEY (required for all LLM calls — Dispatcher,
    Specialists, and Analyst) is present in the environment.  If missing,
    shows a clear sidebar error and returns ``None``.

    Returns:
        The ``AgentOrchestrator`` instance, or ``None`` if setup is incomplete.
    """
    if ss.orchestrator is not None:
        return ss.orchestrator

    # Validate required API keys
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        st.sidebar.error(
            "🔑 **API Keys Missing:** `GROQ_API_KEY`\n\n"
            "Add it to your `.env` file and restart the app."
        )
        return None

    try:
        ss.orchestrator = AgentOrchestrator(data_manager=ss.manager)
        logger.info("AgentOrchestrator created on first use (Groq + Groq).")
        return ss.orchestrator
    except Exception as exc:
        logger.error(f"Failed to instantiate AgentOrchestrator: {exc}", exc_info=True)
        st.sidebar.error(f"⚠️ **Orchestrator Error:** {exc}")
        ss.orchestrator = None
        return None


_init_session_state()

# Ensure tools always operate on the same per-session FactoryDataManager
# instance that the UI and orchestrator use.  This is called on every
# Streamlit rerun so that the module-level reference in tool_registry
# stays in sync with st.session_state.manager.
set_data_manager(ss.manager)


# ---------------------------------------------------------------------------
# Helper: reload FactoryDataManager after CSV upload
# ---------------------------------------------------------------------------

def _reload_data_manager(workspace: str | None = None) -> None:
    """Rebuild the FactoryDataManager for a (possibly new) workspace."""
    ws = workspace or ss.active_workspace
    ss.active_workspace = ws
    ss.manager = FactoryDataManager(workspace=ws)
    set_data_manager(ss.manager)  # Sync tools to the new instance
    ss.orchestrator = None   # Will be rebuilt lazily with the new manager
    ss.chat_history = []
    ss.agent_steps = []
    ss.pending_proposal = None
    ss.pending_route = None
    ss.pending_crisis = None
    ss.schema_profile_summary = None
    ss.schema_profile = None
    ss.suggested_crises = []
    logger.info("Data manager reloaded for workspace '%s'.", ws)


# ---------------------------------------------------------------------------
# Helper: parse committed actions from proposal text
# ---------------------------------------------------------------------------

def _parse_proposal_actions(proposal_text: str) -> list[dict[str, str]]:
    """Extract action lines from the MITIGATION PROPOSAL block.

    Looks for lines between ``ACTIONS TAKEN:`` and the next heading.
    Returns a list of raw action strings for display.

    Args:
        proposal_text: The full text of the agent's final answer.

    Returns:
        List of action description strings.
    """
    match = re.search(
        r"ACTIONS TAKEN:\s*(.*?)(?:FINANCIAL IMPACT|RATIONALE|---)",
        proposal_text,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return []
    block = match.group(1).strip()
    lines = [ln.strip().lstrip("-•*").strip() for ln in block.splitlines() if ln.strip()]
    return lines


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> CrisisEvent | None:
    """Render the Sidebar and return a CrisisEvent if the user triggers one."""
    with st.sidebar:
        st.markdown("## 🏭 Sentinel Control Panel")
        st.divider()

        # ── Workspace Management ───────────────────────────────────────────
        st.markdown("### 📂 Workspaces")

        # Show active workspace with metadata
        meta = FactoryDataManager.get_workspace_metadata(ss.active_workspace)
        if meta:
            updated = meta.get("last_updated", "")[:10]  # date portion
            updates = meta.get("update_count", 0)
            txns = meta.get("transaction_count", 0)
            st.info(
                f"**{ss.active_workspace}**  \n"
                f"{meta.get('row_count', '?')} rows · "
                f"Updated {updated} · "
                f"{updates} upload(s) · "
                f"{txns} transactions"
            )
        else:
            st.info(f"Active: **{ss.active_workspace}**")

        # Switch workspace
        existing_ws = FactoryDataManager.list_workspaces()
        if existing_ws:
            switch_ws = st.selectbox(
                "Switch workspace",
                options=existing_ws,
                index=existing_ws.index(ss.active_workspace) if ss.active_workspace in existing_ws else 0,
                key="ws_selector",
            )
            if switch_ws != ss.active_workspace:
                _reload_data_manager(workspace=switch_ws)
                profile = DynamicSchemaInferencer(ss.manager.get_inventory()).infer()
                ss.schema_profile_summary = profile.as_agent_summary()
                ss.schema_profile = profile
                st.rerun()

        # Upload mode selector
        upload_mode = st.radio(
            "Upload mode",
            options=["New workspace", "Update current workspace"],
            index=0,
            key="upload_mode",
            horizontal=True,
            help=(
                "**New workspace**: Creates a fresh workspace with empty history.  \n"
                "**Update current**: Replaces inventory data but keeps transaction "
                "log and trust scores — ideal for weekly/monthly data refreshes."
            ),
        )

        uploaded_file = st.file_uploader(
            label="Upload CSV",
            type=["csv"],
            key="csv_uploader",
            help="Columns can have any names — the schema engine infers the rules automatically.",
        )
        if uploaded_file is not None:
            last_uploaded = ss.get("_last_uploaded_csv_name", "")
            if uploaded_file.name != last_uploaded:
                try:
                    new_df = pd.read_csv(uploaded_file)

                    if upload_mode == "New workspace":
                        # Derive workspace name from filename
                        ws_name = Path(uploaded_file.name).stem
                        ws_name = re.sub(r"[^\w\-]", "_", ws_name).strip("_") or "uploaded"

                        _reload_data_manager(workspace=ws_name)
                        new_df.to_csv(ss.manager.workspace_dir / "inventory.csv", index=False)
                        _reload_data_manager(workspace=ws_name)
                        ss.manager.save_metadata(original_filename=uploaded_file.name)
                        msg = f"Workspace **{ws_name}** created — {len(new_df)} rows loaded."

                    else:
                        # Update current workspace — preserve history
                        ss.manager.update_inventory_data(new_df)
                        ss.manager.save_metadata(original_filename=uploaded_file.name)
                        # Clear UI state but keep trust scores & transaction log
                        ss.orchestrator = None
                        ss.agent_steps = []
                        ss.chat_history = []
                        ss.pending_proposal = None
                        ss.pending_route = None
                        ss.pending_crisis = None
                        ss.suggested_crises = []
                        msg = (
                            f"Workspace **{ss.active_workspace}** updated — "
                            f"{len(new_df)} rows loaded. "
                            f"Transaction log and trust scores preserved."
                        )

                    profile = DynamicSchemaInferencer(new_df).infer()
                    ss.schema_profile_summary = profile.as_agent_summary()
                    ss.schema_profile = profile
                    ss["_last_uploaded_csv_name"] = uploaded_file.name
                    st.success(f"✅ {msg}")

                    # Auto-scan the newly loaded data
                    with st.spinner("Agents are analyzing data for risks..."):
                        orch = _get_or_create_orchestrator()
                        if orch:
                            ss.suggested_crises = orch.scan_for_crises()
                except Exception as exc:
                    st.error(f"❌ Failed to load CSV: {exc}")
            else:
                st.success(f"✅ Using **{uploaded_file.name}**")

        # Delete workspace
        if existing_ws and len(existing_ws) > 1:
            st.markdown("---")
            del_ws = st.selectbox(
                "Delete workspace",
                options=[w for w in existing_ws if w != ss.active_workspace],
                key="ws_delete_selector",
                help="Cannot delete the currently active workspace.",
            )
            if st.button("🗑️ Delete", key="ws_delete_btn", type="secondary"):
                FactoryDataManager.delete_workspace(del_ws)
                st.success(f"Workspace **{del_ws}** deleted.")
                st.rerun()

        st.divider()

        # ── Trust Scores ──────────────────────────────────────────────────
        st.markdown("### 🎯 Agent Trust Scores")
        trust_data = ss.manager.get_trust_scores()
        agents = trust_data.get("agents", {})
        threshold = config.ROUTING_THRESHOLD

        for agent_id, info in agents.items():
            score: float = float(info.get("trust_score", 0.0))
            pct = int(score * 100)
            fill_class = "crit" if score < threshold else ("warn" if score < 0.75 else "")
            penalties = info.get("recent_penalties", 0)
            col_label, col_score = st.columns([2, 1])
            with col_label:
                emoji = "🔴" if score < threshold else ("🟡" if score < 0.75 else "🟢")
                st.markdown(
                    f"**{emoji} {agent_id.capitalize()}**"
                    f"{'  ⚠️ OVERRIDE ACTIVE' if score < threshold else ''}"
                )
            with col_score:
                st.markdown(f"**{score:.2f}**")
            st.markdown(
                f'<div class="trust-bar-container">'
                f'<div class="trust-bar">'
                f'<div class="trust-fill {fill_class}" style="width:{pct}%"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Decisions: {info.get('total_decisions', 0)} | Penalties: {penalties}")

        st.caption(f"Routing threshold: **{threshold:.2f}**")
        st.divider()

        # ── Crisis Simulator ──────────────────────────────────────────────
        st.markdown("### 🚨 Crisis Event Simulator")

        # Free-text input for custom crisis descriptions
        crisis_text = st.text_area(
            "Describe your crisis or problem:",
            placeholder=(
                "Example: Shipping delays have caused a 2-week backlog on "
                "incoming raw materials. We need to reallocate inventory and "
                "find alternative suppliers."
            ),
            height=100,
            key="crisis_text_input",
        )

        col_sev, col_type = st.columns(2)
        with col_sev:
            severity = st.selectbox(
                "Severity",
                options=["CRITICAL", "HIGH", "MEDIUM"],
                key="crisis_severity",
            )
        with col_type:
            event_type = st.selectbox(
                "Type",
                options=["EXTERNAL_SHOCK", "INTERNAL_FAILURE", "CAPACITY_WARNING", "MARKET_SHOCK"],
                key="crisis_type",
            )

        # Optional: Quick presets (expand for convenience)
        with st.expander("Quick Presets (toy factory demos)"):
            preset_name = st.selectbox(
                "Load a preset:",
                options=["(none)"] + list(MOCK_CRISES.keys()),
                key="crisis_preset",
            )
            if preset_name != "(none)":
                st.caption(f"📋 {MOCK_CRISES[preset_name].description}")

        trigger_crisis: CrisisEvent | None = None
        if st.button("🔴 Trigger Crisis", type="primary", use_container_width=True):
            # Use preset if selected, otherwise build from free text
            if preset_name != "(none)" and not crisis_text.strip():
                trigger_crisis = MOCK_CRISES[preset_name]
            elif crisis_text.strip():
                import uuid as _uuid
                trigger_crisis = CrisisEvent(
                    event_id=f"EVT-{_uuid.uuid4().hex[:8].upper()}",
                    event_type=event_type,
                    severity=severity,
                    description=crisis_text.strip(),
                    affected_entities={},
                )
            else:
                st.warning("Please describe your crisis or select a preset.")

            if trigger_crisis:
                ss.agent_steps = []
                ss.pending_proposal = None
                ss.pending_route = None

        st.divider()

        # ── Suggested Crises (Auto-Detected) ──────────────────────────────
        if getattr(ss, "suggested_crises", None):
            st.markdown("### 🔍 Suggested Crises")
            st.caption("The Risk Scanner detected these potential issues in your data. Click one to trigger it.")
            for i, c in enumerate(ss.suggested_crises):
                if st.button(f"🔴 {c.severity}: {c.description[:80]}...", key=f"sugg_crisis_{i}"):
                    trigger_crisis = c
                    ss.agent_steps = []
                    ss.pending_proposal = None
                    ss.pending_route = None
            st.divider()

        # ── Analyst Panel ─────────────────────────────────────────────────
        st.markdown("### 📊 Run Analyst (Trust Review)")
        st.caption("Analyses transaction history and applies trust score penalties/rewards.")
        if st.button("🔬 Evaluate Agent Performance", use_container_width=True):
            orch = _get_or_create_orchestrator()
            if orch is None:
                st.sidebar.warning("Connect API key (GROQ_API_KEY) in .env first.")
            else:
                with st.spinner("Analyst reviewing transaction ledger..."):
                    verdict = orch.run_analyst(chat_history=ss.chat_history)
                    ss.agent_steps.append({"type": "analyst_verdict", "content": verdict})
                st.rerun()

        st.divider()
        st.caption("Sentinel v1.0 · Groq (All Agents)")
        st.caption(status_summary())

    return trigger_crisis


# ---------------------------------------------------------------------------
# Tab 1: Data & Schema
# ---------------------------------------------------------------------------

def _render_data_tab() -> None:
    """Render the live inventory DataFrame and inferred schema rules."""
    st.markdown("## 📋 Live Inventory — Master Clipboard")

    inv_df = ss.manager.get_inventory()
    if inv_df.empty:
        st.warning("No inventory data loaded. Upload a CSV in the sidebar.")
        return

    # Run or reuse schema inference
    if ss.schema_profile is None:
        inferencer = DynamicSchemaInferencer(inv_df)
        profile = inferencer.infer()
        ss.schema_profile = profile
        ss.schema_profile_summary = profile.as_agent_summary()

    profile = ss.schema_profile

    # Identify dynamic columns from inferred constraint pairs
    mutable_col = None
    limit_col = None
    pk_col = profile.primary_key_column if profile else None
    if profile and profile.constraint_rules:
        mutable_col = profile.constraint_rules[0].mutable_column
        limit_col = profile.constraint_rules[0].limit_column

    # Colour code rows where mutable value > 80% of limit
    def _highlight_stock(row: pd.Series) -> list[str]:
        styles: list[str] = [""] * len(row)
        if mutable_col and limit_col and mutable_col in row.index and limit_col in row.index:
            try:
                ratio = float(row[mutable_col]) / max(float(row[limit_col]), 1)
                if ratio >= 0.90:
                    styles = ["background-color: #3a1a1a"] * len(row)
                elif ratio >= 0.75:
                    styles = ["background-color: #3a2a10"] * len(row)
            except (ValueError, TypeError):
                pass
        return styles

    st.dataframe(
        inv_df.style.apply(_highlight_stock, axis=1),
        use_container_width=True,
        height=320,
    )

    # Dynamic legend based on detected constraint pair
    if mutable_col and limit_col:
        st.caption(
            f"🔴 Red rows = ≥90% capacity | 🟡 Amber rows = ≥75% capacity "
            f"(comparing **{mutable_col}** against **{limit_col}**). "
            f"Data refreshes on every page interaction."
        )
    else:
        st.caption("Data refreshes on every page interaction.")

    st.divider()
    st.markdown("## 🧠 Dynamic Schema Profile")
    st.caption(
        "The Schema Engine inspects the uploaded CSV and infers constraint rules "
        "automatically — no hard-coded column names."
    )

    with st.expander("View Inferred Schema Rules", expanded=True):
        st.code(ss.schema_profile_summary, language="yaml")

    # Dynamic capacity gauges using inferred constraint pair
    if mutable_col and limit_col and pk_col:
        col_count = min(len(inv_df), 4)
        if col_count > 0:
            st.markdown("#### Capacity Gauges")
            cols = st.columns(col_count)
            for idx, (_, row) in enumerate(inv_df.head(col_count).iterrows()):
                with cols[idx]:
                    try:
                        current_val = float(row[mutable_col])
                        limit_val = max(float(row[limit_col]), 1)
                        pct = min(100, int((current_val / limit_val) * 100))
                        color = "#e74c3c" if pct >= 90 else ("#f39c12" if pct >= 75 else "#2ecc71")
                        label = str(row.get(pk_col, f"Row {idx}"))[:20]
                        st.markdown(f"**{label}**")
                        st.markdown(
                            f'<div style="background:#1a1a2a;border-radius:6px;height:12px;margin:4px 0;">'
                            f'<div style="background:{color};width:{pct}%;height:100%;border-radius:6px;"></div>'
                            f'</div>'
                            f'<span style="font-size:0.75rem;color:#888">{pct}% ({int(current_val)} / {int(limit_val)})</span>',
                            unsafe_allow_html=True,
                        )
                    except (ValueError, TypeError):
                        st.caption(f"Row {idx}: non-numeric data")

    st.divider()
    st.markdown("## 📜 Transaction Ledger")
    log_df = ss.manager.get_transaction_log()
    if log_df.empty:
        st.info("No transactions recorded yet. Trigger a crisis to generate activity.")
    else:
        st.dataframe(log_df, use_container_width=True, height=250)


# ---------------------------------------------------------------------------
# Tab 2: Crisis Console
# ---------------------------------------------------------------------------

def _render_chat_bubble(step: dict[str, Any]) -> None:
    """Render a single agent step as a styled message bubble."""
    step_type = step.get("type", "")
    content = step.get("content", "")

    if step_type == "dispatch":
        route: DispatchRoute = step["route"]
        override = route.trust_override_applied
        css_class = "msg-override" if override else "msg-system"
        icon = "⚡" if override else "📡"
        override_text = (
            f"<br><strong>⚠️ TRUST OVERRIDE:</strong> LLM selected "
            f"<code>{route.original_llm_choice}</code> but trust score was below threshold. "
            f"Rerouted to <code>{route.selected_agent}</code>."
            if override else ""
        )
        st.markdown(
            f'<div class="{css_class}">'
            f'{icon} <strong>Dispatcher Routing</strong> → '
            f'<code>{html.escape(route.selected_agent.upper())}</code> '
            f'[{html.escape(route.urgency_tier)}]<br>'
            f'<em>{html.escape(route.delegation_justification)}</em>'
            f'{override_text}'
            f'</div>',
            unsafe_allow_html=True,
        )

    elif step_type == "tool_call":
        tool_name = html.escape(str(step.get("tool", "tool")))
        safe_content = html.escape(content[:200])
        st.markdown(
            f'<div class="msg-tool">🔧 <strong>{tool_name}</strong>('
            f'{safe_content}{"..." if len(content) > 200 else ""})</div>',
            unsafe_allow_html=True,
        )

    elif step_type == "tool_result":
        tool_name = step.get("tool", "tool")
        is_rejection = "SANDBOX REJECTION" in content or "REJECTED" in content
        css_class = "msg-rejected" if is_rejection else "msg-tool"
        icon = "❌" if is_rejection else "✅"
        with st.expander(f"{icon} {tool_name} result", expanded=is_rejection):
            st.code(content[:1200], language="text")

    elif step_type == "final_answer":
        agent_id = step.get("agent_id", "agent")
        has_proposal = "--- MITIGATION PROPOSAL ---" in content
        st.markdown(
            f'<div class="msg-agent">🤖 <strong>{agent_id.capitalize()}</strong> says:</div>',
            unsafe_allow_html=True,
        )
        if has_proposal:
            pre, _, proposal_block = content.partition("--- MITIGATION PROPOSAL ---")
            if pre.strip():
                st.markdown(pre.strip())
            proposal_full = "--- MITIGATION PROPOSAL ---" + proposal_block
            st.markdown(
                f'<div class="msg-safe">{html.escape(proposal_full).replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(content)

    elif step_type == "analyst_verdict":
        st.markdown("### 📊 Analyst Verdict")
        st.markdown(
            f'<div class="msg-system">{html.escape(content).replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

    elif step_type == "human":
        st.markdown(
            f'<div class="msg-human">👤 <strong>Human Operator</strong><br>{html.escape(content)}</div>',
            unsafe_allow_html=True,
        )

    elif step_type == "system":
        st.markdown(
            f'<div class="msg-system">⚙️ {html.escape(content)}</div>',
            unsafe_allow_html=True,
        )

    elif step_type == "auto_commit":
        # F4 — confidence-routed auto-commit notice.
        st.markdown(
            f'<div class="msg-safe">{html.escape(content)}</div>',
            unsafe_allow_html=True,
        )

    elif step_type == "error":
        st.error(f"❌ {content}")


def _render_crisis_console(trigger_crisis: CrisisEvent | None) -> None:
    """Render the Crisis Console tab with chat UI and Human-in-the-Loop controls."""
    st.markdown("## 🚨 Crisis Console")

    # Safety: reset stuck crisis_running flag (e.g. after Streamlit script kill)
    if ss.get("crisis_running") and ss.get("_crisis_start_time"):
        import time
        if time.time() - ss._crisis_start_time > 120:  # 2-minute timeout
            ss.crisis_running = False
            logger.warning("Reset stuck crisis_running flag after timeout.")

    # ── If a new crisis was triggered ────────────────────────────────────
    if trigger_crisis is not None and not ss.crisis_running:
        import time as _time
        ss.crisis_running = True
        ss._crisis_start_time = _time.time()
        crisis = trigger_crisis

        try:
            st.markdown(
                f'<div class="msg-system">🚨 <strong>CRISIS INCOMING:</strong> '
                f'[{crisis.event_id}] — {crisis.description[:200]}</div>',
                unsafe_allow_html=True,
            )

            # Step 1: Dispatch
            with st.spinner("🔀 Dispatcher routing crisis to specialist agent..."):
                orch = _get_or_create_orchestrator()
                if orch is None:
                    st.error("⚠️ Orchestrator not ready. Check GROQ_API_KEY in .env.")
                    return
                route: DispatchRoute = orch.dispatch(crisis=crisis)
                ss.pending_route = route
                ss.pending_crisis = crisis
                ss.agent_steps.append({"type": "dispatch", "route": route})

            # Step 2: Run specialist with streaming steps
            st.markdown(
                f"**Routing to: `{route.selected_agent.upper()}`** — collecting data and "
                f"formulating mitigation plan..."
            )

            progress_placeholder = st.empty()

            with st.spinner(f"🤖 {route.selected_agent.capitalize()} agent working..."):
                for step in orch.run_specialist(
                    route=route,
                    crisis=crisis,
                    chat_history=ss.chat_history,
                ):
                    ss.agent_steps.append(step)
                    if step["type"] == "final_answer":
                        ss.pending_proposal = step["content"]
                        from langchain_core.messages import AIMessage as AI, HumanMessage as HM
                        ss.chat_history.append(HM(content=crisis.description))
                        ss.chat_history.append(AI(content=step["content"]))
                    progress_placeholder.empty()

            # ── F4 — Confidence-routed auto-commit pass ────────────────
            # Runs after the specialist finishes staging. Only commits
            # changes that pass auto_commit_eligible (confidence + trust
            # + delta-fraction); ineligible changes remain staged for
            # human approval. Master switch is config.AUTO_COMMIT_ENABLED.
            _maybe_auto_commit()

        except RuntimeError as exc:
            ss.agent_steps.append({"type": "error", "content": f"Dispatcher failed: {exc}"})
        except Exception as exc:
            ss.agent_steps.append({"type": "error", "content": f"Crisis handling failed: {exc}"})
            logger.error("Crisis handling failed: %s", exc, exc_info=True)
        finally:
            ss.crisis_running = False

        st.rerun()

    # ── Render all accumulated steps ─────────────────────────────────────
    if not ss.agent_steps:
        st.info(
            "👈 Select a crisis scenario from the sidebar and click **Trigger Crisis** "
            "to begin the multi-agent simulation."
        )
    else:
        for step in ss.agent_steps:
            _render_chat_bubble(step)

    st.divider()

    # ── Human-in-the-Loop Approval Panel ─────────────────────────────────
    if ss.pending_proposal and "--- MITIGATION PROPOSAL ---" in ss.pending_proposal:
        st.markdown("---")
        st.markdown("## ✅ Human-in-the-Loop: Approval Required")
        st.warning(
            "The agent has produced a validated Mitigation Proposal. "
            "All proposed state changes were approved by the Shadow Sandbox. "
            "**Your approval is required to commit these changes to the Master Clipboard.**"
        )

        proposal_text = ss.pending_proposal
        actions = _parse_proposal_actions(proposal_text)

        if actions:
            st.markdown("**Proposed Actions:**")
            for action_line in actions:
                st.markdown(f"- {action_line}")

        # Show count of staged changes (these are the changes that did
        # NOT auto-commit — either the master switch is off, or one of
        # the F4 thresholds was not met). Render confidence + skip-reason
        # badges so the operator can see WHY each change is pending.
        staged_count = ss.manager.pending_changes_count()
        if staged_count > 0:
            st.info(
                f"📦 **{staged_count} state change(s)** validated by the Sandbox "
                f"and staged for your approval."
            )
            with st.expander("🔍 Confidence & auto-commit eligibility", expanded=False):
                for ch in ss.manager.pending_changes:
                    conf = ch.get("confidence")
                    conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
                    skip_reason = ch.get("_auto_commit_skip_reason", "—")
                    st.markdown(
                        f"- **{ch.get('row_key')}.{ch.get('target_column')}** "
                        f"`{ch.get('delta', 0):+}` · "
                        f"agent=`{ch.get('agent_id', 'unknown')}` · "
                        f"confidence=`{conf_str}` · "
                        f"_not auto-committed_: {skip_reason}"
                    )

        # ── Monte Carlo uncertainty quantification (optional, pre-commit) ───
        if ss.manager.pending_changes:
            with st.expander("🎲 Monte Carlo Simulation (optional)", expanded=False):
                n_runs = st.slider(
                    "Runs",
                    min_value=100,
                    max_value=5000,
                    value=config.MONTE_CARLO_DEFAULT_RUNS,
                    step=100,
                    key="mc_n_runs",
                )
                if st.button("Run Simulation", key="mc_run_btn"):
                    from src.core.simulator import MonteCarloSimulator
                    sim = MonteCarloSimulator(
                        live_df=ss.manager.get_inventory(),
                        schema=ss.manager.get_schema_profile(),
                        pending_changes=ss.manager.pending_changes,
                        n_runs=n_runs,
                        noise_std_pct=config.MONTE_CARLO_NOISE_STD_PCT,
                    )
                    with st.spinner(f"Running {n_runs} simulations..."):
                        result = sim.run()
                    st.metric("Rejection rate", f"{result.rejection_rate:.1%}")
                    if result.metrics:
                        df_metrics = pd.DataFrame(result.metrics).T
                        st.dataframe(df_metrics)
                    if result.constraint_violations:
                        st.warning(
                            f"Sample violations: {result.constraint_violations[:3]}"
                        )

        col_approve, col_reject, col_spacer = st.columns([2, 2, 6])
        with col_approve:
            if st.button("✅ Approve & Execute", type="primary", use_container_width=True):
                _handle_approval(proposal_text)

        with col_reject:
            if st.button("❌ Reject Plan", use_container_width=True):
                discarded = discard_pending_changes()
                ss.agent_steps.append({
                    "type": "system",
                    "content": f"Human operator REJECTED the mitigation proposal. {discarded} staged change(s) discarded.",
                })
                ss.pending_proposal = None
                ss.pending_route = None
                st.rerun()

    # ── Custom chat input ─────────────────────────────────────────────────
    st.divider()
    col_title, col_clear = st.columns([4, 1])
    with col_title:
        st.markdown("#### 💬 Direct Agent Query")
    with col_clear:
        if st.button("🔄 Clear", key="clear_history_btn", help="Clear chat history and agent steps"):
            ss.chat_history = []
            ss.agent_steps = []
            ss.pending_proposal = None
            ss.pending_route = None
            ss.pending_crisis = None
            st.rerun()

    user_query = st.chat_input(
        placeholder="Ask a question about the inventory or request an action...",
        key="direct_query",
    )
    if user_query:
        _handle_direct_query(user_query)


def _maybe_auto_commit() -> None:
    """F4 — Run a confidence-routed auto-commit pass after the specialist.

    If ``config.AUTO_COMMIT_ENABLED`` is False (default), this is a no-op
    and behaviour is byte-identical to pre-F4 deployments. When enabled,
    we ask ``commit_pending_changes(auto_only=True)`` to commit only the
    high-confidence + high-trust + small-delta changes; everything else
    remains staged for the human operator. Sandbox re-validation runs at
    commit time on every change regardless of which path was taken.
    """
    if not config.AUTO_COMMIT_ENABLED:
        return
    if ss.manager is None or ss.manager.pending_changes_count() == 0:
        return

    try:
        summary = commit_pending_changes(ss.manager, auto_only=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Auto-commit pass failed: %s", exc, exc_info=True)
        ss.agent_steps.append({
            "type": "error",
            "content": f"Auto-commit pass failed: {exc}",
        })
        return

    auto_count = int(summary.get("auto_committed_count", 0))
    if auto_count > 0:
        change_summary = "; ".join(
            f"{c['row_key']}.{c['target_column']} {c['delta']:+.0f} "
            f"(conf={c.get('confidence', 0.0):.2f})"
            for c in summary.get("auto_committed", [])
        )
        ss.agent_steps.append({
            "type": "auto_commit",
            "content": (
                f"⚡ Auto-committed {auto_count} high-confidence change(s) "
                f"without human approval: {change_summary}"
            ),
            "auto_committed": summary.get("auto_committed", []),
        })

    deferred_count = int(summary.get("deferred_count", 0))
    if deferred_count > 0:
        # Soft notice — these are still in the HITL queue for review.
        ss.agent_steps.append({
            "type": "system",
            "content": (
                f"🧑‍⚖️ {deferred_count} change(s) require human approval "
                f"(below auto-commit thresholds)."
            ),
        })


def _handle_approval(proposal_text: str) -> None:
    """Commit all staged state changes to the live data layer."""
    ss.agent_steps.append({
        "type": "system",
        "content": "✅ Human operator APPROVED the plan. Committing staged changes to the Master Clipboard...",
    })

    # Commit staged changes via two-phase commit
    committed = commit_pending_changes()

    if committed:
        change_summary = "; ".join(
            f"{c['row_key']}.{c['target_column']} {c['delta']:+.0f}"
            for c in committed
        )
        ss.agent_steps.append({
            "type": "system",
            "content": (
                f"📌 {len(committed)} change(s) committed: {change_summary}. "
                f"Transaction ledger updated."
            ),
        })
    else:
        ss.agent_steps.append({
            "type": "system",
            "content": "📌 Approval recorded. No pending state changes to commit.",
        })

    # Log the human approval as a special transaction
    if ss.pending_crisis is not None:
        try:
            ss.manager.log_transaction(
                event_id=ss.pending_crisis.event_id,
                agent_id=ss.pending_route.selected_agent if ss.pending_route else "unknown",
                action_schema={"action": "HUMAN_APPROVED", "proposal": proposal_text[:500]},
                financial_impact=0.0,
                sandbox_approved=True,
            )
        except Exception as exc:
            logger.warning("Could not log approval transaction: %s", exc)

    ss.pending_proposal = None
    ss.pending_route = None
    ss.pending_crisis = None
    ss.schema_profile_summary = None
    ss.schema_profile = None   # Force schema re-inference after data changes
    st.rerun()


def _is_action_query(query: str) -> bool:
    """Return True if the query implies a crisis or action that mutates state.

    Informational queries (show, list, what, how many, details, etc.) return
    False so they go through the read-only info path instead of the Dispatcher.
    """
    lower = query.lower().strip()

    # Strong informational signals → always informational
    info_patterns = (
        "show", "list", "display", "what is", "what are", "how many",
        "tell me", "give me", "details", "describe", "summary", "overview",
        "which", "who", "where", "status", "check", "view", "get",
        "print", "fetch", "report",
    )
    if any(lower.startswith(p) or p in lower for p in info_patterns):
        # Only override if there are imperative action commands present.
        # Phrases like "reorder point" are data concepts, not actions.
        action_verbs = (
            "quarantine", "reallocate", "increase", "reduce",
            "transfer", "rush", "halt", "shut down",
            "replenish", "expedite", "divert", "reroute",
        )
        # "reorder" is only an action when NOT followed by "point"/"level"
        has_reorder_action = "reorder" in lower and not re.search(r"reorder\s*(point|level|threshold)", lower)
        # "recall" is only an action when NOT preceded by "about"/"regarding"
        has_recall_action = "recall" in lower and not re.search(r"(about|regarding|of)\s+.*recall", lower)
        # "ship" is only action when NOT part of "shipment"
        has_ship_action = "ship" in lower and "shipment" not in lower and not re.search(r"ship(ping|ped|s)", lower)
        # "move" is only action when NOT part of "movement"
        has_move_action = "move" in lower and "movement" not in lower

        explicit_actions = any(v in lower for v in action_verbs)
        if not (explicit_actions or has_reorder_action or has_recall_action or has_ship_action or has_move_action):
            return False

    # Crisis / action signals
    action_keywords = (
        "crisis", "emergency", "failure", "failed", "broke", "broken",
        "recall", "strike", "disrupted", "disruption", "shortage",
        "stockout", "overflow", "spike", "surge", "halted", "spoil",
        "quarantine", "reorder", "reallocate", "increase production",
        "reduce stock", "transfer", "ship", "rush", "urgent",
        "expedite", "divert", "reroute", "replenish", "expired",
        "defect", "contaminated", "damaged",
    )
    return any(kw in lower for kw in action_keywords)


def _handle_direct_query(query: str) -> None:
    """Route a direct user query — informational or action-oriented."""
    from langchain_core.messages import AIMessage as AI, HumanMessage as HM

    orch = _get_or_create_orchestrator()
    if orch is None:
        ss.agent_steps.append({"type": "error", "content": "Orchestrator not ready. Check API keys in .env."})
        st.rerun()
        return

    ss.agent_steps.append({"type": "human", "content": query})

    if _is_action_query(query):
        # Action/crisis query → full Dispatcher + specialist pipeline
        synthetic_crisis = CrisisEvent(
            event_id="EVT-DIRECT-QUERY",
            event_type="OPERATOR_QUERY",
            severity="LOW",
            description=query,
            affected_entities={},
        )

        with st.spinner("Routing and processing query..."):
            try:
                route = orch.dispatch(crisis=synthetic_crisis)
                ss.agent_steps.append({"type": "dispatch", "route": route})

                for step in orch.run_specialist(
                    route=route,
                    crisis=synthetic_crisis,
                    chat_history=ss.chat_history,
                ):
                    ss.agent_steps.append(step)
                    if step["type"] == "final_answer":
                        ss.pending_proposal = step["content"]
                        ss.pending_route = route
                        ss.pending_crisis = synthetic_crisis
                        ss.chat_history.append(HM(content=query))
                        ss.chat_history.append(AI(content=step["content"]))
                _maybe_auto_commit()
            except Exception as exc:
                ss.agent_steps.append({"type": "error", "content": str(exc)})
    else:
        # Informational query → direct answer without crisis fabrication
        with st.spinner("Looking up data..."):
            try:
                for step in orch.answer_query(
                    query=query,
                    chat_history=ss.chat_history,
                ):
                    ss.agent_steps.append(step)
                    if step["type"] == "final_answer":
                        ss.chat_history.append(HM(content=query))
                        ss.chat_history.append(AI(content=step["content"]))
            except Exception as exc:
                ss.agent_steps.append({"type": "error", "content": str(exc)})

    st.rerun()


# ---------------------------------------------------------------------------
# Main Page Layout
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the Streamlit application."""
    # Render header
    st.markdown(
        """
<div class="sentinel-header">
  <h1>🏭 Sentinel</h1>
  <p>Supply Chain Digital Twin · Multi-Agent Orchestration · Groq LPU</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Render sidebar (returns a CrisisEvent if user clicked Trigger)
    triggered_crisis = _render_sidebar()

    # Main content tabs
    tab_data, tab_console = st.tabs(["📋 Data & Schema", "🚨 Crisis Console"])

    with tab_data:
        _render_data_tab()

    with tab_console:
        _render_crisis_console(trigger_crisis=triggered_crisis)


if __name__ == "__main__":
    main()
