"""
src/core/config.py
==================
Global configuration constants for the Sentinel Supply Chain Digital Twin.

All file paths, financial thresholds, and routing constants are centralized here
so that any component in the system can import a single source of truth instead
of hard-coding magic strings or numbers.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root Paths
# ---------------------------------------------------------------------------

#: Absolute path to the project root (two levels above this file: src/core/ → root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Directory where all persistent data artefacts are stored.
DATA_DIR: Path = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Data File Paths
# ---------------------------------------------------------------------------

#: Primary inventory state file — the Master Clipboard.
INVENTORY_CSV: Path = DATA_DIR / "inventory.csv"

#: Append-only transaction ledger used by the Retrospective Weighting engine.
TRANSACTION_LOG_CSV: Path = DATA_DIR / "transaction_log.csv"

#: JSON file storing per-agent dynamic trust scores updated by the Analyst.
AGENT_TRUST_SCORES_JSON: Path = DATA_DIR / "agent_trust_scores.json"

# ---------------------------------------------------------------------------
# Routing & Trust Score Thresholds
# ---------------------------------------------------------------------------

#: Minimum trust score an agent must hold before the Dispatcher will route
#: tasks to it.  Below this threshold the Dispatcher falls back to the
#: agent's ``preferred_fallback`` or escalates to a human operator.
ROUTING_THRESHOLD: float = 0.50

#: Default trust score assigned to every agent on first initialisation.
DEFAULT_TRUST_SCORE: float = 0.95

#: Lower bound for any agent's trust score (cannot go negative).
MIN_TRUST_SCORE: float = 0.0

#: Upper bound for any agent's trust score.
MAX_TRUST_SCORE: float = 1.0

# ---------------------------------------------------------------------------
# Financial Thresholds
# ---------------------------------------------------------------------------

#: Maximum single-transaction spend (USD) an agent may commit to without
#: triggering an immediate human escalation regardless of trust score.
MAX_SINGLE_TRANSACTION_USD: float = 100_000.0

#: Baseline cost threshold used by the Analyst when evaluating whether a
#: previous decision was financially sub-optimal (triggers penalty scoring
#: when actual spend exceeds expected spend by this factor).
FINANCIAL_PENALTY_FACTOR: float = 1.5

# ---------------------------------------------------------------------------
# Sandbox Validation
# ---------------------------------------------------------------------------

#: Maximum number of retry attempts the Sandbox will allow an agent before
#: it surfaces a failure to the UI instead of endlessly looping.
SANDBOX_MAX_RETRIES: int = 3

# ---------------------------------------------------------------------------
# LLM Configuration — Multi-Provider Setup
# ---------------------------------------------------------------------------

#: Dispatcher LLM: small, fast, strict JSON enforcement (Groq — free tier)
#: llama-3.1-8b-instant: 8B params, ~600 tokens/s on Groq, great at structured output.
#: Alternatives on Groq (all free): llama3-8b-8192, llama-3.3-70b-versatile
DISPATCHER_MODEL: str = "llama-3.1-8b-instant"

#: Specialist LLM: large parameter model with high reasoning capability.
#: Used by Maker, Mover, and Keeper ReAct agents.
#: Note: All Llama models (3.x and 4.x) on Groq emit `<function=...>` XML tool
#: calls which Groq's API rejects. We handle this via fallback parser in orchestrator.
SPECIALIST_MODEL: str = "llama-3.3-70b-versatile"

#: Analyst LLM: large context window for reading full CSV logs (Groq)
#: Used by the Analyst trust-review agent and the CSV Risk Scanner.
ANALYST_MODEL: str = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Agent Identifiers
# ---------------------------------------------------------------------------

#: Canonical string identifiers used to key into trust score records.
AGENT_IDS: tuple[str, ...] = ("maker", "mover", "keeper")

#: Fallback chain: if an agent's trust falls below threshold the Dispatcher
#: will prefer the next agent in each agent's personal fallback chain.
AGENT_FALLBACKS: dict[str, str] = {
    "maker": "mover",
    "mover": "keeper",
    "keeper": "maker",
}
