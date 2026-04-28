"""
src/core/config.py
==================
Global configuration constants for the Sentinel Supply Chain Digital Twin.

All file paths, financial thresholds, and routing constants are centralized here
so that any component in the system can import a single source of truth instead
of hard-coding magic strings or numbers.
"""

import os
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Root Paths
# ---------------------------------------------------------------------------

#: Absolute path to the project root (two levels above this file: src/core/ → root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Directory where all persistent data artefacts are stored.
DATA_DIR: Path = PROJECT_ROOT / "data"

#: Root directory for named workspaces. Each workspace is a subdirectory
#: containing its own inventory.csv, transaction_log.csv, and trust scores.
WORKSPACES_DIR: Path = DATA_DIR / "workspaces"

#: Default workspace name used when no workspace is explicitly selected.
DEFAULT_WORKSPACE: str = "default"

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
# Monte Carlo Simulation
# ---------------------------------------------------------------------------

#: Default number of stochastic runs the Monte Carlo simulator executes when
#: the human approver triggers a pre-commit uncertainty check from the HITL
#: panel.  1000 strikes a balance between distribution stability and UI latency.
MONTE_CARLO_DEFAULT_RUNS: int = 1000

#: Standard deviation (as a fraction of the proposed delta) of the multiplicative
#: Gaussian noise applied to each pending change during simulation.  ``0.10``
#: corresponds to ±10% noise — a realistic supply-chain execution variance.
MONTE_CARLO_NOISE_STD_PCT: float = 0.10

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

# ---------------------------------------------------------------------------
# Observability — Langfuse LLM tracing (auto-enabled when env vars present)
# ---------------------------------------------------------------------------
# These constants document the env-var names the observability layer reads.
# The actual ``os.environ`` lookups happen in
# ``src/observability/tracing.py`` so this module stays import-free of any
# observability runtime. Tracing is auto-enabled when both the public and
# secret keys are present; absence of either disables it (no-op path).

#: Env-var holding the Langfuse public key (required to enable tracing).
LANGFUSE_PUBLIC_KEY_ENV: str = "LANGFUSE_PUBLIC_KEY"

#: Env-var holding the Langfuse secret key (required to enable tracing).
LANGFUSE_SECRET_KEY_ENV: str = "LANGFUSE_SECRET_KEY"

#: Optional self-hosted Langfuse URL. Defaults to https://cloud.langfuse.com
#: when unset (handled by the Langfuse SDK).
LANGFUSE_HOST_ENV: str = "LANGFUSE_HOST"

# ---------------------------------------------------------------------------
# F10 — Cost tracking
# ---------------------------------------------------------------------------
# Per-1k-token rates in USD: (input_rate, output_rate). Update when Groq
# pricing changes. "default" used when model name is not in this map.
MODEL_COST_PER_1K_TOKENS: Dict[str, tuple] = {
    "llama-3.1-8b-instant": (0.00005, 0.00008),
    "llama-3.3-70b-versatile": (0.00059, 0.00079),
    "default": (0.0005, 0.0008),
}

#: Per-session USD budget cap. Once accumulated cost >= this, the next
#: LLM call raises BudgetExceededError. Set high to disable enforcement
#: in development.
MAX_LLM_USD_PER_SESSION: float = float(os.environ.get("SENTINEL_MAX_LLM_USD", "5.00"))

#: Master switch — when False, the tracker still counts but does not
#: raise on budget breach.
COST_TRACKING_ENFORCE: bool = os.environ.get("SENTINEL_COST_ENFORCE", "true").lower() in (
    "1", "true", "yes", "on"
)

# ---------------------------------------------------------------------------
# Feature Flags — Phase 4 (LangGraph migration)
# ---------------------------------------------------------------------------

#: When True, ``AgentOrchestrator.run_specialist`` routes through the new
#: LangGraph ``StateGraph`` in ``src/agents/graph.py`` instead of the
#: legacy manual ReAct loop. Defaults to False so production stays on the
#: proven path; flip to True (via ``SENTINEL_USE_LANGGRAPH=1``) once
#: parity is verified in staging.
USE_LANGGRAPH: bool = os.environ.get(
    "SENTINEL_USE_LANGGRAPH", "false"
).lower() in ("1", "true", "yes", "on")

# ---------------------------------------------------------------------------
# F4 — Confidence-routed HITL ("human-on-the-loop")
# ---------------------------------------------------------------------------
# Auto-commit a staged change WITHOUT human approval iff ALL THREE hold:
#   - agent.confidence            >= AUTO_COMMIT_CONFIDENCE_THRESHOLD
#   - agent.trust_score           >= AUTO_COMMIT_TRUST_THRESHOLD
#   - |delta| / current_value     <= AUTO_COMMIT_MAX_DELTA_FRACTION
# Otherwise the change waits for human approval as before. Sandbox
# re-validation runs at commit time regardless of which path is taken.
#
# AUTO_COMMIT_ENABLED is the master switch — defaults False so existing
# deployments behave identically until explicitly opted-in via the env var
# ``SENTINEL_AUTO_COMMIT_ENABLED``.

#: Minimum self-rated agent confidence required for auto-commit (0.0–1.0).
AUTO_COMMIT_CONFIDENCE_THRESHOLD: float = 0.85

#: Minimum agent trust score required for auto-commit (0.0–1.0).
AUTO_COMMIT_TRUST_THRESHOLD: float = 0.80

#: Maximum |delta| / |current_value| ratio allowed for auto-commit (5%).
AUTO_COMMIT_MAX_DELTA_FRACTION: float = 0.05

#: Master switch for the entire confidence-routed HITL feature. When False
#: (default), auto_commit_eligible always returns False and behaviour is
#: byte-identical to pre-F4 deployments.
AUTO_COMMIT_ENABLED: bool = os.environ.get(
    "SENTINEL_AUTO_COMMIT_ENABLED", "false"
).lower() in ("1", "true", "yes", "on")

# ---------------------------------------------------------------------------
# F5 — Lightweight conversation guardrails layer
# ---------------------------------------------------------------------------
# Wraps Dispatcher INPUT (jailbreak / prompt-injection regex) and Specialist
# / Analyst OUTPUT (PII + secret leak redaction) with a small pluggable
# layer. See ``src/agents/guardrails.py``. The layer is conservative — input
# only blocks on explicit jailbreak signatures, and output WARNs with
# redaction rather than blocking, so a single false positive cannot break a
# crisis response. Default ON; flip ``SENTINEL_GUARDRAILS_ENABLED=false`` to
# disable for local development.

#: Master switch for the F5 guardrails. When False, all ``guard_or_raise``
#: call sites in the orchestrator become no-ops.
GUARDRAILS_ENABLED: bool = os.environ.get(
    "SENTINEL_GUARDRAILS_ENABLED", "true"
).lower() in ("1", "true", "yes", "on")
