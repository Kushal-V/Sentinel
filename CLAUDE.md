# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Sentinel

Sentinel is a **Supply Chain Digital Twin** — a multi-agent LLM orchestration system built with LangChain, Groq, and Streamlit. It uses specialist AI agents (Maker, Mover, Keeper) to respond to supply chain crises, with a Shadow Sandbox preventing hallucinated state changes and a Retrospective Weighting system (trust scores) that adapts agent routing over time.

## Commands

```bash
# Run the app
streamlit run app.py

# Run tests
pytest tests/test_edge_cases.py -v

# Run a single test class
pytest tests/test_edge_cases.py::TestSandboxHallucinationGuard -v

# Install dependencies (uses uv with pyproject.toml)
uv sync

# Install with dev dependencies
uv sync --group dev
```

## Required Environment

- Python >= 3.11
- A `.env` file with `GROQ_API_KEY` (required for all LLM calls — dispatcher, specialists, analyst)
- See `.env.example` for the template

## Architecture

### Core Pipeline Flow

1. **Crisis Event** arrives (user-triggered or auto-detected by Scanner)
2. **Dispatcher** (Groq, llama-3.1-8b-instant) classifies and routes via `with_structured_output(DispatchRoute)` — outputs strict JSON, no free-text
3. **Trust Override** (Claim B): if the chosen agent's trust score < `ROUTING_THRESHOLD` (0.50), routing is programmatically overridden to the fallback agent
4. **Specialist Agent** (Groq, llama-3.3-70b-versatile) runs a manual ReAct tool loop (not `create_react_agent`, which breaks on Groq)
5. **Shadow Sandbox** (Claim A): every `propose_state_change` is validated against a deep-copied DataFrame clone before staging
6. **Two-Phase Commit**: changes are STAGED in `pending_changes` list, then only COMMITTED when the human clicks "Approve" in the UI (HITL)
7. **Analyst** reviews transaction history and adjusts trust scores

### Key Modules

- **`app.py`** — Streamlit entry point. All mutable state lives in `st.session_state`. The `FactoryDataManager` and `AgentOrchestrator` are singletons stored there.
- **`src/agents/orchestrator.py`** — `AgentOrchestrator` holds three Groq LLM clients (dispatcher, specialist, analyst). Implements manual ReAct loop in `run_specialist()` with XML `<function=...>` error recovery for Groq/Llama bug.
- **`src/agents/prompts.py`** — All system prompts. Agents are persona-driven and must follow mandatory tool call order: `get_dataset_schema` -> `query_data` -> `propose_state_change`.
- **`src/core/schema_engine.py`** — `DynamicSchemaInferencer` detects primary keys and constraint pairs (mutable column -> limit column) from arbitrary DataFrames using regex heuristics. This enables multi-tenancy — no hardcoded column names.
- **`src/core/sandbox.py`** — `ShadowSandbox` deep-copies the live DataFrame and validates proposals against inferred constraints (floor >= 0, ceiling <= limit column). Returns `SandboxResult` with SAFE/REJECTED status.
- **`src/core/state_manager.py`** — `FactoryDataManager` owns three persistent artifacts: `inventory.csv`, `transaction_log.csv`, `agent_trust_scores.json`. All writes are guarded by `threading.Lock`. Caches the `SchemaProfile` via `get_schema_profile()`. Holds per-session `pending_changes` list for two-phase commit.
- **`src/tools/tool_registry.py`** — Four LangChain `@tool` functions: `get_dataset_schema`, `query_data`, `propose_state_change`, `ask_other_agent`. Tools are the ONLY interface between LLM agents and deterministic state. Uses a settable singleton (`set_data_manager`/`get_data_manager`) synced to `st.session_state.manager` on every Streamlit rerun. `commit_pending_changes` re-validates each staged change via ShadowSandbox before applying.
- **`src/core/config.py`** — All constants: file paths, thresholds, model names, agent IDs, fallback chains.

### Multi-Tenancy via Schema Inference

The system works with any CSV — toy factory, hospital beds, shipping fleet, server farm, retail. `DynamicSchemaInferencer` heuristically detects:
- Primary key column (regex patterns: `id`, `key`, `sku`, `ref`, etc.)
- Constraint pairs: mutable column (e.g., `current_stock`) -> limit column (e.g., `max_capacity`) via stem extraction and pattern matching
- Cross-stem pairing for columns with no shared stem (data-driven validation: checks that mutable <= limit in >= 75% of rows)

### Agent Tool Call Contract

Agents MUST follow this order (enforced by prompts, validated by sandbox):
1. `get_dataset_schema()` — learn column names (they vary per tenant)
2. `query_data(search_term)` — read current values before proposing changes
3. `propose_state_change(row_key, target_column, delta, justification)` — sandbox-gated, stages for HITL approval

### Test Structure

Tests in `tests/test_edge_cases.py` cover four guarantees:
1. Sandbox rejects overflow above inferred upper bounds
2. Sandbox rejects negative quantities (universal physics floor)
3. Schema inference works across 5 industry domains (toy factory, hospital, shipping, server farm, retail)
4. Trust override fires when agent score < threshold (mocked, no API calls)

The trust override tests mock `ChatGroq` to prevent real API calls during testing.

### Data Files

Persistent state in `data/`:
- `inventory.csv` — Master Clipboard (overwritten on CSV upload)
- `transaction_log.csv` — append-only audit ledger
- `agent_trust_scores.json` — per-agent trust scores with fallback chains

All are auto-generated with mock data on first run if missing.
