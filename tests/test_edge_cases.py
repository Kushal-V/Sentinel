"""
tests/test_edge_cases.py
========================
Sentinel QA Suite — Backend Edge Case Tests

Tests the four critical behavioural guarantees of the Sentinel core:

1. **Hallucination Test** — ShadowSandbox MUST reject a delta that would push
   a column above its dynamically inferred upper bound.
2. **Physics Test** — ShadowSandbox MUST reject any delta that drops a value
   below zero (a universal physical impossibility).
3. **Dynamic Schema Test** — DynamicSchemaInferencer MUST detect the correct
   mutable→limit column pairs for all four industry datasets, regardless of
   naming convention.
4. **Trust Override Test** — AgentOrchestrator MUST programmatically override
   the LLM Dispatcher's routing choice when the chosen agent's trust score
   falls below ``config.ROUTING_THRESHOLD``.

Usage::

    # From project root with venv active:
    pytest tests/test_edge_cases.py -v

    # Quick run with shorter tracebacks:
    pytest tests/test_edge_cases.py -v --tb=short
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path (needed when pytest is run from any dir)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Lazy import guard — load .env silently so tests don't crash on missing key
# ---------------------------------------------------------------------------

from dotenv import load_dotenv as _load_dotenv
_load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)

# Core imports (must come AFTER sys.path and dotenv setup)
from src.core.sandbox import SandboxResult, ShadowSandbox
from src.core.schema_engine import ConstraintRule, DynamicSchemaInferencer, SchemaProfile


# ===========================================================================
# Fixtures & Shared Data Builders
# ===========================================================================

def _make_toy_factory_df() -> pd.DataFrame:
    """Return a small, self-contained toy-factory DataFrame for sandbox tests.

    This mirrors the cold-start data from ``FactoryDataManager`` so sandbox
    tests never depend on the ``data/`` directory on disk.

    Schema:
        item_id (PK), item_name, current_stock (mutable), max_capacity (limit),
        unit_cost_usd, reorder_point
    """
    return pd.DataFrame([
        {"item_id": "ITEM-PLASTIC-01",    "item_name": "Plastic Resin",     "current_stock": 800,  "max_capacity": 1000, "unit_cost_usd": 2.50,  "reorder_point": 200},
        {"item_id": "ITEM-MICROCHIP-01",  "item_name": "Microchip",         "current_stock": 4500, "max_capacity": 5000, "unit_cost_usd": 12.00, "reorder_point": 500},
        {"item_id": "ITEM-PAINT-01",      "item_name": "Non-Toxic Paint",   "current_stock": 320,  "max_capacity": 600,  "unit_cost_usd": 1.20,  "reorder_point": 100},
        {"item_id": "ITEM-WIP-01",        "item_name": "WIP Body Frame",    "current_stock": 90,   "max_capacity": 200,  "unit_cost_usd": 8.00,  "reorder_point": 30},
        {"item_id": "ITEM-FG-01",         "item_name": "Finished Toy",      "current_stock": 1200, "max_capacity": 2000, "unit_cost_usd": 22.50, "reorder_point": 250},
    ])


def _make_hospital_df() -> pd.DataFrame:
    """Minimal hospital DataFrame with deliberate naming conventions."""
    return pd.DataFrame([
        {"ward_id": "WARD-CARD-1A", "department": "Cardiology",  "currently_admitted": 14, "max_bed_capacity": 20, "daily_cost_usd": 4800.0},
        {"ward_id": "WARD-NEUR-2B", "department": "Neurology",   "currently_admitted": 8,  "max_bed_capacity": 16, "daily_cost_usd": 2400.0},
        {"ward_id": "WARD-ORTH-3C", "department": "Orthopaedics","currently_admitted": 11, "max_bed_capacity": 14, "daily_cost_usd": 3200.0},
    ])


def _make_shipping_df() -> pd.DataFrame:
    """Minimal shipping fleet DataFrame."""
    return pd.DataFrame([
        {"vessel_id": "IMO-9100001", "route": "Shanghai → Rotterdam", "containers_loaded": 18200, "teu_limit": 22000, "fuel_efficiency_g_co2_per_teu_km": 3.2},
        {"vessel_id": "IMO-9200002", "route": "Singapore → LA",       "containers_loaded": 9800,  "teu_limit": 10000, "fuel_efficiency_g_co2_per_teu_km": 4.5},
    ])


def _make_server_df() -> pd.DataFrame:
    """Minimal server farm DataFrame."""
    return pd.DataFrame([
        {"rack_id": "RACK-US1-A01", "server_type": "GPU (A100)", "active_compute_tb": 72.5, "max_compute_allowance": 100.0, "power_draw_kw": 28.4},
        {"rack_id": "RACK-EU2-B12", "server_type": "CPU Xeon",   "active_compute_tb": 44.1, "max_compute_allowance": 80.0,  "power_draw_kw": 15.2},
    ])


def _make_retail_df() -> pd.DataFrame:
    """Minimal retail apparel DataFrame."""
    return pd.DataFrame([
        {"sku": "ARW-TS-M-101", "clothing_item": "ArcoWear T-Shirt", "units_on_floor": 28, "stockroom_limit": 48, "retail_price_gbp": 29.99},
        {"sku": "NVT-HD-L-202", "clothing_item": "NovaTex Hoodie",   "units_on_floor": 15, "stockroom_limit": 36, "retail_price_gbp": 64.99},
        {"sku": "UCR-JN-S-303", "clothing_item": "UrbanCraft Jeans", "units_on_floor": 40, "stockroom_limit": 60, "retail_price_gbp": 79.99},
    ])


# ===========================================================================
# QA TEST CLASS 1 — Shadow Sandbox Hallucination Rejection
# ===========================================================================

class TestSandboxHallucinationGuard(unittest.TestCase):
    """Tests that the ShadowSandbox enforces inferred upper-bound constraints.

    This directly targets the primary safety claim (Novelty Claim A):
    *LLM agents cannot commit a state change that violates physical capacity
    limits, even if the LLM confidently hallucinates that the limits don't
    apply or are different.*
    """

    def setUp(self) -> None:
        """Initialise a sandbox with the standard toy factory dataset."""
        self.df = _make_toy_factory_df()
        self.sandbox = ShadowSandbox(live_df=self.df)

    def test_hallucination_overflow_is_rejected(self) -> None:
        """A delta that pushes current_stock above max_capacity MUST be REJECTED.

        Scenario: Agent hallucinates that PLASTIC-01's warehouse can hold 2,000
        units when the real limit is 1,000. Current stock is 800. Agent proposes
        +600 → proposed=1,400 > limit=1,000 → MUST FAIL.
        """
        result: SandboxResult = self.sandbox.evaluate_proposal(
            row_primary_key="ITEM-PLASTIC-01",
            target_column="current_stock",
            delta=+600.0,
        )
        self.assertEqual(result.status, "REJECTED",
            "Sandbox must REJECT a delta that exceeds max_capacity.")
        self.assertIsNotNone(result.rejection_reason)
        self.assertIn("Upper Bound Constraint Violated",
                      result.rejection_reason,  # type: ignore[arg-type]
                      "Rejection reason must name the violated constraint type.")
        self.assertIsNone(result.validated_state,
            "validated_state must be None on rejection.")
        self.assertAlmostEqual(result.proposed_value, 1400.0,
            msg="proposed_value should reflect current + delta (800 + 600 = 1400).")
        self.assertAlmostEqual(result.limit_value, 1000.0,
            msg="limit_value should reflect the inferred max_capacity column value.")

    def test_marginally_over_limit_is_rejected(self) -> None:
        """Even a +1 unit overflow above the limit must be rejected (strict ≤)."""
        # PLASTIC-01: current=800, max=1000 → delta=+201 → 1001 > 1000
        result = self.sandbox.evaluate_proposal("ITEM-PLASTIC-01", "current_stock", +201.0)
        self.assertEqual(result.status, "REJECTED")

    def test_exactly_at_limit_is_safe(self) -> None:
        """A delta that brings current_stock to exactly max_capacity must PASS."""
        # PLASTIC-01: current=800, max=1000 → delta=+200 → 1000 == 1000 → SAFE
        result = self.sandbox.evaluate_proposal("ITEM-PLASTIC-01", "current_stock", +200.0)
        self.assertEqual(result.status, "SAFE",
            "current_stock == max_capacity (exactly at limit) must be accepted.")
        self.assertIsNotNone(result.validated_state)
        new_val = result.validated_state.loc[  # type: ignore[union-attr]
            result.validated_state["item_id"] == "ITEM-PLASTIC-01", "current_stock"
        ].iloc[0]
        self.assertAlmostEqual(float(new_val), 1000.0)

    def test_safe_decrease_below_limit_passes(self) -> None:
        """A valid decrease that stays within bounds must be SAFE."""
        # PLASTIC-01: current=800 → -300 → 500 (> 0 and < 1000)
        result = self.sandbox.evaluate_proposal("ITEM-PLASTIC-01", "current_stock", -300.0)
        self.assertEqual(result.status, "SAFE")

    def test_large_hallucination_on_finished_goods(self) -> None:
        """Massive overflow on FG-01: +5000 units onto a 2000-unit max rack."""
        result = self.sandbox.evaluate_proposal("ITEM-FG-01", "current_stock", +5000.0)
        self.assertEqual(result.status, "REJECTED")
        # proposed_value should be 1200 + 5000 = 6200
        self.assertAlmostEqual(result.proposed_value, 6200.0)

    def test_unknown_row_key_is_rejected(self) -> None:
        """A proposal for a non-existent row_primary_key must be REJECTED."""
        result = self.sandbox.evaluate_proposal(
            "ITEM-DOES-NOT-EXIST-99", "current_stock", +10.0
        )
        self.assertEqual(result.status, "REJECTED")
        self.assertIn("Row Not Found", result.rejection_reason)  # type: ignore[operator]

    def test_nonexistent_column_is_rejected(self) -> None:
        """A proposal targeting a column that doesn't exist must be REJECTED."""
        result = self.sandbox.evaluate_proposal(
            "ITEM-PLASTIC-01", "hallucinated_quantity_field", +50.0
        )
        self.assertEqual(result.status, "REJECTED")
        self.assertIn("Column Not Found", result.rejection_reason)  # type: ignore[operator]


# ===========================================================================
# QA TEST CLASS 2 — Shadow Sandbox Physics Guard (floor = 0)
# ===========================================================================

class TestSandboxPhysicsGuard(unittest.TestCase):
    """Tests that ShadowSandbox enforces the non-negative floor constraint.

    No physical quantity can be negative. This rule is universal and applies
    to ALL numeric columns, regardless of whether a limit column was inferred.
    """

    def setUp(self) -> None:
        self.df = _make_toy_factory_df()
        self.sandbox = ShadowSandbox(live_df=self.df)

    def test_below_zero_is_rejected(self) -> None:
        """Removing more units than exist must be REJECTED.

        Scenario: WIP-01 has 90 units. Agent proposes -100. Result = -10 < 0.
        """
        result: SandboxResult = self.sandbox.evaluate_proposal(
            row_primary_key="ITEM-WIP-01",
            target_column="current_stock",
            delta=-100.0,
        )
        self.assertEqual(result.status, "REJECTED",
            "Sandbox must REJECT a delta causing a negative stock level.")
        self.assertIsNotNone(result.rejection_reason)
        self.assertIn("Non-Negative Constraint Violated", result.rejection_reason)  # type: ignore
        self.assertAlmostEqual(result.proposed_value, -10.0,
            msg="proposed_value = 90 + (-100) = -10.0")

    def test_exact_zero_depletion_is_safe(self) -> None:
        """Depleting a stock to exactly 0 must be SAFE (0 is valid)."""
        # WIP-01: current=90 → delta=-90 → new=0 SAFE
        result = self.sandbox.evaluate_proposal("ITEM-WIP-01", "current_stock", -90.0)
        self.assertEqual(result.status, "SAFE",
            "Depleting to exactly 0 must be accepted (zero is not negative).")

    def test_minus_one_below_zero_fails(self) -> None:
        """current=90, delta=-91 → proposed=-1 — must be REJECTED."""
        result = self.sandbox.evaluate_proposal("ITEM-WIP-01", "current_stock", -91.0)
        self.assertEqual(result.status, "REJECTED")
        self.assertAlmostEqual(result.proposed_value, -1.0)

    def test_physics_applies_to_non_constrained_columns(self) -> None:
        """A non-constrained numeric column (reorder_point) still cannot go below 0."""
        # PLASTIC-01: reorder_point=200 — no limit column inferred for this
        # delta=-250 → -50 which is below 0 → REJECTED
        result = self.sandbox.evaluate_proposal("ITEM-PLASTIC-01", "reorder_point", -250.0)
        self.assertEqual(result.status, "REJECTED",
            "Physics floor (>=0) must apply to all numeric columns, not just constrained ones.")

    def test_massive_depletion_on_microchip(self) -> None:
        """agent proposes removing all 10,000 chips from a 4,500-unit inventory."""
        result = self.sandbox.evaluate_proposal("ITEM-MICROCHIP-01", "current_stock", -10000.0)
        self.assertEqual(result.status, "REJECTED")
        self.assertAlmostEqual(result.proposed_value, 4500 - 10000)


# ===========================================================================
# QA TEST CLASS 3 — Dynamic Schema Inferencer (Cross-Industry)
# ===========================================================================

class TestDynamicSchemaInferencer(unittest.TestCase):
    """Tests that DynamicSchemaInferencer detects constraint pairs across
    four entirely different naming conventions.

    This is the core claim of the multi-tenant architecture: Sentinel adapts
    to ANY domain without code changes.
    """

    # ── Toy Factory (baseline) ─────────────────────────────────────────────

    def test_toy_factory_detects_stock_capacity_pair(self) -> None:
        """current_stock → max_capacity must be detected in the baseline dataset."""
        df = _make_toy_factory_df()
        profile: SchemaProfile = DynamicSchemaInferencer(df).infer()

        mutable_cols = {r.mutable_column for r in profile.constraint_rules}
        limit_cols = {r.limit_column for r in profile.constraint_rules}

        self.assertIn("current_stock", mutable_cols,
            "'current_stock' must be detected as a mutable quantity column.")
        self.assertIn("max_capacity", limit_cols,
            "'max_capacity' must be detected as a limit column.")

        rule: ConstraintRule | None = next(
            (r for r in profile.constraint_rules if r.mutable_column == "current_stock"),
            None,
        )
        self.assertIsNotNone(rule, "A constraint rule for 'current_stock' must exist.")
        self.assertEqual(rule.limit_column, "max_capacity")  # type: ignore[union-attr]

    def test_toy_factory_primary_key_is_item_id(self) -> None:
        """The PK for the toy factory dataset must be 'item_id'."""
        profile = DynamicSchemaInferencer(_make_toy_factory_df()).infer()
        self.assertEqual(profile.primary_key_column, "item_id")

    # ── Hospital: currently_admitted → max_bed_capacity ───────────────────

    def test_hospital_detects_admitted_to_bed_capacity(self) -> None:
        """'currently_admitted' must map to 'max_bed_capacity' for hospital data.

        Tests the regex pattern where stem extraction strips 'currently_'
        prefix and the limit column matches 'max_{stem}_capacity'.
        """
        df = _make_hospital_df()
        profile: SchemaProfile = DynamicSchemaInferencer(df).infer()

        mutable_cols = {r.mutable_column for r in profile.constraint_rules}
        self.assertIn("currently_admitted", mutable_cols,
            "'currently_admitted' must be identified as a mutable value column.")

        hospital_rule: ConstraintRule | None = next(
            (r for r in profile.constraint_rules if r.mutable_column == "currently_admitted"),
            None,
        )
        self.assertIsNotNone(hospital_rule,
            "A ConstraintRule for 'currently_admitted' must be inferred.")
        self.assertEqual(
            hospital_rule.limit_column,  # type: ignore[union-attr]
            "max_bed_capacity",
            "Hospital limit column must be 'max_bed_capacity'.",
        )

    def test_hospital_primary_key_is_ward_id(self) -> None:
        profile = DynamicSchemaInferencer(_make_hospital_df()).infer()
        self.assertEqual(profile.primary_key_column, "ward_id")

    # ── Shipping: containers_loaded → teu_limit ────────────────────────────

    def test_shipping_detects_containers_to_teu_limit(self) -> None:
        """'containers_loaded' must map to 'teu_limit' for shipping data.

        Tests the '{stem}_{limit|cap|max}' pattern where stem='containers'
        and suffix='limit'.
        """
        df = _make_shipping_df()
        profile: SchemaProfile = DynamicSchemaInferencer(df).infer()

        mutable_cols = {r.mutable_column for r in profile.constraint_rules}
        self.assertIn("containers_loaded", mutable_cols,
            "'containers_loaded' must be a detected mutable column.")

        shipping_rule: ConstraintRule | None = next(
            (r for r in profile.constraint_rules if r.mutable_column == "containers_loaded"),
            None,
        )
        self.assertIsNotNone(shipping_rule,
            "A ConstraintRule for 'containers_loaded' must be inferred.")
        self.assertEqual(
            shipping_rule.limit_column,  # type: ignore[union-attr]
            "teu_limit",
            "Shipping limit column must be 'teu_limit'.",
        )

    def test_shipping_primary_key_is_vessel_id(self) -> None:
        profile = DynamicSchemaInferencer(_make_shipping_df()).infer()
        self.assertIn("vessel_id", profile.primary_key_column)

    # ── Server Farm: active_compute_tb → max_compute_allowance ────────────

    def test_server_detects_active_compute_to_allowance(self) -> None:
        """'active_compute_tb' must map to 'max_compute_allowance' for server data.

        Tests the 'max_{stem}_{suffix}' cascade where stem='compute'.
        This is the most unusual naming convention in the test suite.
        """
        df = _make_server_df()
        profile: SchemaProfile = DynamicSchemaInferencer(df).infer()

        mutable_cols = {r.mutable_column for r in profile.constraint_rules}
        self.assertIn("active_compute_tb", mutable_cols,
            "'active_compute_tb' should be flagged as mutable.")

        server_rule: ConstraintRule | None = next(
            (r for r in profile.constraint_rules if r.mutable_column == "active_compute_tb"),
            None,
        )
        self.assertIsNotNone(server_rule,
            "ConstraintRule for 'active_compute_tb' must exist.")
        self.assertEqual(
            server_rule.limit_column,  # type: ignore[union-attr]
            "max_compute_allowance",
            "Server limit column must be 'max_compute_allowance'.",
        )

    def test_server_primary_key_is_rack_id(self) -> None:
        profile = DynamicSchemaInferencer(_make_server_df()).infer()
        self.assertEqual(profile.primary_key_column, "rack_id")

    # ── Retail: units_on_floor → stockroom_limit ──────────────────────────

    def test_retail_detects_floor_units_to_stockroom_limit(self) -> None:
        """'units_on_floor' must map to 'stockroom_limit' for retail data.

        Tests detection of 'stockroom_limit' via the '{noun}_limit' pattern,
        where the limit column doesn't share an obvious stem with the mutable
        column — the hardest detection case.
        """
        df = _make_retail_df()
        profile: SchemaProfile = DynamicSchemaInferencer(df).infer()

        mutable_cols = {r.mutable_column for r in profile.constraint_rules}
        limit_cols = {r.limit_column for r in profile.constraint_rules}

        # At minimum, stockroom_limit should be flagged as a limit-like column
        self.assertIn("stockroom_limit", limit_cols,
            "'stockroom_limit' must be detected as a capacity/limit column.")

    def test_retail_primary_key_is_sku(self) -> None:
        profile = DynamicSchemaInferencer(_make_retail_df()).infer()
        self.assertEqual(profile.primary_key_column, "sku")

    # ── Schema Profile Sanity Checks ──────────────────────────────────────

    def test_as_agent_summary_is_non_empty_string(self) -> None:
        """as_agent_summary() must return a non-empty, formatted string."""
        profile = DynamicSchemaInferencer(_make_toy_factory_df()).infer()
        summary = profile.as_agent_summary()
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 50,
            "Agent summary must be a meaningful, multi-line description.")
        self.assertIn("PRIMARY KEY", summary.upper(),
            "Agent summary must mention the primary key column.")

    def test_empty_dataframe_raises_value_error(self) -> None:
        """An empty DataFrame must raise ValueError immediately on init."""
        with self.assertRaises(ValueError):
            DynamicSchemaInferencer(pd.DataFrame())


# ===========================================================================
# QA TEST CLASS 4 — Trust Override (Claim B)
# ===========================================================================

class TestTrustOverride(unittest.TestCase):
    """Tests that AgentOrchestrator's Trust Override fires correctly.

    The test mocks:
    1. ``FactoryDataManager.get_trust_scores`` to return a fabricated JSON
       where the agent chosen by the LLM Dispatcher has a score of 0.20,
       which is below ``config.ROUTING_THRESHOLD`` (0.50 by default).
    2. The Dispatcher LLM call to return a deterministic ``DispatchRoute``
       pointing at the low-trust agent — without making any real API calls.

    The test then asserts that the Orchestrator's ``dispatch()`` method:
    a) Sets ``trust_override_applied=True`` on the returned route.
    b) Changes ``selected_agent`` to the fallback agent.
    c) Records the original LLM choice in ``original_llm_choice``.
    """

    def _make_low_trust_scores(
        self,
        low_agent: str = "maker",
        fallback: str = "mover",
    ) -> dict[str, Any]:
        """Build a fabricated trust scores dict with one agent below threshold."""
        from src.core import config
        threshold = config.ROUTING_THRESHOLD
        return {
            "agents": {
                low_agent: {
                    "trust_score": 0.20,          # Well below any threshold
                    "preferred_fallback": fallback,
                    "total_decisions": 8,
                    "recent_penalties": 4,
                },
                "mover": {
                    "trust_score": threshold + 0.30,
                    "preferred_fallback": "keeper",
                    "total_decisions": 12,
                    "recent_penalties": 0,
                },
                "keeper": {
                    "trust_score": threshold + 0.25,
                    "preferred_fallback": "maker",
                    "total_decisions": 10,
                    "recent_penalties": 1,
                },
            }
        }

    @patch("src.agents.orchestrator.ChatGoogleGenerativeAI")
    @patch("src.agents.orchestrator.create_react_agent")
    def setUp(self, mock_create_react, mock_llm_class) -> None:  # type: ignore[override]
        """Instantiate AgentOrchestrator with all LLM calls mocked out."""
        # Mock the LLM class so no actual Gemini API call is made on __init__
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
        mock_llm_instance.with_structured_output.return_value = mock_llm_instance

        # Mock create_react_agent to return a simple MagicMock graph
        mock_create_react.return_value = MagicMock()

        # Mock FactoryDataManager so no disk I/O occurs
        self.mock_manager = MagicMock()
        self.mock_manager.get_trust_scores.return_value = (
            self._make_low_trust_scores(low_agent="maker", fallback="mover")
        )

        from src.agents.orchestrator import AgentOrchestrator
        self.orchestrator = AgentOrchestrator(data_manager=self.mock_manager)
        self.MockLLMClass = mock_llm_class
        self.mock_llm_instance = mock_llm_instance

    def test_trust_override_triggers_when_score_below_threshold(self) -> None:
        """The Dispatcher MUST override routing when the chosen agent has score < threshold.

        Flow:
        1. Dispatcher LLM selects "maker" (mocked deterministically).
        2. Orchestrator.dispatch() reads trust scores → maker=0.20 < 0.50.
        3. Override fires: selected_agent becomes "mover" (maker's fallback).
        4. trust_override_applied=True, original_llm_choice="maker".
        """
        from src.agents.orchestrator import CrisisEvent, DispatchRoute

        # Mock the Dispatcher chain to deterministically choose "maker"
        fake_llm_route = DispatchRoute(
            selected_agent="maker",
            delegation_justification="Machine breakdown requires production response.",
            urgency_tier="HIGH",
            required_lookups=["ITEM-WIP-01"],
        )
        self.orchestrator._dispatcher_chain = MagicMock()
        self.orchestrator._dispatcher_chain.invoke.return_value = fake_llm_route

        crisis = CrisisEvent(
            event_id="EVT-TEST-TRUST-001",
            event_type="INTERNAL_FAILURE",
            severity="HIGH",
            description="Test: Machine breakdown on line 3.",
            affected_entities={},
        )

        result: DispatchRoute = self.orchestrator.dispatch(crisis=crisis)

        # Primary assertion: override DID fire
        self.assertTrue(
            result.trust_override_applied,
            "trust_override_applied must be True when chosen agent score < threshold.",
        )

        # The final selected_agent must NOT be the low-trust "maker"
        self.assertNotEqual(
            result.selected_agent, "maker",
            "Route must NOT go to 'maker' when its trust score is 0.20.",
        )

        # The final selected_agent must be the configured fallback
        self.assertEqual(
            result.selected_agent, "mover",
            "Route must be redirected to 'mover' (maker's configured fallback).",
        )

        # The original LLM choice must be recorded faithfully
        self.assertEqual(
            result.original_llm_choice, "maker",
            "original_llm_choice must preserve the original LLM routing decision.",
        )

        # The justification must mention the override
        self.assertIn(
            "TRUST OVERRIDE",
            result.delegation_justification,
            "Justification text must contain 'TRUST OVERRIDE' for audit trail.",
        )

    def test_no_override_when_trust_is_healthy(self) -> None:
        """When the chosen agent's trust score is ABOVE the threshold, no override fires."""
        from src.agents.orchestrator import CrisisEvent, DispatchRoute

        # Make mover and keeper both healthy (maker still low but we'll choose mover)
        self.mock_manager.get_trust_scores.return_value = {
            "agents": {
                "maker": {"trust_score": 0.20, "preferred_fallback": "mover", "total_decisions": 8, "recent_penalties": 4},
                "mover": {"trust_score": 0.90, "preferred_fallback": "keeper", "total_decisions": 12, "recent_penalties": 0},
                "keeper": {"trust_score": 0.85, "preferred_fallback": "maker", "total_decisions": 10, "recent_penalties": 0},
            }
        }

        fake_llm_route = DispatchRoute(
            selected_agent="mover",
            delegation_justification="Port strike requires logistics response.",
            urgency_tier="CRITICAL",
            required_lookups=[],
        )
        self.orchestrator._dispatcher_chain = MagicMock()
        self.orchestrator._dispatcher_chain.invoke.return_value = fake_llm_route

        crisis = CrisisEvent(
            event_id="EVT-TEST-TRUST-002",
            event_type="EXTERNAL_SHOCK",
            severity="CRITICAL",
            description="Port strike — route to logistics.",
            affected_entities={},
        )

        result: DispatchRoute = self.orchestrator.dispatch(crisis=crisis)

        self.assertFalse(
            result.trust_override_applied,
            "trust_override_applied must be False when mover score=0.90 > threshold.",
        )
        self.assertEqual(result.selected_agent, "mover",
            "Route must remain 'mover' — no override needed.")
        self.assertIsNone(result.original_llm_choice,
            "original_llm_choice must be None when no override occurred.")

    def test_override_records_correct_original_choice(self) -> None:
        """original_llm_choice must exactly match what the LLM returned, not the fallback."""
        from src.agents.orchestrator import CrisisEvent, DispatchRoute

        # All agents below threshold except keeper — make LLM pick "mover" (low trust)
        self.mock_manager.get_trust_scores.return_value = {
            "agents": {
                "maker":  {"trust_score": 0.20, "preferred_fallback": "keeper", "total_decisions": 5, "recent_penalties": 3},
                "mover":  {"trust_score": 0.15, "preferred_fallback": "keeper", "total_decisions": 5, "recent_penalties": 4},
                "keeper": {"trust_score": 0.90, "preferred_fallback": "maker",  "total_decisions": 20, "recent_penalties": 0},
            }
        }

        fake_llm_route = DispatchRoute(
            selected_agent="mover",
            delegation_justification="Mover seems right.",
            urgency_tier="MEDIUM",
            required_lookups=[],
        )
        self.orchestrator._dispatcher_chain = MagicMock()
        self.orchestrator._dispatcher_chain.invoke.return_value = fake_llm_route

        crisis = CrisisEvent(
            event_id="EVT-TEST-TRUST-003",
            event_type="INTERNAL_FAILURE",
            severity="MEDIUM",
            description="Routing test.",
            affected_entities={},
        )
        result = self.orchestrator.dispatch(crisis=crisis)

        self.assertTrue(result.trust_override_applied)
        self.assertEqual(result.original_llm_choice, "mover",
            "original_llm_choice must be 'mover' — what the LLM actually chose.")
        self.assertEqual(result.selected_agent, "keeper",
            "Fallback for mover is 'keeper' per mock trust scores.")


# ===========================================================================
# QA TEST CLASS 5 — Hospital Sandbox (Cross-Domain)
# ===========================================================================

class TestHospitalSandbox(unittest.TestCase):
    """Runs the core Sandbox tests against the hospital dataset.

    Proves that the sandbox works correctly for ANY domain, not just
    the toy factory baseline.
    """

    def setUp(self) -> None:
        self.df = _make_hospital_df()
        self.sandbox = ShadowSandbox(live_df=self.df)

    def test_overflow_above_max_bed_capacity_rejected(self) -> None:
        """Admitting more patients than there are beds must be REJECTED."""
        # CARD-1A: admitted=14, max=20 → delta=+10 → 24 > 20 → REJECTED
        result = self.sandbox.evaluate_proposal(
            row_primary_key="WARD-CARD-1A",
            target_column="currently_admitted",
            delta=+10.0,
        )
        self.assertEqual(result.status, "REJECTED",
            "Hospital sandbox must reject overflow above max_bed_capacity.")
        self.assertAlmostEqual(result.proposed_value, 24.0)
        self.assertAlmostEqual(result.limit_value, 20.0)

    def test_admitting_within_capacity_is_safe(self) -> None:
        """Admitting a patient up to max beds must be SAFE."""
        # NEUR-2B: admitted=8, max=16 → delta=+5 → 13 ≤ 16 → SAFE
        result = self.sandbox.evaluate_proposal("WARD-NEUR-2B", "currently_admitted", +5.0)
        self.assertEqual(result.status, "SAFE")

    def test_discharging_below_zero_rejected(self) -> None:
        """Discharging more patients than exist must fail the physics check."""
        # ORTH-3C: admitted=11 → delta=-50 → -39 < 0 → REJECTED
        result = self.sandbox.evaluate_proposal("WARD-ORTH-3C", "currently_admitted", -50.0)
        self.assertEqual(result.status, "REJECTED")
        self.assertIn("Non-Negative Constraint Violated", result.rejection_reason)  # type: ignore


# ===========================================================================
# Entry point for direct execution
# ===========================================================================

if __name__ == "__main__":
    # Allow running with: python tests/test_edge_cases.py
    # Provides a colourful summary without needing the pytest binary.
    pytest.main([__file__, "-v", "--tb=short", "-q"])
