"""
tests/test_simulator.py
=======================
Unit tests for the Monte Carlo simulator (``src/core/simulator.py``).

Covers six guarantees:

1. **Empty pending changes** — return zeroed result, no errors.
2. **Single safe change** — runs complete, p10 ≤ p50 ≤ p90, rejection rate
   stays near zero with safe noise.
3. **Always-invalid change** — proposing a delta beyond the limit yields
   ``rejection_rate == 1.0``.
4. **Multi-tenant invariance** — same simulator works on a hospital schema
   without any factory-specific code path.
5. **Speed budget** — 100 runs on a 3-row factory_df finish in under 1.0 s.
6. **Determinism** — identical seeds produce identical metrics dicts.

Run::

    pytest tests/test_simulator.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.schema_engine import DynamicSchemaInferencer
from src.core.simulator import MonteCarloSimulator, SimulationResult


# ---------------------------------------------------------------------------
# Fixtures — factory + hospital DataFrames so we exercise multi-tenancy
# ---------------------------------------------------------------------------

@pytest.fixture
def factory_df() -> pd.DataFrame:
    """A 3-row toy-factory inventory snapshot."""
    return pd.DataFrame({
        "sku": ["A", "B", "C"],
        "current_stock": [50, 80, 30],
        "max_capacity": [100, 100, 100],
    })


@pytest.fixture
def factory_schema(factory_df: pd.DataFrame):
    """Schema profile inferred from ``factory_df``."""
    return DynamicSchemaInferencer(factory_df).infer()


@pytest.fixture
def hospital_df() -> pd.DataFrame:
    """A 2-row hospital ward snapshot — different domain, same simulator."""
    return pd.DataFrame({
        "ward_id": ["A", "B"],
        "occupied_beds": [15, 8],
        "bed_capacity": [20, 10],
    })


@pytest.fixture
def hospital_schema(hospital_df: pd.DataFrame):
    """Schema profile inferred from ``hospital_df``."""
    return DynamicSchemaInferencer(hospital_df).infer()


# ---------------------------------------------------------------------------
# Helpers — build a pending_change dict matching the staging schema
# ---------------------------------------------------------------------------

def _make_change(
    row_key: str,
    target_column: str,
    delta: float,
    *,
    old_value: float = 0.0,
) -> dict:
    """Construct a staged-change dict in the same shape as
    ``FactoryDataManager.pending_changes`` entries.
    """
    return {
        "row_key": row_key,
        "target_column": target_column,
        "delta": float(delta),
        "justification": "test",
        "old_value": float(old_value),
        "new_value": float(old_value) + float(delta),
        "limit_value": None,
        "financial_impact": 0.0,
        "agent_id": "maker",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmptyPendingChanges:
    """Empty staging queue must short-circuit to a zeroed result."""

    def test_empty_returns_zero_runs(self, factory_df, factory_schema):
        sim = MonteCarloSimulator(
            live_df=factory_df,
            schema=factory_schema,
            pending_changes=[],
            n_runs=500,
            random_seed=42,
        )
        result = sim.run()

        assert isinstance(result, SimulationResult)
        assert result.n_runs == 0
        assert result.rejection_rate == 0.0
        assert result.metrics == {}
        assert result.constraint_violations == []
        assert result.runs_completed == 0


class TestSingleSafeChange:
    """A modest, in-bounds delta should run cleanly with sensible distribution."""

    def test_safe_change_distribution(self, factory_df, factory_schema):
        # A=50 + 10 = 60 against a max of 100 — well within bounds even with
        # 10% multiplicative noise.
        changes = [_make_change("A", "current_stock", +10, old_value=50)]
        sim = MonteCarloSimulator(
            live_df=factory_df,
            schema=factory_schema,
            pending_changes=changes,
            n_runs=300,
            noise_std_pct=0.10,
            random_seed=7,
        )
        result = sim.run()

        assert result.runs_completed == 300
        # Rejection rate should be effectively zero on a +10 delta with 10%
        # noise — even ±5σ won't push us above 100 from 50.
        assert result.rejection_rate < 0.05

        assert "current_stock" in result.metrics, (
            "current_stock must be present in metrics when sandbox accepts."
        )
        stats = result.metrics["current_stock"]
        # Quantile ordering is the core invariant.
        assert stats["p10"] <= stats["p50"] <= stats["p90"]
        # p50 should sit close to (50+80+30) + 10 = 170 (sum across rows).
        assert 165.0 <= stats["p50"] <= 175.0
        assert stats["std"] >= 0.0


class TestAlwaysInvalidChange:
    """A delta that always violates the upper bound must yield 100% rejection."""

    def test_overflow_rejected_every_run(self, factory_df, factory_schema):
        # A=50 + 200 = 250, far above max_capacity=100 — even -5σ noise can't
        # pull this back into the valid region with 10% sigma.
        changes = [_make_change("A", "current_stock", +200, old_value=50)]
        sim = MonteCarloSimulator(
            live_df=factory_df,
            schema=factory_schema,
            pending_changes=changes,
            n_runs=200,
            noise_std_pct=0.10,
            random_seed=11,
        )
        result = sim.run()

        assert result.runs_completed == 200
        assert result.rejection_rate == 1.0
        # We should have captured at least one human-readable violation message.
        assert len(result.constraint_violations) >= 1
        assert any(
            "Upper Bound" in v or "Constraint" in v
            for v in result.constraint_violations
        )


class TestMultiTenantInvariance:
    """The same simulator works on a hospital DataFrame with no code changes."""

    def test_hospital_schema_works_identically(self, hospital_df, hospital_schema):
        # Ward A: 15 occupied / 20 capacity — +3 is always safe.
        changes = [_make_change("A", "occupied_beds", +3, old_value=15)]
        sim = MonteCarloSimulator(
            live_df=hospital_df,
            schema=hospital_schema,
            pending_changes=changes,
            n_runs=200,
            noise_std_pct=0.10,
            random_seed=23,
        )
        result = sim.run()

        assert result.runs_completed == 200
        # Cleanly inside the bound.
        assert result.rejection_rate < 0.05
        # Metric column should be the hospital's mutable column — NOT a
        # factory-specific name.  This is the multi-tenant guarantee.
        assert "occupied_beds" in result.metrics
        assert "current_stock" not in result.metrics

        stats = result.metrics["occupied_beds"]
        assert stats["p10"] <= stats["p50"] <= stats["p90"]


class TestSpeedBudget:
    """Performance regression guard: 100 runs on 3 rows must stay snappy."""

    def test_under_one_second_for_100_runs(self, factory_df, factory_schema):
        changes = [_make_change("A", "current_stock", +5, old_value=50)]
        sim = MonteCarloSimulator(
            live_df=factory_df,
            schema=factory_schema,
            pending_changes=changes,
            n_runs=100,
            random_seed=99,
        )

        t0 = time.perf_counter()
        result = sim.run()
        elapsed = time.perf_counter() - t0

        assert result.runs_completed == 100
        assert elapsed < 1.0, (
            f"100 runs took {elapsed:.3f}s — should be <1.0s on the 3-row factory_df"
        )


class TestDeterminism:
    """Identical seeds must produce bit-identical metrics."""

    def test_same_seed_same_metrics(self, factory_df, factory_schema):
        changes = [_make_change("A", "current_stock", +10, old_value=50)]

        def _run():
            sim = MonteCarloSimulator(
                live_df=factory_df,
                schema=factory_schema,
                pending_changes=changes,
                n_runs=150,
                noise_std_pct=0.10,
                random_seed=2026,
            )
            return sim.run()

        r1 = _run()
        r2 = _run()

        assert r1.rejection_rate == r2.rejection_rate
        assert set(r1.metrics.keys()) == set(r2.metrics.keys())
        for col in r1.metrics:
            for stat in ("p10", "p50", "p90", "mean", "std"):
                assert r1.metrics[col][stat] == r2.metrics[col][stat], (
                    f"Determinism broken on {col}.{stat}: "
                    f"{r1.metrics[col][stat]} != {r2.metrics[col][stat]}"
                )

    def test_different_seeds_diverge(self, factory_df, factory_schema):
        """Sanity check: different seeds produce different metric draws."""
        changes = [_make_change("A", "current_stock", +10, old_value=50)]

        def _run(seed: int):
            sim = MonteCarloSimulator(
                live_df=factory_df,
                schema=factory_schema,
                pending_changes=changes,
                n_runs=150,
                noise_std_pct=0.10,
                random_seed=seed,
            )
            return sim.run()

        r_a = _run(1)
        r_b = _run(2)
        # std should differ between runs with different seeds; if both are 0
        # this means the noise didn't propagate and the test would be vacuous.
        assert r_a.metrics["current_stock"]["std"] != r_b.metrics["current_stock"]["std"]


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_negative_n_runs_rejected(self, factory_df, factory_schema):
        with pytest.raises(ValueError, match="n_runs"):
            MonteCarloSimulator(
                live_df=factory_df,
                schema=factory_schema,
                pending_changes=[],
                n_runs=-1,
            )

    def test_negative_noise_rejected(self, factory_df, factory_schema):
        with pytest.raises(ValueError, match="noise_std_pct"):
            MonteCarloSimulator(
                live_df=factory_df,
                schema=factory_schema,
                pending_changes=[],
                n_runs=10,
                noise_std_pct=-0.05,
            )
