"""Tests for src.core.forecaster — F9 demand forecasting.

These tests exercise the forecaster against a lightweight schema
stand-in (``_MiniSchema``) so they do not depend on the production
``DynamicSchemaInferencer``. The forecaster resolves either schema
shape (``primary_key``/``constraint_pairs`` or
``primary_key_column``/``constraint_rules``) transparently.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so ``from src.core.forecaster import …``
# works under direct ``pytest tests/test_forecaster.py`` invocation (matches
# the convention used by other tests in this suite).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402,F401  (kept for potential numerical assertions)
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.core.forecaster import (  # noqa: E402
    DemandForecaster,
    ForecastResult,  # noqa: F401  (re-exported for downstream import checks)
    FleetForecast,
)


class _MiniSchema:
    """Minimal schema fixture matching the F9 spec."""

    def __init__(self, pk: str, constraint_pairs: dict):
        self.primary_key = pk
        self.constraint_pairs = constraint_pairs


@pytest.fixture
def factory_inventory():
    return pd.DataFrame({
        "sku": ["A", "B", "C"],
        "current_stock": [50, 30, 100],
        "max_capacity": [100, 100, 100],
    })


@pytest.fixture
def factory_schema():
    return _MiniSchema(
        pk="sku",
        constraint_pairs={"current_stock": "max_capacity"},
    )


def test_empty_inventory_returns_empty(factory_schema):
    f = DemandForecaster(pd.DataFrame(), None, factory_schema)
    r = f.forecast_fleet(horizon=5)
    assert isinstance(r, FleetForecast)
    assert not r.per_row


def test_naive_forecast_when_no_log(factory_inventory, factory_schema):
    f = DemandForecaster(factory_inventory, None, factory_schema)
    r = f.forecast_fleet(horizon=5)
    assert "A.current_stock" in r.per_row
    fr = r.per_row["A.current_stock"]
    assert fr.method == "naive"
    assert len(fr.projected_values) == 5
    assert all(v == 50 for v in fr.projected_values)  # flat naive
    assert fr.runs_out_in is None  # never runs out at flat 50


def test_regression_forecast_with_history(factory_inventory, factory_schema):
    log = pd.DataFrame({
        "sku": ["A"] * 6,
        "target_column": ["current_stock"] * 6,
        "new_value": [60, 55, 50, 45, 40, 35],  # decreasing
        "timestamp": pd.date_range("2026-04-01", periods=6),
    })
    f = DemandForecaster(factory_inventory, log, factory_schema)
    r = f.forecast_fleet(horizon=5)
    fr = r.per_row.get("A.current_stock")
    assert fr is not None
    assert fr.method in ("regression", "exp_smoothing")
    # projected should continue downward
    assert fr.projected_values[0] < 35


def test_at_risk_detection(factory_inventory, factory_schema):
    """Row B starting at low stock with declining trend -> at risk."""
    inv = pd.DataFrame({
        "sku": ["B"],
        "current_stock": [10],
        "max_capacity": [100],
    })
    log = pd.DataFrame({
        "sku": ["B"] * 6,
        "target_column": ["current_stock"] * 6,
        "new_value": [30, 25, 20, 15, 12, 10],  # steep decline
        "timestamp": pd.date_range("2026-04-01", periods=6),
    })
    schema = _MiniSchema(
        pk="sku",
        constraint_pairs={"current_stock": "max_capacity"},
    )
    f = DemandForecaster(inv, log, schema)
    r = f.forecast_fleet(horizon=10)
    fr = r.per_row["B.current_stock"]
    assert fr.runs_out_in is not None
    assert "B.current_stock" in r.at_risk_rows


def test_summary_string_format(factory_inventory, factory_schema):
    f = DemandForecaster(factory_inventory, None, factory_schema)
    r = f.forecast_fleet(horizon=5)
    s = r.summary()
    assert "horizon" in s.lower() or "5" in s


def test_multi_tenant_hospital_schema():
    inv = pd.DataFrame({
        "ward": ["A", "B"],
        "occupied_beds": [15, 8],
        "bed_capacity": [20, 10],
    })
    schema = _MiniSchema(
        pk="ward",
        constraint_pairs={"occupied_beds": "bed_capacity"},
    )
    f = DemandForecaster(inv, None, schema)
    r = f.forecast_fleet(horizon=3)
    assert "A.occupied_beds" in r.per_row
    assert "B.occupied_beds" in r.per_row


def test_horizon_clamping():
    inv = pd.DataFrame({"id": ["X"], "stock": [50], "max": [100]})
    schema = _MiniSchema(pk="id", constraint_pairs={"stock": "max"})
    f = DemandForecaster(inv, None, schema)
    r = f.forecast_fleet(horizon=3)
    fr = r.per_row["X.stock"]
    assert len(fr.projected_values) == 3


def test_explicit_target_columns():
    inv = pd.DataFrame({"id": ["X"], "stock": [50], "extra": [10]})
    schema = _MiniSchema(pk="id", constraint_pairs={"stock": "max"})
    f = DemandForecaster(inv, None, schema)
    r = f.forecast_fleet(horizon=2, target_columns=["extra"])
    assert "X.extra" in r.per_row
    assert "X.stock" not in r.per_row
