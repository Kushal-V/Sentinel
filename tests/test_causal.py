"""Unit tests for F2 — causal inference (src.core.causal)."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path (needed when pytest is run from any dir)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.core.causal import CausalAnalyzer, CausalEffect


class _MiniSchema:
    """Minimal schema stub — the analyzer only stores it, never reads it."""

    def __init__(self, pk: str):
        self.primary_key = pk


@pytest.fixture
def linear_df():
    """y = 2*x + noise — strong causal-style relationship."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, size=50)
    y = 2.0 * x + rng.normal(0, 5, size=50)
    return pd.DataFrame({"x": x, "y": y, "z": rng.uniform(0, 10, size=50)})


@pytest.fixture
def empty_df():
    return pd.DataFrame()


def test_empty_inventory_returns_insufficient(empty_df):
    a = CausalAnalyzer(empty_df)
    e = a.estimate_effect("x", "y", 10.0)
    assert e.method == "insufficient_data"
    assert e.confidence == 0.0


def test_missing_columns_returns_insufficient(linear_df):
    a = CausalAnalyzer(linear_df)
    e = a.estimate_effect("missing", "y", 5.0)
    assert e.method == "insufficient_data"


def test_self_treatment_outcome_returns_delta(linear_df):
    a = CausalAnalyzer(linear_df)
    e = a.estimate_effect("y", "y", 7.0)
    assert e.method == "row_lookup"
    assert abs(e.effect - 7.0) < 1e-9


def test_linear_relationship_effect_close_to_two(linear_df):
    """y = 2x → estimated effect of dx=10 should be ≈ 20."""
    a = CausalAnalyzer(linear_df)
    e = a.estimate_effect("x", "y", 10.0)
    assert e.method == "regression"
    assert e.r2 > 0.9
    assert abs(e.effect - 20.0) < 5.0  # tolerance for regression noise
    assert e.confidence > 0.5


def test_negative_intervention():
    df = pd.DataFrame({"a": np.arange(50), "b": np.arange(50) * 3 + 1})
    a = CausalAnalyzer(df)
    e = a.estimate_effect("a", "b", -5.0)
    assert e.effect < 0
    assert abs(e.effect - (-15.0)) < 5.0


def test_small_sample_falls_back_to_correlation():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    a = CausalAnalyzer(df)
    e = a.estimate_effect("x", "y", 1.0)
    assert e.method == "row_lookup"
    assert e.confidence < 0.3


def test_explicit_candidate_features(linear_df):
    a = CausalAnalyzer(linear_df)
    e = a.estimate_effect("x", "y", 5.0, candidate_features=["x", "z"])
    assert e.method == "regression"
    assert e.n_samples == 50


def test_multi_tenant_hospital_columns():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "ward_id": [f"W{i}" for i in range(30)],
        "occupied_beds": rng.integers(0, 20, size=30),
        "bed_capacity": [20] * 30,
        "incoming_patients": rng.integers(0, 10, size=30),
    })
    a = CausalAnalyzer(df)
    e = a.estimate_effect("incoming_patients", "occupied_beds", 5.0)
    assert e.method in ("regression", "row_lookup")


def test_summary_string_format(linear_df):
    a = CausalAnalyzer(linear_df)
    e = a.estimate_effect("x", "y", 3.0)
    s = e.summary()
    assert "x" in s and "y" in s
    assert "effect" in s.lower()


def test_no_unobserved_confounders_disclaimer(linear_df):
    a = CausalAnalyzer(linear_df)
    e = a.estimate_effect("x", "y", 1.0)
    assert "confounders" in e.notes.lower() or "regression" in e.notes.lower()
