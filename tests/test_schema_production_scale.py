"""Schema inference tests on production-scale datasets and naming patterns.

Covers the bugs surfaced by 5_data_center_capacity.csv:
  - _used / _total pair detection on multi-token stems (memory_gb, storage_tb)
  - _current / _budget pair detection on power/network columns
  - _gb / _tb / _watts / _gbps unit suffixes preserved as part of stem
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make `src` importable when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.schema_engine import DynamicSchemaInferencer


def _rules(df: pd.DataFrame) -> set[tuple[str, str]]:
    profile = DynamicSchemaInferencer(df).infer()
    return {(r.mutable_column, r.limit_column) for r in profile.constraint_rules}


def _pk(df: pd.DataFrame) -> str:
    return DynamicSchemaInferencer(df).infer().primary_key_column


# ---------------------------------------------------------------------------
# New naming patterns
# ---------------------------------------------------------------------------

def test_used_total_pair_with_unit_suffix():
    df = pd.DataFrame({
        "node_id": [f"n{i}" for i in range(10)],
        "memory_gb_used": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "memory_gb_total": [128] * 10,
    })
    assert ("memory_gb_used", "memory_gb_total") in _rules(df)


def test_used_total_pair_storage_tb():
    df = pd.DataFrame({
        "id": [f"d{i}" for i in range(10)],
        "storage_tb_used": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "storage_tb_total": [50] * 10,
    })
    assert ("storage_tb_used", "storage_tb_total") in _rules(df)


def test_current_budget_pair_power():
    df = pd.DataFrame({
        "rack_id": [f"r{i}" for i in range(10)],
        "power_watts_current": [100, 200, 300, 400, 500, 600, 700, 800, 900, 950],
        "power_watts_budget": [1000] * 10,
    })
    assert ("power_watts_current", "power_watts_budget") in _rules(df)


def test_current_budget_pair_network():
    df = pd.DataFrame({
        "iface_id": [f"i{i}" for i in range(10)],
        "network_gbps_current": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
        "network_gbps_budget": [100] * 10,
    })
    assert ("network_gbps_current", "network_gbps_budget") in _rules(df)


def test_quota_term_recognized_as_limit():
    df = pd.DataFrame({
        "user_id": [f"u{i}" for i in range(10)],
        "api_calls_used": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "api_calls_quota": [1000] * 10,
    })
    assert ("api_calls_used", "api_calls_quota") in _rules(df)


# ---------------------------------------------------------------------------
# Production-scale dataset (5_data_center_capacity.csv) end-to-end
# ---------------------------------------------------------------------------

DATA_CENTER_CSV = Path(__file__).resolve().parent.parent / "test_datasets" / "5_data_center_capacity.csv"


@pytest.mark.skipif(not DATA_CENTER_CSV.exists(), reason="dataset not generated")
def test_data_center_dataset_loads_at_scale():
    df = pd.read_csv(DATA_CENTER_CSV)
    assert len(df) == 1500
    assert df.shape[1] == 23


@pytest.mark.skipif(not DATA_CENTER_CSV.exists(), reason="dataset not generated")
def test_data_center_pk_inferred():
    df = pd.read_csv(DATA_CENTER_CSV)
    assert _pk(df) == "server_id"


@pytest.mark.skipif(not DATA_CENTER_CSV.exists(), reason="dataset not generated")
def test_data_center_all_five_constraint_pairs_detected():
    df = pd.read_csv(DATA_CENTER_CSV)
    rules = _rules(df)
    expected = {
        ("cpu_cores_used", "cpu_cores_total"),
        ("memory_gb_used", "memory_gb_total"),
        ("storage_tb_used", "storage_tb_total"),
        ("power_watts_current", "power_watts_budget"),
        ("network_gbps_current", "network_gbps_budget"),
    }
    missing = expected - rules
    assert not missing, f"missing constraint pairs: {missing}"


@pytest.mark.skipif(not DATA_CENTER_CSV.exists(), reason="dataset not generated")
def test_data_center_engineered_crisis_scenarios_present():
    """The generator embeds crisis rows. Verify they are present so a
    runtime can demo Sentinel's response."""
    df = pd.read_csv(DATA_CENTER_CSV)

    power_breach = (df["power_watts_current"] >= 0.95 * df["power_watts_budget"]).sum()
    mem_breach = (df["memory_gb_used"] >= 0.92 * df["memory_gb_total"]).sum()
    stor_breach = (df["storage_tb_used"] >= 0.93 * df["storage_tb_total"]).sum()
    failed = (df["status"] == "FAILED").sum()
    overdue = (df["last_maintenance_days_ago"] > 180).sum()

    assert power_breach >= 20, f"too few power-breach rows: {power_breach}"
    assert mem_breach >= 10, f"too few memory-breach rows: {mem_breach}"
    assert stor_breach >= 10, f"too few storage-breach rows: {stor_breach}"
    assert failed >= 10, f"too few failed nodes: {failed}"
    assert overdue >= 50, f"too few overdue maintenance rows: {overdue}"


# ---------------------------------------------------------------------------
# Backward compatibility — existing industries unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected_pk,must_contain", [
    ("1_pharmaceutical_warehouse.csv", "item_id", {("current_stock", "max_capacity")}),
    ("2_electronics_manufacturing.csv", "part_id", {("current_stock", "max_capacity")}),
    ("3_food_cold_chain.csv", "sku_id", {("current_stock_kg", "max_capacity_kg")}),
])
def test_existing_industries_unchanged(filename, expected_pk, must_contain):
    path = Path(__file__).resolve().parent.parent / "test_datasets" / filename
    if not path.exists():
        pytest.skip(f"{filename} not present")
    df = pd.read_csv(path)
    assert _pk(df) == expected_pk
    rules = _rules(df)
    assert must_contain.issubset(rules), f"missing {must_contain - rules}"
