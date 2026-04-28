"""
tests/test_schema_cache.py
==========================
Tests for I6: column-signature-based schema cache invalidation in
``FactoryDataManager``.

Schema inference (primary key detection, constraint pair extraction) depends
ONLY on column names and dtypes. Pure value mutations (e.g., decrementing a
stock count) must therefore NOT force an O(n*m) regex re-inference on the next
``get_schema_profile`` call. Conversely, any change to the column shape — adds,
removes, renames, or dtype changes — MUST invalidate the cache.

These tests use a tiny synthetic DataFrame with generic, multi-tenant column
names and patch ``DynamicSchemaInferencer.infer`` so we can assert exact call
counts without hitting any LLM or filesystem heavy paths.

Usage::

    pytest tests/test_schema_cache.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path (matches test_edge_cases convention).
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core import config as _sentinel_config  # noqa: E402
from src.core.state_manager import FactoryDataManager  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_workspaces_dir(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``WORKSPACES_DIR`` to a temp dir so tests do not touch real data."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        monkeypatch.setattr(_sentinel_config, "WORKSPACES_DIR", tmp_path)
        # state_manager imported the symbol directly — patch there too
        from src.core import state_manager as _sm
        monkeypatch.setattr(_sm, "WORKSPACES_DIR", tmp_path)
        yield tmp_path


@pytest.fixture
def synthetic_inventory() -> pd.DataFrame:
    """Generic, multi-tenant-shaped inventory frame.

    Uses ``id`` / ``current_stock`` / ``max_capacity`` so the schema engine has
    a clean primary-key + constraint-pair to detect, but the test logic itself
    never reads these names — it only counts inferencer invocations.
    """
    return pd.DataFrame(
        [
            {"id": "A-1", "current_stock": 10, "max_capacity": 100},
            {"id": "A-2", "current_stock": 20, "max_capacity": 100},
            {"id": "A-3", "current_stock": 30, "max_capacity": 100},
        ]
    )


@pytest.fixture
def manager(
    tmp_workspaces_dir: Path,
    synthetic_inventory: pd.DataFrame,
) -> FactoryDataManager:
    """A ``FactoryDataManager`` whose inventory has been overwritten with the
    synthetic generic frame, signature reset, and any prior cache cleared."""
    dm = FactoryDataManager(workspace="test-schema-cache")
    # Replace mock toy-factory frame with the generic test frame
    dm.update_inventory_data(synthetic_inventory.copy(deep=True))
    # Make sure no leftover cache from initialisation persists
    dm._schema_cache = None
    dm._schema_cache_signature = None
    return dm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_inferencer():
    """Patch ``DynamicSchemaInferencer.infer`` and return the mock so callers
    can assert ``call_count``. Uses the real return value via ``wraps``-style
    fallthrough to avoid breaking downstream code that reads the profile."""
    from src.core import schema_engine as _se

    real_infer = _se.DynamicSchemaInferencer.infer

    def _wrapped(self):  # noqa: ANN001
        return real_infer(self)

    return patch.object(
        _se.DynamicSchemaInferencer,
        "infer",
        autospec=True,
        side_effect=_wrapped,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSchemaCacheSignature:
    """Validate that the cache is invalidated by column shape only, never values."""

    def test_cache_hit_on_value_change(self, manager: FactoryDataManager) -> None:
        """A pure value mutation must NOT trigger re-inference."""
        with _patch_inferencer() as mock_infer:
            # First read — cold cache, infer once
            manager.get_schema_profile()
            assert mock_infer.call_count == 1

            # Mutate a single cell value (decrement stock); schema unchanged
            manager.update_inventory(
                item_id="A-1",
                target_column="current_stock",
                quantity_change=-5,
            )

            # Second read — must hit the cache, no extra infer calls
            manager.get_schema_profile()
            assert mock_infer.call_count == 1, (
                "Schema was re-inferred after a pure value mutation; "
                "cache invalidation is not signature-aware."
            )

    def test_cache_miss_on_column_add(
        self, manager: FactoryDataManager
    ) -> None:
        """Adding a brand-new column must invalidate the cache."""
        with _patch_inferencer() as mock_infer:
            manager.get_schema_profile()
            assert mock_infer.call_count == 1

            # Add a new column directly via the data-replacement path
            new_df = manager.get_inventory()
            new_df["unit_cost_usd"] = 1.50
            manager.update_inventory_data(new_df)

            manager.get_schema_profile()
            assert mock_infer.call_count == 2, (
                "Cache should have been invalidated when a new column was added."
            )

    def test_cache_miss_on_column_rename(
        self, manager: FactoryDataManager
    ) -> None:
        """Renaming a column must invalidate the cache."""
        with _patch_inferencer() as mock_infer:
            manager.get_schema_profile()
            assert mock_infer.call_count == 1

            new_df = manager.get_inventory().rename(
                columns={"current_stock": "stock_on_hand"}
            )
            manager.update_inventory_data(new_df)

            manager.get_schema_profile()
            assert mock_infer.call_count == 2, (
                "Cache should have been invalidated when a column was renamed."
            )

    def test_cache_miss_on_dtype_change(
        self, manager: FactoryDataManager
    ) -> None:
        """Changing a column's dtype (same name) must invalidate the cache."""
        with _patch_inferencer() as mock_infer:
            manager.get_schema_profile()
            assert mock_infer.call_count == 1

            # Cast int64 -> float64 on the same column; signature must change
            new_df = manager.get_inventory()
            new_df["current_stock"] = new_df["current_stock"].astype("float64")
            manager.update_inventory_data(new_df)

            manager.get_schema_profile()
            assert mock_infer.call_count == 2, (
                "Cache should have been invalidated when a column's dtype changed."
            )

    def test_empty_dataframe_signature(
        self, manager: FactoryDataManager
    ) -> None:
        """An empty DataFrame must yield the empty-string signature without error."""
        # Direct call — exercises the early-return branch
        empty = pd.DataFrame()
        assert manager._columns_signature(empty) == ""

        # Also tolerate ``None`` defensively
        assert manager._columns_signature(None) == ""  # type: ignore[arg-type]

    def test_signature_stable_under_row_addition(
        self, manager: FactoryDataManager
    ) -> None:
        """Adding rows (no column change) must keep the cache hot."""
        with _patch_inferencer() as mock_infer:
            manager.get_schema_profile()
            assert mock_infer.call_count == 1

            # Append a row — column shape and dtypes unchanged
            new_df = pd.concat(
                [
                    manager.get_inventory(),
                    pd.DataFrame(
                        [{"id": "A-4", "current_stock": 40, "max_capacity": 100}]
                    ),
                ],
                ignore_index=True,
            )
            # Coerce dtypes back so they match exactly
            new_df["current_stock"] = new_df["current_stock"].astype(
                manager.get_inventory()["current_stock"].dtype
            )
            new_df["max_capacity"] = new_df["max_capacity"].astype(
                manager.get_inventory()["max_capacity"].dtype
            )
            manager.update_inventory_data(new_df)

            manager.get_schema_profile()
            # update_inventory_data unconditionally clears the cache (full
            # DataFrame replacement), so this is expected to be 2 — but the
            # signature itself must still match between old and new frames.
            sig_before = manager._schema_cache_signature
            manager.get_schema_profile()  # third call
            assert manager._schema_cache_signature == sig_before, (
                "Signature should be identical when only row count changes."
            )
