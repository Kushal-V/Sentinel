"""
tests/test_auto_commit.py
=========================
Tests for F4 — Confidence-routed HITL ("human-on-the-loop").

Covers four behavioural guarantees:

1. ``auto_commit_eligible`` returns the correct (eligible, reason) tuple
   for the full predicate matrix:
     - master switch off                  → not eligible
     - confidence below threshold         → not eligible
     - trust below threshold              → not eligible
     - delta exceeds max_delta_fraction   → not eligible
     - missing row_key                    → not eligible (defensive)
     - all thresholds met + flag on       → eligible

2. ``commit_pending_changes(auto_only=True)`` partitions correctly:
     - eligible changes are committed
     - ineligible changes are re-staged (with skip-reason attached)
     - the function returns a summary dict

3. Sandbox re-validation still fires on the auto-commit fast-path —
   if the live state mutates between propose and commit so the change
   becomes invalid, auto-commit must REJECT it (re-stage), not silently
   apply an out-of-bounds update.

4. Transaction log records ``auto_committed=True`` and the agent's
   confidence value for auto-committed rows.

Tests use a tmp-dir-isolated ``FactoryDataManager`` so they never touch
real workspace data on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core import config as sentinel_config
from src.core.state_manager import FactoryDataManager
from src.tools.tool_registry import (
    auto_commit_eligible,
    commit_pending_changes,
    set_data_manager,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_manager(tmp_path, monkeypatch):
    """Spin up a FactoryDataManager in a tmp workspace.

    Each test gets a fresh on-disk workspace so state cannot leak.
    Also wires the singleton in ``tool_registry`` so module-level helpers
    that fall back to ``get_data_manager()`` see the test instance.
    """
    monkeypatch.setattr(sentinel_config, "WORKSPACES_DIR", tmp_path)
    import src.core.state_manager as sm
    monkeypatch.setattr(sm, "WORKSPACES_DIR", tmp_path)
    dm = FactoryDataManager(workspace="test_f4_auto_commit")
    set_data_manager(dm)
    yield dm


@pytest.fixture
def auto_commit_on(monkeypatch):
    """Force ``AUTO_COMMIT_ENABLED`` True for the duration of a test.

    Restores the original value via monkeypatch teardown. Tests that
    assert flag-off behaviour deliberately do NOT use this fixture.
    """
    monkeypatch.setattr(sentinel_config, "AUTO_COMMIT_ENABLED", True)
    yield


@pytest.fixture
def healthy_trust(isolated_manager):
    """Bump the maker agent's trust score above the auto-commit threshold."""
    # Default is 0.95 from DEFAULT_TRUST_SCORE — already above 0.80.
    # This fixture is a no-op but documents intent.
    return isolated_manager


def _build_staged_change(
    row_key: str = "ITEM-PLASTIC-01",
    target_column: str = "current_stock",
    delta: float = 100.0,           # 100 / 8500 ≈ 1.18% — under 5% cap
    confidence: float = 0.95,
    agent_id: str = "maker",
) -> dict:
    """Build a synthetic staged-change dict matching propose_state_change output."""
    return {
        "row_key": row_key,
        "target_column": target_column,
        "delta": delta,
        "justification": "test",
        "old_value": 8500.0,
        "new_value": 8500.0 + delta,
        "limit_value": 15000.0,
        "financial_impact": 0.0,
        "agent_id": agent_id,
        "confidence": confidence,
    }


# ===========================================================================
# CLASS 1 — auto_commit_eligible predicate matrix
# ===========================================================================

class TestAutoCommitEligible:
    """Exercises every branch of the eligibility decision tree."""

    def test_eligible_when_all_thresholds_met(self, isolated_manager, auto_commit_on):
        change = _build_staged_change(
            confidence=0.95,    # ≥ 0.85
            delta=100.0,        # 100/8500 ≈ 1.18% ≤ 5%
            agent_id="maker",   # default trust 0.95 ≥ 0.80
        )
        eligible, reason = auto_commit_eligible(change, isolated_manager)
        assert eligible, f"Expected eligible, got reason: {reason}"
        assert "all thresholds met" in reason

    def test_ineligible_when_master_switch_off(self, isolated_manager, monkeypatch):
        """Master switch off → ineligible regardless of other inputs."""
        monkeypatch.setattr(sentinel_config, "AUTO_COMMIT_ENABLED", False)
        change = _build_staged_change(confidence=1.0, delta=1.0)
        eligible, reason = auto_commit_eligible(change, isolated_manager)
        assert not eligible
        assert "disabled" in reason.lower()

    def test_ineligible_when_confidence_below_threshold(
        self, isolated_manager, auto_commit_on
    ):
        change = _build_staged_change(confidence=0.50)  # < 0.85
        eligible, reason = auto_commit_eligible(change, isolated_manager)
        assert not eligible
        assert "confidence" in reason.lower()

    def test_ineligible_when_trust_below_threshold(
        self, isolated_manager, auto_commit_on
    ):
        # Drive maker's trust below 0.80 by applying a large penalty.
        # Default is 0.95, so -0.20 → 0.75 < 0.80.
        isolated_manager.update_trust_score("maker", -0.20)
        change = _build_staged_change(agent_id="maker", confidence=0.95)
        eligible, reason = auto_commit_eligible(change, isolated_manager)
        assert not eligible
        assert "trust" in reason.lower()

    def test_ineligible_when_delta_exceeds_max_fraction(
        self, isolated_manager, auto_commit_on
    ):
        # current=8500, threshold 5% → delta>425 ineligible. Use 1000.
        change = _build_staged_change(delta=1000.0, confidence=0.95)
        eligible, reason = auto_commit_eligible(change, isolated_manager)
        assert not eligible
        assert "delta" in reason.lower()

    def test_ineligible_when_row_key_missing(
        self, isolated_manager, auto_commit_on
    ):
        """Defensive: unknown row_key → not eligible (no silent commit)."""
        change = _build_staged_change(row_key="DOES-NOT-EXIST-99")
        eligible, reason = auto_commit_eligible(change, isolated_manager)
        assert not eligible
        # Either resolves to "not found" or "row/column not resolvable"
        assert "not found" in reason.lower() or "not resolvable" in reason.lower()

    def test_ineligible_when_current_value_is_zero(
        self, isolated_manager, auto_commit_on
    ):
        """Zero baseline → relative delta undefined → defensive default ineligible."""
        # Decrement plastic stock all the way to 0 first.
        isolated_manager.update_inventory(
            item_id="ITEM-PLASTIC-01",
            target_column="current_stock",
            quantity_change=-8500,
        )
        change = _build_staged_change(delta=10.0, confidence=0.99)
        eligible, reason = auto_commit_eligible(change, isolated_manager)
        assert not eligible
        assert "0" in reason or "undefined" in reason.lower()


# ===========================================================================
# CLASS 2 — commit_pending_changes(auto_only=True) partitioning
# ===========================================================================

class TestAutoOnlyCommit:
    """Tests the partition behaviour of the auto-commit pass."""

    def test_all_eligible_are_committed_queue_empties(
        self, isolated_manager, auto_commit_on
    ):
        eligible_change = _build_staged_change(
            row_key="ITEM-PLASTIC-01",
            delta=50.0,
            confidence=0.95,
        )
        isolated_manager.add_pending_change(eligible_change)
        assert isolated_manager.pending_changes_count() == 1

        summary = commit_pending_changes(isolated_manager, auto_only=True)
        assert summary["auto_committed_count"] == 1
        assert summary["deferred_count"] == 0
        assert isolated_manager.pending_changes_count() == 0

        # Inventory must reflect the commit.
        inv = isolated_manager.get_inventory()
        new_stock = float(inv[inv["item_id"] == "ITEM-PLASTIC-01"]["current_stock"].iloc[0])
        assert new_stock == 8500.0 + 50.0

    def test_all_ineligible_are_re_staged_no_commit(
        self, isolated_manager, auto_commit_on
    ):
        # Confidence below threshold → ineligible.
        ineligible = _build_staged_change(confidence=0.30)
        isolated_manager.add_pending_change(ineligible)

        summary = commit_pending_changes(isolated_manager, auto_only=True)
        assert summary["auto_committed_count"] == 0
        assert summary["deferred_count"] == 1
        # Still staged, with skip-reason annotation.
        assert isolated_manager.pending_changes_count() == 1
        staged = isolated_manager.pending_changes
        assert "_auto_commit_skip_reason" in staged[0]
        assert "confidence" in staged[0]["_auto_commit_skip_reason"].lower()

        # Inventory unchanged.
        inv = isolated_manager.get_inventory()
        new_stock = float(inv[inv["item_id"] == "ITEM-PLASTIC-01"]["current_stock"].iloc[0])
        assert new_stock == 8500.0

    def test_mixed_partition_committed_and_deferred(
        self, isolated_manager, auto_commit_on
    ):
        ok_change = _build_staged_change(
            row_key="ITEM-PLASTIC-01", delta=10.0, confidence=0.99,
        )
        bad_change = _build_staged_change(
            row_key="ITEM-MICROCHIP-01", delta=10.0, confidence=0.20,
        )
        isolated_manager.add_pending_change(ok_change)
        isolated_manager.add_pending_change(bad_change)

        summary = commit_pending_changes(isolated_manager, auto_only=True)
        assert summary["auto_committed_count"] == 1
        assert summary["deferred_count"] == 1
        # The ineligible one is still staged.
        assert isolated_manager.pending_changes_count() == 1
        remaining = isolated_manager.pending_changes
        assert remaining[0]["row_key"] == "ITEM-MICROCHIP-01"

    def test_flag_off_auto_only_commits_nothing(self, isolated_manager, monkeypatch):
        """Master switch off → auto_only commits nothing, queue intact."""
        monkeypatch.setattr(sentinel_config, "AUTO_COMMIT_ENABLED", False)
        ok_change = _build_staged_change(confidence=1.0, delta=1.0)
        isolated_manager.add_pending_change(ok_change)

        summary = commit_pending_changes(isolated_manager, auto_only=True)
        assert summary["auto_committed_count"] == 0
        assert summary["deferred_count"] == 1
        assert isolated_manager.pending_changes_count() == 1

    def test_hitl_path_still_works_when_flag_off(self, isolated_manager):
        """``commit_pending_changes()`` (no args) is the legacy HITL path
        and must keep returning a list of committed dicts."""
        change = _build_staged_change(delta=10.0)
        isolated_manager.add_pending_change(change)
        committed = commit_pending_changes(isolated_manager)
        assert isinstance(committed, list)
        assert len(committed) == 1
        assert isolated_manager.pending_changes_count() == 0


# ===========================================================================
# CLASS 3 — Sandbox re-validation still applies on auto-commit
# ===========================================================================

class TestAutoCommitSandboxRevalidation:
    """The sandbox safety guarantee must hold on the F4 fast-path."""

    def test_stale_change_is_rejected_even_when_eligible(
        self, isolated_manager, auto_commit_on
    ):
        """If the live state mutates so the staged change would now overflow,
        auto-commit must REJECT it via sandbox re-validation."""
        # PLASTIC-01: current=8500, max_capacity=15000.
        # Stage a +200 change (well under any threshold).
        change = _build_staged_change(
            row_key="ITEM-PLASTIC-01",
            target_column="current_stock",
            delta=200.0,
            confidence=0.99,
        )
        isolated_manager.add_pending_change(change)

        # Now ANOTHER process maxes out the stock to 15000.
        # delta=+6500 lands us exactly at the cap.
        isolated_manager.update_inventory(
            item_id="ITEM-PLASTIC-01",
            target_column="current_stock",
            quantity_change=6500,
        )

        summary = commit_pending_changes(isolated_manager, auto_only=True)

        # Sandbox catches the now-overflow at commit time. The change is
        # rejected (not committed), and re-staged for human review. The
        # eligibility check itself may or may not flag it — what matters
        # is that nothing actually got past the sandbox.
        assert summary["auto_committed_count"] == 0, (
            "Stale state must NOT be auto-committed — sandbox re-validation "
            "must fire on the F4 fast-path."
        )
        # Inventory is still at 15000 from the manual mutation, not 15200.
        inv = isolated_manager.get_inventory()
        new_stock = float(inv[inv["item_id"] == "ITEM-PLASTIC-01"]["current_stock"].iloc[0])
        assert new_stock == 15000.0


# ===========================================================================
# CLASS 4 — Transaction log captures auto_committed + confidence
# ===========================================================================

class TestTransactionLogAutoCommitFields:
    """Ensure the F4 transaction-log columns are populated correctly."""

    def test_auto_commit_writes_auto_committed_and_confidence(
        self, isolated_manager, auto_commit_on
    ):
        change = _build_staged_change(
            row_key="ITEM-PLASTIC-01", delta=25.0, confidence=0.93,
        )
        isolated_manager.add_pending_change(change)
        commit_pending_changes(isolated_manager, auto_only=True)

        log = isolated_manager.get_transaction_log()
        # Filter to AGENT_ACTION rows (skip any rejection logs).
        agent_rows = log[log["event_id"] == "AGENT_ACTION"]
        assert len(agent_rows) == 1, "Exactly one AGENT_ACTION row expected."

        row = agent_rows.iloc[0]
        assert bool(row["auto_committed"]) is True, (
            "auto_committed must be True for rows committed via the F4 fast-path."
        )
        assert float(row["confidence"]) == pytest.approx(0.93)

    def test_hitl_commit_writes_auto_committed_false(self, isolated_manager):
        """HITL-committed rows must have auto_committed=False."""
        change = _build_staged_change(delta=10.0, confidence=0.50)
        isolated_manager.add_pending_change(change)
        commit_pending_changes(isolated_manager)  # no auto_only

        log = isolated_manager.get_transaction_log()
        agent_rows = log[log["event_id"] == "AGENT_ACTION"]
        assert len(agent_rows) == 1
        assert bool(agent_rows.iloc[0]["auto_committed"]) is False

    def test_log_transaction_no_confidence_records_sentinel(self, isolated_manager):
        """When confidence isn't passed, the sentinel -1.0 is recorded."""
        isolated_manager.log_transaction(
            event_id="TEST_EVENT",
            agent_id="maker",
            action_schema={"foo": "bar"},
            financial_impact=0.0,
            sandbox_approved=True,
        )
        log = isolated_manager.get_transaction_log()
        last = log[log["event_id"] == "TEST_EVENT"].iloc[-1]
        assert float(last["confidence"]) == pytest.approx(-1.0)
        assert bool(last["auto_committed"]) is False
