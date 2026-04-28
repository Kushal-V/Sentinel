"""
tests/test_pending_changes_lock.py
==================================
Tests for I5 — Lock pending_changes list.

Verifies that ``FactoryDataManager`` exposes the staging queue only via
lock-guarded helpers (``add_pending_change``, ``get_pending_changes``,
``clear_pending_changes``, ``pending_changes_count``) and that the
backward-compatibility ``pending_changes`` property returns a deep copy
that callers cannot use to corrupt internal state.

The concurrent-append test exercises the threading guarantee: two
threads each appending N items must yield exactly 2*N items in the
final queue, with no lost updates due to interleaved list mutations.
"""

from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.state_manager import FactoryDataManager
from src.core import config as _config


@pytest.fixture
def isolated_manager(tmp_path, monkeypatch):
    """Spin up a FactoryDataManager rooted in a tmp workspaces dir.

    Each test gets a fresh on-disk workspace so state cannot leak between
    tests, and the cold-start mock data is generated fresh.
    """
    monkeypatch.setattr(_config, "WORKSPACES_DIR", tmp_path)
    # state_manager imported the symbol directly, so patch the binding too.
    import src.core.state_manager as sm
    monkeypatch.setattr(sm, "WORKSPACES_DIR", tmp_path)
    dm = FactoryDataManager(workspace="test_pending_lock")
    yield dm


def _make_change(n: int) -> dict:
    """Build a synthetic staged change for tests."""
    return {
        "row_key": f"ITEM-{n:04d}",
        "target_column": "current_stock",
        "delta": -1.0 * n,
        "justification": f"test change {n}",
        "old_value": 100.0,
        "new_value": 100.0 - n,
        "limit_value": 1000.0,
        "financial_impact": -1.5 * n,
        "agent_id": "maker",
    }


# ---------------------------------------------------------------------------
# Basic API behaviour
# ---------------------------------------------------------------------------

def test_add_pending_change_appends_and_is_visible(isolated_manager):
    """add_pending_change persists the change and get_pending_changes returns it."""
    dm = isolated_manager
    assert dm.get_pending_changes() == []

    change = _make_change(1)
    dm.add_pending_change(change)

    snapshot = dm.get_pending_changes()
    assert len(snapshot) == 1
    assert snapshot[0]["row_key"] == "ITEM-0001"
    assert snapshot[0]["delta"] == -1.0


def test_clear_pending_changes_drains_and_resets(isolated_manager):
    """clear_pending_changes returns the drained snapshot and empties the queue."""
    dm = isolated_manager
    for n in range(5):
        dm.add_pending_change(_make_change(n))
    assert dm.pending_changes_count() == 5

    drained = dm.clear_pending_changes()

    assert len(drained) == 5
    assert [c["row_key"] for c in drained] == [f"ITEM-{n:04d}" for n in range(5)]
    assert dm.pending_changes_count() == 0
    assert dm.get_pending_changes() == []


def test_clear_on_empty_queue_returns_empty_list(isolated_manager):
    """clear_pending_changes is safe to call on an empty queue."""
    dm = isolated_manager
    assert dm.clear_pending_changes() == []
    assert dm.pending_changes_count() == 0


# ---------------------------------------------------------------------------
# Backward-compat property — must return a deep copy
# ---------------------------------------------------------------------------

def test_pending_changes_property_returns_a_copy(isolated_manager):
    """Mutating the list returned by the property does NOT affect the manager."""
    dm = isolated_manager
    dm.add_pending_change(_make_change(1))

    snapshot = dm.pending_changes
    # Mutate the snapshot in three ways: append, pop, in-place edit.
    snapshot.append(_make_change(99))
    snapshot.pop(0)
    if snapshot:
        snapshot[0]["row_key"] = "TAMPERED"

    # Internal state must be untouched.
    fresh = dm.get_pending_changes()
    assert len(fresh) == 1
    assert fresh[0]["row_key"] == "ITEM-0001"
    assert dm.pending_changes_count() == 1


def test_pending_changes_property_is_read_only_view(isolated_manager):
    """Direct dict mutation on a returned snapshot is also isolated."""
    dm = isolated_manager
    dm.add_pending_change(_make_change(7))

    snap_a = dm.pending_changes
    snap_a[0]["justification"] = "MUTATED"

    snap_b = dm.pending_changes
    assert snap_b[0]["justification"] == "test change 7"


# ---------------------------------------------------------------------------
# pending_changes_count consistency
# ---------------------------------------------------------------------------

def test_pending_changes_count_matches_len(isolated_manager):
    """pending_changes_count tracks the queue length through every operation."""
    dm = isolated_manager
    assert dm.pending_changes_count() == 0

    dm.add_pending_change(_make_change(1))
    assert dm.pending_changes_count() == 1

    dm.add_pending_change(_make_change(2))
    dm.add_pending_change(_make_change(3))
    assert dm.pending_changes_count() == 3
    assert len(dm.get_pending_changes()) == 3

    dm.clear_pending_changes()
    assert dm.pending_changes_count() == 0


# ---------------------------------------------------------------------------
# Thread-safety — concurrent appends must not lose entries
# ---------------------------------------------------------------------------

def test_concurrent_appends_preserve_all_entries(isolated_manager):
    """Two threads appending N items each yields exactly 2*N items."""
    dm = isolated_manager
    N = 200

    def worker(offset: int) -> None:
        for i in range(N):
            dm.add_pending_change(_make_change(offset + i))

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(N,))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert dm.pending_changes_count() == 2 * N
    snapshot = dm.get_pending_changes()
    keys = {c["row_key"] for c in snapshot}
    expected = {f"ITEM-{n:04d}" for n in range(2 * N)}
    assert keys == expected


def test_concurrent_add_and_clear_is_consistent(isolated_manager):
    """A clear interleaved with appends never observes partial list state."""
    dm = isolated_manager
    N = 100

    drained_total: list[int] = []
    drained_lock = threading.Lock()

    def appender() -> None:
        for i in range(N):
            dm.add_pending_change(_make_change(i))

    def clearer() -> None:
        # Drain a few times during the appends; collect what we drained
        # so we can assert the union with the final residue equals N.
        for _ in range(5):
            drained = dm.clear_pending_changes()
            with drained_lock:
                drained_total.append(len(drained))

    threads = [
        threading.Thread(target=appender),
        threading.Thread(target=clearer),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final_residue = dm.pending_changes_count()
    assert sum(drained_total) + final_residue == N
