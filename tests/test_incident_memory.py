"""Unit tests for ``src/core/incident_memory.py`` (Feature F3 — RAG over log).

Covers:
* Empty store + empty query graceful handling.
* TF-IDF retrieval relevance (similar incidents rank higher).
* JSONL persistence across IncidentMemory instances on the same path.
* ``top_k`` clamping when fewer records exist.
* Thread-safety under concurrent ``add`` calls.
* End-to-end observer integration with ``FactoryDataManager.log_transaction``.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on sys.path when pytest is invoked from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.incident_memory import IncidentMemory  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Return a per-test workspace directory under pytest's tmp_path."""
    return tmp_path / "wkspc_test"


# ---------------------------------------------------------------------------
# Core IncidentMemory tests
# ---------------------------------------------------------------------------


def test_empty_query_returns_empty(tmp_workspace: Path) -> None:
    """Querying an empty store returns ``[]`` and ``count`` is 0."""
    mem = IncidentMemory(tmp_workspace)
    assert mem.query("anything", top_k=5) == []
    assert mem.count() == 0


def test_empty_query_string_returns_empty(tmp_workspace: Path) -> None:
    """Empty / whitespace queries are handled gracefully."""
    mem = IncidentMemory(tmp_workspace)
    mem.add(
        {
            "transaction_id": "t1",
            "timestamp": "2026-04-01",
            "justification": "supplier delay",
        }
    )
    assert mem.query("", top_k=3) == []
    assert mem.query("   ", top_k=3) == []


def test_add_and_query_finds_similar(tmp_workspace: Path) -> None:
    """The most semantically similar record ranks first."""
    mem = IncidentMemory(tmp_workspace)
    mem.add(
        {
            "transaction_id": "t1",
            "timestamp": "2026-04-01",
            "type": "STOCK_LOW",
            "sku": "WIDGET_A",
            "justification": "supplier delay caused shortage",
        }
    )
    mem.add(
        {
            "transaction_id": "t2",
            "timestamp": "2026-04-15",
            "type": "DEMAND_SPIKE",
            "sku": "WIDGET_B",
            "justification": "marketing campaign drove unexpected demand",
        }
    )
    results = mem.query("supplier shortage", top_k=2)
    assert len(results) == 2
    top_desc = results[0]["description"].lower()
    assert "supplier" in top_desc or "delay" in top_desc or "shortage" in top_desc
    # Similarity must be a float in [0, 1].
    assert 0.0 <= results[0]["similarity"] <= 1.0


def test_persistence_across_instances(tmp_workspace: Path) -> None:
    """A second IncidentMemory pointed at the same path sees prior records."""
    mem1 = IncidentMemory(tmp_workspace)
    mem1.add(
        {
            "transaction_id": "x",
            "timestamp": "2026-04-20",
            "type": "test",
            "justification": "persistent record",
        }
    )
    mem2 = IncidentMemory(tmp_workspace)
    assert mem2.count() == 1
    # And the reloaded record should be queryable.
    results = mem2.query("persistent", top_k=1)
    assert len(results) == 1
    assert "persistent" in results[0]["description"].lower()


def test_top_k_clamps_to_records(tmp_workspace: Path) -> None:
    """Asking for more records than exist returns however many exist."""
    mem = IncidentMemory(tmp_workspace)
    mem.add(
        {
            "transaction_id": "a",
            "timestamp": "x",
            "justification": "alpha",
        }
    )
    results = mem.query("alpha", top_k=100)
    assert len(results) == 1


def test_top_k_clamps_lower_bound(tmp_workspace: Path) -> None:
    """``top_k`` of 0 / negative is clamped up to 1."""
    mem = IncidentMemory(tmp_workspace)
    mem.add(
        {
            "transaction_id": "a",
            "timestamp": "x",
            "justification": "beta",
        }
    )
    results = mem.query("beta", top_k=0)
    assert len(results) == 1


def test_thread_safety_concurrent_adds(tmp_workspace: Path) -> None:
    """20 threads adding concurrently — final count must be exactly 20."""
    mem = IncidentMemory(tmp_workspace)

    def worker(i: int) -> None:
        mem.add(
            {
                "transaction_id": f"t{i}",
                "timestamp": "t",
                "justification": f"event {i}",
            }
        )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert mem.count() == 20


def test_description_handles_action_schema_json_string(tmp_workspace: Path) -> None:
    """Nested JSON-encoded ``action_schema`` payloads are flattened into the description."""
    import json

    mem = IncidentMemory(tmp_workspace)
    mem.add(
        {
            "transaction_id": "n1",
            "timestamp": "2026-04-22",
            "agent_id": "mover",
            "action_schema": json.dumps(
                {
                    "row_key": "ROUTE_42",
                    "justification": "highway closure rerouting",
                }
            ),
        }
    )
    results = mem.query("highway closure", top_k=1)
    assert len(results) == 1
    assert "highway" in results[0]["description"].lower()


def test_metadata_preserved_verbatim(tmp_workspace: Path) -> None:
    """The full transaction dict round-trips through ``metadata``."""
    mem = IncidentMemory(tmp_workspace)
    mem.add(
        {
            "transaction_id": "z9",
            "timestamp": "2026-04-25",
            "agent_id": "keeper",
            "justification": "buffer",
            "financial_impact": -123.45,
        }
    )
    results = mem.query("buffer", top_k=1)
    assert results[0]["metadata"]["agent_id"] == "keeper"
    assert results[0]["metadata"]["financial_impact"] == -123.45
    assert results[0]["transaction_id"] == "z9"


def test_non_dict_input_ignored(tmp_workspace: Path) -> None:
    """Non-dict transactions are dropped without raising."""
    mem = IncidentMemory(tmp_workspace)
    mem.add("not a dict")  # type: ignore[arg-type]
    mem.add(None)  # type: ignore[arg-type]
    assert mem.count() == 0


# ---------------------------------------------------------------------------
# Integration: FactoryDataManager observer
# ---------------------------------------------------------------------------


def test_observer_registration_and_invocation(tmp_workspace: Path) -> None:
    """``register_transaction_observer`` callbacks fire on log_transaction."""
    # Build a FactoryDataManager backed entirely by the temp workspace —
    # we monkey-patch WORKSPACES_DIR so no real ``data/`` writes happen.
    from src.core import state_manager as sm_mod

    with patch.object(sm_mod, "WORKSPACES_DIR", tmp_workspace.parent):
        dm = sm_mod.FactoryDataManager(workspace=tmp_workspace.name)

        captured: list[dict] = []
        dm.register_transaction_observer(lambda txn: captured.append(txn))

        dm.log_transaction(
            event_id="EVT-OBS-1",
            agent_id="maker",
            action_schema={"justification": "observer test"},
            financial_impact=0.0,
            sandbox_approved=True,
        )

        assert len(captured) == 1
        assert captured[0]["event_id"] == "EVT-OBS-1"
        assert captured[0]["agent_id"] == "maker"


def test_failing_observer_does_not_break_logging(tmp_workspace: Path) -> None:
    """A raising observer is isolated — log_transaction still completes."""
    from src.core import state_manager as sm_mod

    with patch.object(sm_mod, "WORKSPACES_DIR", tmp_workspace.parent):
        dm = sm_mod.FactoryDataManager(workspace=tmp_workspace.name)

        def boom(_txn: dict) -> None:
            raise RuntimeError("observer is angry")

        dm.register_transaction_observer(boom)

        # Must not raise
        dm.log_transaction(
            event_id="EVT-OBS-2",
            agent_id="mover",
            action_schema={"justification": "isolation test"},
            financial_impact=0.0,
            sandbox_approved=True,
        )

        # Ledger still received the row.
        log_df = dm.get_transaction_log()
        assert any(log_df["event_id"] == "EVT-OBS-2")


def test_get_incident_memory_is_per_workspace(tmp_workspace: Path) -> None:
    """``get_incident_memory`` returns a per-workspace, lazily-created instance."""
    from src.core import state_manager as sm_mod

    with patch.object(sm_mod, "WORKSPACES_DIR", tmp_workspace.parent):
        dm = sm_mod.FactoryDataManager(workspace=tmp_workspace.name)
        mem_a = dm.get_incident_memory()
        mem_b = dm.get_incident_memory()
        assert mem_a is mem_b  # cached singleton per manager
        assert mem_a is not None
        assert mem_a.count() == 0

        # Logging a transaction should index it via the observer hook.
        dm.log_transaction(
            event_id="EVT-OBS-3",
            agent_id="keeper",
            action_schema={"justification": "warehouse capacity reached"},
            financial_impact=0.0,
            sandbox_approved=True,
        )
        assert mem_a.count() == 1
        results = mem_a.query("warehouse capacity", top_k=1)
        assert len(results) == 1
        assert "warehouse" in results[0]["description"].lower()
