"""
src/core/state_manager.py
=========================
The deterministic data engine for the Sentinel Supply Chain Digital Twin.

``FactoryDataManager`` is the single source of truth for ALL runtime state.
Agents NEVER hold state internally; they interact with the factory's physical
reality exclusively through the methods exposed here.  Every public method
either reads a deep copy of the data (preventing accidental mutation of the
live frame) or writes atomically to disk after performing an in-memory update.

Thread-safety note
------------------
Streamlit re-runs the entire script on every widget interaction, which can
cause concurrent I/O if users click rapidly.  A ``threading.Lock`` guards all
write operations so partial file writes cannot corrupt the CSV/JSON artefacts.

Usage
-----
>>> from src.core.state_manager import FactoryDataManager
>>> dm = FactoryDataManager()
>>> inv = dm.get_inventory()
>>> dm.update_inventory("ITEM-001", quantity_change=-50)
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import (
    AGENT_FALLBACKS,
    AGENT_IDS,
    DATA_DIR,
    DEFAULT_TRUST_SCORE,
    INVENTORY_CSV,
    MAX_TRUST_SCORE,
    MIN_TRUST_SCORE,
    TRANSACTION_LOG_CSV,
    AGENT_TRUST_SCORES_JSON,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inventory Schema
# ---------------------------------------------------------------------------

#: Expected columns and their pandas dtypes for the inventory DataFrame.
_INVENTORY_DTYPE: dict[str, str] = {
    "item_id": "str",
    "item_type": "str",
    "location_id": "str",
    "current_stock": "int64",
    "max_capacity": "int64",
    "unit_cost_usd": "float64",
}

#: Expected columns and their dtypes for the transaction log DataFrame.
_TRANSACTION_LOG_DTYPE: dict[str, str] = {
    "transaction_id": "str",
    "timestamp": "str",
    "event_id": "str",
    "agent_id": "str",
    "action_schema": "str",   # Serialised JSON string
    "financial_impact": "float64",
    "sandbox_approved": "bool",
}


# ---------------------------------------------------------------------------
# Mock Data Generators
# ---------------------------------------------------------------------------

def _build_mock_inventory() -> pd.DataFrame:
    """Return a realistic toy factory inventory DataFrame for cold-start.

    Returns:
        A 7-row DataFrame with schema matching ``_INVENTORY_DTYPE``.
    """
    records = [
        {
            "item_id":        "ITEM-PLASTIC-01",
            "item_type":      "RAW_MATERIAL",
            "location_id":    "WH-A-RAW",
            "current_stock":  8_500,
            "max_capacity":   15_000,
            "unit_cost_usd":  0.45,
        },
        {
            "item_id":        "ITEM-MICROCHIP-01",
            "item_type":      "RAW_MATERIAL",
            "location_id":    "WH-A-RAW",
            "current_stock":  3_200,
            "max_capacity":   6_000,
            "unit_cost_usd":  2.80,
        },
        {
            "item_id":        "ITEM-PACKAGING-01",
            "item_type":      "RAW_MATERIAL",
            "location_id":    "WH-B-PACK",
            "current_stock":  12_000,
            "max_capacity":   20_000,
            "unit_cost_usd":  0.15,
        },
        {
            "item_id":        "ITEM-PAINTPIGMENT-01",
            "item_type":      "RAW_MATERIAL",
            "location_id":    "WH-A-RAW",
            "current_stock":  1_400,
            "max_capacity":   3_000,
            "unit_cost_usd":  1.20,
        },
        {
            "item_id":        "ITEM-WIP-BODYFRAME-01",
            "item_type":      "WORK_IN_PROGRESS",
            "location_id":    "FACTORY-FLOOR-1",
            "current_stock":  650,
            "max_capacity":   1_000,
            "unit_cost_usd":  4.75,
        },
        {
            "item_id":        "ITEM-FG-TOY-DELUXE-01",
            "item_type":      "FINISHED_GOOD",
            "location_id":    "WH-C-FG",
            "current_stock":  2_300,
            "max_capacity":   5_000,
            "unit_cost_usd":  18.99,
        },
        {
            "item_id":        "ITEM-FG-TOY-BASIC-01",
            "item_type":      "FINISHED_GOOD",
            "location_id":    "WH-C-FG",
            "current_stock":  4_100,
            "max_capacity":   8_000,
            "unit_cost_usd":  9.49,
        },
    ]
    df = pd.DataFrame(records)
    for col, dtype in _INVENTORY_DTYPE.items():
        df[col] = df[col].astype(dtype)
    return df


def _build_empty_transaction_log() -> pd.DataFrame:
    """Return an empty transaction log DataFrame with the correct schema.

    Returns:
        An empty DataFrame with columns matching ``_TRANSACTION_LOG_DTYPE``.
    """
    df = pd.DataFrame(columns=list(_TRANSACTION_LOG_DTYPE.keys()))
    for col, dtype in _TRANSACTION_LOG_DTYPE.items():
        df[col] = df[col].astype(dtype)
    return df


def _build_default_trust_scores() -> dict[str, Any]:
    """Return a default trust score dictionary for a cold-start scenario.

    Returns:
        Nested dict matching the ``agent_trust_scores.json`` schema.
    """
    return {
        "global_metrics": {
            "routing_threshold": 0.50,
            "last_analysis_run": datetime.now(tz=timezone.utc).isoformat(),
        },
        "agents": {
            agent_id: {
                "trust_score": DEFAULT_TRUST_SCORE,
                "total_decisions": 0,
                "recent_penalties": 0,
                "preferred_fallback": AGENT_FALLBACKS[agent_id],
            }
            for agent_id in AGENT_IDS
        },
    }


# ---------------------------------------------------------------------------
# FactoryDataManager
# ---------------------------------------------------------------------------

class FactoryDataManager:
    """Single source of truth for all Sentinel factory state.

    This class owns three persistent artefacts:
    * ``inventory.csv``         — real-time stock levels with capacity limits.
    * ``transaction_log.csv``   — append-only ledger of every committed action.
    * ``agent_trust_scores.json`` — dynamic per-agent routing weights.

    On instantiation the class will auto-generate realistic mock data for any
    artefact that does not yet exist on disk, allowing the application to boot
    without manual data entry.

    All write methods acquire ``_write_lock`` before mutating in-memory state
    and flushing to disk, preventing partial writes under concurrent Streamlit
    re-runs.

    Attributes:
        _inventory (pd.DataFrame): Live inventory state (never exposed directly).
        _transaction_log (pd.DataFrame): Growing ledger of committed actions.
        _trust_scores (dict): Agent routing weights loaded from JSON.
        _write_lock (threading.Lock): Guards all state-mutating operations.
    """

    def __init__(self) -> None:
        """Initialise the data manager.

        Creates the ``data/`` directory if absent, then loads each artefact from
        disk.  If an artefact does not exist a realistic mock dataset is written
        to disk first so that all subsequent operations have a consistent base.

        Raises:
            OSError: If the data directory cannot be created.
            ValueError: If an existing CSV contains unexpected columns.
        """
        self._write_lock: threading.Lock = threading.Lock()
        self._ensure_data_directory()

        self._inventory: pd.DataFrame = self._load_or_create_inventory()
        self._transaction_log: pd.DataFrame = self._load_or_create_transaction_log()
        self._trust_scores: dict[str, Any] = self._load_or_create_trust_scores()

        logger.info(
            "FactoryDataManager initialised. Inventory rows: %d | "
            "Log rows: %d | Agents tracked: %d",
            len(self._inventory),
            len(self._transaction_log),
            len(self._trust_scores.get("agents", {})),
        )

    # ------------------------------------------------------------------
    # Private: Initialisation helpers
    # ------------------------------------------------------------------

    def _ensure_data_directory(self) -> None:
        """Create the ``data/`` directory (and parents) if it does not exist.

        Raises:
            OSError: Propagated from ``Path.mkdir`` if creation fails due to
                permission errors or invalid path.
        """
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug("Data directory confirmed at: %s", DATA_DIR)

    def _load_or_create_inventory(self) -> pd.DataFrame:
        """Load inventory from disk, or create and persist a mock dataset.

        Returns:
            The inventory DataFrame with dtypes enforced.

        Raises:
            ValueError: If the CSV exists but is missing required columns.
        """
        if INVENTORY_CSV.exists():
            # Let pandas infer dtypes naturally instead of forcing str.
            # This allows DynamicSchemaInferencer to accurately detect numeric columns.
            df = pd.read_csv(INVENTORY_CSV)
            logger.info("Loaded dynamic inventory from %s (%d rows).", INVENTORY_CSV, len(df))
            return df

        logger.warning("inventory.csv not found. Generating mock dataset.")
        df = _build_mock_inventory()
        df.to_csv(INVENTORY_CSV, index=False)
        logger.info("Mock inventory written to %s.", INVENTORY_CSV)
        return df

    def _load_or_create_transaction_log(self) -> pd.DataFrame:
        """Load transaction log from disk, or create an empty schema-compliant frame.

        Returns:
            The transaction log DataFrame.

        Raises:
            ValueError: If the CSV exists but is missing required columns.
        """
        if TRANSACTION_LOG_CSV.exists():
            df = pd.read_csv(TRANSACTION_LOG_CSV, dtype=str)
            if df.empty:
                # File exists but is empty (header only) — acceptable.
                df = _build_empty_transaction_log()
            else:
                missing_cols = set(_TRANSACTION_LOG_DTYPE) - set(df.columns)
                if missing_cols:
                    raise ValueError(
                        f"transaction_log.csv is missing required columns: {missing_cols}"
                    )
                for col, dtype in _TRANSACTION_LOG_DTYPE.items():
                    df[col] = df[col].astype(dtype)
            logger.info(
                "Loaded transaction log from %s (%d rows).",
                TRANSACTION_LOG_CSV,
                len(df),
            )
            return df

        logger.warning("transaction_log.csv not found. Creating empty ledger.")
        df = _build_empty_transaction_log()
        df.to_csv(TRANSACTION_LOG_CSV, index=False)
        logger.info("Empty transaction log written to %s.", TRANSACTION_LOG_CSV)
        return df

    def _load_or_create_trust_scores(self) -> dict[str, Any]:
        """Load trust scores from disk, or create and persist default values.

        Returns:
            The full trust score dictionary.

        Raises:
            json.JSONDecodeError: If the JSON file is present but malformed.
        """
        if AGENT_TRUST_SCORES_JSON.exists():
            with AGENT_TRUST_SCORES_JSON.open("r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded trust scores from %s.", AGENT_TRUST_SCORES_JSON)
            return data

        logger.warning("agent_trust_scores.json not found. Generating defaults.")
        data = _build_default_trust_scores()
        with AGENT_TRUST_SCORES_JSON.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Default trust scores written to %s.", AGENT_TRUST_SCORES_JSON)
        return data

    # ------------------------------------------------------------------
    # Private: Persistence helpers
    # ------------------------------------------------------------------

    def _flush_inventory(self) -> None:
        """Write the in-memory inventory frame to disk atomically.

        Must be called while ``_write_lock`` is already held by the caller.
        """
        self._inventory.to_csv(INVENTORY_CSV, index=False)
        logger.debug("Inventory flushed to %s.", INVENTORY_CSV)

    def _flush_transaction_log(self) -> None:
        """Append the in-memory log frame to disk.

        Must be called while ``_write_lock`` is already held by the caller.
        """
        self._transaction_log.to_csv(TRANSACTION_LOG_CSV, index=False)
        logger.debug("Transaction log flushed to %s.", TRANSACTION_LOG_CSV)

    def _flush_trust_scores(self) -> None:
        """Serialise the trust score dict to JSON on disk.

        Must be called while ``_write_lock`` is already held by the caller.
        """
        with AGENT_TRUST_SCORES_JSON.open("w", encoding="utf-8") as f:
            json.dump(self._trust_scores, f, indent=2)
        logger.debug("Trust scores flushed to %s.", AGENT_TRUST_SCORES_JSON)

    # ------------------------------------------------------------------
    # Public: Read operations
    # ------------------------------------------------------------------

    def get_inventory(self) -> pd.DataFrame:
        """Return a deep copy of the current inventory DataFrame.

        Agents and tools must ALWAYS call this method instead of accessing
        the underlying frame directly.  The deep copy prevents accidental
        mutation of live state outside a controlled write path.

        Returns:
            A full deep copy of the inventory DataFrame.

        Example:
            >>> dm = FactoryDataManager()
            >>> inv = dm.get_inventory()
            >>> print(inv.columns.tolist())
        """
        return self._inventory.copy(deep=True)

    def _detect_primary_key(self) -> str:
        """Detect the primary key column dynamically using schema inference.

        Returns:
            The name of the primary key column detected in the current dataset.
        """
        from src.core.schema_engine import DynamicSchemaInferencer
        profile = DynamicSchemaInferencer(self._inventory).infer()
        return profile.primary_key_column

    def get_item(self, item_id: str) -> dict[str, Any]:
        """Return the details of a single inventory item as a plain dictionary.

        Uses dynamic primary key detection — works with any column name
        (``item_id``, ``sku``, ``bed_id``, etc.).

        Args:
            item_id: The unique identifier of the item to retrieve
                (must match a value in the primary key column exactly).

        Returns:
            A dictionary representation of the item's row.

        Raises:
            ValueError: If the item does not exist in the inventory.
        """
        pk_col = self._detect_primary_key()
        mask = self._inventory[pk_col] == item_id
        if not mask.any():
            valid_ids = self._inventory[pk_col].tolist()
            raise ValueError(
                f"Item '{item_id}' not found in column '{pk_col}'. "
                f"Valid IDs: {valid_ids[:20]}{'...' if len(valid_ids) > 20 else ''}"
            )
        row = self._inventory.loc[mask].iloc[0]
        return row.to_dict()

    def get_transaction_log(self) -> pd.DataFrame:
        """Return a deep copy of the full transaction log.

        Returns:
            A deep copy of the transaction log DataFrame, ordered by timestamp
            ascending.

        Example:
            >>> dm = FactoryDataManager()
            >>> log = dm.get_transaction_log()
        """
        return self._transaction_log.copy(deep=True)

    def get_trust_scores(self) -> dict[str, Any]:
        """Return the current agent trust score dictionary.

        Returns:
            A deep copy of the trust score structure so callers cannot
            accidentally mutate the in-memory state.
        """
        import copy
        return copy.deepcopy(self._trust_scores)

    # ------------------------------------------------------------------
    # Public: Write operations
    # ------------------------------------------------------------------

    def update_inventory(self, item_id: str, target_column: str, quantity_change: int) -> bool:
        """Update a numeric column for a given row and persist to disk.

        This method is fully dynamic — it detects the primary key column
        and respects any schema-inferred constraints (e.g., stock vs. capacity).
        No hardcoded column names are used.

        Args:
            item_id: Value in the primary key column identifying the row.
            target_column: The numeric column to update.
            quantity_change: Signed integer delta to apply.

        Returns:
            ``True`` on successful update and disk flush.

        Raises:
            ValueError: If the row or column doesn't exist, or constraints are violated.
        """
        with self._write_lock:
            pk_col = self._detect_primary_key()
            mask = self._inventory[pk_col] == item_id
            if not mask.any():
                raise ValueError(
                    f"Cannot update inventory: '{item_id}' not found in column '{pk_col}'."
                )

            if target_column not in self._inventory.columns:
                raise ValueError(
                    f"Column '{target_column}' does not exist. "
                    f"Available: {list(self._inventory.columns)}"
                )

            idx = self._inventory.index[mask][0]
            current = float(self._inventory.at[idx, target_column])
            new_value = current + quantity_change

            # Check for schema-inferred constraint (e.g., stock <= capacity)
            from src.core.schema_engine import DynamicSchemaInferencer
            profile = DynamicSchemaInferencer(self._inventory).infer()
            for rule in profile.constraint_rules:
                if rule.mutable_column == target_column:
                    limit_value = float(self._inventory.at[idx, rule.limit_column])
                    if new_value > limit_value:
                        raise ValueError(
                            f"CONSTRAINT VIOLATION: Update would set '{target_column}' to "
                            f"{new_value}, exceeding '{rule.limit_column}' limit of {limit_value}. "
                            f"Current value: {current}."
                        )

            if new_value < 0:
                raise ValueError(
                    f"CONSTRAINT VIOLATION: Update would set '{target_column}' to "
                    f"{new_value}, which is below 0. Current value: {current}."
                )

            self._inventory.at[idx, target_column] = new_value
            self._flush_inventory()

            logger.info(
                "Inventory updated | pk=%s | col=%s | old=%.2f | new=%.2f | limit=%.2f",
                item_id,
                target_column,
                current,
                new_value,
                new_value,  # placeholder if no limit
            )
            return True

    def log_transaction(
        self,
        event_id: str,
        agent_id: str,
        action_schema: dict[str, Any],
        financial_impact: float,
        sandbox_approved: bool,
    ) -> None:
        """Append a new record to the transaction ledger and flush to disk.

        Args:
            event_id: The identifier of the triggering crisis event
                (e.g., ``"EVT-8921-PORT-STRIKE"``).
            agent_id: The canonical identifier of the agent that authored
                the decision (must be one of ``"maker"``, ``"mover"``,
                ``"keeper"``).
            action_schema: A dictionary capturing what the agent decided to do.
                This will be JSON-serialised for storage.
            financial_impact: The net financial consequence of the action in USD.
                Positive values are savings; negative values are costs.
            sandbox_approved: ``True`` if the Sandbox validated the action
                before it was committed; ``False`` if the record is being
                written to capture a rejection event.

        Returns:
            None

        Raises:
            ValueError: If ``agent_id`` is not a recognised agent identifier.

        Example:
            >>> dm = FactoryDataManager()
            >>> dm.log_transaction(
            ...     event_id="EVT-001",
            ...     agent_id="mover",
            ...     action_schema={"action": "REROUTE", "route": "TRUCK-14"},
            ...     financial_impact=-4_500.00,
            ...     sandbox_approved=True,
            ... )
        """
        if agent_id not in AGENT_IDS:
            raise ValueError(
                f"Unknown agent_id '{agent_id}'. "
                f"Valid agent IDs are: {list(AGENT_IDS)}"
            )

        record: dict[str, Any] = {
            "transaction_id": str(uuid.uuid4()),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "event_id": event_id,
            "agent_id": agent_id,
            "action_schema": json.dumps(action_schema),
            "financial_impact": float(financial_impact),
            "sandbox_approved": sandbox_approved,
        }

        with self._write_lock:
            new_row = pd.DataFrame([record]).astype(
                {col: dtype for col, dtype in _TRANSACTION_LOG_DTYPE.items()}
            )
            self._transaction_log = pd.concat(
                [self._transaction_log, new_row], ignore_index=True
            )
            self._flush_transaction_log()

        logger.info(
            "Transaction logged | event=%s | agent=%s | financial_impact=%.2f | approved=%s",
            event_id,
            agent_id,
            financial_impact,
            sandbox_approved,
        )

    def update_trust_score(self, agent_id: str, score_delta: float) -> dict[str, Any]:
        """Modify an agent's trust score by ``score_delta`` and persist to JSON.

        The resulting score is bounded within ``[MIN_TRUST_SCORE, MAX_TRUST_SCORE]``
        (i.e., ``[0.0, 1.0]``) to prevent runaway growth or negative scores.

        Args:
            agent_id: The canonical identifier of the agent whose score is being
                adjusted (must be one of ``"maker"``, ``"mover"``, ``"keeper"``).
            score_delta: Signed float representing the penalty (negative) or
                reward (positive) to apply.  For example, ``-0.15`` for a bad
                financial decision; ``+0.05`` for an exemplary outcome.

        Returns:
            A dictionary containing the agent's updated trust score entry, e.g.::

                {
                    "trust_score": 0.80,
                    "total_decisions": 143,
                    "recent_penalties": 3,
                    "preferred_fallback": "keeper",
                }

        Raises:
            ValueError: If ``agent_id`` is not a recognised agent identifier.

        Example:
            >>> dm = FactoryDataManager()
            >>> updated = dm.update_trust_score("mover", score_delta=-0.15)
            >>> print(updated["trust_score"])
        """
        if agent_id not in AGENT_IDS:
            raise ValueError(
                f"Unknown agent_id '{agent_id}'. "
                f"Valid agent IDs are: {list(AGENT_IDS)}"
            )

        with self._write_lock:
            agent_entry: dict[str, Any] = self._trust_scores["agents"][agent_id]
            old_score: float = float(agent_entry["trust_score"])
            new_score: float = round(
                max(MIN_TRUST_SCORE, min(MAX_TRUST_SCORE, old_score + score_delta)), 4
            )
            agent_entry["trust_score"] = new_score
            agent_entry["total_decisions"] = int(agent_entry["total_decisions"]) + 1

            if score_delta < 0:
                agent_entry["recent_penalties"] = (
                    int(agent_entry["recent_penalties"]) + 1
                )

            self._trust_scores["global_metrics"]["last_analysis_run"] = (
                datetime.now(tz=timezone.utc).isoformat()
            )
            self._flush_trust_scores()

        logger.info(
            "Trust score updated | agent=%s | delta=%+.4f | old=%.4f | new=%.4f",
            agent_id,
            score_delta,
            old_score,
            new_score,
        )
        return dict(agent_entry)
