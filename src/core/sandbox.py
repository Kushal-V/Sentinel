"""
src/core/sandbox.py
===================
The Shadow Sandbox — Novelty Claim A for Sentinel.

``ShadowSandbox`` implements the "Imagination Room" pattern described in the
architecture blueprint: before any agent is allowed to mutate the Master
Clipboard, it must pass a mathematical gauntlet inside a fully isolated,
in-memory clone of the current state.

Why this matters
----------------
LLMs are stochastic.  An agent might "decide" to move 10,000 units of plastic
resin into a warehouse that only holds 8,000.  Without a deterministic guard,
this hallucinated decision would corrupt the live data.

How it works
------------
1. ``ShadowSandbox.__init__`` deep-copies the live DataFrame and runs it
   through ``DynamicSchemaInferencer`` to extract the active constraint rules.
   No column names are hard-coded.
2. ``evaluate_proposal`` applies the proposed delta to the in-memory clone.
3. Dynamic constraint rules are evaluated:
   * ``new_value >= 0`` (no negative physical quantities — universal rule).
   * ``new_value <= row[limit_column]`` (only if a limit column was inferred
     for the target column).
4. If all assertions pass, the Sandbox returns a SUCCESS payload containing
   the validated shadow DataFrame.  The caller (LangChain tool) then commits
   the change to the live ``FactoryDataManager``.
5. If any assertion fails, the Sandbox returns a rich FAILURE payload with
   the exact violated constraint, specific numeric values, and the column
   names — allowing the LLM to understand *why* it was rejected and generate
   a corrected proposal within ``config.SANDBOX_MAX_RETRIES``.

Thread-safety
-------------
``ShadowSandbox`` is intentionally stateless across calls.  Each call to
``evaluate_proposal`` operates on a freshly-cloned copy of the DataFrame
taken at construction time.  Concurrent Streamlit callbacks cannot interfere
with each other's sandbox sessions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from src.core.schema_engine import DynamicSchemaInferencer, SchemaProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result Types
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    """Encapsulates the outcome of a single ``evaluate_proposal`` call.

    Attributes:
        status: ``"SAFE"`` if the proposal passed all constraints;
            ``"REJECTED"`` otherwise.
        validated_state: A deep copy of the DataFrame *after* applying the
            delta.  Populated only when ``status == "SAFE"``; ``None`` on
            rejection.
        rejection_reason: A detailed human- and LLM-readable explanation of
            exactly which constraint was violated and by how much.  Populated
            only when ``status == "REJECTED"``; ``None`` on success.
        proposed_value: The numeric value the agent requested.
        current_value: The numeric value in the live DataFrame before the delta.
        limit_value: The applicable upper bound, if one was inferred for the
            target column.  ``None`` if no limit column was detected.
    """
    status: Literal["SAFE", "REJECTED"]
    validated_state: pd.DataFrame | None
    rejection_reason: str | None
    proposed_value: float
    current_value: float
    limit_value: float | None


# ---------------------------------------------------------------------------
# ShadowSandbox
# ---------------------------------------------------------------------------

class ShadowSandbox:
    """Deterministic mathematical validator for agent-proposed state changes.

    Operates on an in-memory clone of the live inventory so the live Master
    Clipboard is never touched until a proposal has been fully validated.

    The sandbox is *schema-agnostic*: it accepts whatever DataFrame is handed
    to it and delegates constraint discovery to ``DynamicSchemaInferencer``.

    Attributes:
        _shadow_df: Deep copy of the live DataFrame at instantiation time.
        _profile: Inferred schema rules  (primary key + constraint pairs).
    """

    def __init__(self, live_df: pd.DataFrame) -> None:
        """Initialise the sandbox with a snapshot of the live data.

        Args:
            live_df: The live inventory DataFrame retrieved from
                ``FactoryDataManager.get_inventory()``.  A deep copy is made
                immediately — any subsequent mutations to ``live_df`` will NOT
                affect the sandbox session.

        Raises:
            ValueError: Propagated from ``DynamicSchemaInferencer`` if the
                DataFrame is empty or has no columns.
        """
        self._shadow_df: pd.DataFrame = live_df.copy(deep=True)
        inferencer = DynamicSchemaInferencer(self._shadow_df)
        self._profile: SchemaProfile = inferencer.infer()

        logger.info(
            "ShadowSandbox initialised | pk='%s' | constraint_rules=%d",
            self._profile.primary_key_column,
            len(self._profile.constraint_rules),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def schema_profile(self) -> SchemaProfile:
        """Return the inferred schema profile (read-only).

        Returns:
            The ``SchemaProfile`` produced during initialisation.
        """
        return self._profile

    # ------------------------------------------------------------------
    # Core Validation
    # ------------------------------------------------------------------

    def evaluate_proposal(
        self,
        row_primary_key: str,
        target_column: str,
        delta: float,
    ) -> SandboxResult:
        """Evaluate an agent's proposed state-change inside the sandbox.

        This is the single critical path all agent actions must pass through
        before any live data is mutated.

        Args:
            row_primary_key: The value in the primary-key column that
                identifies the target row.  For the toy factory this would be
                something like ``"ITEM-PLASTIC-01"``; for a hospital it might
                be ``"BED-ICU-04"``.
            target_column: The name of the numeric column the agent wishes to
                modify.  Must be a real column in the DataFrame.
            delta: Signed float representing the proposed change.  Positive
                values increase the quantity; negative values decrease it.

        Returns:
            A ``SandboxResult`` with ``status="SAFE"`` and a validated shadow
            DataFrame on success, or ``status="REJECTED"`` with a descriptive
            ``rejection_reason`` on failure.

        Example:
            >>> sandbox = ShadowSandbox(dm.get_inventory())
            >>> result = sandbox.evaluate_proposal("ITEM-PLASTIC-01", "current_stock", -200)
            >>> if result.status == "SAFE":
            ...     dm.update_inventory("ITEM-PLASTIC-01", -200)
        """
        # ── 1. Validate inputs ─────────────────────────────────────────────
        pk_col = self._profile.primary_key_column

        row_mask = self._shadow_df[pk_col].astype(str) == str(row_primary_key)
        if not row_mask.any():
            valid_keys = self._shadow_df[pk_col].tolist()
            reason = (
                f"SANDBOX REJECTION — Row Not Found: "
                f"No row with {pk_col}='{row_primary_key}' exists in the dataset. "
                f"Valid keys are: {valid_keys}"
            )
            logger.warning(reason)
            return SandboxResult(
                status="REJECTED",
                validated_state=None,
                rejection_reason=reason,
                proposed_value=float("nan"),
                current_value=float("nan"),
                limit_value=None,
            )

        if target_column not in self._shadow_df.columns:
            valid_cols = self._shadow_df.columns.tolist()
            reason = (
                f"SANDBOX REJECTION — Column Not Found: "
                f"Column '{target_column}' does not exist. "
                f"Valid columns are: {valid_cols}"
            )
            logger.warning(reason)
            return SandboxResult(
                status="REJECTED",
                validated_state=None,
                rejection_reason=reason,
                proposed_value=float("nan"),
                current_value=float("nan"),
                limit_value=None,
            )

        if not pd.api.types.is_numeric_dtype(self._shadow_df[target_column]):
            reason = (
                f"SANDBOX REJECTION — Non-Numeric Column: "
                f"Column '{target_column}' has dtype "
                f"'{self._shadow_df[target_column].dtype}'. "
                f"Only numeric columns can be modified by agents."
            )
            logger.warning(reason)
            return SandboxResult(
                status="REJECTED",
                validated_state=None,
                rejection_reason=reason,
                proposed_value=float("nan"),
                current_value=float("nan"),
                limit_value=None,
            )

        # ── 2. Apply delta to shadow clone ─────────────────────────────────
        row_idx = self._shadow_df.index[row_mask][0]
        current_value: float = float(self._shadow_df.at[row_idx, target_column])
        proposed_value: float = current_value + delta
        limit_value: float | None = None

        # ── 3. Universal floor constraint (no negative physical quantities) ─
        if proposed_value < 0:
            reason = (
                f"SANDBOX REJECTION — Non-Negative Constraint Violated: "
                f"Applying delta {delta:+.2f} to '{target_column}' "
                f"(current={current_value:.2f}) would yield {proposed_value:.2f}, "
                f"which is below the minimum allowed value of 0. "
                f"You must reduce the magnitude of your delta."
            )
            logger.warning(reason)
            return SandboxResult(
                status="REJECTED",
                validated_state=None,
                rejection_reason=reason,
                proposed_value=proposed_value,
                current_value=current_value,
                limit_value=0.0,
            )

        # ── 4. Dynamic upper-bound constraint (if a limit column was inferred)
        applicable_rule = next(
            (r for r in self._profile.constraint_rules if r.mutable_column == target_column),
            None,
        )

        if applicable_rule is not None:
            limit_col_name = applicable_rule.limit_column
            if limit_col_name in self._shadow_df.columns:
                limit_value = float(self._shadow_df.at[row_idx, limit_col_name])
                if proposed_value > limit_value:
                    reason = (
                        f"SANDBOX REJECTION — Upper Bound Constraint Violated: "
                        f"Applying delta {delta:+.2f} to '{target_column}' "
                        f"(current={current_value:.2f}) would yield {proposed_value:.2f}, "
                        f"which exceeds the inferred limit column '{limit_col_name}' "
                        f"value of {limit_value:.2f} for row '{row_primary_key}'. "
                        f"Maximum allowable increase is "
                        f"{(limit_value - current_value):.2f}."
                    )
                    logger.warning(reason)
                    return SandboxResult(
                        status="REJECTED",
                        validated_state=None,
                        rejection_reason=reason,
                        proposed_value=proposed_value,
                        current_value=current_value,
                        limit_value=limit_value,
                    )

        # ── 5. All constraints passed — commit to shadow clone ─────────────
        self._shadow_df.at[row_idx, target_column] = proposed_value
        validated_snapshot = self._shadow_df.copy(deep=True)

        logger.info(
            "Sandbox SAFE | pk=%s | col=%s | delta=%+.2f | "
            "old=%.2f | new=%.2f | limit=%s",
            row_primary_key,
            target_column,
            delta,
            current_value,
            proposed_value,
            f"{limit_value:.2f}" if limit_value is not None else "none",
        )

        return SandboxResult(
            status="SAFE",
            validated_state=validated_snapshot,
            rejection_reason=None,
            proposed_value=proposed_value,
            current_value=current_value,
            limit_value=limit_value,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_row_as_dict(self, row_primary_key: str) -> dict[str, Any]:
        """Return a specific row from the shadow DataFrame as a plain dict.

        Useful for agents wanting to inspect a row's current (shadow) state
        after a previous ``evaluate_proposal`` call within the same session.

        Args:
            row_primary_key: The primary key value of the row to fetch.

        Returns:
            A dictionary representation of the row's current shadow state.

        Raises:
            ValueError: If the ``row_primary_key`` does not exist.
        """
        pk_col = self._profile.primary_key_column
        mask = self._shadow_df[pk_col].astype(str) == str(row_primary_key)
        if not mask.any():
            raise ValueError(
                f"Row '{row_primary_key}' not found in shadow DataFrame "
                f"(primary key column: '{pk_col}')."
            )
        return self._shadow_df.loc[mask].iloc[0].to_dict()
