"""
src/core/simulator.py
=====================
Monte Carlo simulator over pending state changes.

Runs ``N`` stochastic futures by perturbing each pending change's delta with
Gaussian noise, validating each run against the live constraint model via
``ShadowSandbox``, and aggregating distributions per mutable column.

Why this exists
---------------
The HITL approval panel previously surfaced staged changes deterministically:
the human approver saw the post-state numbers as if they were certain.  In
reality, supply-chain interventions are noisy — supplier deliveries vary,
shipping times wobble, demand spikes.  Industry-standard 2026 supply-chain
digital twins quantify this uncertainty by simulating thousands of stochastic
futures before commit.

This module implements that capability without coupling to any specific
schema.  It reuses the existing ``ShadowSandbox`` for per-run validation and
discovers tracked columns dynamically from ``SchemaProfile.constraint_rules``,
so the simulator works equally well on toy-factory inventory, hospital beds,
shipping containers, server-farm capacity, or retail SKUs.

Algorithm
---------
For each of ``n_runs`` independent simulations:
    1. Spawn a fresh ``ShadowSandbox`` over a deep-copy of the live DataFrame.
    2. For each pending change, perturb its delta by a Gaussian factor
       ``(1 + N(0, noise_std_pct))`` and submit it to the sandbox.
    3. If the sandbox rejects the noisy delta, mark the run as having a
       constraint violation and continue with the un-mutated state.
    4. After all changes are attempted, record the value of every mutable
       column (per ``schema.constraint_rules``) at the simulated end-state.

After all runs complete, aggregate per-column distributions into
``p10 / p50 / p90 / mean / std`` and an overall rejection rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.sandbox import ShadowSandbox
from src.core.schema_engine import SchemaProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Aggregated outcome of a Monte Carlo simulation batch.

    Attributes:
        n_runs: The requested number of stochastic runs.
        rejection_rate: Fraction of runs that experienced at least one
            sandbox rejection across their pending changes.  In ``[0.0, 1.0]``.
        metrics: Per-mutable-column distribution stats.  Outer key is the
            column name (e.g. ``"current_stock"``); inner dict has keys
            ``p10``, ``p50``, ``p90``, ``mean``, ``std``.
        constraint_violations: A small sample of human-readable violation
            messages collected during simulation (capped to avoid memory
            blow-up).  Useful for surfacing in the UI as a warning.
        runs_completed: Number of simulations actually executed (``n_runs``
            in the normal case; ``0`` if there were no pending changes).
    """
    n_runs: int
    rejection_rate: float
    metrics: Dict[str, Dict[str, float]]
    constraint_violations: List[str] = field(default_factory=list)
    runs_completed: int = 0


# ---------------------------------------------------------------------------
# MonteCarloSimulator
# ---------------------------------------------------------------------------

#: Maximum number of rejection-reason strings to retain in the result.  Keeps
#: memory bounded for long simulations.
_MAX_VIOLATION_SAMPLES: int = 10


class MonteCarloSimulator:
    """Run stochastic futures of staged state changes against the sandbox.

    The simulator is **schema-agnostic**: it discovers which columns to
    track via ``schema.constraint_pairs`` (from the inferred profile) and
    never references hard-coded column names.

    Attributes:
        _live_df: Deep-copied snapshot of the live inventory at construction.
        _schema: The inferred ``SchemaProfile`` for the dataset.
        _pending_changes: List of staged change dicts (same shape as
            ``FactoryDataManager.pending_changes`` entries).
        _n_runs: Requested number of simulations.
        _noise_std_pct: Standard deviation of the multiplicative Gaussian
            noise applied to each delta (e.g. ``0.10`` => 10% noise).
        _rng: ``numpy.random.Generator`` driving all randomness; deterministic
            when ``random_seed`` is supplied.
        _tracked_columns: List of mutable column names sourced from the
            schema profile's constraint rules.
    """

    def __init__(
        self,
        live_df: pd.DataFrame,
        schema: SchemaProfile,
        pending_changes: List[Dict[str, Any]],
        n_runs: int = 1000,
        noise_std_pct: float = 0.10,
        random_seed: Optional[int] = None,
    ) -> None:
        """Construct a simulator with a frozen snapshot of the live data.

        Args:
            live_df: The current live inventory DataFrame.  A deep copy is
                taken; subsequent mutations to the caller's frame will not
                leak into the simulation.
            schema: The inferred schema profile for ``live_df``.  Used to
                discover the set of mutable columns to aggregate over.
            pending_changes: Staged change dicts (same shape as
                ``FactoryDataManager.pending_changes`` entries).  Each must
                expose ``row_key``, ``target_column``, and ``delta`` keys.
            n_runs: Number of stochastic simulations to execute.  Must be
                non-negative.
            noise_std_pct: Standard deviation of the multiplicative Gaussian
                noise applied to each delta (fraction, e.g. ``0.10`` for
                10% noise).  Must be non-negative.
            random_seed: Optional seed for deterministic reproducibility
                (set this in tests).  When ``None``, the default RNG is used
                with non-deterministic entropy.

        Raises:
            ValueError: If ``n_runs`` is negative or ``noise_std_pct`` is
                negative.
        """
        if n_runs < 0:
            raise ValueError(f"n_runs must be non-negative, got {n_runs}")
        if noise_std_pct < 0:
            raise ValueError(
                f"noise_std_pct must be non-negative, got {noise_std_pct}"
            )

        # Deep copy so we never mutate the caller's frame.
        self._live_df: pd.DataFrame = live_df.copy(deep=True)
        self._schema: SchemaProfile = schema
        # Shallow-copy the list of dicts; we never mutate change entries themselves.
        self._pending_changes: List[Dict[str, Any]] = list(pending_changes)
        self._n_runs: int = int(n_runs)
        self._noise_std_pct: float = float(noise_std_pct)
        self._rng: np.random.Generator = np.random.default_rng(random_seed)

        # Track every mutable column the schema knows about.  This is the
        # universal multi-tenant hook: never hardcode column names.
        self._tracked_columns: List[str] = [
            rule.mutable_column for rule in self._schema.constraint_rules
        ]

        logger.info(
            "MonteCarloSimulator initialised | n_runs=%d | noise_std=%.3f | "
            "pending=%d | tracked_cols=%d",
            self._n_runs,
            self._noise_std_pct,
            len(self._pending_changes),
            len(self._tracked_columns),
        )

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def run(self) -> SimulationResult:
        """Execute ``n_runs`` simulations and aggregate the results.

        Returns:
            A ``SimulationResult`` with per-column distribution metrics,
            overall rejection rate, and a sample of violation messages.

            If there are no pending changes, returns an empty result
            (``n_runs=0``, ``rejection_rate=0.0``, ``metrics={}``,
            ``runs_completed=0``).
        """
        # Empty-staging short-circuit.  Honours the "n_runs=0" contract from
        # the spec — there is nothing to perturb if nothing is staged.
        if not self._pending_changes:
            return SimulationResult(
                n_runs=0,
                rejection_rate=0.0,
                metrics={},
                constraint_violations=[],
                runs_completed=0,
            )

        # Per-column samples accumulated across all runs.  We use plain lists
        # (rather than pre-allocated NumPy arrays) because some runs may not
        # touch certain rows, but aggregation stats can still operate on the
        # cross-run sample of column means without issue.
        # We aggregate the *full mutable column* (sum across rows) per run,
        # which gives a stable per-column scalar distribution regardless of
        # how many rows each pending change touches.
        column_samples: Dict[str, List[float]] = {
            col: [] for col in self._tracked_columns
        }

        rejection_count: int = 0
        violation_samples: List[str] = []

        for _ in range(self._n_runs):
            sandbox = ShadowSandbox(self._live_df)
            run_state: pd.DataFrame = self._live_df.copy(deep=True)
            run_had_rejection: bool = False

            for change in self._pending_changes:
                delta = float(change["delta"])
                # Multiplicative Gaussian noise.  delta_noisy = delta * (1 + eps)
                noise = float(self._rng.normal(0.0, self._noise_std_pct))
                delta_noisy = delta * (1.0 + noise)

                result = sandbox.evaluate_proposal(
                    row_primary_key=change["row_key"],
                    target_column=change["target_column"],
                    delta=delta_noisy,
                )

                if result.status == "REJECTED":
                    run_had_rejection = True
                    if len(violation_samples) < _MAX_VIOLATION_SAMPLES:
                        violation_samples.append(
                            result.rejection_reason or "Sandbox rejection"
                        )
                    # Skip applying this change to the local run_state — the
                    # sandbox's internal _shadow_df is also unchanged because
                    # evaluate_proposal only mutates a per-call copy.
                    continue

                # Mirror the sandbox-validated state back into our run_state
                # so subsequent column aggregation sees the post-change values.
                if result.validated_state is not None:
                    run_state = result.validated_state

            if run_had_rejection:
                rejection_count += 1

            # Snapshot tracked columns at end-of-run.  Sum over rows so we get
            # a single scalar per column per run — the natural unit for a
            # distribution histogram in the HITL panel.
            for col in self._tracked_columns:
                if col in run_state.columns:
                    try:
                        col_total = float(
                            pd.to_numeric(run_state[col], errors="coerce")
                            .fillna(0.0)
                            .sum()
                        )
                    except (TypeError, ValueError):
                        col_total = float("nan")
                    column_samples[col].append(col_total)

        # Aggregate distributions.
        metrics: Dict[str, Dict[str, float]] = {}
        for col, samples in column_samples.items():
            if not samples:
                continue
            arr = np.asarray(samples, dtype=float)
            # Drop NaNs defensively before computing percentiles.
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                continue
            metrics[col] = {
                "p10": float(np.percentile(arr, 10)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=0)),
            }

        rejection_rate = (
            rejection_count / self._n_runs if self._n_runs > 0 else 0.0
        )

        logger.info(
            "MonteCarlo complete | runs=%d | rejection_rate=%.3f | "
            "metric_cols=%d | violation_samples=%d",
            self._n_runs,
            rejection_rate,
            len(metrics),
            len(violation_samples),
        )

        return SimulationResult(
            n_runs=self._n_runs,
            rejection_rate=rejection_rate,
            metrics=metrics,
            constraint_violations=violation_samples,
            runs_completed=self._n_runs,
        )
