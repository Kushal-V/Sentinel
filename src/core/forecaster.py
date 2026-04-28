"""Demand forecasting over historical inventory snapshots.

Builds a per-row forecast using two simple methods:
  - Exponential smoothing (no extra deps, defensive) for short series.
  - sklearn LinearRegression on time index for medium series.

Reads transaction history (the existing append-only ledger) to derive
per-row demand series, then projects N steps ahead. Identifies rows at
risk of stock-out by comparing forecast horizon vs current_stock /
average_consumption.

Multi-tenant: works on any DataFrame with a primary key column and at
least one mutable numeric column (auto-detected via SchemaProfile).
NO hardcoded SKU / column names.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

_LOG = logging.getLogger(__name__)

_MIN_SERIES_LEN_FOR_REGRESSION = 5
_DEFAULT_HORIZON = 7  # forecast 7 steps ahead


# ---------------------------------------------------------------------------
# Schema attribute resolution helpers
# ---------------------------------------------------------------------------
# Sentinel's real ``SchemaProfile`` exposes
#   - ``primary_key_column``        (str)
#   - ``constraint_rules``          (list[ConstraintRule])
# but the F9 spec's tests use a lightweight stand-in that exposes
#   - ``primary_key``               (str)
#   - ``constraint_pairs``          (dict[mutable -> limit])
# Resolve both shapes through these getters so the forecaster works
# unchanged with the production schema profile and the test fixture.

def _resolve_primary_key(schema: Any) -> Optional[str]:
    """Return the primary key column name from either schema flavour."""
    if schema is None:
        return None
    return (
        getattr(schema, "primary_key", None)
        or getattr(schema, "primary_key_column", None)
    )


def _resolve_constraint_pairs(schema: Any) -> Dict[str, str]:
    """Return a dict[mutable_col -> limit_col] from either schema flavour."""
    if schema is None:
        return {}
    pairs = getattr(schema, "constraint_pairs", None)
    if isinstance(pairs, dict):
        return dict(pairs)
    rules = getattr(schema, "constraint_rules", None) or []
    out: Dict[str, str] = {}
    for r in rules:
        m = getattr(r, "mutable_column", None)
        l = getattr(r, "limit_column", None)
        if m and l:
            out[m] = l
    return out


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    row_key: str
    column: str
    horizon: int
    method: str  # "regression" | "exp_smoothing" | "naive"
    history_length: int
    last_value: float
    projected_values: List[float]  # one per horizon step
    avg_consumption: float  # rate change per step from history
    runs_out_in: Optional[int]  # None if not at risk
    confidence: float  # 0-1, lower with shorter history
    notes: str = ""


@dataclass
class FleetForecast:
    horizon: int
    per_row: Dict[str, ForecastResult] = field(default_factory=dict)
    at_risk_rows: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Forecast horizon: {self.horizon} steps"]
        if self.at_risk_rows:
            lines.append(f"At-risk rows ({len(self.at_risk_rows)}):")
            for k in self.at_risk_rows[:10]:
                fr = self.per_row[k]
                lines.append(
                    f"  - {k}: last={fr.last_value:.1f}, "
                    f"runs out in {fr.runs_out_in} step(s), "
                    f"method={fr.method}, conf={fr.confidence:.2f}"
                )
        else:
            lines.append("No rows currently at risk over the forecast horizon.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DemandForecaster
# ---------------------------------------------------------------------------

class DemandForecaster:
    """Compute fleet-wide forecasts from a transaction ledger.

    Usage:
        f = DemandForecaster(inventory_df, transaction_log_df, schema_profile)
        forecast = f.forecast_fleet(horizon=7, target_columns=["current_stock"])
        print(forecast.summary())
    """

    def __init__(
        self,
        inventory: pd.DataFrame,
        transaction_log: Optional[pd.DataFrame],
        schema: Any,  # SchemaProfile (real or test stand-in)
    ):
        self._inventory = (
            inventory.copy(deep=True) if inventory is not None else pd.DataFrame()
        )
        self._log = (
            transaction_log.copy(deep=True)
            if transaction_log is not None
            else pd.DataFrame()
        )
        self._schema = schema

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forecast_fleet(
        self,
        horizon: int = _DEFAULT_HORIZON,
        target_columns: Optional[List[str]] = None,
    ) -> FleetForecast:
        """Forecast each row's mutable column over the horizon."""
        if self._inventory is None or self._inventory.empty:
            return FleetForecast(horizon=horizon)

        pk = _resolve_primary_key(self._schema)
        if not pk or pk not in self._inventory.columns:
            _LOG.warning("Forecaster: no primary key, returning empty forecast")
            return FleetForecast(horizon=horizon)

        # Auto-detect target columns from schema if not provided
        if target_columns is None:
            constraint_pairs = _resolve_constraint_pairs(self._schema)
            target_columns = list(constraint_pairs.keys()) if constraint_pairs else []
        if not target_columns:
            _LOG.warning(
                "Forecaster: no constraint pairs found; using first numeric column"
            )
            numeric = self._inventory.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            target_columns = numeric[:1] if numeric else []

        result = FleetForecast(horizon=horizon)
        for _, row in self._inventory.iterrows():
            row_key = str(row[pk])
            for col in target_columns:
                if col not in self._inventory.columns:
                    continue
                fr = self._forecast_single(row_key, col, horizon)
                if fr is not None:
                    result.per_row[f"{row_key}.{col}"] = fr
                    if fr.runs_out_in is not None and fr.runs_out_in <= horizon:
                        result.at_risk_rows.append(f"{row_key}.{col}")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_value(self, row_key: str, column: str) -> Optional[float]:
        """Return the current value for row_key.column, or None if missing."""
        pk = _resolve_primary_key(self._schema)
        if not pk or self._inventory is None or self._inventory.empty:
            return None
        mask = self._inventory[pk].astype(str) == row_key
        if not mask.any():
            return None
        try:
            return float(self._inventory.loc[mask].iloc[0][column])
        except (KeyError, ValueError, TypeError):
            return None

    def _forecast_single(
        self,
        row_key: str,
        column: str,
        horizon: int,
    ) -> Optional[ForecastResult]:
        """Forecast a single row.column for ``horizon`` steps ahead."""
        series = self._build_series(row_key, column)

        if len(series) < 2:
            current = self._current_value(row_key, column)
            if current is None:
                current = float(series[-1]) if series else 0.0
            return ForecastResult(
                row_key=row_key,
                column=column,
                horizon=horizon,
                method="naive",
                history_length=len(series),
                last_value=float(current),
                projected_values=[float(current)] * horizon,
                avg_consumption=0.0,
                runs_out_in=None,
                confidence=0.2,
                notes="insufficient history",
            )

        last_value = float(series[-1])
        x = np.arange(len(series)).reshape(-1, 1).astype(float)
        y = np.array(series, dtype=float)

        if len(series) >= _MIN_SERIES_LEN_FOR_REGRESSION:
            model = LinearRegression().fit(x, y)
            future_x = np.arange(
                len(series), len(series) + horizon
            ).reshape(-1, 1).astype(float)
            projected = model.predict(future_x).tolist()
            slope = float(model.coef_[0])
            r2 = float(model.score(x, y))
            method = "regression"
            confidence = max(0.3, min(0.95, 0.5 + 0.5 * r2))
        else:
            # Exponential smoothing
            alpha = 0.5
            smoothed = float(y[0])
            for v in y[1:]:
                smoothed = alpha * float(v) + (1 - alpha) * smoothed
            slope = float((y[-1] - y[0]) / max(1, len(y) - 1))
            projected = [
                float(smoothed + slope * (i + 1)) for i in range(horizon)
            ]
            method = "exp_smoothing"
            confidence = 0.4

        # Detect when projected hits zero or below
        runs_out_in: Optional[int] = None
        for i, v in enumerate(projected):
            if v <= 0:
                runs_out_in = i + 1
                break

        # avg_consumption is positive when stock is decreasing
        avg_consumption = max(0.0, -slope)

        return ForecastResult(
            row_key=row_key,
            column=column,
            horizon=horizon,
            method=method,
            history_length=len(series),
            last_value=last_value,
            projected_values=[float(v) for v in projected],
            avg_consumption=avg_consumption,
            runs_out_in=runs_out_in,
            confidence=confidence,
        )

    def _build_series(self, row_key: str, column: str) -> List[float]:
        """Reconstruct the value series for a row+column from transaction log.

        Heuristic: filter log entries that mention this row_key and target
        column, accumulate deltas chronologically. If log is empty or the
        row is never seen, fall back to the current inventory value as a
        single-point series.
        """
        pk = _resolve_primary_key(self._schema)

        if self._log is None or self._log.empty:
            current = self._current_value(row_key, column) if pk else None
            return [float(current)] if current is not None else []

        # Best-effort: log columns vary across tenants/sources. Search for
        # row_key mention in any plausible identifier column.
        id_col_names = {"row_key", "sku", "item", "id"}
        if pk:
            id_col_names.add(pk.lower())
        id_cols = [c for c in self._log.columns if c.lower() in id_col_names]
        target_col_filter = [
            c for c in self._log.columns
            if c.lower() in ("target_column", "column", "field")
        ]
        delta_cols = [
            c for c in self._log.columns
            if c.lower() in ("delta", "change", "amount")
        ]
        new_value_cols = [
            c for c in self._log.columns
            if c.lower() in ("new_value", "after_value", "value_after")
        ]

        # Sort by timestamp if a time-like column is present.
        ts_col = next(
            (
                c for c in self._log.columns
                if "time" in c.lower() or "date" in c.lower()
            ),
            None,
        )
        log = self._log.sort_values(ts_col) if ts_col else self._log

        # Reconstruct via after_value if present; else current ± deltas.
        running: Optional[float] = self._current_value(row_key, column)

        series: List[float] = []
        for _, entry in log.iterrows():
            # match on row_key in any id col
            if id_cols and not any(
                str(entry.get(c, "")) == row_key for c in id_cols
            ):
                continue
            # match on target column when the log carries a column field
            if target_col_filter and not any(
                str(entry.get(c, "")) == column for c in target_col_filter
            ):
                continue
            if new_value_cols:
                v = entry.get(new_value_cols[0])
                if pd.notna(v):
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    series.append(fv)
                    running = fv
                    continue
            if delta_cols and running is not None:
                d = entry.get(delta_cols[0])
                if pd.notna(d):
                    try:
                        running = float(running) + float(d)
                    except (TypeError, ValueError):
                        continue
                    series.append(running)

        # If we couldn't reconstruct, fall back to current value as a
        # single-point series so the forecaster never returns empty for a
        # row that exists in inventory.
        if not series and running is not None:
            series = [running]
        return series
