"""Lightweight causal inference for supply-chain what-if analysis.

NOT a full causal-inference framework (DoWhy / EconML / CausalNex). This
module answers ONE question:

    "If column X on row R changed by amount D, what would happen to
     downstream column Y on the same or related row?"

Method: train a simple sklearn LinearRegression of Y on (X plus other
candidate features), produce two predictions:
  - actual: at the current X value
  - counterfactual: at X + D

The difference is the estimated causal effect of the proposed change.
Interpretation is correlational unless the data has a true experimental
structure -- but it is enough to surface "supplier delay -> downstream
inventory drop" type insights for an LLM agent to reason about.

Multi-tenant: works on any inventory DataFrame. Treat-column,
outcome-column, and intervention magnitude come from the caller.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

_LOG = logging.getLogger(__name__)

_MIN_SAMPLES = 5  # absolute minimum rows for regression-based estimate


@dataclass
class CausalEffect:
    """Estimated causal effect of an intervention on an outcome column."""

    treatment_column: str
    outcome_column: str
    intervention_delta: float
    actual_outcome: float
    counterfactual_outcome: float
    effect: float  # counterfactual - actual
    method: str  # "regression" | "row_lookup" | "insufficient_data"
    n_samples: int
    r2: float
    confidence: float  # 0-1
    notes: str = ""

    def summary(self) -> str:
        """Render a compact human/LLM-readable one-liner of the effect."""
        return (
            f"Causal effect of {self.treatment_column} += {self.intervention_delta} "
            f"on {self.outcome_column}: actual={self.actual_outcome:.2f}, "
            f"counterfactual={self.counterfactual_outcome:.2f}, "
            f"effect={self.effect:+.2f} (method={self.method}, "
            f"n={self.n_samples}, R²={self.r2:.2f}, conf={self.confidence:.2f}). "
            f"{self.notes}"
        )


class CausalAnalyzer:
    """Estimate counterfactual outcomes via linear regression.

    Multi-tenant by construction: the caller supplies treatment / outcome
    column names and an optional explicit feature list, so the analyzer
    never assumes any particular schema.
    """

    def __init__(self, inventory: pd.DataFrame, schema=None):
        self._inventory = (
            inventory.copy(deep=True) if inventory is not None else pd.DataFrame()
        )
        self._schema = schema

    def estimate_effect(
        self,
        treatment_column: str,
        outcome_column: str,
        intervention_delta: float,
        candidate_features: Optional[List[str]] = None,
    ) -> CausalEffect:
        """Estimate the causal effect of perturbing ``treatment_column`` by
        ``intervention_delta`` on ``outcome_column``.

        Defensive contract:
          * Empty inventory -> ``insufficient_data``.
          * Missing treatment / outcome column -> ``insufficient_data``.
          * Treatment == outcome -> trivial ``row_lookup`` (effect == delta).
          * < ``_MIN_SAMPLES`` valid rows -> ``row_lookup`` correlation
            fallback.
          * Otherwise -> sklearn ``LinearRegression`` on the candidate
            feature matrix evaluated at sample means.
        """
        df = self._inventory
        if df.empty:
            return self._insufficient(
                treatment_column,
                outcome_column,
                intervention_delta,
                reason="empty inventory",
            )
        if treatment_column not in df.columns or outcome_column not in df.columns:
            return self._insufficient(
                treatment_column,
                outcome_column,
                intervention_delta,
                reason=(
                    f"column missing: treatment={treatment_column in df.columns}, "
                    f"outcome={outcome_column in df.columns}"
                ),
            )
        if treatment_column == outcome_column:
            # trivial: effect is just delta
            try:
                actual = float(df[outcome_column].astype(float).mean())
            except Exception:
                actual = 0.0
            return CausalEffect(
                treatment_column=treatment_column,
                outcome_column=outcome_column,
                intervention_delta=intervention_delta,
                actual_outcome=actual,
                counterfactual_outcome=actual + intervention_delta,
                effect=intervention_delta,
                method="row_lookup",
                n_samples=len(df),
                r2=1.0,
                confidence=1.0,
                notes=(
                    "treatment and outcome are the same column; "
                    "effect equals intervention"
                ),
            )

        # Build feature matrix --------------------------------------------
        if candidate_features is None:
            candidate_features = [
                c
                for c in df.select_dtypes(include=[np.number]).columns
                if c != outcome_column
            ]
        else:
            candidate_features = [c for c in candidate_features if c in df.columns]
        if treatment_column not in candidate_features:
            candidate_features = [treatment_column] + candidate_features

        X_df = df[candidate_features].copy()
        # Coerce candidate features to numeric — non-numeric becomes NaN and
        # is dropped by the validity mask below. This keeps the regressor
        # safe when callers pass mixed-type columns by mistake.
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
        try:
            y = pd.to_numeric(df[outcome_column], errors="coerce").astype(float)
        except Exception:
            return self._insufficient(
                treatment_column,
                outcome_column,
                intervention_delta,
                reason=f"outcome column '{outcome_column}' is not numeric-coercible",
            )

        # Drop rows with NaN
        valid = ~(X_df.isna().any(axis=1) | y.isna())
        X_df = X_df[valid]
        y = y[valid]

        if len(X_df) < _MIN_SAMPLES:
            # Fallback: treat effect as proportional to historical correlation
            return self._fallback_correlation(
                df,
                treatment_column,
                outcome_column,
                intervention_delta,
                len(X_df),
            )

        # Fit regression --------------------------------------------------
        try:
            model = LinearRegression().fit(X_df.values, y.values)
        except Exception as e:  # noqa: BLE001
            _LOG.exception("Regression fit failed: %s", e)
            return self._fallback_correlation(
                df,
                treatment_column,
                outcome_column,
                intervention_delta,
                len(X_df),
            )

        # Predict at sample mean (actual) and mean+delta on treatment (counterfactual)
        means = X_df.mean()
        actual_x = means.values.reshape(1, -1)
        cf_x_series = means.copy()
        cf_x_series[treatment_column] = means[treatment_column] + intervention_delta
        cf_x = cf_x_series.values.reshape(1, -1)

        actual_pred = float(model.predict(actual_x)[0])
        cf_pred = float(model.predict(cf_x)[0])
        effect = cf_pred - actual_pred

        # R^2 on training (interpretation: explanatory power, not predictive)
        try:
            r2 = float(r2_score(y, model.predict(X_df.values)))
        except Exception:
            r2 = 0.0
        # Confidence: combine sample size + R²
        conf = max(
            0.1,
            min(
                0.95,
                0.4
                + 0.4 * max(0.0, r2)
                + 0.2 * min(1.0, len(X_df) / 50.0),
            ),
        )

        return CausalEffect(
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            intervention_delta=intervention_delta,
            actual_outcome=actual_pred,
            counterfactual_outcome=cf_pred,
            effect=effect,
            method="regression",
            n_samples=int(len(X_df)),
            r2=r2,
            confidence=conf,
            notes=(
                "Effect derived from linear regression at sample means. "
                "Causal interpretation requires no unobserved confounders."
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fallback_correlation(
        self,
        df: pd.DataFrame,
        t: str,
        o: str,
        d: float,
        n: int,
    ) -> CausalEffect:
        """Crude correlation-scaled estimate when regression isn't viable."""
        if t in df.columns and o in df.columns and len(df) > 1:
            try:
                pair = df[[t, o]].apply(pd.to_numeric, errors="coerce")
                if pair.isna().all().all():
                    corr = 0.0
                else:
                    corr_val = pair.corr().iloc[0, 1]
                    corr = 0.0 if pd.isna(corr_val) else float(corr_val)
            except Exception:
                corr = 0.0
        else:
            corr = 0.0

        try:
            actual = (
                float(pd.to_numeric(df[o], errors="coerce").mean())
                if o in df.columns and not df.empty
                else 0.0
            )
            if pd.isna(actual):
                actual = 0.0
        except Exception:
            actual = 0.0

        # Crude effect estimate: corr * (delta * std_o / std_t)
        try:
            std_o = float(pd.to_numeric(df[o], errors="coerce").std() or 0.0)
            std_t_raw = float(pd.to_numeric(df[t], errors="coerce").std() or 0.0)
            std_t = std_t_raw if std_t_raw else 1.0
            est = corr * (d * std_o / std_t)
            if pd.isna(est):
                est = 0.0
        except Exception:
            est = 0.0

        return CausalEffect(
            treatment_column=t,
            outcome_column=o,
            intervention_delta=d,
            actual_outcome=actual,
            counterfactual_outcome=actual + est,
            effect=est,
            method="row_lookup",
            n_samples=n,
            r2=0.0,
            confidence=0.2,
            notes="Insufficient samples for regression; correlation-based estimate.",
        )

    def _insufficient(
        self,
        t: str,
        o: str,
        d: float,
        reason: str,
    ) -> CausalEffect:
        """Construct an ``insufficient_data`` result with zeroed metrics."""
        return CausalEffect(
            treatment_column=t,
            outcome_column=o,
            intervention_delta=d,
            actual_outcome=0.0,
            counterfactual_outcome=0.0,
            effect=0.0,
            method="insufficient_data",
            n_samples=0,
            r2=0.0,
            confidence=0.0,
            notes=reason,
        )
