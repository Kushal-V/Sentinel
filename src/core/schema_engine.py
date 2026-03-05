"""
src/core/schema_engine.py
=========================
Dynamic Schema Inference Engine for the Sentinel Digital Twin.

``DynamicSchemaInferencer`` inspects any arbitrary ``pandas`` DataFrame and
extracts the structural rules needed to validate proposed state changes
without any hard-coded column names.  This makes Sentinel a multi-tenant
platform: the same validation engine works for toy factory stock, hospital
beds, retail SKUs, shipping containers, or any domain-specific inventory.

Algorithm Overview
------------------
1. **Primary Key Detection** — Heuristically identify the column that uniquely
   identifies each row (handles names like ``item_id``, ``sku``, ``bed_id``,
   ``container_ref``, etc.).
2. **Constraint Pair Mapping** — For every numeric column, apply a cascade of
   regex patterns to discover its corresponding upper-limit column (e.g.,
   ``beds_occupied`` → ``max_beds``, ``stock`` → ``stock_capacity``).
3. **Type Filtering** — Only numeric columns can participate in mathematical
   constraints.  String/datetime columns are catalogued but excluded from
   limit checks.

The returned ``SchemaProfile`` dataclass is the authoritative contract
consumed by ``ShadowSandbox`` during proposal evaluation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — regex patterns for constraint pair detection
# ---------------------------------------------------------------------------

# Patterns for detecting a "maximum / limit" column given a *stem* string.
# Applied in priority order — first match wins for each candidate column.
_LIMIT_SUFFIX_PATTERNS: Final[list[str]] = [
    r"^max[_\s]?{stem}$",            # max_beds, max_stock
    r"^{stem}[_\s]?max$",            # beds_max, stock_max
    r"^max[_\s]?{stem}[_\s]?.*$",   # max_bed_capacity, max_compute_allowance
    r"^{stem}[_\s]?(limit|cap|capacity|ceiling|total|size|maximum|upper|allowance)$",
    r"^(limit|cap|capacity|ceiling|total|size|maximum|upper|allowance)[_\s]?{stem}$",
    r"^max[_\s]?(limit|cap|capacity|ceiling|total|size|maximum|upper|allowance)[_\s]?{stem}$",
]

# Regex that identifies a column as likely being a *limit-side* column.
# Used in cross-stem pair detection.
_LIMIT_COLUMN_PATTERN: Final[re.Pattern] = re.compile(
    r"(^max[_\s]|[_\s]max$|[_\s]?limit$|[_\s]?capacity$|[_\s]?cap$"
    r"|[_\s]?ceiling$|[_\s]?allowance$|[_\s]?maximum$|[_\s]?upper$"
    r"|[_\s]?room$|^teu[_\s])",
    re.IGNORECASE,
)

# Minimum fraction of rows where mutable_value <= limit_value to
# consider a cross-stem pairing valid (data-driven validation).
_CROSS_STEM_MIN_RATIO: Final[float] = 0.75

# Patterns for detecting a "current / in-use" column — helps us decide which
# of two paired columns is the "mutable" one vs the "limit" one.
_CURRENT_PREFIX_PATTERNS: Final[list[str]] = [
    r"^(current|curr)[_\s]?(.+)$",
    r"^(stock|inventory|level|used|occupied|in_use|on_hand|qty|quantity)[_\s]?(.+)$",
    r"^(.+)[_\s]?(stock|inventory|level|used|occupied|in_use|on_hand|qty|quantity)$",
]

# Heuristics for primary key column detection — evaluated top-to-bottom.
_PK_KEYWORD_PATTERNS: Final[list[str]] = [
    r"\bid\b",         # item_id, id, bed_id
    r"\bkey\b",        # item_key, row_key
    r"\bsku\b",        # sku, product_sku
    r"\bref\b",        # container_ref
    r"\bcode\b",       # item_code
    r"\bname\b",       # Only if no pure ID column exists
    r"\bslug\b",
]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstraintRule:
    """A single validated constraint between a mutable and a limit column.

    Attributes:
        mutable_column: The column whose value changes with agent actions
            (e.g., ``"beds_occupied"``).
        limit_column: The column holding the upper bound for ``mutable_column``
            (e.g., ``"max_beds"``).
        min_value: The universal lower bound (always 0 — no negative physical
            quantities).
    """
    mutable_column: str
    limit_column: str
    min_value: float = 0.0


@dataclass
class SchemaProfile:
    """Fully inferred structural description of an arbitrary DataFrame.

    Produced by ``DynamicSchemaInferencer.infer()`` and consumed by
    ``ShadowSandbox``.

    Attributes:
        primary_key_column: Name of the column treated as the row identifier.
        numeric_columns: All columns with numeric dtypes.
        constraint_rules: List of ``ConstraintRule`` pairs discovered.
        column_dtypes: Mapping of *column name* → pandas dtype string.
        unconstrained_numeric_columns: Numeric columns with no detected
            upper-bound partner (still subject to the ``>= 0`` floor).
    """
    primary_key_column: str
    numeric_columns: list[str] = field(default_factory=list)
    constraint_rules: list[ConstraintRule] = field(default_factory=list)
    column_dtypes: dict[str, str] = field(default_factory=dict)
    unconstrained_numeric_columns: list[str] = field(default_factory=list)

    def as_agent_summary(self) -> str:
        """Return a compact, LLM-readable summary of the schema.

        This is the string surfaced to the agent by ``get_dataset_schema``
        so the LLM knows exactly which column names and constraints exist
        before proposing any state change.

        Returns:
            A multi-line plain-text summary of the schema profile.
        """
        lines: list[str] = [
            f"PRIMARY KEY COLUMN : {self.primary_key_column}",
            "",
            "ALL COLUMNS (name → dtype):",
        ]
        for col, dtype in self.column_dtypes.items():
            lines.append(f"  • {col} ({dtype})")

        lines += ["", "DETECTED CONSTRAINT RULES:"]
        if self.constraint_rules:
            for rule in self.constraint_rules:
                lines.append(
                    f"  • {rule.mutable_column} must be in "
                    f"[{rule.min_value}, row['{rule.limit_column}']"
                )
        else:
            lines.append("  • No upper-bound pairs detected.")

        if self.unconstrained_numeric_columns:
            lines += ["", "UNCONSTRAINED NUMERIC COLUMNS (floor = 0 only):"]
            for col in self.unconstrained_numeric_columns:
                lines.append(f"  • {col}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DynamicSchemaInferencer
# ---------------------------------------------------------------------------

class DynamicSchemaInferencer:
    """Infer constraint rules from any arbitrary pandas DataFrame at runtime.

    No hard-coded column names are used.  The inferencer relies entirely on
    regex-based heuristics applied to the observed column names and their dtypes.

    Example:
        >>> df = pd.read_csv("hospital_beds.csv")
        >>> inferencer = DynamicSchemaInferencer(df)
        >>> profile = inferencer.infer()
        >>> print(profile.as_agent_summary())
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Store the DataFrame to be analysed.

        Args:
            df: Any pandas DataFrame.  Can have arbitrarily named columns;
                the inferencer will operate on whatever columns are present.

        Raises:
            ValueError: If ``df`` is empty or has no columns.
        """
        if df.empty or df.columns.empty:
            raise ValueError(
                "DynamicSchemaInferencer received an empty DataFrame. "
                "Cannot infer schema from zero rows or zero columns."
            )
        self._df = df

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def infer(self) -> SchemaProfile:
        """Run the full inference pipeline and return a ``SchemaProfile``.

        Steps:
            1. Catalogue all column dtypes.
            2. Identify the primary key column.
            3. Separate numeric vs non-numeric columns.
            4. Discover constraint pairs among numeric columns.
            5. Identify unconstrained numeric columns (no upper-bound partner).

        Returns:
            A fully populated ``SchemaProfile`` instance.
        """
        column_dtypes: dict[str, str] = {
            col: str(self._df[col].dtype)
            for col in self._df.columns
        }

        pk_column: str = self._detect_primary_key(list(self._df.columns))

        numeric_columns: list[str] = [
            col for col in self._df.columns
            if pd.api.types.is_numeric_dtype(self._df[col])
        ]

        constraint_rules, limit_columns_used = self._detect_constraint_pairs(
            numeric_columns
        )

        # Any numeric column not participating in a rule as the *mutable* side
        # is still subject to the non-negative floor constraint.
        mutable_cols_in_rules: set[str] = {r.mutable_column for r in constraint_rules}
        unconstrained: list[str] = [
            c for c in numeric_columns
            if c not in mutable_cols_in_rules and c not in limit_columns_used
        ]

        profile = SchemaProfile(
            primary_key_column=pk_column,
            numeric_columns=numeric_columns,
            constraint_rules=constraint_rules,
            column_dtypes=column_dtypes,
            unconstrained_numeric_columns=unconstrained,
        )

        logger.info(
            "Schema inference complete | pk='%s' | numeric_cols=%d | "
            "constraint_rules=%d | unconstrained=%d",
            pk_column,
            len(numeric_columns),
            len(constraint_rules),
            len(unconstrained),
        )
        return profile

    # ------------------------------------------------------------------
    # Private: Primary Key Detection
    # ------------------------------------------------------------------

    def _detect_primary_key(self, columns: list[str]) -> str:
        """Heuristically identify the primary key column.

        Strategy:
            1. Try each ``_PK_KEYWORD_PATTERNS`` regex against each column name
               (case-insensitive).  Return the first match.
            2. Fall back to the first non-numeric column if no pattern matches.
            3. Last resort: return the very first column regardless of type.

        Args:
            columns: List of column names from the DataFrame.

        Returns:
            The name of the inferred primary key column.
        """
        lower_map: dict[str, str] = {col.lower(): col for col in columns}

        for pattern in _PK_KEYWORD_PATTERNS:
            for lower_col, orig_col in lower_map.items():
                if re.search(pattern, lower_col):
                    logger.debug("PK detected via pattern '%s': column '%s'", pattern, orig_col)
                    return orig_col

        # Fallback: first non-numeric column
        for col in columns:
            if not pd.api.types.is_numeric_dtype(self._df[col]):
                logger.debug("PK fallback (first non-numeric): column '%s'", col)
                return col

        # Last resort: first column
        logger.warning(
            "Could not detect a primary key column by heuristic. "
            "Defaulting to first column: '%s'",
            columns[0],
        )
        return columns[0]

    # ------------------------------------------------------------------
    # Private: Constraint Pair Detection
    # ------------------------------------------------------------------

    def _detect_constraint_pairs(
        self, numeric_columns: list[str]
    ) -> tuple[list[ConstraintRule], set[str]]:
        """Discover (mutable, limit) column pairs among numeric columns.

        For each candidate numeric column, derive its *stem* (stripped of
        common prefixes/suffixes like ``current_``, ``_level``, etc.) and try
        to match it against every other numeric column using
        ``_LIMIT_SUFFIX_PATTERNS``.

        Args:
            numeric_columns: All numeric column names in the DataFrame.

        Returns:
            A 2-tuple of:
            * ``list[ConstraintRule]`` — all discovered constraint pairs.
            * ``set[str]`` — column names identified as *limit* columns
              (these are excluded from the unconstrained list).
        """
        rules: list[ConstraintRule] = []
        limit_columns_used: set[str] = set()
        # Track which columns have already been assigned as a mutable to avoid
        # mapping the same column to two different limits.
        assigned_mutable: set[str] = set()

        col_lower: dict[str, str] = {c.lower(): c for c in numeric_columns}

        for candidate_lower, candidate_orig in col_lower.items():
            if candidate_orig in assigned_mutable or candidate_orig in limit_columns_used:
                continue

            stem = self._extract_stem(candidate_lower)
            if not stem:
                continue

            limit_col = self._find_limit_column(stem, candidate_lower, col_lower)
            if limit_col and limit_col != candidate_orig:
                rules.append(
                    ConstraintRule(
                        mutable_column=candidate_orig,
                        limit_column=limit_col,
                    )
                )
                assigned_mutable.add(candidate_orig)
                limit_columns_used.add(limit_col)
                logger.debug(
                    "Constraint pair found: '%s' → max '%s'",
                    candidate_orig,
                    limit_col,
                )

        # Second pass: cross-stem pairing for columns with no shared stem word.
        # Handles cases like current_stock/max_capacity or containers_loaded/teu_limit.
        cross_rules = self._detect_cross_stem_pairs(
            numeric_columns=numeric_columns,
            already_assigned_mutable=assigned_mutable,
            already_limit=limit_columns_used,
        )
        for rule in cross_rules:
            rules.append(rule)
            assigned_mutable.add(rule.mutable_column)
            limit_columns_used.add(rule.limit_column)

        return rules, limit_columns_used

    def _detect_cross_stem_pairs(
        self,
        numeric_columns: list[str],
        already_assigned_mutable: set[str],
        already_limit: set[str],
    ) -> list[ConstraintRule]:
        """Second-pass: pair mutable columns with limit columns via data evidence.

        This pass handles cases where the mutable and limit columns share NO
        common stem (e.g., ``containers_loaded`` / ``teu_limit`` or
        ``current_stock`` / ``max_capacity``).  It:

        1. Identifies unpaired numeric columns that look like a limit column
           by name (matching ``_LIMIT_COLUMN_PATTERN``).
        2. For each unmatched mutable candidate, computes the fraction of rows
           where ``mutable_value <= limit_candidate_value``.
        3. If this fraction exceeds ``_CROSS_STEM_MIN_RATIO``, records the pair.

        Args:
            numeric_columns: All numeric column names in the DataFrame.
            already_assigned_mutable: Mutable columns already paired in pass 1.
            already_limit: Limit columns already used in pass 1.

        Returns:
            Additional ``ConstraintRule`` pairs discovered by cross-stem matching.
        """
        cross_rules: list[ConstraintRule] = []

        # Candidate limit columns: look like a limit AND not yet used
        candidate_limits = [
            col for col in numeric_columns
            if col not in already_limit
            and _LIMIT_COLUMN_PATTERN.search(col)
        ]

        if not candidate_limits:
            return cross_rules

        # Candidate mutable columns: not already assigned AND don't look like limits
        candidate_mutables = [
            col for col in numeric_columns
            if col not in already_assigned_mutable
            and col not in already_limit
            and not _LIMIT_COLUMN_PATTERN.search(col)
        ]

        used_limits: set[str] = set()
        used_mutables: set[str] = set()

        for mut_col in candidate_mutables:
            if mut_col in used_mutables:
                continue

            best_limit: str | None = None
            best_ratio: float = _CROSS_STEM_MIN_RATIO - 0.001  # min to beat

            for lim_col in candidate_limits:
                if lim_col in used_limits:
                    continue
                # Data-driven validation: check mutable <= limit in most rows
                try:
                    valid_mask = self._df[mut_col].notna() & self._df[lim_col].notna()
                    n_valid = valid_mask.sum()
                    if n_valid == 0:
                        continue
                    ratio = float(
                        (self._df.loc[valid_mask, mut_col] <= self._df.loc[valid_mask, lim_col]).sum()
                    ) / n_valid
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_limit = lim_col
                except (TypeError, KeyError):
                    continue

            if best_limit is not None:
                cross_rules.append(ConstraintRule(
                    mutable_column=mut_col,
                    limit_column=best_limit,
                ))
                used_mutables.add(mut_col)
                used_limits.add(best_limit)
                logger.debug(
                    "Cross-stem pair: '%s' → limit '%s' (ratio=%.2f)",
                    mut_col, best_limit, best_ratio,
                )

        return cross_rules

    def _extract_stem(self, col_lower: str) -> str:
        """Strip common level/current prefixes and suffixes to get the core stem.

        Examples:
            ``"current_stock"``      → ``"stock"``
            ``"currently_admitted"`` → ``"admitted"``
            ``"beds_occupied"``      → ``"beds"``
            ``"inventory_level"``    → ``"inventory"``
            ``"containers_loaded"``  → ``"containers"``
            ``"active_compute_tb"``  → ``"compute"``

        Args:
            col_lower: Lower-cased column name.

        Returns:
            The extracted stem string, or the original string if no known
            prefix/suffix was found.  Returns empty string if the column is
            already a "limit" style name (``max_``, ``_capacity``, etc.)
            to prevent reverse-mapping.
        """
        # Skip columns that look like they ARE the limit side already
        limit_indicators = (
            "max", "maximum", "limit", "capacity", "cap",
            "ceiling", "upper", "total", "size", "allowance",
        )
        parts = re.split(r"[_\s]", col_lower)
        if parts[0] in limit_indicators or parts[-1] in limit_indicators:
            return ""

        # Strip known level/current tokens from start
        strip_prefixes = (
            "currently", "current", "curr",
            "used", "occupied", "in_use", "active",
        )
        stem = col_lower
        for prefix in strip_prefixes:
            stem = re.sub(rf"^{prefix}[_\s]?", "", stem)

        # Strip known level/current tokens from end
        strip_suffixes = (
            "level", "used", "occupied", "in_use", "on_hand",
            "qty", "quantity", "count", "loaded", "admitted",
            "_tb", "_gb", "_kb",  # unit suffixes for compute resources
        )
        for suffix in strip_suffixes:
            stem = re.sub(rf"[_\s]?{re.escape(suffix)}$", "", stem)

        return stem.strip("_").strip()

    def _find_limit_column(
        self,
        stem: str,
        candidate_lower: str,
        col_lower_map: dict[str, str],
    ) -> str | None:
        """Try each limit-column pattern with the given stem against the column pool.

        Args:
            stem: The extracted core stem of the candidate mutable column.
            candidate_lower: Lower-cased name of the mutable candidate column
                (used to avoid self-matching).
            col_lower_map: Dict of ``{lower_name: original_name}`` for all
                numeric columns.

        Returns:
            The *original-case* column name of the matched limit column, or
            ``None`` if no match was found.
        """
        if not stem:
            return None

        escaped_stem = re.escape(stem)
        for pattern_template in _LIMIT_SUFFIX_PATTERNS:
            pattern = pattern_template.format(stem=escaped_stem)
            for lower_col, orig_col in col_lower_map.items():
                if lower_col == candidate_lower:
                    continue  # never self-match
                if re.fullmatch(pattern, lower_col, flags=re.IGNORECASE):
                    return orig_col
        return None
