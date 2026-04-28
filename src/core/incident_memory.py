"""Incident memory: TF-IDF semantic search over transaction history.

Per-workspace persistent index. On each crisis, the Analyst (or
specialists via tool) can query for past incidents semantically similar
to the current situation, expanding context beyond the 50-row CSV
truncation in orchestrator.run_analyst.

Design notes
------------
* Persistence is plain JSONL at ``<workspace>/incidents/store.jsonl``,
  one record per line.  Cheap to inspect and diff in Git or by humans.
* Index is rebuilt lazily on the next ``query`` call after any ``add``.
  This is fine for sub-10k record stores; swap to chroma/qdrant via the
  same interface for larger scale.
* Multi-tenant by construction: one ``IncidentMemory`` per workspace
  directory.  No global state; no shared vectorizer between workspaces.
* Description text is built generically from the transaction dict so
  this module is agnostic to whatever industry schema the workspace
  happens to be using (factory, hospital, retail, ...).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_LOG = logging.getLogger(__name__)


@dataclass
class IncidentRecord:
    """A single past-incident record stored in the semantic index.

    Attributes:
        transaction_id: Unique identifier copied from the source transaction.
        timestamp: ISO-8601 timestamp string copied from the source transaction.
        description: Natural-language summary built from the transaction
            fields; this is what the TF-IDF vectorizer fits on.
        metadata: The full original transaction dict, preserved verbatim
            for downstream consumers (analyst, debug tooling, etc.).
    """

    transaction_id: str
    timestamp: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _coerce_str(value: Any) -> str:
    """Best-effort string coercion for description fields."""
    if value is None:
        return ""
    return str(value)


def _build_description(transaction: dict[str, Any]) -> str:
    """Compose a generic searchable description from a transaction dict.

    The text is intentionally schema-agnostic — we walk every key/value
    pair on the dict and emit ``key=value`` segments for any scalar
    value.  This keeps the module reusable across industries (factory,
    hospital, retail, etc.) without hardcoding column names like
    ``sku`` or ``delta``.

    Nested ``action_schema`` payloads (when present as a dict or as a
    JSON-encoded string) are flattened so their keys also contribute
    to the searchable text.

    Args:
        transaction: Arbitrary dict describing a logged transaction.

    Returns:
        A single-line ``" | "``-joined string suitable for TF-IDF input.
    """
    parts: list[str] = []
    for key, value in transaction.items():
        # Special-case the action_schema field: it commonly holds the
        # most diagnostic information (justification, target column,
        # rejection reason).  Flatten one level so its contents become
        # first-class search terms.
        if key == "action_schema":
            payload: Any = value
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except (json.JSONDecodeError, ValueError):
                    payload = None
            if isinstance(payload, dict):
                for sub_k, sub_v in payload.items():
                    if isinstance(sub_v, (str, int, float, bool)):
                        parts.append(f"{sub_k}={_coerce_str(sub_v)}")
            elif isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}={_coerce_str(value)}")
            continue

        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}={_coerce_str(value)}")
    return " | ".join(parts)


class IncidentMemory:
    """TF-IDF semantic store over transaction descriptions.

    Persists records as JSONL at ``<workspace>/incidents/store.jsonl``.
    Index rebuilt lazily on query (cheap for <10k records). For larger
    scale swap to chroma/qdrant via the same interface.
    """

    def __init__(self, workspace_path: Path) -> None:
        """Initialize and load any pre-existing JSONL store.

        Args:
            workspace_path: Filesystem directory of the workspace this
                memory belongs to.  An ``incidents/`` subdirectory is
                created underneath if missing.
        """
        self._workspace = Path(workspace_path)
        self._store_dir = self._workspace / "incidents"
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._store_file = self._store_dir / "store.jsonl"

        self._records: list[IncidentRecord] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: Any = None
        # ``True`` whenever new records have been appended since the
        # last ``_build_index`` call — triggers a lazy rebuild on query.
        self._dirty: bool = True
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, transaction: dict[str, Any]) -> None:
        """Append a transaction record to the index.

        The on-disk JSONL is updated synchronously.  The in-memory
        TF-IDF matrix is invalidated; it will be rebuilt on the next
        ``query`` call.

        Args:
            transaction: Dict with at minimum a ``transaction_id``
                and ``timestamp`` field.  Any additional scalar fields
                contribute to the searchable description.
        """
        if not isinstance(transaction, dict):
            _LOG.warning("IncidentMemory.add ignored non-dict input: %r", transaction)
            return

        description = _build_description(transaction)
        record = IncidentRecord(
            transaction_id=_coerce_str(transaction.get("transaction_id", "")),
            timestamp=_coerce_str(transaction.get("timestamp", "")),
            description=description,
            metadata={k: v for k, v in transaction.items()},
        )

        with self._lock:
            self._records.append(record)
            self._dirty = True
            try:
                with self._store_file.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(record), default=str) + "\n")
            except OSError:
                _LOG.exception("Failed to persist incident record to %s", self._store_file)

    def query(self, crisis_summary: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the ``top_k`` records most semantically similar to the input.

        Args:
            crisis_summary: Free-text description of the current crisis.
            top_k: Maximum number of records to return.  Clamped to
                ``[1, len(records)]``.

        Returns:
            List of dicts ``{"description", "similarity", "metadata",
            "timestamp", "transaction_id"}``, ordered by descending
            similarity.  Empty list if no records exist or the query
            string is empty/whitespace.
        """
        if not crisis_summary or not crisis_summary.strip():
            return []

        with self._lock:
            if not self._records:
                return []

            if self._dirty or self._vectorizer is None or self._matrix is None:
                self._build_index()

            # _build_index can still leave a None vectorizer if every
            # record's description was empty (vocabulary error).  Guard.
            if self._vectorizer is None or self._matrix is None:
                return []

            try:
                query_vec = self._vectorizer.transform([crisis_summary])
            except ValueError:
                # e.g., empty vocabulary mismatch.  Bail gracefully.
                return []

            sims = cosine_similarity(query_vec, self._matrix)[0]
            k = max(1, min(int(top_k), len(self._records)))
            # ``argsort`` ascending → take the last k and reverse for descending.
            top_idx = np.argsort(sims)[-k:][::-1]

            results: list[dict[str, Any]] = []
            for idx in top_idx:
                rec = self._records[int(idx)]
                results.append(
                    {
                        "transaction_id": rec.transaction_id,
                        "timestamp": rec.timestamp,
                        "description": rec.description,
                        "similarity": float(sims[int(idx)]),
                        "metadata": dict(rec.metadata),
                    }
                )
            return results

    def count(self) -> int:
        """Return the number of records currently indexed."""
        with self._lock:
            return len(self._records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Restore records from JSONL on disk (best-effort)."""
        if not self._store_file.exists():
            return
        loaded = 0
        try:
            with self._store_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        _LOG.warning("Skipping malformed JSONL line in %s", self._store_file)
                        continue
                    rec = IncidentRecord(
                        transaction_id=_coerce_str(payload.get("transaction_id", "")),
                        timestamp=_coerce_str(payload.get("timestamp", "")),
                        description=_coerce_str(payload.get("description", "")),
                        metadata=dict(payload.get("metadata") or {}),
                    )
                    self._records.append(rec)
                    loaded += 1
        except OSError:
            _LOG.exception("Failed to read incident store at %s", self._store_file)
            return
        if loaded:
            self._dirty = True
            _LOG.info("IncidentMemory loaded %d records from %s", loaded, self._store_file)

    def _build_index(self) -> None:
        """Fit TfidfVectorizer on all descriptions. Called lazily."""
        descriptions = [r.description for r in self._records]
        # Filter out completely empty docs so the vectorizer doesn't
        # raise ``empty vocabulary``; replace with a single space token.
        sanitized = [d if d.strip() else " " for d in descriptions]
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                stop_words=None,
            )
            matrix = vectorizer.fit_transform(sanitized)
        except ValueError:
            # All documents stop-worded out / empty vocabulary.  Mark
            # the index as built (so we don't loop) but unusable.
            _LOG.warning("IncidentMemory could not fit vectorizer (empty vocabulary).")
            self._vectorizer = None
            self._matrix = None
            self._dirty = False
            return

        self._vectorizer = vectorizer
        self._matrix = matrix
        self._dirty = False
        _LOG.debug("IncidentMemory rebuilt index over %d records.", len(descriptions))
