"""Per-session LLM cost tracker.

A LangChain BaseCallbackHandler that counts input/output tokens from
ChatGroq responses, accumulates per-session totals, estimates USD cost,
and raises BudgetExceededError when MAX_LLM_USD_PER_SESSION is breached.

Costs derived from config.MODEL_COST_PER_1K_TOKENS — a dict mapping
model name to (input_per_1k, output_per_1k) tuples.

Thread-safe; one tracker instance per orchestrator instance.
"""
from __future__ import annotations
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler

_LOG = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    """Raised when accumulated session cost exceeds the configured limit."""


@dataclass
class CostSnapshot:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_usd: float = 0.0
    n_calls: int = 0
    by_model: Dict[str, Dict[str, float]] = field(default_factory=dict)


class CostTracker(BaseCallbackHandler):
    """Callback handler that accumulates per-session LLM token usage + cost."""

    def __init__(
        self,
        max_usd: float,
        rates: Dict[str, tuple[float, float]],
        enforce: bool = True,
    ):
        super().__init__()
        self._max_usd = max_usd
        self._rates = rates  # {model_name: (input_per_1k, output_per_1k)}
        self._enforce = enforce
        self._lock = threading.Lock()
        self._snap = CostSnapshot()

    def reset(self) -> None:
        with self._lock:
            self._snap = CostSnapshot()

    def snapshot(self) -> CostSnapshot:
        with self._lock:
            return CostSnapshot(
                input_tokens=self._snap.input_tokens,
                output_tokens=self._snap.output_tokens,
                total_tokens=self._snap.total_tokens,
                estimated_usd=self._snap.estimated_usd,
                n_calls=self._snap.n_calls,
                by_model={k: dict(v) for k, v in self._snap.by_model.items()},
            )

    def is_over_budget(self) -> bool:
        with self._lock:
            return self._snap.estimated_usd >= self._max_usd

    def assert_within_budget(self) -> None:
        if self._enforce and self.is_over_budget():
            raise BudgetExceededError(
                f"Session LLM cost ${self._snap.estimated_usd:.4f} >= cap ${self._max_usd:.2f}. "
                f"Reset via the sidebar or raise MAX_LLM_USD_PER_SESSION."
            )

    # --- BaseCallbackHandler hooks ---

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        # Pre-call budget guard
        self.assert_within_budget()

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self.assert_within_budget()

    def on_llm_end(self, response, **kwargs) -> None:
        # Extract token usage from response
        usage = self._extract_usage(response)
        if usage is None:
            return
        in_tok, out_tok, model = usage
        with self._lock:
            self._snap.input_tokens += in_tok
            self._snap.output_tokens += out_tok
            self._snap.total_tokens += in_tok + out_tok
            self._snap.n_calls += 1
            rates = self._rates.get(model) or self._rates.get("default", (0.0, 0.0))
            cost = (in_tok / 1000.0) * rates[0] + (out_tok / 1000.0) * rates[1]
            self._snap.estimated_usd += cost
            mb = self._snap.by_model.setdefault(model or "unknown", {
                "input_tokens": 0, "output_tokens": 0, "usd": 0.0, "calls": 0
            })
            mb["input_tokens"] += in_tok
            mb["output_tokens"] += out_tok
            mb["usd"] += cost
            mb["calls"] += 1

    def _extract_usage(self, response) -> Optional[tuple[int, int, str]]:
        """Best-effort extraction of (input_tokens, output_tokens, model_name)."""
        try:
            # LangChain LLMResult
            generations = getattr(response, "generations", None)
            llm_output = getattr(response, "llm_output", None) or {}
            usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
            model = llm_output.get("model_name") or llm_output.get("model") or "unknown"
            in_tok = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            out_tok = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            if not in_tok and not out_tok and generations:
                # try to read from generation message metadata
                for gen_list in generations:
                    for gen in gen_list:
                        msg = getattr(gen, "message", None)
                        meta = getattr(msg, "usage_metadata", None) if msg else None
                        if meta:
                            in_tok = int(meta.get("input_tokens", 0)) or in_tok
                            out_tok = int(meta.get("output_tokens", 0)) or out_tok
                            break
            return (in_tok, out_tok, model)
        except Exception:
            _LOG.exception("Failed to extract token usage from LLM response")
            return None
