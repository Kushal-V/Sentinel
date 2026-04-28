"""
tests/test_analyst_structured.py
================================
Sentinel QA Suite — Analyst structured-output regression tests (Item I2).

These tests pin down the contract that replaced the fragile free-text regex
parser in ``AgentOrchestrator.run_analyst``:

1. ``with_structured_output(AnalystReview)`` is used to bind the LLM, so the
   Analyst is forced to emit a Pydantic-validated schema rather than prose
   that downstream regex could miss or misinterpret.
2. Valid trust-delta updates are forwarded to ``FactoryDataManager.update_trust_score``.
3. Updates referencing an unknown agent (one that does not appear in the
   live trust roster) are skipped with a warning — they MUST NOT be applied
   and MUST NOT raise.
4. An empty updates list is a no-op (zero writer calls, no error).
5. Pydantic schema validation rejects deltas outside [-0.15, 0.15] at
   construction time.
6. The writer-side clamp (``MIN_TRUST_SCORE`` / ``MAX_TRUST_SCORE``) still
   applies — the schema bounds do not bypass the global clamp.

Usage::

    pytest tests/test_analyst_structured.py -v
"""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Silently load .env so ChatGroq doesn't bail at import; tests still mock it.
from dotenv import load_dotenv as _load_dotenv  # noqa: E402

_load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)


# ===========================================================================
# Helper: build a non-empty transaction log so run_analyst doesn't short-circuit
# ===========================================================================

def _make_transaction_log() -> pd.DataFrame:
    """Return a small, non-empty transaction log so ``run_analyst`` proceeds.

    The exact contents do not matter to these tests — the LLM is mocked. We
    just need ``log_df.empty`` to be False so ``run_analyst`` does not
    short-circuit before attempting structured output.
    """
    return pd.DataFrame([
        {
            "transaction_id": "TX-1",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "event_id": "EVT-1",
            "agent_id": "maker",
            "action_schema": "{}",
            "financial_impact": -1500.0,
            "sandbox_approved": True,
        },
        {
            "transaction_id": "TX-2",
            "timestamp": "2026-01-01T00:05:00+00:00",
            "event_id": "EVT-2",
            "agent_id": "mover",
            "action_schema": "{}",
            "financial_impact": -8000.0,
            "sandbox_approved": False,
        },
    ])


def _trust_scores_roster() -> dict[str, Any]:
    """A minimal, canonical trust-scores dict matching the real on-disk shape."""
    return {
        "global_metrics": {
            "routing_threshold": 0.50,
            "last_analysis_run": "2026-01-01T00:00:00+00:00",
        },
        "agents": {
            "maker":  {"trust_score": 0.90, "preferred_fallback": "mover",  "total_decisions": 10, "recent_penalties": 1},
            "mover":  {"trust_score": 0.80, "preferred_fallback": "keeper", "total_decisions": 12, "recent_penalties": 2},
            "keeper": {"trust_score": 0.85, "preferred_fallback": "maker",  "total_decisions":  8, "recent_penalties": 0},
        },
    }


# ===========================================================================
# Pydantic schema validation tests (no LLM, no orchestrator instantiation)
# ===========================================================================

class TestTrustDeltaUpdateSchema(unittest.TestCase):
    """Direct Pydantic-level guarantees on the new schemas."""

    def test_delta_within_bounds_constructs_cleanly(self) -> None:
        """Deltas inside [-0.15, 0.15] must construct successfully."""
        from src.agents.orchestrator import TrustDeltaUpdate

        for delta in (-0.15, -0.05, 0.0, 0.05, 0.15):
            upd = TrustDeltaUpdate(agent_id="maker", delta=delta, reasoning="ok")
            self.assertAlmostEqual(upd.delta, delta)

    def test_delta_above_upper_bound_raises_validation_error(self) -> None:
        """delta=0.5 must raise ValidationError (le=0.15)."""
        from src.agents.orchestrator import TrustDeltaUpdate

        with self.assertRaises(ValidationError):
            TrustDeltaUpdate(agent_id="x", delta=0.5, reasoning="y")

    def test_delta_below_lower_bound_raises_validation_error(self) -> None:
        """delta=-0.5 must raise ValidationError (ge=-0.15)."""
        from src.agents.orchestrator import TrustDeltaUpdate

        with self.assertRaises(ValidationError):
            TrustDeltaUpdate(agent_id="x", delta=-0.5, reasoning="y")

    def test_analyst_review_default_empty_updates(self) -> None:
        """AnalystReview with no updates must default to an empty list."""
        from src.agents.orchestrator import AnalystReview

        review = AnalystReview(summary="nothing to do")
        self.assertEqual(review.updates, [])
        self.assertEqual(review.summary, "nothing to do")


# ===========================================================================
# run_analyst() behavioural tests — mock LLM, mock data manager
# ===========================================================================

class TestRunAnalystStructured(unittest.TestCase):
    """Tests the new ``run_analyst`` contract with mocked LLM and data manager.

    Mocks ``ChatGroq`` so no real API call is made. The mocked LLM's
    ``with_structured_output(AnalystReview)`` returns a stub LLM whose
    ``invoke`` returns a real ``AnalystReview`` instance — exercising the
    real downstream loop in ``run_analyst``.
    """

    @patch("src.agents.orchestrator.ChatGroq")
    def setUp(self, mock_groq_class) -> None:  # type: ignore[override]
        # Mock ChatGroq so __init__ does not require an API key
        mock_llm_instance = MagicMock()
        mock_groq_class.return_value = mock_llm_instance
        mock_llm_instance.with_structured_output.return_value = mock_llm_instance
        mock_llm_instance.bind_tools.return_value = mock_llm_instance

        # Mock FactoryDataManager so no disk I/O happens
        self.mock_manager = MagicMock()
        self.mock_manager.get_transaction_log.return_value = _make_transaction_log()
        self.mock_manager.get_trust_scores.return_value = _trust_scores_roster()
        # Default writer behaviour: echo back a sane updated entry
        self.mock_manager.update_trust_score.return_value = {
            "trust_score": 0.85,
            "total_decisions": 11,
            "recent_penalties": 1,
            "preferred_fallback": "mover",
        }

        from src.agents.orchestrator import AgentOrchestrator
        self.orchestrator = AgentOrchestrator(data_manager=self.mock_manager)

        # Capture the same MagicMock that ChatGroq() returned so tests can
        # rebind .invoke() / .with_structured_output() per scenario.
        self.mock_llm = mock_llm_instance

    # -------------------------- helper --------------------------

    def _bind_structured_review(self, review_obj: Any) -> MagicMock:
        """Rebind the analyst's structured-output chain to return ``review_obj``.

        Returns the stub chain mock so callers can assert call counts.
        """
        stub_chain = MagicMock()
        stub_chain.invoke.return_value = review_obj
        # The orchestrator calls self._analyst_llm.with_structured_output(AnalystReview)
        # so we must intercept it on the analyst LLM specifically.
        self.orchestrator._analyst_llm = MagicMock()
        self.orchestrator._analyst_llm.with_structured_output.return_value = stub_chain
        return stub_chain

    # -------------------------- tests ---------------------------

    def test_two_valid_plus_one_unknown_agent(self) -> None:
        """Two updates for known agents apply; one unknown agent_id is skipped + warned."""
        from src.agents.orchestrator import AnalystReview, TrustDeltaUpdate

        review = AnalystReview(
            updates=[
                TrustDeltaUpdate(agent_id="maker", delta=0.05, reasoning="cost-efficient mitigation"),
                TrustDeltaUpdate(agent_id="mover", delta=-0.10, reasoning="sandbox rejections"),
                TrustDeltaUpdate(agent_id="ghost", delta=0.10, reasoning="hallucinated agent"),
            ],
            summary="Maker rewarded; Mover penalised; ghost ignored.",
        )
        stub_chain = self._bind_structured_review(review)

        with self.assertLogs("src.agents.orchestrator", level="WARNING") as captured:
            result_summary = self.orchestrator.run_analyst(chat_history=[])

        # The structured chain was invoked exactly once
        stub_chain.invoke.assert_called_once()

        # Exactly two writer calls — the unknown agent must have been skipped
        self.assertEqual(self.mock_manager.update_trust_score.call_count, 2)

        actual_calls = {
            call.kwargs.get("agent_id", call.args[0] if call.args else None):
                call.kwargs.get("score_delta", call.args[1] if len(call.args) > 1 else None)
            for call in self.mock_manager.update_trust_score.call_args_list
        }
        self.assertIn("maker", actual_calls)
        self.assertIn("mover", actual_calls)
        self.assertNotIn("ghost", actual_calls)
        self.assertAlmostEqual(actual_calls["maker"], 0.05)
        self.assertAlmostEqual(actual_calls["mover"], -0.10)

        # A warning was logged for the unknown agent
        warning_messages = "\n".join(captured.output)
        self.assertIn("ghost", warning_messages)

        # The returned summary is the AnalystReview.summary string (UI contract)
        self.assertEqual(result_summary, "Maker rewarded; Mover penalised; ghost ignored.")

    def test_empty_updates_list_is_noop(self) -> None:
        """No updates → zero writer calls, no error, summary returned."""
        from src.agents.orchestrator import AnalystReview

        review = AnalystReview(updates=[], summary="All agents performing within tolerance.")
        stub_chain = self._bind_structured_review(review)

        result = self.orchestrator.run_analyst(chat_history=[])

        stub_chain.invoke.assert_called_once()
        self.assertEqual(self.mock_manager.update_trust_score.call_count, 0)
        self.assertEqual(result, "All agents performing within tolerance.")

    def test_unknown_agent_only_no_writes(self) -> None:
        """If every update references an unknown agent, no writes happen."""
        from src.agents.orchestrator import AnalystReview, TrustDeltaUpdate

        review = AnalystReview(
            updates=[
                TrustDeltaUpdate(agent_id="phantom", delta=0.10, reasoning="not real"),
                TrustDeltaUpdate(agent_id="wraith",  delta=-0.10, reasoning="also not real"),
            ],
            summary="Hallucinated names skipped.",
        )
        self._bind_structured_review(review)

        with self.assertLogs("src.agents.orchestrator", level="WARNING") as captured:
            result = self.orchestrator.run_analyst(chat_history=[])

        self.assertEqual(self.mock_manager.update_trust_score.call_count, 0)
        joined = "\n".join(captured.output)
        self.assertIn("phantom", joined)
        self.assertIn("wraith", joined)
        self.assertEqual(result, "Hallucinated names skipped.")

    def test_empty_transaction_log_short_circuits(self) -> None:
        """If the ledger is empty, run_analyst must short-circuit without LLM calls."""
        self.mock_manager.get_transaction_log.return_value = pd.DataFrame()
        # Bind a chain that would scream if it were ever called.
        sentinel_chain = MagicMock()
        sentinel_chain.invoke.side_effect = AssertionError("LLM must not be invoked on empty ledger")
        self.orchestrator._analyst_llm = MagicMock()
        self.orchestrator._analyst_llm.with_structured_output.return_value = sentinel_chain

        result = self.orchestrator.run_analyst(chat_history=[])

        sentinel_chain.invoke.assert_not_called()
        self.assertEqual(self.mock_manager.update_trust_score.call_count, 0)
        self.assertIn("nothing to evaluate", result.lower())

    def test_writer_clamping_is_not_bypassed(self) -> None:
        """The Pydantic bound is [-0.15, 0.15], but the writer's MIN/MAX clamp must still run.

        We can't construct an out-of-range delta via the schema (Pydantic
        rejects it — covered by ``TestTrustDeltaUpdateSchema``). Instead, we
        verify that ``update_trust_score`` is called with the unmodified
        in-range delta from the schema — i.e., ``run_analyst`` does not
        re-clamp prematurely or otherwise bypass the writer.

        The writer itself (``FactoryDataManager.update_trust_score``)
        applies the global ``[MIN_TRUST_SCORE, MAX_TRUST_SCORE]`` clamp.
        """
        from src.agents.orchestrator import AnalystReview, TrustDeltaUpdate

        review = AnalystReview(
            updates=[TrustDeltaUpdate(agent_id="keeper", delta=0.15, reasoning="max boost")],
            summary="Keeper rewarded at the schema bound.",
        )
        self._bind_structured_review(review)

        self.orchestrator.run_analyst(chat_history=[])

        self.mock_manager.update_trust_score.assert_called_once()
        call = self.mock_manager.update_trust_score.call_args
        delta_passed = call.kwargs.get("score_delta", call.args[1] if len(call.args) > 1 else None)
        # The exact in-range value flows through to the writer.
        self.assertAlmostEqual(delta_passed, 0.15)

    def test_llm_failure_returns_error_string(self) -> None:
        """If the LLM raises, run_analyst must return a graceful error string."""
        stub_chain = MagicMock()
        stub_chain.invoke.side_effect = RuntimeError("Groq is down")
        self.orchestrator._analyst_llm = MagicMock()
        self.orchestrator._analyst_llm.with_structured_output.return_value = stub_chain

        result = self.orchestrator.run_analyst(chat_history=[])

        self.assertIn("Analyst execution failed", result)
        # No writes attempted on failure
        self.assertEqual(self.mock_manager.update_trust_score.call_count, 0)


# ===========================================================================
# Entry point for direct execution
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-q"])
