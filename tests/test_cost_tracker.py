import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from src.observability.cost_tracker import CostTracker, BudgetExceededError, CostSnapshot


def _fake_response(input_tokens=10, output_tokens=20, model="test-model"):
    """Build a minimal object that _extract_usage can read."""
    class _LLMResult:
        def __init__(self):
            self.generations = []
            self.llm_output = {
                "token_usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
                "model_name": model,
            }
    return _LLMResult()


def test_snapshot_starts_zero():
    t = CostTracker(max_usd=1.0, rates={"default": (0.001, 0.002)})
    s = t.snapshot()
    assert s.input_tokens == 0 and s.output_tokens == 0 and s.estimated_usd == 0.0


def test_accumulates_tokens_and_cost():
    t = CostTracker(max_usd=1.0, rates={"test-model": (0.001, 0.002)})
    t.on_llm_end(_fake_response(input_tokens=1000, output_tokens=2000, model="test-model"))
    s = t.snapshot()
    assert s.input_tokens == 1000
    assert s.output_tokens == 2000
    # cost = 1.0 * 0.001 + 2.0 * 0.002 = 0.001 + 0.004 = 0.005
    assert abs(s.estimated_usd - 0.005) < 1e-9
    assert s.n_calls == 1


def test_unknown_model_uses_default_rate():
    t = CostTracker(max_usd=1.0, rates={"default": (0.0001, 0.0002)})
    t.on_llm_end(_fake_response(input_tokens=1000, output_tokens=1000, model="unseen"))
    s = t.snapshot()
    assert abs(s.estimated_usd - (0.0001 + 0.0002)) < 1e-9


def test_budget_exceeded_raises_on_next_call():
    t = CostTracker(max_usd=0.001, rates={"m": (0.01, 0.01)})
    t.on_llm_end(_fake_response(input_tokens=1000, output_tokens=1000, model="m"))  # ~0.02
    assert t.is_over_budget()
    with pytest.raises(BudgetExceededError):
        t.on_llm_start(serialized={}, prompts=[])


def test_enforce_false_does_not_raise():
    t = CostTracker(max_usd=0.001, rates={"m": (0.01, 0.01)}, enforce=False)
    t.on_llm_end(_fake_response(input_tokens=1000, output_tokens=1000, model="m"))
    # Should not raise
    t.on_llm_start(serialized={}, prompts=[])
    t.on_chat_model_start(serialized={}, messages=[])


def test_reset_clears_counters():
    t = CostTracker(max_usd=1.0, rates={"m": (0.001, 0.001)})
    t.on_llm_end(_fake_response(model="m"))
    t.reset()
    s = t.snapshot()
    assert s.input_tokens == 0 and s.estimated_usd == 0.0


def test_thread_safety_concurrent_accumulation():
    import threading
    t = CostTracker(max_usd=1000.0, rates={"m": (0.001, 0.001)})
    def worker():
        for _ in range(100):
            t.on_llm_end(_fake_response(input_tokens=10, output_tokens=10, model="m"))
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for th in threads: th.start()
    for th in threads: th.join()
    s = t.snapshot()
    assert s.n_calls == 500
    assert s.input_tokens == 5000


def test_by_model_breakdown():
    t = CostTracker(max_usd=10.0, rates={"a": (0.001, 0.002), "b": (0.005, 0.006)})
    t.on_llm_end(_fake_response(input_tokens=1000, output_tokens=1000, model="a"))
    t.on_llm_end(_fake_response(input_tokens=500, output_tokens=500, model="b"))
    s = t.snapshot()
    assert "a" in s.by_model and "b" in s.by_model
    assert s.by_model["a"]["calls"] == 1
    assert s.by_model["b"]["calls"] == 1


def test_extracts_from_usage_metadata_fallback():
    """If llm_output is empty but generation has usage_metadata, use it."""
    from langchain_core.messages import AIMessage

    class _Gen:
        def __init__(self, msg): self.message = msg
    class _Result:
        def __init__(self, gens):
            self.generations = gens
            self.llm_output = {}

    msg = AIMessage(content="hi", usage_metadata={"input_tokens": 30, "output_tokens": 20, "total_tokens": 50})
    res = _Result([[_Gen(msg)]])
    t = CostTracker(max_usd=1.0, rates={"default": (0.001, 0.002)})
    t.on_llm_end(res)
    s = t.snapshot()
    assert s.input_tokens == 30
    assert s.output_tokens == 20
