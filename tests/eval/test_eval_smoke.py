"""
tests/eval/test_eval_smoke.py
=============================

Smoke tests for the Sentinel evaluation harness skeleton.

These tests intentionally exercise only the *plumbing*:

* the YAML loader handles missing files gracefully
* ``scenarios.yaml`` ships at least one well-formed scenario
* a mock-mode run produces a populated :class:`ScenarioResult`
* metric helpers return sane values on a populated result

All tests are tagged with ``@pytest.mark.eval`` so they are opt-in via
``pytest -m eval`` and never run during the default ``pytest`` invocation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from . import metrics
from .runner import ScenarioResult, load_scenarios, run_scenario


SCENARIOS_PATH = Path(__file__).parent / "scenarios.yaml"


@pytest.mark.eval
def test_loader_handles_missing_file() -> None:
    """Loader must return ``[]`` (not raise) when the file is absent."""
    assert load_scenarios(Path("/does/not/exist.yaml")) == []


@pytest.mark.eval
def test_at_least_one_scenario_in_yaml() -> None:
    """``scenarios.yaml`` ships and contains at least one named entry."""
    scenarios = load_scenarios(SCENARIOS_PATH)
    assert len(scenarios) >= 1
    assert all("name" in s for s in scenarios), (
        "every scenario entry must have a 'name' field"
    )
    # All scenarios must declare a crisis with a `type` key.
    assert all(
        isinstance(s.get("crisis"), dict) and "type" in s["crisis"]
        for s in scenarios
    )


@pytest.mark.eval
def test_smoke_run_with_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the first YAML scenario in mock mode and inspect the result."""
    del monkeypatch  # included for signature parity with the spec
    scenarios = load_scenarios(SCENARIOS_PATH)
    assert scenarios, "scenarios.yaml must contain at least one scenario"

    first = scenarios[0]
    result = run_scenario(first, orchestrator=None, mock_mode=True)

    assert isinstance(result, ScenarioResult)
    assert result.name == first["name"]
    assert result.passed, f"scenario {first['name']!r} failed: {result.failures}"
    assert result.actual["selected_agent"] in {"maker", "mover", "keeper"}
    assert len(result.actual["tool_calls"]) >= 3


@pytest.mark.eval
def test_trust_override_scenario_flips_routing() -> None:
    """The trust-override scenario must end up on the configured fallback."""
    scenarios = load_scenarios(SCENARIOS_PATH)
    target = next(
        (s for s in scenarios if s["name"] == "trust_override_fires_when_low_score"),
        None,
    )
    assert target is not None, "trust override scenario must exist"

    result = run_scenario(target, mock_mode=True)
    assert result.passed, result.failures
    assert result.actual["trust_override_applied"] is True
    assert result.actual["selected_agent"] == target["expectations"]["selected_agent"]


@pytest.mark.eval
def test_metric_helpers_return_sane_values() -> None:
    """``correctness``, ``tool_call_count``, and ``latency_ms`` must be usable."""
    scenarios = load_scenarios(SCENARIOS_PATH)
    first = scenarios[0]
    result = run_scenario(first, mock_mode=True)

    assert 0.0 <= metrics.correctness(first, result.actual) <= 1.0
    assert metrics.tool_call_count(first, result.actual) >= 3
    assert metrics.latency_ms(first, result.actual) == 0.0


@pytest.mark.eval
def test_live_mode_is_a_stub() -> None:
    """Live mode is intentionally not wired up in the skeleton phase."""
    scenarios = load_scenarios(SCENARIOS_PATH)
    result = run_scenario(scenarios[0], orchestrator=None, mock_mode=False)
    assert result.passed is False
    assert any("live mode" in f for f in result.failures)
