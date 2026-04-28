"""
tests/eval/metrics.py
=====================

Pluggable metric functions for the Sentinel evaluation harness.

Each metric is a *pure* function with the uniform signature::

    metric(scenario_dict, actual_result_dict) -> number_or_bool

The runner produces ``actual_result_dict`` (see ``ScenarioResult.actual``);
the scenario dict comes straight from ``scenarios.yaml``.  Metric authors
should not mutate either argument.

This is deliberately a small set in the skeleton phase — DeepEval / Ragas
metrics (faithfulness, answer-relevancy, contextual precision) plug in here
later by adding more functions with the same signature.
"""

from __future__ import annotations

from typing import Any


def correctness(scenario: dict[str, Any], actual: dict[str, Any]) -> float:
    """Fraction of scenario expectations that the actual result satisfies.

    Returns a value in ``[0.0, 1.0]``.  An expectation is "satisfied" when
    the corresponding key in ``actual`` matches by ``==``.  Missing keys in
    ``actual`` always count as a miss.

    A scenario with zero expectations returns ``1.0`` (vacuously correct).
    """
    expectations = scenario.get("expectations", {}) or {}
    if not expectations:
        return 1.0

    hits = 0
    total = 0
    for key, expected in expectations.items():
        total += 1
        if key == "min_tool_calls":
            actual_calls = len(actual.get("tool_calls", []))
            if actual_calls >= int(expected):
                hits += 1
            continue
        if key in actual and actual[key] == expected:
            hits += 1

    return hits / total if total else 1.0


def tool_call_count(scenario: dict[str, Any], actual: dict[str, Any]) -> int:
    """Number of tool invocations recorded in the actual result."""
    del scenario  # unused — included for uniform signature
    return len(actual.get("tool_calls", []))


def latency_ms(scenario: dict[str, Any], actual: dict[str, Any]) -> float:
    """Stub latency metric.

    Returns 0.0 in the skeleton phase.  Live-mode runner integrations should
    populate ``actual["latency_ms"]`` and this function will then surface it.
    """
    del scenario
    return float(actual.get("latency_ms", 0.0))
